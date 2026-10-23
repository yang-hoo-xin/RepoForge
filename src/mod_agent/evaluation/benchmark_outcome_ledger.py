"""Read-only aggregation for registered, task-level Product outcome ledgers.

The ledger is append-only at the record level.  A ``run_family_id`` may have
multiple revisions, but only its latest revision contributes to a snapshot.
This lets a later record invalidate or correct an earlier claim without
rewriting the original evidence.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = "benchmark-outcome-ledger.v1"
SUMMARY_SCHEMA_VERSION = "benchmark-outcome-summary.v1"

VALIDITIES = {"valid", "invalid"}
EXECUTION_STATUSES = {"executed", "not_executed", "environment_blocked"}
EXECUTION_ORIGINS = {"product", "developer_assisted"}
INTERVENTION_KINDS = ("developer", "user", "operational")
COST_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "model_calls",
    "provider_sends",
    "tool_calls",
    "source_retrieval_calls",
    "judge_calls",
    "retry_count",
    "elapsed_seconds",
    "human_wait_seconds",
)


# This is part of the public contract, not merely implementation commentary.
# Every rule carries its reason so reports can explain why a record was rejected.
SCHEMA_RULES = (
    {
        "id": "R1",
        "rule": "The ledger and every record use one exact versioned shape.",
        "why": "为什么要有它：缺字段、拼写漂移或混用版本不能被静默解释成零或失败。",
    },
    {
        "id": "R2",
        "rule": "Each registered task has one Product primary run family in every registered arm.",
        "why": "为什么要有它：冻结分母必须是任务×臂槽位，重复运行不能增加独立任务数。",
    },
    {
        "id": "R3",
        "rule": "Run-family identity is immutable and only its greatest revision is current.",
        "why": "为什么要有它：更正应保留历史，同时不能让修订次数膨胀交付或 Repair 分母。",
    },
    {
        "id": "R4",
        "rule": "Validity, execution status, outcome and execution origin are independent fields.",
        "why": "为什么要有它：无效证据、未执行、环境阻塞、失败和开发者辅助是不同事实。",
    },
    {
        "id": "R5",
        "rule": "Only valid executed records can contribute Product outcome metrics.",
        "why": "为什么要有它：无效或未运行记录中的成功声明不能变成产品成绩。",
    },
    {
        "id": "R6",
        "rule": "Delivery and first-attempt success remain nullable and logically ordered.",
        "why": "为什么要有它：未知不能转成失败，首次成功也不能超过最终交付。",
    },
    {
        "id": "R7",
        "rule": "Runtime requirement, execution and pass are recorded separately.",
        "why": "为什么要有它：需要 Runtime 却未运行的任务绝不能被统计为 Runtime 通过。",
    },
    {
        "id": "R8",
        "rule": "Repair eligibility, entry and pass are recorded per run family; only entered repairs form the pass denominator.",
        "why": "为什么要有它：Recovery 分母应是真正进入修订的有效运行族，不是 revision、未进入修订的任务或环境尾迹。",
    },
    {
        "id": "R9",
        "rule": "Developer, user and operational interventions have separate nullable counts.",
        "why": "为什么要有它：三类介入对自主性的含义不同，缺测也不能冒充零介入。",
    },
    {
        "id": "R10",
        "rule": "A complete cost row supplies every cost field; partial rows retain known values.",
        "why": "为什么要有它：完整成本和已知下界必须并存，失败得早不能伪装成节省成本。",
    },
    {
        "id": "R11",
        "rule": "Time-to-delivery and terminal failure stage follow the final delivery outcome.",
        "why": "为什么要有它：失败没有交付耗时，成功也没有终态失败阶段。",
    },
    {
        "id": "R12",
        "rule": "Every revision has exact absolute local evidence references and SHA-256 values.",
        "why": "为什么要有它：快照必须能回到不可混淆的原始证据，路径或内容漂移要失败关闭。",
    },
    {
        "id": "R13",
        "rule": "Final rates require the whole registered primary cohort to be resolved.",
        "why": "为什么要有它：registered snapshot 可以随时看，但十题未完成时不能发布 final rate。",
    },
    {
        "id": "R14",
        "rule": "Final task-arm outcome uses the greatest-repeat valid terminal run family.",
        "why": "为什么要有它：早期无效或环境尾迹不应遮蔽修正结果，早期成功也不能永久遮蔽后续正式回归失败。",
    },
    {
        "id": "R15",
        "rule": "Final intervention rates use valid executed Product attempts; pre-execution events are reported separately.",
        "why": "为什么要有它：无效记录和准备期人工操作不能污染产品自主性，但其实际成本也不能被丢弃。",
    },
)

_RULES = {row["id"]: row for row in SCHEMA_RULES}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_KEYS = {"schema", "registered_task_ids", "records"}
_RECORD_KEYS = {
    "record_id",
    "run_family_id",
    "revision",
    "repeat",
    "task_id",
    "arm",
    "primary",
    "validity",
    "execution_status",
    "execution_origin",
    "status_reason",
    "delivery",
    "first_attempt_success",
    "runtime",
    "repair",
    "interventions",
    "cost",
    "time_to_delivery_seconds",
    "failure_stage",
    "evidence_refs",
}


def _fail(rule_id: str, detail: str) -> None:
    rule = _RULES[rule_id]
    raise ValueError(f"{rule_id}: {detail} {rule['why']}")


def _is_bool_or_none(value: object) -> bool:
    return value is None or type(value) is bool


def _is_nonnegative_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _require_exact_keys(value: object, keys: set[str], rule_id: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(rule_id, f"{label} keys must be exactly {sorted(keys)}.")
    return value


def validate_evidence_reference(reference: object) -> None:
    """Validate reference shape without reading the referenced bytes."""
    value = _require_exact_keys(reference, {"path", "sha256"}, "R12", "Evidence reference")
    path, digest = value["path"], value["sha256"]
    if not isinstance(path, str) or not path or path.startswith(("\\\\", "//")):
        _fail("R12", "Evidence paths must be non-network local strings.")
    if not Path(path).is_absolute():
        _fail("R12", "Evidence paths must be absolute.")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _fail("R12", "Evidence sha256 must be 64 lowercase hexadecimal characters.")


def _validate_measurement(measurement: object, *, label: str) -> None:
    value = _require_exact_keys(measurement, {"required", "count"}, "R9", label)
    required, count = value["required"], value["count"]
    if not _is_bool_or_none(required):
        _fail("R9", f"{label}.required must be true, false or null.")
    if count is not None and not _is_nonnegative_int(count):
        _fail("R9", f"{label}.count must be a nonnegative integer or null.")
    if required is False and count != 0:
        _fail("R9", f"{label} known false requires an explicit zero count.")
    if required is True and (count is None or count < 1):
        _fail("R9", f"{label} known true requires a positive count.")
    if required is None and count is not None:
        _fail("R9", f"{label} unknown cannot carry a count.")


def _validate_record(record: object, registered: set[str]) -> Mapping[str, Any]:
    row = _require_exact_keys(record, _RECORD_KEYS, "R1", "Ledger record")
    for key in ("record_id", "run_family_id", "task_id", "arm"):
        if not isinstance(row[key], str) or not row[key]:
            _fail("R1", f"{key} must be a non-empty string.")
    if row["task_id"] not in registered:
        _fail("R2", f"Unknown task_id {row['task_id']!r}.")
    if type(row["revision"]) is not int or row["revision"] < 1:
        _fail("R3", "revision must be a positive integer.")
    if type(row["repeat"]) is not int or row["repeat"] < 1:
        _fail("R3", "repeat must be a positive integer.")
    if type(row["primary"]) is not bool:
        _fail("R2", "primary must be a boolean.")
    if row["validity"] not in VALIDITIES:
        _fail("R4", f"Unsupported validity {row['validity']!r}.")
    if row["execution_status"] not in EXECUTION_STATUSES:
        _fail("R4", f"Unsupported execution_status {row['execution_status']!r}.")
    if row["execution_origin"] not in EXECUTION_ORIGINS:
        _fail("R4", f"Unsupported execution_origin {row['execution_origin']!r}.")
    reason = row["status_reason"]
    reason_required = (
        row["validity"] == "invalid"
        or row["execution_status"] != "executed"
        or row["delivery"] is None
    )
    if reason_required and (not isinstance(reason, str) or not reason):
        _fail("R4", "Invalid, unexecuted, environment-blocked and nonterminal records require status_reason.")
    if not reason_required and reason is not None:
        _fail("R4", "A valid executed record has no status_reason.")
    if not _is_bool_or_none(row["delivery"]) or not _is_bool_or_none(row["first_attempt_success"]):
        _fail("R6", "delivery and first_attempt_success must be true, false or null.")

    runtime = _require_exact_keys(row["runtime"], {"required", "executed", "passed"}, "R7", "runtime")
    if not all(_is_bool_or_none(runtime[key]) for key in runtime):
        _fail("R7", "Runtime fields must be true, false or null.")
    repair = _require_exact_keys(row["repair"], {"eligible", "attempted", "passed"}, "R8", "repair")
    if not all(_is_bool_or_none(repair[key]) for key in repair):
        _fail("R8", "Repair fields must be true, false or null.")

    interventions = _require_exact_keys(
        row["interventions"], set(INTERVENTION_KINDS), "R9", "interventions"
    )
    for kind in INTERVENTION_KINDS:
        _validate_measurement(interventions[kind], label=f"interventions.{kind}")

    cost = _require_exact_keys(row["cost"], {"complete", *COST_FIELDS}, "R10", "cost")
    if type(cost["complete"]) is not bool:
        _fail("R10", "cost.complete must be a boolean.")
    for key in COST_FIELDS:
        value = cost[key]
        if value is not None and not _is_nonnegative_number(value):
            _fail("R10", f"cost.{key} must be a finite nonnegative number or null.")
        if key not in {"elapsed_seconds", "human_wait_seconds"} and type(value) is float and not value.is_integer():
            _fail("R10", f"cost.{key} must be an integer count when it is known.")
    if cost["complete"] and any(cost[key] is None for key in COST_FIELDS):
        _fail("R10", "A complete cost row cannot omit a cost field.")

    ttd = row["time_to_delivery_seconds"]
    if ttd is not None and not _is_nonnegative_number(ttd):
        _fail("R11", "time_to_delivery_seconds must be finite and nonnegative or null.")
    stage = row["failure_stage"]
    if stage is not None and (not isinstance(stage, str) or not stage):
        _fail("R11", "failure_stage must be a non-empty string or null.")

    references = row["evidence_refs"]
    if not isinstance(references, list) or not references:
        _fail("R12", "Every record requires at least one evidence reference.")
    seen_refs: set[tuple[str, str]] = set()
    for reference in references:
        validate_evidence_reference(reference)
        identity = (reference["path"], reference["sha256"])
        if identity in seen_refs:
            _fail("R12", "A record cannot repeat the same evidence reference.")
        seen_refs.add(identity)

    # Invalid records retain raw outcome/autonomy claims for audit.  Logical
    # outcome rules apply only when evidence validity says the record is usable.
    if row["validity"] == "valid":
        if row["execution_status"] != "executed":
            if row["delivery"] is not None or row["first_attempt_success"] is not None:
                _fail("R5", "A valid non-executed record cannot claim a Product outcome.")
            if repair != {"eligible": None, "attempted": None, "passed": None}:
                _fail("R8", "A valid non-executed record has no Repair outcome.")
            if runtime["required"] is True and not (
                runtime["executed"] is False and runtime["passed"] is False
            ):
                _fail("R7", "Required Runtime on a non-executed task must record not-executed and not-passed.")
        if row["first_attempt_success"] is True and row["delivery"] is not True:
            _fail("R6", "First-attempt success requires final delivery.")
        if runtime["required"] is False and (runtime["executed"] is not False or runtime["passed"] is not None):
            _fail("R7", "Runtime not required must be explicit not-executed with no pass value.")
        if runtime["required"] is None and (runtime["executed"] is not None or runtime["passed"] is not None):
            _fail("R7", "Unknown Runtime applicability cannot imply execution or pass.")
        if runtime["passed"] is True and not (runtime["required"] is True and runtime["executed"] is True):
            _fail("R7", "Runtime pass requires both required=true and executed=true.")
        if runtime["required"] is True and runtime["executed"] is False and runtime["passed"] is not False:
            _fail("R7", "A required but unexecuted Runtime must be recorded as not passed.")
        if row["delivery"] is True and runtime["required"] is True and not (
            runtime["executed"] is True and runtime["passed"] is True
        ):
            _fail("R7", "Delivery cannot pass while required Runtime acceptance did not pass.")
        if repair["eligible"] is False and (repair["attempted"] is not False or repair["passed"] is not None):
            _fail("R8", "Repair ineligible must be explicit not-attempted with no pass value.")
        if repair["eligible"] is None and (repair["attempted"] is not None or repair["passed"] is not None):
            _fail("R8", "Unknown Repair eligibility cannot imply an attempt or pass.")
        if repair["passed"] is True and not (
            repair["eligible"] is True
            and repair["attempted"] is True
            and row["delivery"] is True
            and row["first_attempt_success"] is False
        ):
            _fail("R8", "Repair pass requires an eligible attempted recovery after first-attempt failure and delivery.")
        if row["first_attempt_success"] is True and repair["eligible"] is not False:
            _fail("R8", "First-attempt success is not Repair-eligible.")
        if row["delivery"] is True and row["failure_stage"] is not None:
            _fail("R11", "A delivered outcome cannot have a terminal failure_stage.")
        if row["delivery"] is not True and row["time_to_delivery_seconds"] is not None:
            _fail("R11", "Only a delivered outcome can have time_to_delivery_seconds.")
        if row["delivery"] is not False and row["failure_stage"] is not None:
            _fail("R11", "Only a failed outcome can have a terminal failure_stage.")
        if row["execution_origin"] == "developer_assisted" and interventions["developer"]["required"] is not True:
            _fail("R9", "Developer-assisted execution requires a positive developer intervention record.")
    return row


def validate_outcome_ledger(ledger: object) -> None:
    """Validate the versioned ledger without opening any evidence reference."""
    value = _require_exact_keys(ledger, _LEDGER_KEYS, "R1", "Ledger")
    if value["schema"] != LEDGER_SCHEMA_VERSION:
        _fail("R1", f"Unsupported ledger schema {value['schema']!r}.")
    task_ids = value["registered_task_ids"]
    if not isinstance(task_ids, list) or not task_ids:
        _fail("R2", "registered_task_ids must be a non-empty list.")
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        _fail("R2", "Every registered task ID must be a non-empty string.")
    if len(set(task_ids)) != len(task_ids):
        _fail("R2", "Registered task IDs must be unique.")
    records = value["records"]
    if not isinstance(records, list) or not records:
        _fail("R1", "records must be a non-empty list.")

    registered = set(task_ids)
    validated = [_validate_record(record, registered) for record in records]
    record_ids: set[str] = set()
    revision_ids: set[tuple[str, int]] = set()
    path_hashes: dict[str, str] = {}
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validated:
        if row["record_id"] in record_ids:
            _fail("R3", f"Duplicate record_id {row['record_id']!r}.")
        record_ids.add(row["record_id"])
        revision_identity = (row["run_family_id"], row["revision"])
        if revision_identity in revision_ids:
            _fail("R3", f"Duplicate run-family revision {revision_identity!r}.")
        revision_ids.add(revision_identity)
        families[row["run_family_id"]].append(row)
        for reference in row["evidence_refs"]:
            previous = path_hashes.setdefault(reference["path"], reference["sha256"])
            if previous != reference["sha256"]:
                _fail("R12", f"Evidence path {reference['path']!r} has conflicting hashes.")

    immutable_keys = ("task_id", "arm", "execution_origin", "repeat", "primary")
    slots: set[tuple[str, str, str, int]] = set()
    latest: list[Mapping[str, Any]] = []
    for family_id, revisions in families.items():
        identity = tuple(revisions[0][key] for key in immutable_keys)
        if any(tuple(row[key] for key in immutable_keys) != identity for row in revisions[1:]):
            _fail("R3", f"Run family {family_id!r} changed immutable identity.")
        slot = identity[:3] + (identity[3],)
        if slot in slots:
            _fail("R3", f"Two run families occupy task/arm/origin/repeat slot {slot!r}.")
        slots.add(slot)
        latest.append(max(revisions, key=lambda row: row["revision"]))

    primary_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    primary_slots: set[tuple[str, str]] = set()
    for row in latest:
        if row["primary"]:
            if row["execution_origin"] != "product":
                _fail("R2", "Developer-assisted run families cannot be primary Product outcomes.")
            slot = (row["task_id"], row["arm"])
            if slot in primary_slots:
                _fail("R2", f"Primary task/arm slot {slot!r} has more than one run family.")
            primary_slots.add(slot)
            primary_by_task[row["task_id"]].append(row)
    arm_sets = [{row["arm"] for row in primary_by_task[task_id]} for task_id in task_ids]
    if any(not arms for arms in arm_sets) or any(arms != arm_sets[0] for arms in arm_sets[1:]):
        _fail("R2", "Every registered task must have exactly the same non-empty primary arm set.")


def _latest_records(ledger: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger["records"]:
        families[row["run_family_id"]].append(row)
    latest = [max(rows, key=lambda row: row["revision"]) for rows in families.values()]
    return latest, len(ledger["records"]) - len(latest)


def verify_evidence_references(ledger: object) -> dict[str, Any]:
    """Read and hash every distinct evidence reference after schema validation."""
    validate_outcome_ledger(ledger)
    references: dict[tuple[str, str], Mapping[str, str]] = {}
    occurrences = 0
    for row in ledger["records"]:
        for reference in row["evidence_refs"]:
            occurrences += 1
            references[(reference["path"], reference["sha256"])] = reference
    for reference in references.values():
        path = Path(reference["path"])
        try:
            raw = path.read_bytes()
        except OSError as error:
            _fail("R12", f"Cannot read evidence {path}: {type(error).__name__}: {error}")
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            _fail("R12", f"Evidence hash mismatch: {path}")
    return {
        "status": "verified",
        "record_count": len(ledger["records"]),
        "reference_occurrences": occurrences,
        "distinct_reference_count": len(references),
    }


def _classified_status(row: Mapping[str, Any]) -> str:
    return "invalid" if row["validity"] == "invalid" else row["execution_status"]


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    observed = Counter(_classified_status(row) for row in rows)
    return {
        status: observed.get(status, 0)
        for status in ("executed", "not_executed", "environment_blocked", "invalid")
    }


def _usable(row: Mapping[str, Any]) -> bool:
    return row["validity"] == "valid" and row["execution_status"] == "executed"


def _boolean_metric(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], bool | None],
    *,
    cohort_complete: bool,
) -> dict[str, Any]:
    known = [value(row) for row in rows if _usable(row) and type(value(row)) is bool]
    true_count = sum(item is True for item in known)
    unknown_rows = [row for row in rows if not (_usable(row) and type(value(row)) is bool)]
    complete = cohort_complete and not unknown_rows
    return {
        "confirmed_true": true_count,
        "confirmed_false": sum(item is False for item in known),
        "observed_denominator": len(known),
        "registered_denominator": len(rows),
        "unknown_count": len(unknown_rows),
        "unknown_by_status": dict(Counter(_classified_status(row) for row in unknown_rows)),
        "observed_rate_among_known": true_count / len(known) if known else None,
        "final_numerator": true_count if complete else None,
        "final_denominator": len(rows) if complete else None,
        "final_rate": true_count / len(rows) if complete and rows else None,
        "is_final": complete,
    }


def _conditional_pass_metric(
    rows: Sequence[Mapping[str, Any]],
    section: str,
    eligibility_key: str,
    *,
    cohort_complete: bool,
) -> dict[str, Any]:
    eligibility_known = [
        row for row in rows if _usable(row) and type(row[section][eligibility_key]) is bool
    ]
    eligible = [row for row in eligibility_known if row[section][eligibility_key] is True]
    passes = [row[section]["passed"] for row in eligible if type(row[section]["passed"]) is bool]
    unknown_eligibility = len(rows) - len(eligibility_known)
    unknown_pass = len(eligible) - len(passes)
    complete = cohort_complete and unknown_eligibility == 0 and unknown_pass == 0
    passed = sum(value is True for value in passes)
    result = {
        "registered_denominator": len(rows),
        "confirmed_eligible": len(eligible),
        "confirmed_not_eligible": len(eligibility_known) - len(eligible),
        "unknown_eligibility": unknown_eligibility,
        "confirmed_passed": passed,
        "confirmed_failed": sum(value is False for value in passes),
        "observed_pass_denominator": len(passes),
        "unknown_pass_among_eligible": unknown_pass,
        "observed_pass_rate_among_known": passed / len(passes) if passes else None,
        "final_numerator": passed if complete else None,
        "final_denominator": len(eligible) if complete else None,
        "final_rate": passed / len(eligible) if complete and eligible else None,
        "is_final": complete,
    }
    if section == "runtime":
        result["required_but_not_executed"] = sum(
            row["runtime"]["required"] is True and row["runtime"]["executed"] is False
            for row in rows if row["validity"] == "valid"
        )
    return result


def _repair_recovery_metric(
    rows: Sequence[Mapping[str, Any]], *, cohort_complete: bool
) -> dict[str, Any]:
    usable = [row for row in rows if _usable(row)]
    entered = [row for row in usable if row["repair"]["attempted"] is True]
    pass_values = [
        row["repair"]["passed"] for row in entered
        if type(row["repair"]["passed"]) is bool
    ]
    unresolved_eligibility = sum(row["repair"]["eligible"] is None for row in usable)
    unresolved_entry = sum(
        row["repair"]["eligible"] is True and row["repair"]["attempted"] is None
        for row in usable
    )
    unknown_pass = len(entered) - len(pass_values)
    complete = (
        cohort_complete
        and unresolved_eligibility == 0
        and unresolved_entry == 0
        and unknown_pass == 0
    )
    passed = sum(value is True for value in pass_values)
    excluded = Counter(
        _classified_status(row) for row in rows if not _usable(row)
    )
    return {
        "denominator_kind": "valid_executed_run_families_with_repair_attempted_true",
        "registered_run_family_count": len(rows),
        "valid_executed_run_family_count": len(usable),
        "confirmed_eligible": sum(row["repair"]["eligible"] is True for row in usable),
        "confirmed_not_eligible": sum(row["repair"]["eligible"] is False for row in usable),
        "eligible_not_entered": sum(
            row["repair"]["eligible"] is True and row["repair"]["attempted"] is False
            for row in usable
        ),
        "unresolved_eligibility": unresolved_eligibility,
        "unresolved_entry": unresolved_entry,
        "entered_repair_denominator": len(entered),
        "confirmed_passed": passed,
        "confirmed_failed": sum(value is False for value in pass_values),
        "observed_pass_denominator": len(pass_values),
        "unknown_pass_among_entered": unknown_pass,
        "excluded_or_unresolved_by_status": {
            status: excluded.get(status, 0)
            for status in ("invalid", "environment_blocked", "not_executed")
        },
        "observed_pass_rate_among_known": passed / len(pass_values) if pass_values else None,
        "final_numerator": passed if complete else None,
        "final_denominator": len(entered) if complete else None,
        "final_rate": passed / len(entered) if complete and entered else None,
        "is_final": complete,
    }


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _time_to_delivery(
    rows: Sequence[Mapping[str, Any]], delivery: Mapping[str, Any]
) -> dict[str, Any]:
    successful = [row for row in rows if _usable(row) and row["delivery"] is True]
    known = [row["time_to_delivery_seconds"] for row in successful if row["time_to_delivery_seconds"] is not None]
    complete = delivery["is_final"] and len(known) == len(successful)
    return {
        "eligible_deliveries": len(successful),
        "known_count": len(known),
        "unknown_success_count": len(successful) - len(known),
        "observed": _distribution(known),
        "final": _distribution(known) if complete else None,
        "is_final": complete,
    }


def _failure_stages(
    rows: Sequence[Mapping[str, Any]], delivery: Mapping[str, Any]
) -> dict[str, Any]:
    failures = [row for row in rows if _usable(row) and row["delivery"] is False]
    known = [row["failure_stage"] for row in failures if row["failure_stage"] is not None]
    counts = dict(sorted(Counter(known).items()))
    complete = delivery["is_final"] and len(known) == len(failures)
    return {
        "confirmed_failure_denominator": len(failures),
        "known_stage_count": len(known),
        "unknown_stage_count": len(failures) - len(known),
        "observed_distribution": counts,
        "final_distribution": counts if complete else None,
        "is_final": complete,
    }


def _cost_metric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    known: list[float | int] = []
    complete_values: list[float | int] = []
    complete = True
    for row in rows:
        value = row["cost"][key]
        if value is not None:
            known.append(value)
        if row["cost"]["complete"] is not True or value is None:
            complete = False
        else:
            complete_values.append(value)
    return {
        "known_lower_bound": sum(known) if known else None,
        "known_rows": len(known),
        "unknown_rows": len(rows) - len(known),
        "complete_total": sum(complete_values) if complete and rows else None,
        "is_complete": complete and bool(rows),
    }


def _total_tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known_components: list[float | int] = []
    fully_known_rows = 0
    complete = True
    totals: list[float | int] = []
    for row in rows:
        input_tokens, output_tokens = row["cost"]["input_tokens"], row["cost"]["output_tokens"]
        known_components.extend(value for value in (input_tokens, output_tokens) if value is not None)
        if input_tokens is not None and output_tokens is not None:
            fully_known_rows += 1
            totals.append(input_tokens + output_tokens)
        else:
            complete = False
        if row["cost"]["complete"] is not True:
            complete = False
    return {
        "known_component_lower_bound": sum(known_components) if known_components else None,
        "fully_known_rows": fully_known_rows,
        "unknown_rows": len(rows) - fully_known_rows,
        "complete_total": sum(totals) if complete and rows else None,
        "is_complete": complete and bool(rows),
    }


def _cost_summary(
    rows: Sequence[Mapping[str, Any]], delivery: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = {key: _cost_metric(rows, key) for key in COST_FIELDS}
    metrics["total_tokens"] = _total_tokens(rows)
    valid_rows = [row for row in rows if row["validity"] == "valid"]
    valid_product_metrics = {key: _cost_metric(valid_rows, key) for key in COST_FIELDS}
    valid_product_metrics["total_tokens"] = _total_tokens(valid_rows)
    all_complete = bool(rows) and all(metric["is_complete"] for metric in metrics.values())
    successes = delivery["confirmed_true"] if delivery["is_final"] else None
    per_success_complete = all_complete and successes is not None and successes > 0
    per_success = {
        key: metric["complete_total"] / successes if per_success_complete else None
        for key, metric in metrics.items()
    }
    if not delivery["is_final"]:
        reason = "delivery_rate_not_final"
    elif not all_complete:
        reason = "included_attempt_cost_missing_or_invalid"
    elif not successes:
        reason = "no_successful_delivery"
    else:
        reason = None
    return {
        "scope": "all selected run-family attempt costs, including failed outcomes",
        "included_run_families": len(rows),
        "invalid_run_family_costs_included_in_all_attempt_lower_bound": sum(
            row["validity"] == "invalid" for row in rows
        ),
        "metrics": metrics,
        "valid_product_metrics": valid_product_metrics,
        "all_included_attempt_costs_complete": all_complete,
        "cost_per_successful_delivery": {
            "complete": per_success_complete,
            "successful_delivery_denominator": successes if per_success_complete else None,
            "metrics": per_success,
            "reason": reason,
        },
    }


def _intervention_summary(
    rows: Sequence[Mapping[str, Any]], *, cohort_complete: bool
) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    for kind in INTERVENTION_KINDS:
        metric = _boolean_metric(
            rows,
            lambda row, kind=kind: row["interventions"][kind]["required"],
            cohort_complete=cohort_complete,
        )
        known_counts = [
            row["interventions"][kind]["count"]
            for row in rows
            if _usable(row) and row["interventions"][kind]["count"] is not None
        ]
        count_complete = cohort_complete and len(known_counts) == len(rows)
        metric["event_count"] = {
            "known_lower_bound": sum(known_counts) if known_counts else None,
            "known_rows": len(known_counts),
            "unknown_rows": len(rows) - len(known_counts),
            "complete_total": sum(known_counts) if count_complete else None,
            "is_final": count_complete,
        }
        by_kind[kind] = metric

    def any_required(row: Mapping[str, Any]) -> bool | None:
        values = [row["interventions"][kind]["required"] for kind in INTERVENTION_KINDS]
        if any(value is True for value in values):
            return True
        if all(value is False for value in values):
            return False
        return None

    return {
        "by_kind": by_kind,
        "any_intervention": _boolean_metric(rows, any_required, cohort_complete=cohort_complete),
    }


def _primary_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cohort_complete = all(_usable(row) for row in rows)
    delivery = _boolean_metric(rows, lambda row: row["delivery"], cohort_complete=cohort_complete)
    return {
        "denominator_kind": "registered_independent_tasks",
        "registered_task_count": len(rows),
        "cohort_execution_complete": cohort_complete,
        "status_counts": _status_counts(rows),
        "delivery": delivery,
        "first_attempt_success": _boolean_metric(
            rows, lambda row: row["first_attempt_success"], cohort_complete=cohort_complete
        ),
        "runtime_acceptance": _conditional_pass_metric(
            rows, "runtime", "required", cohort_complete=cohort_complete
        ),
        "repair_recovery": _repair_recovery_metric(rows, cohort_complete=cohort_complete),
        "intervention": _intervention_summary(rows, cohort_complete=cohort_complete),
        "time_to_delivery_seconds": _time_to_delivery(rows, delivery),
        "failure_stage": _failure_stages(rows, delivery),
        "cost": _cost_summary(rows, delivery),
    }


def _task_arm_delivery(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    outcomes: list[bool | None] = []
    for family_rows in by_task.values():
        terminal = [
            row for row in family_rows
            if _usable(row) and type(row["delivery"]) is bool
        ]
        latest = max(terminal, key=lambda row: row["repeat"]) if terminal else None
        outcomes.append(latest["delivery"] if latest is not None else None)
    known = [value for value in outcomes if type(value) is bool]
    successes = sum(value is True for value in known)
    complete = len(known) == len(outcomes)
    return {
        "semantics": "latest valid terminal delivery by repeat per task arm; repeats never add successes",
        "confirmed_successes": successes,
        "confirmed_failures": sum(value is False for value in known),
        "observed_denominator": len(known),
        "registered_task_arm_denominator": len(outcomes),
        "unknown_count": len(outcomes) - len(known),
        "observed_rate_among_known": successes / len(known) if known else None,
        "final_rate": successes / len(outcomes) if complete and outcomes else None,
        "is_final": complete,
    }


def _combined_interventions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        row for row in rows
        if _usable(row) and row["execution_origin"] == "product"
    ]
    combined: dict[str, Any] = {}
    for kind in INTERVENTION_KINDS:
        if not rows:
            combined[kind] = {"required": None, "count": None}
            continue
        required = [row["interventions"][kind]["required"] for row in rows]
        counts = [row["interventions"][kind]["count"] for row in rows]
        if any(value is True for value in required):
            combined_required = True
        elif all(value is False for value in required):
            combined_required = False
        else:
            combined_required = None
        combined[kind] = {
            "required": combined_required,
            "count": sum(counts) if all(value is not None for value in counts) else None,
        }
    return combined


def _excluded_preexecution_interventions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    excluded = [
        row for row in rows
        if not (_usable(row) and row["execution_origin"] == "product")
    ]
    by_kind: dict[str, Any] = {}
    all_known: list[int] = []
    unknown_components = 0
    for kind in INTERVENTION_KINDS:
        values = [row["interventions"][kind]["count"] for row in excluded]
        known = [value for value in values if value is not None]
        all_known.extend(known)
        unknown_components += len(values) - len(known)
        by_kind[kind] = {
            "known_lower_bound": sum(known) if known else None,
            "known_rows": len(known),
            "unknown_rows": len(values) - len(known),
        }
    statuses = Counter(_classified_status(row) for row in excluded)
    return {
        "excluded_run_family_count": len(excluded),
        "excluded_by_status": dict(sorted(statuses.items())),
        "by_kind": by_kind,
        "known_event_lower_bound": sum(all_known) if all_known else None,
        "unknown_event_components": unknown_components,
    }


def _final_task_rows(
    rows: Sequence[Mapping[str, Any]], task_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    projected: list[Mapping[str, Any]] = []
    for task_id in task_ids:
        family_rows = sorted(by_task[task_id], key=lambda row: row["repeat"])
        terminal = [
            row for row in family_rows
            if _usable(row) and type(row["delivery"]) is bool
        ]
        chosen = max(terminal, key=lambda row: row["repeat"]) if terminal else family_rows[-1]
        projection = dict(chosen)
        projection["interventions"] = _combined_interventions(family_rows)
        projected.append(projection)
    return projected


def _final_product_summary(
    rows: Sequence[Mapping[str, Any]], task_ids: Sequence[str]
) -> dict[str, Any]:
    task_rows = _final_task_rows(rows, task_ids)
    cohort_complete = all(_usable(row) and type(row["delivery"]) is bool for row in task_rows)
    delivery = _boolean_metric(
        task_rows, lambda row: row["delivery"], cohort_complete=cohort_complete
    )
    return {
        "denominator_kind": "registered_independent_tasks_after_task_arm_collapse",
        "registered_task_count": len(task_rows),
        "registered_run_family_count": len(rows),
        "cohort_final_outcome_complete": cohort_complete,
        "task_status_counts": _status_counts(task_rows),
        "delivery": delivery,
        "runtime_acceptance": _conditional_pass_metric(
            task_rows, "runtime", "required", cohort_complete=cohort_complete
        ),
        "repair_recovery": _repair_recovery_metric(rows, cohort_complete=cohort_complete),
        "intervention": _intervention_summary(task_rows, cohort_complete=cohort_complete),
        "excluded_preexecution_intervention_events": _excluded_preexecution_interventions(rows),
        "time_to_delivery_seconds": _time_to_delivery(task_rows, delivery),
        "failure_stage": _failure_stages(task_rows, delivery),
        "cost": _cost_summary(rows, delivery),
    }


def _scope_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_delivery = _task_arm_delivery(rows)
    return {
        "registered_run_family_count": len(rows),
        "independent_task_count": len({row["task_id"] for row in rows}),
        "status_counts": _status_counts(rows),
        "task_arm_delivery": task_delivery,
        "repair_recovery": _repair_recovery_metric(
            rows, cohort_complete=task_delivery["is_final"]
        ),
        "cost": _cost_summary(rows, {
            "confirmed_true": task_delivery["confirmed_successes"],
            "is_final": task_delivery["is_final"],
        }),
    }


def aggregate_outcome_ledger(ledger: object) -> dict[str, Any]:
    """Validate and aggregate an in-memory ledger without reading or writing files."""
    validate_outcome_ledger(ledger)
    latest, superseded = _latest_records(ledger)
    primary = [row for row in latest if row["primary"]]
    primary_arms = sorted({row["arm"] for row in primary})
    primary_by_slot = {(row["task_id"], row["arm"]): row for row in primary}
    ordered_by_arm = {
        arm: [primary_by_slot[(task_id, arm)] for task_id in ledger["registered_task_ids"]]
        for arm in primary_arms
    }
    scopes: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in latest:
        scopes[row["execution_origin"]][row["arm"]].append(row)
    by_scope = {
        origin: {arm: _scope_summary(rows) for arm, rows in sorted(arms.items())}
        for origin, arms in sorted(scopes.items())
    }
    primary_summaries = {
        arm: _primary_summary(rows) for arm, rows in ordered_by_arm.items()
    }
    final_summaries = {
        arm: _final_product_summary(
            [row for row in latest if row["execution_origin"] == "product" and row["arm"] == arm],
            ledger["registered_task_ids"],
        )
        for arm in primary_arms
    }
    return {
        "schema": SUMMARY_SCHEMA_VERSION,
        "source_schema": LEDGER_SCHEMA_VERSION,
        "schema_rules": [dict(rule) for rule in SCHEMA_RULES],
        "registered_snapshot": {
            "registered_task_ids": list(ledger["registered_task_ids"]),
            "registered_task_count": len(ledger["registered_task_ids"]),
            "primary_arms": primary_arms,
            "registered_slot_count": len(primary),
            "status_counts": _status_counts(primary),
            "ledger_record_count": len(ledger["records"]),
            "run_family_count": len(latest),
            "superseded_revision_count": superseded,
            "primary_final_delivery_available_by_arm": {
                arm: summary["delivery"]["is_final"] for arm, summary in final_summaries.items()
            },
        },
        "first_registered_by_arm": primary_summaries,
        "final_product_by_arm": final_summaries,
        "by_scope": by_scope,
    }


__all__ = [
    "COST_FIELDS",
    "EXECUTION_ORIGINS",
    "EXECUTION_STATUSES",
    "INTERVENTION_KINDS",
    "LEDGER_SCHEMA_VERSION",
    "SCHEMA_RULES",
    "SUMMARY_SCHEMA_VERSION",
    "VALIDITIES",
    "aggregate_outcome_ledger",
    "validate_evidence_reference",
    "validate_outcome_ledger",
    "verify_evidence_references",
    "project_native_outcome",
]


def _outcome_ref_key(ref):
    if not isinstance(ref, Mapping):
        return None
    path, digest = ref.get("path"), ref.get("sha256", ref.get("artifact_hash"))
    normalized = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalized):
        normalized = normalized.casefold()
    return (normalized, digest) if path and digest else None


def _outcome_refs(rows):
    unique = {}
    for ref in rows:
        key = _outcome_ref_key(ref)
        if key:
            unique[key] = {k: ref[k] for k in ("path", "sha256", "artifact_hash") if k in ref}
    return list(unique.values())


def _outcome_fact_rows(facts, key):
    rows = {}
    for row in facts.get(key, []):
        identity = _outcome_ref_key(row.get("ref"))
        if identity and isinstance(row.get("value"), Mapping):
            if identity in rows and rows[identity]["value"] != row["value"]:
                raise ValueError("NATIVE_OUTCOME_CONFLICTING_FACT")
            rows[identity] = row
    return list(rows.values())


def _outcome_current_context(rows):
    # 为什么要有它：显式更正取代旧聚合，不能按文件时间挑选有利结果。
    superseded = {_outcome_ref_key(row["value"].get(key)) for row in rows
                  for key in ("supersedes_aggregate_ref", "supersedes_ref")}
    return [row for row in rows if _outcome_ref_key(row["ref"]) not in superseded]


def _outcome_time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _outcome_union(intervals):
    total, end = 0.0, None
    for start, finish in sorted(intervals):
        total += finish - max(start, end) if end is not None and finish > end else finish - start if end is None else 0
        end = max(finish, end) if end is not None else finish
    return total


def _outcome_interventions(contexts, *, partition=True):
    events, unknown = {}, []
    for row in contexts:
        value, ref = row["value"], row["ref"]
        supplied = value.get("interventions", value.get("intervention_events", []))
        supplied = list(supplied) if isinstance(supplied, list) else []
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, Mapping):
                attributes = current.get("attributes", {})
                if isinstance(attributes, Mapping) and isinstance(attributes.get("intervention"), Mapping):
                    supplied.append(attributes["intervention"])
                pending.extend(v for k, v in current.items() if k in {"stages", "data", "observations", "events", "trace", "execution_observations"})
                if current and all(isinstance(v, Mapping) and "attributes" in v for v in current.values()):
                    pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        if value.get("event_id") and value.get("actor") in {"human", "outer_codex"}:
            supplied = [*supplied, value] if isinstance(supplied, list) else [value]
        for event in supplied if isinstance(supplied, list) else []:
            if not isinstance(event, Mapping) or not event.get("event_id"):
                unknown.append("intervention_without_stable_event_id")
                continue
            event_id = event["event_id"]
            selected = {key: event.get(key) for key in ("event_id", "run_id", "person_id", "actor", "kind", "target", "subject_ref", "cost_scope", "evidence_refs",
                "request_ref", "patch_ref", "timing_ref", "timing_basis", "active_seconds", "started_at", "finished_at", "rework")}
            if event_id in events and any(events[event_id].get(k) != v for k, v in selected.items()):
                unknown.append("conflicting_intervention:" + str(event_id))
                events[event_id]["conflict"] = True
            else:
                events.setdefault(event_id, {**selected, "refs": []})["refs"].append(ref)
        if value.get("developer_assistance"):
            unknown.append("unstructured_developer_assistance_requires_event_classification")
    intervals, declared = defaultdict(list), defaultdict(list)
    observed_human, timed = 0, 0
    for event in events.values():
        event["refs"] = _outcome_refs(event["refs"])
        event["technical_assistance"] = event["actor"] in {"human", "outer_codex"} and event["kind"] in {
            "technical_hint", "patch_edit", "root_cause", "fix", "rework"}
        if event["actor"] != "human" or event.get("conflict"):
            continue
        observed_human += 1
        person, basis = event["person_id"], event["timing_basis"]
        start, finish = _outcome_time(event["started_at"]), _outcome_time(event["finished_at"])
        if not person or basis not in {"measured", "declared"}:
            continue
        if start is not None and finish is not None and finish >= start and (
                event["active_seconds"] is None or event["active_seconds"] == finish - start):
            intervals[person].append((start, finish))
            timed += 1
        elif _is_nonnegative_number(event["active_seconds"]):
            declared[person].append(event["active_seconds"])
            timed += 1
    seconds = sum(_outcome_union(value) for value in intervals.values())
    ambiguous = any(len(values) > 1 or person in intervals for person, values in declared.items())
    # 为什么要有它：无区间的多个申报可能重叠；展示申报值但不能相加伪装完整人分钟。
    if not ambiguous:
        seconds += sum(sum(values) for values in declared.values())
    result = {"events": list(events.values()), "observed_event_count": len(events),
        "technical_assistance_observed": any(e["technical_assistance"] for e in events.values()),
        "total_technical_assistance": None, "coverage": "observed_events_only_not_complete_history",
        "human_active_seconds": seconds if observed_human and timed == observed_human and not ambiguous and not unknown else None,
        "measured_interval_union_seconds": sum(_outcome_union(v) for v in intervals.values()) if intervals else None,
        "declared_seconds_by_person": dict(declared), "unknown": sorted(set(unknown +
            (["unobserved_human_history"] if not events else []) +
            (["declared_time_overlap_unknown"] if ambiguous else []) +
            (["human_timing_missing"] if timed != observed_human else []))),
        "refs": _outcome_refs(ref for e in events.values() for ref in e["refs"])}
    result["human_active_time_scope"] = "observed_explicit_events_only_not_whole_task"
    if partition:
        result["by_cost_scope"] = {}
        for scope in {event.get("cost_scope") or "unknown" for event in events.values()}:
            rows = [{"ref": event["refs"][0], "value": {"interventions": [event]}} for event in events.values()
                    if (event.get("cost_scope") or "unknown") == scope]
            timing = _outcome_interventions(rows, partition=False)
            result["by_cost_scope"][scope] = {k: timing[k] for k in ("observed_event_count", "human_active_seconds",
                "measured_interval_union_seconds", "declared_seconds_by_person", "unknown", "refs")}
        # 为什么要有它：任务、评测和平台开发的人时分别报告，跨分区合计不冒充每任务人分钟。
        if len(result["by_cost_scope"]) > 1 or "unknown" in result["by_cost_scope"]:
            result["human_active_seconds"] = None
            result["unknown"].append("use_partitioned_human_time_not_mixed_total")
    return result


def _outcome_costs(report, natives, contexts):
    executions = [*report.get("executions", []), *report.get("linked_executions", [])]
    turns, conflicts, disputed = {}, [], set()
    for execution in executions:
        for turn in execution.get("turns", []):
            key = (_outcome_ref_key(turn.get("request_ref")), turn.get("reservation_id"))
            if key[0] is None:
                continue
            old = turns.get(key)
            if key in disputed:
                continue
            if old and _outcome_ref_key(old.get("response_ref")) and _outcome_ref_key(turn.get("response_ref")) and _outcome_ref_key(turn.get("response_ref")) != _outcome_ref_key(old.get("response_ref")):
                turns[key] = {**old, "usage": None}
                conflicts.append("conflicting_response_for_request")
                disputed.add(key)
            elif not old or turn.get("response_observed"):
                turns[key] = turn
    costs = report.get("cost", {})
    sums = {}
    for field in ("input_tokens", "output_tokens", "cached_input_tokens"):
        if turns:
            amounts = [(turn.get("usage") or {}).get(field) for turn in turns.values()]
            known = [n for n in amounts if _is_nonnegative_int(n)]
            sums[field] = {"known_sum": sum(known), "unknown_calls": len(amounts) - len(known)}
        else:
            original = costs.get(field, {})
            sums[field] = {"known_sum": original.get("known_sum"), "unknown_calls": original.get("unknown_requests")}
    learning_calls, learning_unknown_by_origin, learning_refs = {}, {}, []
    available = {_outcome_ref_key(row["ref"]): row["value"] for row in contexts}
    native_keys = {_outcome_ref_key(row["ref"]) for row in natives}
    for row in contexts:
        value = row["value"]
        origin = _outcome_ref_key(value.get("original_execution_ref"))
        if origin not in native_keys or "model_calls" not in value or "budget_after" not in value:
            continue
        learning_refs.append(row["ref"])
        receipt_keys = set()
        for call in value.get("call_receipts", []):
            key = _outcome_ref_key(call.get("response_ref"))
            if key:
                receipt_keys.add(key)
            if key is None or key in learning_calls:
                continue
            response = available.get(key, {})
            response = response.get("data", {}).get("response", response.get("response", response))
            usage = response.get("usage")
            learning_calls[key] = usage if usage and usage == call.get("usage") else None
        count = value.get("model_calls")
        if _is_nonnegative_int(count):
            learning_unknown_by_origin[origin] = max(learning_unknown_by_origin.get(origin, 0), count - len(receipt_keys), 0)
    learning_unknown = sum(learning_unknown_by_origin.values())
    for field in sums:
        values = [(usage or {}).get(field) for usage in learning_calls.values()]
        known = [n for n in values if _is_nonnegative_int(n)]
        if known:
            sums[field]["known_sum"] = (sums[field]["known_sum"] or 0) + sum(known)
        if sums[field]["unknown_calls"] is not None:
            sums[field]["unknown_calls"] += len(values) - len(known) + learning_unknown
    reserved = {}
    for row in natives:
        value = row["value"]
        count = (value.get("budget_after") or {}).get("consumed")
        if _is_nonnegative_int(count):
            contract = (value.get("run_id"), value.get("task_contract_ref"))
            reserved[contract] = max(count, reserved.get(contract, 0))
    responded = sum(turn.get("response_observed") is True for turn in turns.values()) if turns else None
    task = {**sums, "total_tokens": (sums["input_tokens"]["known_sum"] + sums["output_tokens"]["known_sum"])
        if all(sums[k]["known_sum"] is not None for k in ("input_tokens", "output_tokens")) else None,
        "unique_requests": len(turns) if turns else costs.get("requests"), "observed_responses": responded,
        "send_attempts": sum(t.get("send_attempt_observed") is True for t in turns.values()) if turns else None,
        "reserved_budget_calls": sum(reserved.values()) if reserved else None,
        "unknown_reserved_calls": max(0, sum(reserved.values()) - (responded or 0)) if reserved else None,
        "known_learning_responses": sum(v is not None for v in learning_calls.values()),
        "unknown_learning_calls": sum(v is None for v in learning_calls.values()) + learning_unknown,
        "cached_input_included_in_input": True, "scope": "observed_related_runs_native_requests_and_bound_learning_not_whole_task",
        "total_tokens_is_known_subtotal": True,
        "selected_run_id": report.get("run_id"),
        "included_run_ids": sorted({r["value"]["run_id"] for r in natives if r["value"].get("run_id")}),
        "selected_run_request_cost": costs,
        "complete": False, "money": None, "observed_processes": [], "known_process_wall_seconds": None,
        "time_observations": [],
        "refs": _outcome_refs([*(row["ref"] for row in natives), *learning_refs]),
        "unknown": sorted(set(["whole_task_cost", "money", *conflicts]))}
    partitions = {"task_runtime": task, "independent_evaluation": {"complete": False, "model_calls": None,
        "observations": [], "money": None, "unknown": ["full_evaluation_cost"]},
        "platform_development": {"complete": False, "model_calls": None, "money": None,
        "unknown": ["platform_development_cost"]}}
    processes = {}
    for row in contexts:
        value = row["value"]
        if value.get("schema") == "task-agent-time-observation.v2":
            task["time_observations"].append({"ref": row["ref"],
                **{k: value.get(k) for k in ("native_execution_ref", "origin_ref", "event", "status",
                    "timing_status", "clock_basis", "scope", "resource_event_ref", "observation_sequence")},
                "observed_elapsed_ms": value.get("duration_ms") if _is_nonnegative_number(value.get("duration_ms")) else None,
                "whole_task_finished": False, "claim": "recorded_boundary_not_added_to_process_or_human_time"})
        if "independent_acceptance_passed" in value or (value.get("status") == "behavior_failed" and "model_calls" in value):
            partitions["independent_evaluation"]["observations"].append({"ref": row["ref"],
                "purpose": "independent_acceptance" if "independent_acceptance_passed" in value else "public_diagnostic_not_independent_acceptance",
                **{k: value.get(k) for k in ("model_calls", "tests_executed", "status", "total_results", "outcomes")}})
        process = value.get("process", value)
        if not isinstance(process, Mapping) or not _is_nonnegative_number(process.get("wall_seconds")):
            continue
        argv = process.get("argv", [])
        matched = [n["value"].get("run_id") for n in natives if n["value"].get("run_id") and
                   any(n["value"]["run_id"] in str(arg) for arg in argv)]
        if len(set(matched)) != 1:
            task["unknown"].append("process_scope_unresolved")
            continue
        key = (process.get("process_id"), process.get("started_at"), tuple(argv))
        processes[key] = {"run_id": matched[0], "wall_seconds": process["wall_seconds"],
            "started_at": process.get("started_at"), "finished_at": process.get("finished_at"), "ref": row["ref"]}
    task["observed_processes"] = list(processes.values())
    # 为什么要有它：进程区间可能嵌套，时间只按观测区间并集计，不把摘要耗时再次相加。
    intervals = [(_outcome_time(p["started_at"]), _outcome_time(p["finished_at"])) for p in processes.values()]
    if intervals and all(a is not None and b is not None and b >= a for a, b in intervals):
        task["known_process_wall_seconds"] = sum(p["wall_seconds"] for p in processes.values()) if all(
            b <= c or d <= a for i, (a, b) in enumerate(intervals) for c, d in intervals[i + 1:]) else _outcome_union(intervals)
    observations = partitions["independent_evaluation"]["observations"]
    if observations and all(o["model_calls"] == 0 for o in observations):
        partitions["independent_evaluation"]["model_calls"] = 0
    return partitions


def project_native_outcome(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a read-only result from caller-verified records, without creating a ledger.

    为什么要有它：历史原件不必伪造预登记槽位；正常收尾与只读查询共用事实口径。
    """
    report = facts.get("report", {})
    natives = _outcome_fact_rows(facts, "native_facts")
    contexts = _outcome_current_context(_outcome_fact_rows(facts, "context_facts"))
    native_keys = {_outcome_ref_key(row["ref"]) for row in natives}
    runs = sorted({row["value"]["run_id"] for row in natives if row["value"].get("run_id")})
    task_ids = sorted({str(row["value"].get("task_id", row["value"].get("task"))) for row in contexts
                       if row["value"].get("task_id", row["value"].get("task"))})
    relations, conditions, acceptances, deliveries, boundaries = [], [], [], [], []
    for row in natives:
        value = row["value"]
        if value.get("whole_task_delivered") is False:
            deliveries.append({"ref": row["ref"], "delivered": False, "run_id": value.get("run_id"),
                "candidate_source_id": value.get("candidate_source_id"), "native_execution_ref": row["ref"],
                "basis": "explicit_not_delivered_at_native_boundary"})
        conditions.append({"ref": row["ref"], **{key: value.get(key) for key in (
            "run_id", "protocol", "model_profile", "expected_model_identity", "expected_verification_identity", "worker_binding")},
            "tool_definition_request_ref": next((turn.get("request_ref") for turn in value.get("provider_turns", [])
                if turn.get("request_ref")), None),
            "source_identity": {"ref": row["ref"], "pointer": "/source_before",
                "recorded": value.get("source_before") is not None}, "freeze_status": "not_established"})
    for row in contexts:
        value, ref = row["value"], row["ref"]
        if value.get("product_commit") or value.get("code_snapshot_ref") or value.get("owner_refs"):
            conditions.append({"ref": ref, "kind": "explicit_version_record",
                **{k: value.get(k) for k in ("native_execution_ref", "product_commit", "code_snapshot_ref", "owner_refs", "new_task")},
                "binding": "recorded_context_relation_not_current_checkout_version"})
        if value.get("schema") == "run_outcome_boundary_v1":
            boundaries.append({"ref": ref, **{k: value.get(k) for k in (
                "run_id", "request_id", "event", "status", "execution_status", "error_type",
                "original_terminal_ref", "resumed_from_checkpoint_ref", "task_contract_ref")},
                "result": {k: value.get("result", {}).get(k) for k in ("status", "candidate_source_id",
                    "whole_task_delivered", "native_execution_ref", "receipt_ref", "v2_checkpoint_ref")}})
        boundary_result = value.get("result", {}) if value.get("schema") == "run_outcome_boundary_v1" else value
        if boundary_result.get("whole_task_delivered") is False:
            deliveries.append({"ref": ref, "delivered": False, "candidate_source_id": boundary_result.get("candidate_source_id"),
                "run_id": value.get("run_id"), "native_execution_ref": boundary_result.get("native_execution_ref"),
                "basis": "explicit_not_delivered_at_observed_boundary"})
        if _outcome_ref_key(value.get("original_native_ref")) in native_keys:
            relations.append({"ref": ref, "original_native_ref": value["original_native_ref"],
                "kind": "new_exposed_diagnostic" if value.get("new_task") is True and value.get("original_task_resumed") is False else "explicit_parent",
                "exposure": value.get("exposure"), "starting_candidate_ref": value.get("starting_candidate_ref"),
                "starting_candidate_source_id": value.get("starting_candidate_source_id")})
        if "independent_acceptance_passed" in value:
            checked = row.get("verification", {})
            acceptance_verified = checked.get("status") == "verified_independent_acceptance" and (
                _outcome_ref_key(checked.get("native_execution_ref")) == _outcome_ref_key(value.get("native_execution_ref"))) and (
                checked.get("candidate_source_id") == value.get("candidate_source_id")) and _outcome_ref_key(checked.get("judge_ref")) is not None
            acceptances.append({"ref": ref, "verification_status": "verified_independent_acceptance" if acceptance_verified else "reported_acceptance_not_judge_verified",
                "verified_judge_ref": checked.get("judge_ref") if acceptance_verified else None,
                **{key: value.get(key) for key in ("native_execution_ref", "candidate_ref",
                "candidate_source_id", "independent_acceptance_passed", "preservation_suite_passed", "preservation_scope",
                "full_selected_suite_passed", "tests_executed", "total_results", "outcomes", "supersedes_aggregate_ref",
                "qualification_ref", "aggregation_script_ref", "judge_ref", "judge_conditions")}})
        if type(value.get("delivered")) is bool:
            deliveries.append({"ref": ref, "delivered": value["delivered"], "candidate_source_id": value.get("candidate_source_id"),
                "native_execution_ref": value.get("native_execution_ref"), "basis": "reported_not_delivery_validation"})
        checked = row.get("verification", {})
        delivery = checked.get("value", {})
        if checked.get("status") == "verified_delivery" and delivery.get("status") == "delivered" and delivery.get("delivered_artifact_refs"):
            verification = delivery.get("verification", {})
            if verification.get("passed") is True and verification.get("candidate_source_id") == delivery.get("candidate_source_id") and verification.get("native_verification_ref"):
                deliveries.append({"ref": ref, "delivered": True, "candidate_source_id": delivery["candidate_source_id"],
                    "native_execution_ref": delivery.get("native_execution_ref"), "basis": "validated_delivery",
                    "delivered_artifact_refs": delivery["delivered_artifact_refs"]})
    selected_keys = {_outcome_ref_key(row["ref"]) for row in natives if row["value"].get("run_id") == report.get("run_id")}
    disputed_ids = {(_outcome_ref_key(a["native_execution_ref"]), a["candidate_source_id"]) for a in acceptances for b in acceptances
        if a["candidate_source_id"] == b["candidate_source_id"] and _outcome_ref_key(a["native_execution_ref"]) == _outcome_ref_key(b["native_execution_ref"]) and any(a[k] != b[k] for k in (
            "independent_acceptance_passed", "preservation_suite_passed"))}
    accepted_ids = {(_outcome_ref_key(a["native_execution_ref"]), a["candidate_source_id"]) for a in acceptances
                   if a["candidate_source_id"] and _outcome_ref_key(a["native_execution_ref"]) in native_keys and
                   (_outcome_ref_key(a["native_execution_ref"]), a["candidate_source_id"]) not in disputed_ids and
                   a["verification_status"] == "verified_independent_acceptance" and
                   a["independent_acceptance_passed"] is True and a["preservation_suite_passed"] is True and a["tests_executed"] is True}
    confirmed_related = [d for d in deliveries if d["basis"] == "validated_delivery" and
                         (_outcome_ref_key(d["native_execution_ref"]), d["candidate_source_id"]) in accepted_ids]
    confirmed = [d for d in confirmed_related if _outcome_ref_key(d["native_execution_ref"]) in selected_keys]
    explicit_no = any(d["delivered"] is False and (_outcome_ref_key(d.get("native_execution_ref")) in selected_keys or
                      d.get("run_id") == report.get("run_id")) for d in deliveries)
    correct = True if confirmed else False if explicit_no else None
    terminal = {key: value for key, value in (report.get("terminal") or {}).items()
                if key in {"run_id", "status", "whole_task_delivered", "current_stage", "completion_classification", "delivery_ref"}}
    outcomes = {"scope": "selected_run_not_parent_diagnostic_outcome", "terminal": terminal, "boundaries": boundaries, "independent_acceptance": acceptances,
        "delivery": deliveries, "correct_delivery": correct,
        "confirmed_correct_deliveries": len({d["candidate_source_id"] for d in confirmed}),
        "confirmed_related_candidate_deliveries": len({(_outcome_ref_key(d["native_execution_ref"]), d["candidate_source_id"]) for d in confirmed_related}),
        "unknown": (["conflicting_independent_acceptance"] if disputed_ids else []) +
            ([] if correct is not None else ["same_candidate_independent_acceptance_and_delivery_required"]),
        "refs": _outcome_refs([*(a["ref"] for a in acceptances), *(d["ref"] for d in deliveries)])}
    feedback = [link for execution in [*report.get("executions", []), *report.get("linked_executions", [])]
                for link in execution.get("feedback_links", [])]
    feedback.extend(report.get("initial_feedback_links", []))
    feedback = [{k: link.get(k) for k in ("turn", "call_id", "name", "candidate_ref", "candidate_source_id",
        "check_ref", "feedback_ref", "status", "request_ref", "provider_request_ref", "response_ref",
        "matches_recorded_tool_result", "included_in_prepared_request", "included_in_provider_request",
        "source_ref", "diagnostic_fields", "diagnostic_details", "missing_fields", "response_observed", "launch_ref",
        "content_fields", "delivery_kind", "content_scope") if k in link} for link in feedback]
    candidates = [a for a in report.get("artifacts", []) if a.get("schema") == "task-agent-candidate.v2"]
    candidate_ids = {a.get("candidate_source_id") for a in candidates if a.get("candidate_source_id")}
    costs = _outcome_costs(report, natives, contexts)
    successes = outcomes["confirmed_correct_deliveries"]
    costs["per_successful_delivery"] = {"value": None, "confirmed_successes": successes,
        "reason": "zero_confirmed_success_no_finite_estimate" if not successes else "whole_task_cost_incomplete"}
    segments = [{"ref": row["ref"], **{k: row["value"].get(k) for k in ("run_id", "request_id", "task_contract_ref", "status", "candidate_ref")}} for row in natives]
    attempts = {}
    for segment in segments:
        identity = (segment["run_id"], segment["request_id"]) if segment["run_id"] else _outcome_ref_key(segment["ref"])
        item = attempts.setdefault(identity, {"run_id": segment["run_id"], "request_id": segment["request_id"],
            "identity_basis": "explicit_run_request_not_registered_trial_repeat", "native_execution_refs": []})
        item["native_execution_refs"].append(segment["ref"])
    return {"schema_version": "native_run_outcome_v1", "derived": True, "read_only": True,
        "identity": {"selected_run_id": report.get("run_id"), "related_run_ids": runs,
            "related_run_count": len(runs), "source_problem_ids": task_ids, "source_problem_count": len(task_ids) if task_ids else None,
            "registration": "historical_or_unregistered", "registered_task_denominator": None,
            "attempts": list(attempts.values()), "native_segments": segments,
            "relations": relations, "recovery": report.get("lineage", []), "refs": _outcome_refs(row["ref"] for row in natives)},
        "conditions": conditions, "result": outcomes,
        "feedback_revision": {"observed_feedback_links": feedback, "candidate_source_ids": sorted(candidate_ids),
            "distinct_candidate_sources": len(candidate_ids), "effective_repairs": None, "eligible_attempts": None,
            "unknown": ["feedback_presence_not_diagnostic_sufficiency", "code_change_not_business_repair"],
            "counting_rule": "plan_updates_and_identical_source_resubmissions_are_not_code_revision"},
        "costs": costs, "interventions": _outcome_interventions(contexts),
        "coverage": {"complete": False, "query_complete": report.get("complete", False), "issues": report.get("issues", []),
            "missing": ["registered_comparison", "complete_human_history", "whole_task_cost"],
            "refs": _outcome_refs([*(r["ref"] for r in natives), *(r["ref"] for r in contexts)]),
            "meaning": "reference_consistency_not_external_authenticity_or_business_acceptance"}}
