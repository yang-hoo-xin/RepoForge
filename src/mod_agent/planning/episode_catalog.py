"""Immutable public episode registrations and provenance-based run selection.

为什么要有它：新经历需要进入共享选择入口，同一运行的不同视图不能被当作多次独立经验，原件仍须完整保留。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.task_memory import _select

_ROLES = {"context", "supporting", "opposing", "inapplicable"}


def require(value, message):
    if not value:
        raise ValueError("EPISODE_CATALOG_" + message)


def _catalog(catalog_id):
    require(isinstance(catalog_id, str) and bool(catalog_id.strip()) and len(catalog_id) <= 128,
            "ID_INVALID")
    return "sx_ec_" + canonical_hash(catalog_id)[:8] + "_"


def normalize_selection(selection=None):
    selection = {} if selection is None else deepcopy(selection)
    require(isinstance(selection, dict) and not set(selection) - {"required_episode_refs", "run_choices"},
            "SELECTION_INVALID")
    required, choices = selection.get("required_episode_refs", []), selection.get("run_choices", {})
    require(isinstance(required, list) and isinstance(choices, dict)
            and all(isinstance(key, str) and key for key in choices), "SELECTION_INVALID")
    for reference in [*required, *choices.values()]:
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}
                and all(isinstance(reference[key], str) and reference[key] for key in reference), "REFERENCE_INVALID")
    require(len({canonical_hash(item) for item in required}) == len(required), "REQUIRED_REFERENCE_DUPLICATED")
    return {"required_episode_refs": required, "run_choices": choices}


def _episode(memory, reference):
    from src.mod_agent.planning.learning_request import resolve_task_context
    from src.mod_agent.planning.strategy_experience import _episode_observations

    episode = read_bound_json(reference)
    require(episode.get("schema_version") == "public_strategy_episode_v1"
            and episode.get("visibility") == "public" and episode.get("data_use") == "strategy_learning",
            "PUBLIC_EPISODE_REQUIRED")
    require(isinstance(episode.get("run_id"), str) and bool(episode["run_id"])
            and isinstance(episode.get("task_contract_ref"), str) and bool(episode["task_contract_ref"]),
            "RUN_IDENTITY_REQUIRED")
    request = episode.get("request_id")
    require(request is None or isinstance(request, str) and bool(request), "REQUEST_IDENTITY_INVALID")
    documents = memory._public_sources(episode["evidence_refs"])
    _episode_observations(episode, documents)
    if episode.get("task_context_selections"):
        resolve_task_context(episode, documents)
    references = {row["evidence_id"]: row for row in episode["evidence_refs"]}
    facts, material, selected_facts = [], [], []
    envelope_fields = {"status", "candidate_ref", "check_ref", "unknown_effects", "native_terminal",
                       "business_result", "stop_attribution", "collector_scope"}
    for selector in [*episode["selections"], *episode.get("task_context_selections", [])]:
        identity = selector["evidence_id"]
        selected = _select(documents[identity], selector["locator"])
        witness = {"evidence_id": identity, "locator": selector["locator"], "role": selector.get("role"),
            "action_index": selector.get("action_index"), "value_sha256": canonical_hash(selected)}
        fingerprint = canonical_hash(witness)
        facts.append(fingerprint)
        selected_facts.append({**witness, "fact_hash": fingerprint,
            "source_ref": {key: references[identity][key] for key in ("path", "sha256")}})
        if (identity == "technical_outcome" and selector.get("role") == "technical_outcome"
                and len(selector["locator"]) == 1 and selector["locator"][0] in envelope_fields):
            continue
        document = documents[identity]
        if isinstance(document, dict) and document.get("schema") == "task-agent-public-semantic-validation.v1":
            if selector["locator"] in (["source_checkpoint_ref"], ["source_execution_ref"]):
                continue
            if selector["locator"] == ["events"]:
                require(isinstance(selected, list), "SEMANTIC_EVENTS_INVALID")
                material.extend(canonical_hash({"semantic_event": {key: value for key, value in event.items()
                    if key not in {"action_indices", "outside_completed_prefix_call_ids"}}}) for event in selected)
                continue
        material.append(fingerprint)
    execution = None
    if episode.get("execution_ref") is not None:
        matches = [row for row in episode["evidence_refs"] if row.get("path") == episode["execution_ref"].get("path")
                   and row.get("sha256") == episode["execution_ref"].get("sha256")]
        require(len(matches) == 1, "EXECUTION_NOT_REGISTERED_PUBLIC")
        execution = documents[matches[0]["evidence_id"]]
        require(isinstance(execution, dict) and execution.get("run_id") == episode["run_id"]
                and execution.get("request_id") == request, "EXECUTION_TASK_MISMATCH")
    actions = None
    terminal = False
    if execution is not None:
        raw = execution.get("completed_tool_records")
        require(isinstance(raw, list), "COMPLETED_RECORDS_REQUIRED")
        actions, seen = [], set()
        for row in raw:
            require(isinstance(row, dict) and isinstance(row.get("call_id"), str)
                    and row["call_id"] not in seen and isinstance(row.get("name"), str), "ACTION_IDENTITY_INVALID")
            seen.add(row["call_id"])
            identities = {}
            for key in ("response_ref", "arguments_ref", "execution_receipt_ref"):
                value = row.get(key)
                require(isinstance(value, dict) and isinstance(value.get("sha256"), str), "ACTION_PROVENANCE_REQUIRED")
                identities[key] = value["sha256"]
            actions.append(canonical_hash({"call_id": row["call_id"], "name": row["name"], **identities}))
        terminal = execution.get("schema_version") in {"task-agent-product-execution.v1", "task-agent-product-execution.v2"}
    identity = {"run_id": episode["run_id"], "request_id": request,
                "task_contract_ref": episode["task_contract_ref"]}
    return episode, {"identity": identity, "selected_fact_hashes": sorted(set(facts)),
        "selected_facts": selected_facts, "material_fact_hashes": sorted(set(material)),
        "completed_action_hashes": actions, "native_terminal": terminal,
        "episode_content_sha256": reference["sha256"],
        "outcome_signature": canonical_hash({key: execution.get(key) for key in (
            "status", "candidate_ref", "check_ref", "unknown_effects", "native_terminal")}) if execution is not None else None}


def register_public_episode(memory, catalog_id, episode_ref, source_role="context"):
    prefix = _catalog(catalog_id)
    require(source_role in _ROLES, "SOURCE_ROLE_INVALID")
    episode, coverage = _episode(memory, episode_ref)
    identity = canonical_hash({"episode_ref": episode_ref, "source_role": source_role})
    return memory._write(prefix + identity[:16], "episode_catalog_registration", {
        "logical_namespace": "episode_catalog_registration:" + canonical_hash(catalog_id) + ":" + identity,
        "catalog_id": catalog_id, "episode_ref": deepcopy(episode_ref),
        **coverage["identity"], "source_role": source_role, "coverage": coverage,
        "visibility": "public", "data_use": "strategy_learning", "information_gain": episode.get("information_gain", []),
        "scope": "Source registration only; no inferred strategy, task success or extra execution authority."})


def _action_covers(candidate, other):
    a, b = candidate["coverage"], other["coverage"]
    if a["identity"] != b["identity"] or a["completed_action_hashes"] is None or b["completed_action_hashes"] is None:
        return False
    return (a["completed_action_hashes"][:len(b["completed_action_hashes"])] == b["completed_action_hashes"]
            and set(b["material_fact_hashes"]) <= set(a["material_fact_hashes"]))


def _covers(candidate, other):
    a, b = candidate["coverage"], other["coverage"]
    if a["identity"] != b["identity"]:
        return False
    if a["episode_content_sha256"] == b["episode_content_sha256"]:
        return True
    # 为什么要有它：required覆盖指全部已选择公开值，不能只以行动前缀相同而吞掉新增停止或失败事实。
    return _action_covers(candidate, other) and set(b["selected_fact_hashes"]) <= set(a["selected_fact_hashes"])


def _choose(entries, selection):
    entries = sorted(entries, key=lambda row: (row["run_id"], canonical_hash(row["episode_ref"]), row["source_role"]))
    groups = {}
    for entry in entries:
        groups.setdefault(entry["run_id"], []).append(entry)
    require(set(selection["run_choices"]) <= set(groups), "UNKNOWN_RUN_CHOICE")
    selected, roles, runs = [], [], []
    represented = set()
    for run_id, variants in groups.items():
        choice = selection["run_choices"].get(run_id)
        if choice is not None:
            matches = [row for row in variants if row["episode_ref"] == choice]
            require(len(matches) == 1, "EXPLICIT_CHOICE_AMBIGUOUS_OR_UNREGISTERED")
            chosen, reason = matches[0], "explicit_registered_view"
        else:
            candidates = [row for row in variants if all(_covers(row, other) for other in variants)]
            reason = "verified_all_selected_value_coverage" if len(variants) > 1 else "only_registered_view"
            if not candidates:
                candidates = [row for row in variants if row["coverage"]["native_terminal"]
                    and all(_action_covers(row, other) and (not other["coverage"]["native_terminal"]
                        or _covers(row, other)) for other in variants)]
                reason = "native_terminal_representative_with_explicit_omissions_not_fact_superset"
            require(bool(candidates), "RUN_VIEWS_CONFLICT_OR_COVERAGE_UNPROVEN")
            chosen = sorted(candidates, key=lambda row: (not row["coverage"]["native_terminal"],
                canonical_hash(row["episode_ref"])))[0]
        require(all(row["source_role"] == chosen["source_role"] for row in variants
                    if row["episode_ref"] == chosen["episode_ref"]), "SOURCE_ROLE_CONFLICT")
        covered = [row["episode_ref"] for row in variants if _covers(chosen, row)]
        represented.update(canonical_hash(reference) for reference in covered)
        selected.append(chosen["episode_ref"])
        roles.append(chosen["source_role"])
        runs.append({"run_id": run_id, "representative_ref": chosen["episode_ref"],
            "variant_refs": [row["episode_ref"] for row in variants], "selection_reason": reason,
            "covered_refs": covered,
            "excluded_refs": [row["episode_ref"] for row in variants if row["episode_ref"] not in covered],
            "omitted_selected_facts": [{"episode_ref": row["episode_ref"],
                "facts": [fact for fact in row["coverage"]["selected_facts"]
                    if fact["fact_hash"] not in set(chosen["coverage"]["selected_fact_hashes"])],
                "reason": "Representative selection does not merge these original selected values."}
                for row in variants if row["episode_ref"] not in covered]})
    require(all(canonical_hash(reference) in represented for reference in selection["required_episode_refs"]),
            "REQUIRED_EPISODE_NOT_REPRESENTED")
    return selected, roles, runs


def freeze_catalog(memory, catalog_id, *, applicability, learning_request_ref=None, selection=None):
    prefix, selection = _catalog(catalog_id), normalize_selection(selection)
    entries = []
    for feature in memory.store.list_feature_ids():
        if not feature.startswith(prefix):
            continue
        for reference in memory.store.list_feature_artifact_revisions(feature, "strategy_experience"):
            entry = memory.read(reference)
            require(entry.get("kind") == "episode_catalog_registration" and entry.get("catalog_id") == catalog_id,
                    "REGISTRATION_IDENTITY_CHANGED")
            _, coverage = _episode(memory, entry["episode_ref"])
            require(coverage == entry["coverage"], "REGISTRATION_SOURCE_CHANGED")
            entries.append({"registration_ref": asdict(reference), **{key: deepcopy(entry[key]) for key in (
                "episode_ref", "run_id", "request_id", "task_contract_ref", "source_role", "coverage")}})
    require(bool(entries), "EMPTY")
    entries.sort(key=lambda row: (row["run_id"], canonical_hash(row["episode_ref"]), row["source_role"]))
    selected, roles, runs = _choose(entries, selection)
    value = {"catalog_id": catalog_id, "selection": selection, "applicability": deepcopy(applicability),
        "learning_request_ref": deepcopy(learning_request_ref), "entries": entries, "runs": runs,
        "selected_episode_refs": selected, "selected_source_roles": roles,
        "required_episode_refs": selection["required_episode_refs"], "frozen": True,
        "scope": "All validated registered runs, once each; original variants and explicit exclusions retained. "
            "Source roles are declarations, not verified positive or negative transfer."}
    selection_ref = memory._write("catalog_selection_" + canonical_hash(value), "episode_catalog_selection", value)
    frozen_ref = memory.freeze_episode_set(selected, applicability=applicability,
        source_roles=roles, learning_request_ref=learning_request_ref)
    frozen = {key: value for key, value in memory.read(frozen_ref).items()
              if key not in {"kind", "schema_version", "logical_namespace"}}
    frozen.update(catalog_id=catalog_id, catalog_selection_ref=asdict(selection_ref))
    result = memory._write("catalog_set_" + canonical_hash(frozen), "episode_set", frozen)
    verify_catalog_selection(memory, result)
    return result


def verify_catalog_selection(memory, frozen_set_ref):
    """Recheck immutable selected registrations, without consulting a growing catalog.

    为什么要有它：新经历只能影响下一次选择，不能改变已经冻结的提炼或B/B+条件。
    """
    frozen = memory.read(frozen_set_ref)
    require(frozen.get("kind") == "episode_set" and frozen.get("frozen") is True, "FROZEN_SET_REQUIRED")
    receipt = memory.read(frozen["catalog_selection_ref"])
    require(receipt.get("kind") == "episode_catalog_selection" and receipt.get("frozen") is True
            and receipt["catalog_id"] == frozen["catalog_id"]
            and receipt["selection"] == normalize_selection(receipt["selection"])
            and receipt["applicability"] == frozen["applicability"]
            and receipt["learning_request_ref"] == frozen.get("learning_request_ref")
            and receipt["selected_episode_refs"] == [row["episode_ref"] for row in frozen["episodes"]]
            and receipt["selected_source_roles"] == [row["source_role"] for row in frozen["episodes"]],
            "FROZEN_SELECTION_CHANGED")
    for row in receipt["entries"]:
        registered = memory.read(row["registration_ref"])
        require(registered.get("kind") == "episode_catalog_registration"
                and registered.get("catalog_id") == receipt["catalog_id"]
                and all(registered[key] == row[key] for key in (
                    "episode_ref", "run_id", "request_id", "task_contract_ref", "source_role", "coverage")),
                "REGISTRATION_CHANGED")
        _, coverage = _episode(memory, row["episode_ref"])
        require(coverage == row["coverage"], "REGISTRATION_SOURCE_CHANGED")
    selected, roles, runs = _choose(receipt["entries"], receipt["selection"])
    require(receipt["selected_episode_refs"] == selected and receipt["selected_source_roles"] == roles
            and receipt["runs"] == runs
            and receipt["required_episode_refs"] == receipt["selection"]["required_episode_refs"],
            "RUN_SELECTION_CHANGED")
    return frozen
