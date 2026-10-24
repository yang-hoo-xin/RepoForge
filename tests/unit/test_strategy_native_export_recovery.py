"""Normal CLI recovery export uses initial conditions and the final candidate."""
from pathlib import Path

import pytest

from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.native_execution_lineage import validate_native_execution_lineage
from src.mod_agent.evaluation.strategy_native_export import (
    export_strategy_execution,
    validate_native_strategy_execution,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, load_v2_artifact
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_task_agent_v2_delivery import delivery_case as delivery_case
from tests.unit.test_task_agent_v2_recovery import (
    test_cli_resumes_applied_public_failure_from_actual_source_and_requires_new_approval as run_recovery,
)
from tests.unit.test_task_agent_v2_recovery import (
    test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product as run_interruption,
)


@pytest.fixture
def short_workspace(tmp_path_factory):
    """Retain controlled recovery originals under the selected short pytest root."""
    return tmp_path_factory.mktemp("r")


def test_normal_cli_rebase_exports_original_conditions_final_delivery_and_cumulative_cost(delivery_case, monkeypatch, capsys):
    root, initial_checkpoint_ref, _isolated, _prepared = delivery_case
    initial_checkpoint = load_v2_artifact(root, initial_checkpoint_ref)
    initial_ref = initial_checkpoint["handoff"]["execution_ref"]
    initial_native = read_bound_json(initial_ref)
    initial_state = TaskAgentV2State(root, initial_native["task_contract_ref"])
    baseline = initial_state.load(initial_state.contract["baseline_manifest_ref"])
    original_request = initial_state.load(initial_state.contract["original_request_ref"])["original_request"]
    baseline_hash = canonical_hash({path: row["sha256"] for path, row in baseline["files"].items()})
    run_recovery(delivery_case, monkeypatch, capsys)
    natives = [(path, read_bound_json(_ref(path))) for path in root.glob("task-agent-execution/resume-*/execution.json")]
    assert natives
    final_path, final_native = next((path, value) for path, value in natives if value.get("status") == "candidate_ready")
    final_ref = _ref(final_path)
    lineage = validate_native_execution_lineage(artifact_store_root=root, native_execution_ref=final_ref, initial_native_ref=initial_ref)
    assert lineage["initial_native_ref"] == initial_ref
    assert lineage["final_state"].contract_ref != initial_state.contract_ref
    receipts = [(path.relative_to(root).as_posix(), load_v2_artifact(root, path.relative_to(root).as_posix()))
        for path in (root / "v2/delivery-receipt").glob("*.json")]
    built_ref, _built = next((ref, value) for ref, value in receipts if value.get("status") == "built"
        and value["candidate_ref"] == final_native["candidate_ref"])
    source = _save(root / "diagnostic-source.json", {"task_id": "diagnostic", "task_family": "controlled",
        "source_version": "controlled-original-baseline", "source_files_sha256": baseline_hash,
        "original_request_sha256": canonical_hash(original_request)})
    exposure = _save(root / "diagnostic-exposure.json", {"task_id": "diagnostic", "source_ref": source,
        "attempt": 1, "exposure": "exposed"})
    protocol = _save(root / "diagnostic-protocol.json", {"schema_version": "strategy_trial_protocol_v1",
        "invariants": {}, "invariants_sha256": canonical_hash({}), "task_families": ["controlled"],
        "planned_pairs": [{"task_id": "diagnostic",
            "task_family": "controlled", "attempt": 1, "source_ref": source, "exposure_ref": exposure,
            "current_context": {"runtime": "dotnet", "target_domain": "csharp_repository"}}]})
    call_id = next(row["call_id"] for row in reversed(final_native["completed_tool_records"]) if row["name"] == "finish")
    exported_ref = export_strategy_execution(native_execution_ref=final_ref, call_id=call_id,
        artifact_store_root=root, output_root=root / "resumed-export", protocol_ref=protocol,
        task_id="diagnostic", arm="B", attempt=1, delivery_receipt_ref=built_ref)
    exported = read_bound_json(exported_ref)
    assert exported["planned_source_binding"] == "source_and_request_verified"
    assert exported["actual_common_conditions"]["task"]["source_files_sha256"] == baseline_hash
    assert exported["candidate_state"]["task_contract_ref"] == final_native["task_contract_ref"]
    assert exported["known_cost"]["model_calls"] == final_native["budget_after"]["consumed"]
    assert exported["resource_evidence"]["known_usage_sums"]["input_tokens"] == sum(
        row["usage"]["input_tokens"] for row in final_native["provider_turns"] if row.get("usage"))
    assert read_bound_json(exported["delivery_ref"])["status"] == "delivered"
    assert validate_native_strategy_execution(exported_ref)["status"] == "native_execution_verified"


def test_normal_cli_checkpoint_only_origin_exports_without_inventing_initial_native(repository_workspace, monkeypatch, capsys):
    run_interruption(repository_workspace, monkeypatch, capsys, "checkpoint_completed")
    paths = list((repository_workspace / "runs").glob("*/a/task-agent-execution/resume-*/execution.json"))
    assert len(paths) == 1
    reference = _ref(paths[0])
    root = paths[0].parents[2]
    result = validate_native_execution_lineage(artifact_store_root=root, native_execution_ref=reference)
    assert result["initial_native_ref"] is None and result["initial_checkpoint_ref"]
    assert result["chain"][-1]["kind"] == "checkpoint_only_origin"
    state = result["initial_state"]
    baseline = state.load(state.contract["baseline_manifest_ref"])
    source = _save(root / "diagnostic-source.json", {
        "task_id": "diagnostic", "task_family": "controlled", "source_version": "controlled-original-baseline",
        "source_files_sha256": canonical_hash({path: row["sha256"] for path, row in baseline["files"].items()}),
        "original_request_sha256": canonical_hash(state.load(state.contract["original_request_ref"])["original_request"])})
    exposure = _save(root / "diagnostic-exposure.json", {"task_id": "diagnostic", "source_ref": source,
        "attempt": 1, "exposure": "exposed"})
    protocol = _save(root / "diagnostic-protocol.json", {"schema_version": "strategy_trial_protocol_v1",
        "invariants": {}, "invariants_sha256": canonical_hash({}), "task_families": ["controlled"],
        "planned_pairs": [{"task_id": "diagnostic",
            "task_family": "controlled", "attempt": 1, "source_ref": source, "exposure_ref": exposure,
            "current_context": {"runtime": "dotnet", "target_domain": "csharp_repository"}}]})
    native = result["final_native"]
    call_id = next(row["call_id"] for row in reversed(native["completed_tool_records"]) if row["name"] == "finish")
    exported = export_strategy_execution(native_execution_ref=reference, call_id=call_id, artifact_store_root=root,
        output_root=root / "checkpoint-only-export", protocol_ref=protocol, task_id="diagnostic", arm="B", attempt=1)
    assert validate_native_strategy_execution(exported)["status"] == "native_execution_verified"
    evidence = read_bound_json(exported)["execution_lineage"]
    assert evidence["initial_native_ref"] is None and evidence["initial_checkpoint_ref"] == result["initial_checkpoint_ref"]
    with pytest.raises(ValueError, match="INITIAL_EXECUTION_NOT_REACHED"):
        validate_native_execution_lineage(artifact_store_root=root, native_execution_ref=reference,
            initial_native_ref={"path": str(Path(root) / "nonexistent/execution.json"), "sha256": "0" * 64})
