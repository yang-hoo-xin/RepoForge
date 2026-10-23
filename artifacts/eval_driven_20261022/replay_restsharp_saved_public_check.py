"""Recheck saved model bytes in a new workspace after a checker path repair.

为什么要有它：先用原候选证明检查器修复，保留未知旧结果；不靠重新解题验证执行器。
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryBackend, _IGNORED, _snapshot
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.task_agent_v2_state import TaskAgentV2State

NATIVE = Path("C:/tmp/rfext-restsharp-v2-b1/runs/dsh_7e85b512903f4692a7812b24edeadefe/a/task-agent-execution/execution.json")
ROOT = Path("C:/tmp/rfext-restsharp-v2-b1-check-replay")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-interrupted-check", action="store_true")
    args = parser.parse_args()
    native_ref = _ref(NATIVE)
    native = read_bound_json(native_ref)
    stores = [parent for parent in NATIVE.parents if (parent / native["task_contract_ref"]).is_file()]
    if len(stores) != 1 or native["status"] != "unknown":
        raise ValueError("SAVED_UNKNOWN_NATIVE_REQUIRED")
    state = TaskAgentV2State(stores[0], native["task_contract_ref"])
    candidate = state.validate_candidate(native["candidate_ref"])
    baseline = state.load(state.contract["baseline_manifest_ref"])
    original = Path(baseline["source_root"])
    before = _snapshot(original)
    materialized = state.materialize(native["candidate_ref"])
    source = ROOT / "source"
    if args.continue_interrupted_check:
        if not source.is_dir() or (ROOT / "result.json").exists():
            raise ValueError("INCOMPLETE_REPLAY_REQUIRED")
        _save(ROOT / "interruption-observed.json", {"original_process_handle": 35069,
            "handle_missing": True, "matching_processes_found": False, "previous_result_missing": True,
            "previous_exit_code": None, "previous_termination_cause": "unknown",
            "new_check_uses_new_output_directory": True, "model_calls": 0})
    else:
        ROOT.mkdir()
        shutil.copytree(original, source, ignore=shutil.ignore_patterns(*_IGNORED))
        for path, text in candidate["files"].items():
            if path not in state.contract["allowed_edit_paths"]:
                raise ValueError("UNAUTHORIZED_CANDIDATE_EDIT")
            (source / path).write_bytes(text.encode("utf-8"))
    expected = {row["path"]: row["sha256"] for row in before}
    expected.update({path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in candidate["files"].items()})
    if {row["path"]: row["sha256"] for row in _snapshot(source)} != expected:
        raise ValueError("REPLAY_SOURCE_INPUTS_CHANGED")
    if any((source / path).read_bytes() != text.encode("utf-8") for path, text in materialized.items()):
        raise ValueError("SAVED_CANDIDATE_BYTES_CHANGED")
    backend = CSharpRepositoryBackend(profile_ref=state.contract["execution_profile_ref"], source_root=source,
        workspace_root=ROOT, output_root=ROOT / "checks")
    result = backend.run_build_test(source, "saved-restsharp-public-replay", "isolated_diagnostic_replay")
    if _snapshot(original) != before or _ref(NATIVE) != native_ref:
        raise ValueError("ORIGINAL_EVIDENCE_CHANGED")
    evidence = _save(ROOT / "result.json", {"schema_version": "saved_candidate_public_recheck_v1",
        "native_execution_ref": native_ref, "original_status": native["status"],
        "candidate_ref": native["candidate_ref"], "candidate_source_id": candidate["candidate_source_id"],
        "status": result["status"], "verification_summary": result["verification_summary"],
        "report_ref": result["report_ref"], "model_calls": 0,
        "original_task_resumed": False, "old_check_rewritten": False, "model_received_replay_feedback": False})
    print(json.dumps({"result_ref": evidence, "status": result["status"], "verification_summary": result["verification_summary"]}))


if __name__ == "__main__":
    main()
