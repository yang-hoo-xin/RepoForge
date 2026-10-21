"""Native origin and immutable post-Apply checks; no real provider or compiler."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.execution.task_delegation import _native_verification_binding
from tests.unit import test_task_agent_delivery_binding as fixtures


@pytest.fixture
def delivery(short_workspace, monkeypatch, request):
    return fixtures.delivery.__wrapped__(short_workspace, monkeypatch, request)


def arguments(delivery):
    case, values = delivery
    return case, dict(request=SimpleNamespace(isolated_workspace_root=str(values["root"]),
        artifact_store_root=str(case.inputs.store.root), run_id=case.state["run_id"],
        request_id=case.state["request_id"]), plan_ref=values["report"]["implementation_plan_ref"],
        code_identity=values["report"]["materialized_code_identity"], proposal=values["proposal"],
        reference=values["verification"], report=values["report"])


@pytest.mark.parametrize("delivery", ["baseline", "current"], indirect=True)
def test_native_origin_admitted_before_apply_and_immutable_after_apply(delivery, monkeypatch):
    case, args = arguments(delivery)
    assert all(row.get("generation", {}) == {} for row in args["proposal"]["diff_intent"])
    binding = _native_verification_binding(**args)
    assert binding["model_response_sha256"] == args["report"]["model_response_sha256"]
    assert binding["immutable_file_refs"]
    assert _native_verification_binding(**args, recorded_binding=binding) == binding
    case.inputs.behavior.write_text("Applied target is no longer the baseline", encoding="utf-8")
    with pytest.raises(ValueError):
        _native_verification_binding(**args, recorded_binding=binding)
    monkeypatch.setattr("src.mod_agent.task_agent_execution.validate_task_agent_handoff",
        lambda *_args, **_kwargs: pytest.fail("Post-Apply must not re-materialize B0"))
    assert _native_verification_binding(**args, recorded_binding=binding, before_apply=False) == binding


@pytest.mark.parametrize("change", ["run", "request", "plan", "code", "report", "response", "proposal", "approval"])
def test_native_candidate_identity_cannot_be_substituted(delivery, change):
    _, args = arguments(delivery)
    args = deepcopy(args)
    if change in {"run", "request"}:
        setattr(args["request"], change + "_id", "other")
    elif change == "plan":
        args["plan_ref"]["artifact_hash"] = "e" * 64
    elif change == "code":
        args["code_identity"]["code_sha256"] = "e" * 64
    elif change == "report":
        args["reference"]["sha256"] = "e" * 64
    elif change == "response":
        args["report"]["model_response_sha256"] = "e" * 64
    elif change == "proposal":
        args["proposal"]["diff_intent"][0]["rationale"] = "not the actual response"
    else:
        args["proposal"]["context_pack"]["task_agent_lineage"]["approval_inherited"] = True
    with pytest.raises(ValueError, match="TASK_DELEGATION_"):
        _native_verification_binding(**args)


@pytest.mark.parametrize("delivery", ["current"], indirect=True)
@pytest.mark.parametrize("change", ["missing", "empty", "loop", "derived_patch", "parent_candidate", "worker"])
def test_post_apply_origin_requires_unchanged_admitted_witnesses(delivery, change):
    case, args = arguments(delivery)
    binding = _native_verification_binding(**args)
    if change == "missing":
        binding = None
    elif change == "empty":
        binding["immutable_file_refs"] = []
    else:
        if change == "loop":
            selected = next(ref for ref in binding["immutable_file_refs"] if ref["path"].endswith("loop-result.json"))
        elif change == "derived_patch":
            candidate = json.loads(Path(binding["candidate_ref"]["path"]).read_bytes())
            selected = candidate["model_origin"]["derived_content_ref"]
            assert selected in binding["immutable_file_refs"]
        elif change == "parent_candidate":
            selected = next(ref for ref in binding["immutable_file_refs"] if Path(ref["path"]).name.startswith("candidate-")
                and not Path(ref["path"]).name.startswith("candidate-check") and ref != binding["candidate_ref"])
        else:
            selected = next(ref for ref in binding["immutable_file_refs"] if ref["path"].endswith("worker.json"))
        Path(selected["path"]).write_bytes(b"changed immutable origin")
    case.inputs.behavior.write_text("already applied", encoding="utf-8")
    with pytest.raises(ValueError, match="TASK_DELEGATION_"):
        _native_verification_binding(**args, recorded_binding=binding, before_apply=False)
