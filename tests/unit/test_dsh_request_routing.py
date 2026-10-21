"""Natural task selection reaches existing owners without granting new authority."""
import io
import json
from pathlib import Path

import pytest

from scripts import dsh_request_routing as routing
from scripts import dsh_run_request as bridge
from src.libs.llm import ChatResponse
from src.mod_agent.planning.task_intent import ASSESSMENT_FIELDS, validate_intent
from tests.unit import test_dsh_new_project as new_project_tests
from tests.unit.test_dsh_run_request import execution_config  # noqa: F401

profile = new_project_tests.profile


def decision(action, target="", name="", question=""):
    return dict(action=action, target_id=target, project_name=name, clarification=question)


class Model:
    model = "qwen3:4b-instruct"

    def __init__(self, value, during=None):
        self.value, self.during, self.calls = value, during, []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.during:
            self.during()
        value = self.value if isinstance(self.value, dict) and "assessment" in self.value else {
            "assessment": {key: False for key in ASSESSMENT_FIELDS}, "decision": self.value}
        native_schema = kwargs["response_format"]
        properties = native_schema.get("properties") or native_schema["oneOf"][0]["properties"]
        if isinstance(value["assessment"], dict) and set(value["assessment"]) == set(ASSESSMENT_FIELDS):
            roles = properties["assessment"]["properties"]["catalogue_roles"]["properties"]
            selected = value["decision"] or {}
            current = selected.get("target_id") if selected.get("action") in {"modify_project", "resume_task"} else None
            value = {**value, "assessment": {"catalogue_roles": {
                key: "current_target" if key == current else "not_referenced" for key in roles}, **value["assessment"]}}
        if "execution_start_anchors" in properties:
            value = {"execution_start_anchors": [], **value}
        if "unit_roles" in properties:
            value = {"unit_roles": {key: "keep" for key in properties["unit_roles"]["properties"]}, **value}
        return ChatResponse(json.dumps(value), self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 20})


def invoke(profile, model, **kwargs):
    return routing.run_natural_request(execution_config=str(profile), description="原始自然语言需求",
        llm=model, identity_reader=lambda *_: {"model": model.model, "digest": "a" * 64}, **kwargs)


def capture_product(monkeypatch):
    calls = []

    def execute(argv):
        calls.append(argv)
        print(json.dumps({"status": "awaiting_approval", "run_id": argv[argv.index("--run-id") + 1]}))
        return 0

    monkeypatch.setattr(bridge.cli, "main", execute)
    return calls


def test_trusted_executor_receives_original_allocated_argv_without_default_dispatch(profile, monkeypatch):
    monkeypatch.setattr(bridge.cli, "main", lambda *_: pytest.fail("default transport must not run"))
    calls = []
    def executor(argv):
        calls.append(argv)
        print(json.dumps({"status": "awaiting_approval", "run_id": argv[argv.index("--run-id") + 1]}))
        return 0
    result = invoke(profile, Model(decision("modify_project", target="configured_project")), product_executor=executor)
    assert result["status"] == "awaiting_approval" and len(calls) == 1
    argv = calls[0]
    assert argv[-1] == "原始自然语言需求" and argv[0] == "run-request"
    assert "--defer-approval" in argv


def test_natural_new_request_initializes_neutral_project_and_dispatches_literal_request(profile, monkeypatch):
    calls = capture_product(monkeypatch)
    model = Model(decision("new_project", name="远征计时器"))
    result = invoke(profile, model)
    assert result["status"] == "awaiting_approval" and len(calls) == len(model.calls) == 1
    argv = calls[0]
    target = Path(argv[argv.index("--target-project") + 1])
    assert target != profile.parent / "Example" and target.is_dir()
    assert "override void Load()" in (target / "Plugin.cs").read_text()
    assert argv[-1] == "原始自然语言需求"
    assert argv[argv.index("--task-model-call-limit") + 1] == "9"
    assert result["budget"]["new_task_combined_maximum"] == 10
    assert result["routing_usage"] == {"input_tokens": 100, "output_tokens": 20}
    saved = Path(result["routing_record_path"])
    assert json.loads(saved.read_text(encoding="utf-8")) == result
    assert len(list(saved.parent.glob("model_trace_bodies/*.response.txt"))) == 1


def test_modify_uses_configured_source_and_original_permission_flags(profile, monkeypatch):
    calls = capture_product(monkeypatch)
    result = invoke(profile, Model(decision("modify_project", target="configured_project")))
    assert result["status"] == "awaiting_approval"
    argv = calls[0]
    assert argv[argv.index("--target-project") + 1] == str(profile.parent / "Example")
    assert "--defer-approval" in argv and "--allow-source-disclosure" not in argv


def _meter_profile(profile):
    document = json.loads(profile.read_text(encoding="utf-8"))
    arguments = document["arguments"]
    for flag in ("--task-model-call-limit", "--stage-model-call-limits", "--model-call-budget-policy"):
        if flag in arguments:
            index = arguments.index(flag)
            del arguments[index:index + 2]
    arguments.extend(["--model-call-budget-policy", "meter_only"])
    profile.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("action", ["modify_project", "new_project"])
@pytest.mark.parametrize("conversation", [None, "meter-conversation"])
def test_meter_only_routing_needs_no_numeric_quota_or_new_approval(profile, monkeypatch, action, conversation):
    _meter_profile(profile)
    calls = capture_product(monkeypatch)
    model = Model(decision(action, target="configured_project" if action == "modify_project" else "",
                           name="计量测试" if action == "new_project" else ""))
    result = invoke(profile, model, conversation_id=conversation)
    assert result["status"] == "awaiting_approval"
    assert len(calls) == len(model.calls) == result["routing_model_calls_reserved"] == 1
    assert result["routing_usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert result["budget"]["policy"] == "meter_only"
    assert result["budget"]["new_task_combined_maximum"] is None
    assert result["budget"]["product_new_task_call_limit"] is None
    assert "--task-model-call-limit" not in calls[0] and "--stage-model-call-limits" not in calls[0]
    assert "--defer-approval" in calls[0] and "--allow-source-disclosure" not in calls[0]
    if conversation is not None:
        assert result["budget"]["conversation_reserved_routing_plus_new_product_maximum"] is None
        assert result["cumulative_routing_calls"] == 1
        assert result["cumulative_routing_usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert json.loads(Path(result["routing_record_path"]).read_text(encoding="utf-8")) == result


def test_meter_conversation_preserves_provider_failure_and_unknown_usage(profile, monkeypatch):
    _meter_profile(profile)
    product = {"status": "blocked", "task_agent_outcome": {
        "status": "blocked", "termination": {"code": "TASK_AGENT_PROVIDER_TURN_FAILED"},
        "provider_turns": [{"turn": 7, "status": "failed", "error_type": "AgentTurnProviderError"}],
        "usage": None,
    }}
    calls = []
    def executor(argv):
        calls.append(argv)
        print(json.dumps(product))
        return 0
    model = Model(decision("modify_project", target="configured_project"))
    result = invoke(profile, model, product_executor=executor, conversation_id="meter-failure", turn_id="first")
    assert result["status"] == "blocked" and result["product_result"] == product
    assert result["budget"]["conversation_reserved_routing_plus_new_product_maximum"] is None
    assert result["cumulative_routing_calls"] == 1
    assert result["product_result"]["task_agent_outcome"]["usage"] is None
    assert json.loads(Path(result["routing_record_path"]).read_text(encoding="utf-8")) == result
    replay = invoke(profile, Model(None), product_executor=executor, conversation_id="meter-failure", turn_id="first")
    assert replay == result and len(calls) == 1


@pytest.mark.parametrize("value,status", [
    (decision("clarify", question="要修改哪个工程？"), "needs_clarification"),
    (decision("modify_project", target="C:/outside"), "routing_failed"),
    ({**decision("new_project", name="A"), "allow_build": True}, "routing_failed"),
])
def test_ambiguity_or_unbound_model_output_cannot_start_product(profile, value, status):
    model = Model(value)
    result = invoke(profile, model)
    assert result["status"] == status and result["product_owner_invoked"] is False
    assert len(model.calls) == 1
    assert not list((profile.parent / "runs").glob(".dsh-inputs/*"))


def test_profile_drift_after_model_stops_before_task_allocation(profile):
    model = Model(decision("new_project", name="A"), during=lambda: profile.write_text(profile.read_text() + " "))
    result = invoke(profile, model)
    assert result["reason"] == "TASK_INTENT_CONTEXT_CHANGED" and not result["product_owner_invoked"]


def test_preview_preserves_model_evidence_but_does_not_initialize(profile):
    result = invoke(profile, Model(decision("new_project", name="A")), preview_only=True)
    assert result["status"] == "route_selected" and not result["product_owner_invoked"]
    assert not (profile.parent / "runs/new_mod_sources").exists()


def test_resume_dispatches_selected_original_owner_without_allocating_new_budget(profile, monkeypatch):
    context = {"projects": [], "tasks": [{"id": "prior"}], "active_task_id": "prior"}
    monkeypatch.setattr(routing, "routing_context", lambda **_: (context, {}))
    calls = []
    monkeypatch.setattr(bridge, "advance_task", lambda **kw: calls.append(kw) or {"status": "blocked"})
    monkeypatch.setattr(bridge, "prepare_new_task", lambda **_: pytest.fail("Must retain the same task"))
    result = invoke(profile, Model(decision("resume_task", target="prior")),
        active_run_id="prior", allocation_config="authorized.json")
    assert result["status"] == "blocked" and len(calls) == 1
    assert calls[0]["run_id"] == "prior" and calls[0]["allocation_config"] == "authorized.json"
    assert result["product_execution_started"] is None


def test_new_requirement_on_selected_task_uses_followup_owner(profile, monkeypatch):
    context = {"projects": [{"id": "prior"}], "tasks": [{"id": "prior"}], "active_task_id": "prior"}
    monkeypatch.setattr(routing, "routing_context", lambda **_: (context, {}))
    selected = bridge.prepare_new_task(execution_config=str(profile), description="原始自然语言需求")
    calls = []
    monkeypatch.setattr(bridge, "prepare_followup_task", lambda **kw: calls.append(kw) or selected, raising=False)
    captured = capture_product(monkeypatch)
    result = invoke(profile, Model(decision("modify_project", target="prior")), active_run_id="prior")
    assert result["status"] == "awaiting_approval" and calls[0]["parent_run_id"] == "prior"
    assert calls[0]["description"] == captured[0][-1] == "原始自然语言需求"


def test_model_cannot_cross_action_target_sets():
    context = {"projects": [{"id": "project"}], "tasks": [{"id": "task"}]}
    with pytest.raises(ValueError, match="TARGET_NOT_AVAILABLE"):
        validate_intent(decision("resume_task", target="project"), context)


def test_public_cli_selects_routing_without_rewriting_text_or_adding_authority(profile, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(routing, "run_natural_request", lambda **kwargs: calls.append(kwargs) or {"status": "needs_clarification"})
    monkeypatch.setattr(bridge.sys, "stdin", io.TextIOWrapper(io.BytesIO("新建一个Mod".encode())))
    assert bridge.main(["--execution-config", str(profile), "--route-request", "--active-run-id", "prior"]) == 0
    assert calls == [dict(execution_config=str(profile), description="新建一个Mod", active_run_id="prior",
        allocation_config=None, delivery_root=None, parent_execution_config=None, preview_only=False)]
    assert json.loads(capsys.readouterr().out)["status"] == "needs_clarification"


def test_recorded_fixed_profile_can_route_but_cannot_relaunch_original_run(profile):
    document = json.loads(profile.read_text())
    document["arguments"] += ["--run-id", "prior", "--request-id", "request_prior"]
    profile.write_text(json.dumps(document))
    (profile.parent / "runs/prior").mkdir(parents=True)
    result = invoke(profile, Model(decision("new_project", name="A")), preview_only=True)
    assert result["status"] == "route_selected"
    with pytest.raises(ValueError, match="RUN_ALREADY_EXISTS"):
        bridge.build_product_argv(execution_config=str(profile), description="original")


def test_structured_product_block_is_not_relabelled_unknown(profile, monkeypatch):
    product = {"run_id": "prior", "product_execution_result": {"execution_disposition": "platform_blocked"}}
    def execute(_):
        print(json.dumps(product))
        return 2
    monkeypatch.setattr(bridge.cli, "main", execute)
    result = invoke(profile, Model(decision("modify_project", target="configured_project")))
    assert result["status"] == "platform_blocked" and result["product_result"] == product


def test_natural_resume_recognizes_prepared_project_after_interruption(profile, monkeypatch):
    selected = bridge.prepare_new_task(execution_config=str(profile), description="原始新建需求", project_name="已准备工程")
    _, options = bridge._configuration_arguments(selected)
    run_id = options["--run-id"]
    context, _ = routing.routing_context(execution_config=str(profile), active_run_id=run_id)
    assert context["tasks"][0]["original_request"] == "原始新建需求"
    assert context["tasks"][0]["status"] == "prepared_not_started"
    calls = capture_product(monkeypatch)
    result = invoke(profile, Model(decision("resume_task", target=run_id)), active_run_id=run_id)
    assert result["status"] == "awaiting_approval" and calls[0][-1] == "原始新建需求"
    assert calls[0][calls[0].index("--run-id") + 1] == run_id
