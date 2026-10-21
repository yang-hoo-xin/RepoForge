"""Normal entry selection and fail-closed authority; no real model or compiler."""
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.patch_worker_configuration import SCHEMA_VERSION, PatchWorkerConfiguration
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from tests.unit.test_dsh_run_request import (
    _arguments,
    _call,
    _install_execution_boundaries,
)
from tests.unit.test_dsh_run_request import execution_config as execution_config


def selection(**updates):
    values = dict(technical_executor="task_agent", constraint_policy="source_priority_v1",
        task_agent_allowed_edit_paths=("Plugin.cs",), local_ollama=True, defer_approval=True,
        run_gtfo_runtime=False, source_retrieval_mode="rg_live", model_call_budget_policy="task_total_only",
        patch_worker_configuration=PatchWorkerConfiguration(PatchAgentLoopPolicy(), True, {}, object()),
        planning_target_evidence_ids=(), repair_ollama_model=None)
    values.update(updates)
    return local_entry._request_technical_execution_selection(**values)


def test_default_selection_has_no_extra_fields():
    assert selection(technical_executor="workflow", constraint_policy=None,
                     task_agent_allowed_edit_paths=()) is None


def test_task_selection_retains_explicit_scope():
    value = selection()
    assert value["allowed_edit_paths_relative_to_project"] == ["Plugin.cs"]
    assert value["scope"] == "common_initial_planning_then_continuous_technical_loop"


def test_current_candidate_mode_is_explicit_and_requires_task_executor():
    assert selection(task_agent_candidate_mode="current")["candidate_mode"] == "current"
    assert "candidate_mode" not in selection()
    with pytest.raises(local_entry.LocalEntryError, match="MODE_WITHOUT_EXECUTOR"):
        selection(technical_executor="workflow", task_agent_allowed_edit_paths=(), task_agent_candidate_mode="current")
    with pytest.raises(local_entry.LocalEntryError, match="CANDIDATE_MODE_INVALID"):
        selection(task_agent_candidate_mode="invented")


@pytest.mark.parametrize("paths", [(), ("../Plugin.cs",), ("/Plugin.cs",), ("C:/Plugin.cs",),
    ("sub\\Plugin.cs",), ("a/./Plugin.cs",), ("Plugin.cs", "plugin.cs"), ("a/CON.txt",),
    ("Plugin.cs:stream",), ("Plugin.cs ",), ("bin/a.cs",), ["Plugin.cs"], (True,)])
def test_task_scope_rejects_ambiguous_paths(paths):
    with pytest.raises((ValueError, local_entry.LocalEntryError)):
        selection(task_agent_allowed_edit_paths=paths)


@pytest.mark.parametrize("changed", [dict(local_ollama=False), dict(defer_approval=False),
    dict(run_gtfo_runtime=True), dict(repair_ollama_model="other"), dict(constraint_policy=None),
    dict(source_retrieval_mode="indexed_source"), dict(model_call_budget_policy="stage_and_total"),
    dict(patch_worker_configuration=None), dict(planning_target_evidence_ids=("source-1",)),
    dict(technical_executor="workflow")])
def test_unimplemented_combinations_rejected_before_execution(changed):
    with pytest.raises((ValueError, local_entry.LocalEntryError)):
        selection(**changed)


@pytest.mark.parametrize("value", ['{}', 'null', 'true', '[1]', '[]'])
def test_cli_scope_requires_nonempty_string_list(value):
    with pytest.raises(ValueError):
        cli._task_agent_edit_paths(value)


def test_normal_cli_source_priority_reaches_plan_and_approval(execution_config, monkeypatch, capsys):
    target, roles, builds = _install_execution_boundaries(execution_config.parent, monkeypatch)
    before = (target / "Plugin.cs").read_bytes()
    _arguments(execution_config, lambda values: [*values, "--constraint-policy", "source_priority_v1"])
    assert _call(execution_config, "Add a marker in Plugin.Load", monkeypatch) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "awaiting_approval", result
    assert result["actual_write_performed"] is False
    assert (target / "Plugin.cs").read_bytes() == before
    root = Path(result["run_root"])
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["technical_execution"] == {"technical_executor": "workflow", "constraint_policy": "source_priority_v1"}
    plan = json.loads(Path(result["artifacts"]["implementation_plan"]).read_text(encoding="utf-8"))["data"]
    assert plan["constraint_context"]["policy"] == "source_priority_v1"
    assert roles and builds and result["patch_execution_receipts"]


@pytest.mark.parametrize("candidate_mode,replan,runner_selection,malformed_first,meter_only,search_replan", [
    ("baseline", False, False, False, False, False), ("current", False, False, False, False, False),
    ("current", True, False, False, False, False), ("current", True, True, False, False, False),
    ("current", True, True, True, False, False), ("current", True, True, True, True, False),
    ("current", True, False, False, False, True)],
    ids=["baseline", "current", "current-replan", "development-runner-current-replan",
         "malformed-first-then-current-repair", "meter-only-normal-repair",
         "search-evidence-replan"])
def test_normal_cli_task_agent_uses_actual_task_loop_and_defers_apply(
        execution_config, monkeypatch, capsys, candidate_mode, replan, runner_selection, malformed_first, meter_only,
        search_replan):
    target, roles, builds = _install_execution_boundaries(execution_config.parent, monkeypatch)
    if search_replan:
        # Scope: a second authorized source file that is absent from the initial Load plan.
        # 为什么要有它：正常入口必须证明搜索证据能经Evidence和Planning产生新绑定，不能只复用初始绑定假装已接通。
        (target / "Configuration.cs").write_text(
            'namespace Example; public static class Configuration { public const string ConfigKey = "marker"; }\n',
            encoding="utf-8")
        import httpx

        fixture_transport = httpx.HTTPTransport.handle_request

        def search_reassessment_transport(transport, request):
            if request.method == "POST" and request.url.path == "/api/chat":
                payload = json.loads(request.content)
                contexts = []
                for message in payload.get("messages", []):
                    if message.get("role") != "user":
                        continue
                    try:
                        context = json.loads(message["content"])
                        if "patch_agent_context_encoding" in context:
                            context = local_entry._restore_patch_agent_context(context)
                        contexts.append(context)
                    except (TypeError, ValueError):
                        continue
                query_context = next((row for row in contexts
                    if row.get("message_type") == "patch_upstream_evidence_queries"), None)
                content = None
                if query_context is not None:
                    query = {"query": "ConfigKey",
                        "requirement_id": query_context["requirements"][0]["requirement_id"]}
                    if query_context.get("query_contract_version") in {
                            "requirement_queries_v2", "requirement_queries_v3"}:
                        query.update(gap_ids=[], current_fact_needed="Existing configuration source ownership.",
                            modification_impact="Planning needs the exact current configuration edit location.",
                            authorized_source="target_source", metadata_queries=[])
                    content = {"queries": [query]}
                findings = next((row for row in contexts
                    if row.get("message_type") == "product_requirement_findings"
                    and row.get("semantic_feedback", {}).get("current_source_observations")), None)
                if findings is not None:
                    evidence_ids = [row["evidence_id"] for row in findings["source_observations"]
                                    if row["source_type"] == "target_source"]
                    config_id = next(row["evidence_id"] for row in findings["source_observations"]
                                     if "ConfigKey" in row.get("excerpt", ""))
                    evidence_ids = [config_id, *(value for value in evidence_ids if value != config_id)]
                    if findings.get("native_findings_contract") == "planning_readiness_v1":
                        content = {"findings": [{"relation_index": relation["relation_index"],
                            "planning_readiness": "ready", "current_evidence_ids": evidence_ids,
                            "current_basis": "Fixture source identifies both the retained and new edit locations.",
                            "requested_behavior_status": "not_yet_implemented", "blocking_current_facts": [],
                            "edit_target_evidence_ids": evidence_ids} for relation in findings["relations"]]}
                    else:
                        content = {"findings": [{"relation_index": relation["relation_index"],
                            "relationship": "support", "evidence_ids": evidence_ids,
                            "edit_target_evidence_ids": evidence_ids, "unresolved_gaps": []}
                            for relation in findings["relations"]]}
                if content is not None:
                    return httpx.Response(200, request=request, json={"model": payload["model"],
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
            return fixture_transport(transport, request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", search_reassessment_transport)
    derived_hint = "Preserve all existing Load behavior even when the requested marker changes."
    if malformed_first:
        from tests.unit import test_dsh_run_request as entry_fixture

        original_projection = entry_fixture._formal_coverage_model_response

        def projection_with_overbroad_hint(request):
            response = original_projection(request)
            if request.get("message_type") == "product_requirement_projection":
                origins = [row["input_obligation_index"] for row in request["input_obligations"]
                    if "preservation" in row["allowed_roles"]]
                assert origins
                # Scope: model-derived overgeneralization, not a new user instruction.
                # 为什么要有它：正常入口必须真的包含矛盾推断才能验证去除传播，空列表通过不能证明这一点。
                response["requirements"].append({"role": "preservation", "text": derived_hint,
                    "input_obligation_indexes": origins, "query_leads": ["Load"]})
            return response

        monkeypatch.setattr(entry_fixture, "_formal_coverage_model_response", projection_with_overbroad_hint)
    before = (target / "Plugin.cs").read_bytes()
    factory = execution_config.parent / "python_oracle.py"
    expected_failure_observation = {"actual": 0, "expected": 1, "operation": "Load",
        "input": {"marker": "runtime"}, "before": {"markers": 0}, "after": {"markers": 0}, "delta": 0}
    # Scope: synthetic normal-entry diagnostics, not a C5 answer or real behavior run.
    # 为什么要有它：新增运行观察必须真实穿过共享入口到达后续模型请求，不只证明报告字典可以保存字段。
    factory.write_text(
        "from pathlib import Path\nfrom tests.unit.test_task_agent_tools import QualifiedPythonOracle, save\n"
        "class DetailedOracle(QualifiedPythonOracle):\n"
        "    def __call__(self, context):\n"
        "        report = super().__call__(context)\n"
        "        if report['observations']:\n"
        f"            report['observations'][0]['observed'] = {expected_failure_observation!r}\n"
        "            report.pop('report_ref')\n"
        "            report['report_ref'] = save(self.root, report)\n"
        "        return report\n"
        "def make(root):\n    return DetailedOracle(Path(root))\n", encoding="utf-8")
    worker = execution_config.parent / "worker.json"
    worker.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "enable_recovery_loop": True,
        "policy": asdict(PatchAgentLoopPolicy(max_total_model_turns=None if meter_only else 18,
            max_tool_rounds=None if meter_only else 6, max_observation_bytes=None if meter_only else 32768,
            enable_supporting_reads=True, allow_upstream_request=True,
            require_source_inspection=False, finalize_after_first_observation=False,
            model_call_budget_policy="meter_only" if meter_only else "task_total_only")),
        "behavior_evaluator": {"factory_name": "make", "factory_ref": {"path": str(factory),
            "sha256": hashlib.sha256(factory.read_bytes()).hexdigest()}, "kwargs": {"root": "allocated"}}}), encoding="utf-8")
    edit_paths = '["Plugin.cs", "Configuration.cs"]' if search_replan else '["Plugin.cs"]'
    selected_options = ["--constraint-policy", "source_priority_v1", "--technical-executor", "task_agent",
        "--task-agent-candidate-mode", candidate_mode, "--task-agent-edit-paths", edit_paths]
    if runner_selection:
        # Scope: use the experiment's real selection builder, replacing only this test's source scope.
        # 为什么要有它：手工传current的入口测试无法发现runner漏参；外部模型和编译仍由已有脚本夹具替代。
        module_path = Path("artifacts/eval_driven_20261016/agent-first-comparison/e2e/c5/run_c5.py")
        spec = importlib.util.spec_from_file_location("current_mode_entry_runner", module_path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        monkeypatch.setattr(driver, "ALLOW", ("Plugin.cs",))
        runner_protocol = ({"mode": "current_code_metered", "resource_policy": "meter_only",
                            "product_call_limit": None} if meter_only else
                           {"mode": "current_code_development",
                            "development_authorization_ref": {"fixture": "offline_only"}, "product_call_limit": 19})
        runner_arguments = driver.protocol_arm_arguments(runner_protocol, "B",
            source=target, workspace=execution_config.parent / "runs", settings=Path("config/settings.yaml"),
            dependency_ref={"path": "fixture_not_executed"}, worker_ref={"path": str(worker)})
        option_names = {"--constraint-policy", "--technical-executor", "--task-agent-candidate-mode", "--task-agent-edit-paths"}
        selected_options = [item for index, option in enumerate(runner_arguments) if option in option_names
                            for item in runner_arguments[index:index + 2]]
    def execution_arguments(values):
        policy = "meter_only" if meter_only else "task_total_only"
        result = [policy if value == "stage_and_total" else "18" if value == "9" else value for value in values]
        if meter_only:
            for flag in ("--task-model-call-limit", "--stage-model-call-limits"):
                index = result.index(flag)
                del result[index:index + 2]
        return [*result, *selected_options, "--patch-worker-config", str(worker)]

    _arguments(execution_config, execution_arguments)
    configuration = json.loads(execution_config.read_text(encoding="utf-8"))
    configuration["worker_path_bindings"] = [{"path": ["behavior_evaluator", "kwargs", "root"],
                                             "relative_path": "a/oracle"}]
    execution_config.write_text(json.dumps(configuration), encoding="utf-8")
    from src.libs.llm.agent_turn import OllamaChatAgentAdapter, _LoopbackHttpJsonSender
    from src.mod_agent.task_agent_tools import TaskAgentSession
    original_send = _LoopbackHttpJsonSender.__call__
    original_bind = OllamaChatAgentAdapter.bind_context_identity
    prefix_bindings = []
    def observed_binding(client, **binding):
        prefix_bindings.append(binding)
        return original_bind(client, **binding)
    monkeypatch.setattr(OllamaChatAgentAdapter, "bind_context_identity", observed_binding)
    task_turns = []
    revision_state = {}
    completion_errors = []
    original_completion = TaskAgentSession.completion_check

    def observed_completion(session, **kwargs):
        try:
            return original_completion(session, **kwargs)
        except Exception as error:
            completion_errors.append({"type": type(error).__name__, "message": str(error)})
            raise

    monkeypatch.setattr(TaskAgentSession, "completion_check", observed_completion)

    def scripted_task_send(sender, url, payload, headers):
        names = {tool["function"]["name"] for tool in payload.get("tools", [])}
        if "candidate_submit" not in names:
            if replan and "upstream_assessment" in str(payload.get("messages")):
                revision_state["planning_request"] = payload
            response = original_send(sender, url, payload, headers)
            if "verification_questions" in payload.get("format", {}).get("properties", {}):
                selected = json.loads(response["message"]["content"])
                selected["verification_questions"] = selected.pop("missing_evidence")
                response["message"]["content"] = json.dumps(selected)
            return response
        task_turns.append(payload)
        turn = len(task_turns) - int(malformed_first and len(task_turns) > 1)
        if turn == 1:
            if malformed_first and len(task_turns) == 2:
                feedback = json.loads(next(row["content"] for row in reversed(payload["messages"]) if row["role"] == "tool"))
                assert feedback["status"] == "rejected_not_executed"
                assert feedback["code"] == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
            assert ("candidate_read" in names) == (candidate_mode == "current")
            submission_schema = next(tool["function"]["parameters"] for tool in payload["tools"]
                                     if tool["function"]["name"] == "candidate_submit")
            assert "parent_candidate_id" not in json.dumps(submission_schema)
            assert set(submission_schema["required"]) == {"context_id", "patch_json"}
            messages = [json.loads(row["content"]) for row in payload["messages"]
                        if row["role"] == "user" and row["content"].startswith("{")]
            contract = next(row for row in messages if "required_change_units" in row)
            if malformed_first:
                model_context = contract["constraint_context"]
                assert model_context["projection_version"] == "source_priority_model_view_v2"
                assert "model_preservation_hints" not in model_context
                assert all("text" not in row for row in model_context["omitted_model_preservation_hints"])
                assert derived_hint not in json.dumps(payload)
            context_id = next(row["context_id"] for row in messages if "context_id" in row)
            if candidate_mode == "current":
                assert submission_schema["properties"]["context_id"]["enum"] == [context_id]
            changes = [{"edit_binding_id": unit["edit_binding_id"],
                "after_text": contract["edit_binding_contexts"][unit["edit_binding_id"]]["source_excerpt"].replace(
                    "{}", "{ /* " + ("RecordWrongMarker();" if candidate_mode == "current" else "RecordRuntimeMarker();") + " */ }", 1),
                "rationale": "Scripted entry check", "rollback_hint": "Restore bytes"}
                for unit in contract["required_change_units"]]
            revision_state.update(context_id=context_id, binding_id=changes[0]["edit_binding_id"])
            name, args = "candidate_submit", {"context_id": context_id, "patch_json": json.dumps({"changes": changes})}
            if malformed_first and len(task_turns) == 1:
                from jsonschema import Draft202012Validator

                validator = Draft202012Validator(submission_schema)
                assert validator.is_valid(args)
                # Scope: the exact outer-form error observed in the real 2026-10-20 run.
                # 为什么要有它：普通空父字段不能在正常入口被当成不可恢复执行失败；测试不替模型修业务代码。
                args.update(parent_candidate_id=None, parent_code_sha256=None)
                assert not validator.is_valid(args)
        else:
            last = json.loads(next(row["content"] for row in reversed(payload["messages"]) if row["role"] == "tool"))
            observed = last["value"]
            if replan and turn == 3:
                assert observed["status"] == "behavior_failed"
                assert observed["observations"][0]["observed"] == expected_failure_observation
                if search_replan:
                    name, args = "source_search", {"context_id": revision_state["context_id"],
                        "query": "ConfigKey"}
                    return {"model": payload["model"], "message": {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "entry-task-" + str(len(task_turns)), "type": "function", "function": {"name": name, "arguments": args}}]},
                        "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}
                context = next(json.loads(row["content"]) for row in payload["messages"]
                    if row["role"] == "user" and row["content"].startswith("{")
                    and "allowed_read_binding_ids" in row["content"])
                name, args = "upstream", {"context_id": revision_state["context_id"], "request": {
                    "required_owner": "planning", "reason": "Reconsider the plan after the concrete marker failure",
                    "source_refs": ["binding://" + context["allowed_read_binding_ids"][0]]}}
                return {"model": payload["model"], "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "entry-task-" + str(len(task_turns)), "type": "function", "function": {"name": name, "arguments": args}}]},
                    "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}
            if search_replan and turn == 4:
                selected = next(row for row in observed["items"]
                                if row["source_path"].endswith("Configuration.cs"))
                revision_state["search_evidence_id"] = selected["evidence_id"]
                name, args = "upstream", {"context_id": revision_state["context_id"], "request": {
                    "required_owner": "evidence", "reason": "Admit the searched configuration source before replanning",
                    "source_refs": [selected["reassessment_ref"]]}}
                return {"model": payload["model"], "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "entry-task-" + str(len(task_turns)), "type": "function", "function": {"name": name, "arguments": args}}]},
                    "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}
            upstream_turn = 5 if search_replan else 4
            if replan and turn >= upstream_turn:
                if turn == upstream_turn:
                    assert observed["upstream_outcome"] == "replanned"
                    assert observed["current_code_preserved"] and observed["new_check_required"]
                    binding_id = observed["allowed_edit_binding_ids"][0]
                    if search_replan:
                        contracts = [json.loads(row["content"])
                                     for row in observed["candidate_submission_contract_messages"]
                                     if row["role"] == "user"]
                        contract = next(row for row in contracts if "required_change_units" in row)
                        binding_id = next(unit["edit_binding_id"] for unit in contract["required_change_units"]
                            if "ConfigKey" in contract["edit_binding_contexts"][unit["edit_binding_id"]]["source_excerpt"])
                    revision_state.update(context_id=observed["context_id"],
                        binding_id=binding_id,
                        transferred_candidate_id=observed["current_candidate_id"])
                turn -= 2 if search_replan else 1
            if turn == 2:
                if candidate_mode == "current":
                    submit_schema = next(tool["function"]["parameters"] for tool in payload["tools"]
                                         if tool["function"]["name"] == "candidate_submit")
                    assert set(submit_schema["required"]) == {
                        "context_id", "parent_candidate_id", "parent_code_sha256", "edits"}
                    assert "patch_json" not in submit_schema["properties"]
                    assert submit_schema["properties"]["context_id"]["enum"] == [revision_state["context_id"]]
                    assert submit_schema["properties"]["parent_candidate_id"]["enum"] == [observed["candidate_id"]]
                    assert submit_schema["properties"]["parent_code_sha256"]["enum"] == [observed["code_sha256"]]
                revision_state["first_candidate_id"] = observed["candidate_id"]
                name, args = "candidate_check", {"candidate_id": observed["candidate_id"]}
            elif candidate_mode == "current" and turn == 3:
                if not replan:
                    assert observed["status"] == "behavior_failed"
                    assert observed["observations"][0]["observed"] == expected_failure_observation
                name, args = "candidate_read", {"context_id": revision_state["context_id"],
                    "version": revision_state.get("transferred_candidate_id", revision_state["first_candidate_id"]),
                    "path": "generated_mods/Example/Configuration.cs" if search_replan
                            else "generated_mods/Example/Plugin.cs",
                    "compare_to": "baseline"}
            elif candidate_mode == "current" and turn == 4:
                expected_before = "ConfigKey" if search_replan else "RecordWrongMarker();"
                expected_after = "ConfigKey /* RecordRuntimeMarker(); */" if search_replan else "RecordRuntimeMarker();"
                assert expected_before in observed["text"]
                name, args = "candidate_submit", {"context_id": revision_state["context_id"],
                    "parent_candidate_id": observed["version"], "parent_code_sha256": observed["code_sha256"],
                    "edits": [{"edit_binding_id": revision_state["binding_id"], "before_text": expected_before,
                        "after_text": expected_after, "rationale": "Correct the actual marker failure"}]}
            else:
                name = "candidate_check" if candidate_mode == "current" and turn == 5 else "finish"
                args = {"candidate_id": observed["candidate_id"]}
        return {"model": payload["model"], "message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "entry-task-" + str(len(task_turns)), "type": "function", "function": {"name": name, "arguments": args}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(_LoopbackHttpJsonSender, "__call__", scripted_task_send)
    from scripts import dsh_run_request as bridge

    # Scope: exercise the real new-task allocation before consuming a worker template.
    # 为什么要有它：模板中的相对占位值不是执行目录，必须由原入口绑定新任务路径，不能放宽完成时的工件身份检查。
    allocated = bridge.prepare_new_task(execution_config=str(execution_config), description="Add a marker in Plugin.Load")
    exit_code = _call(Path(allocated), "Add a marker in Plugin.Load", monkeypatch)
    result = json.loads(capsys.readouterr().out)
    assert not completion_errors, completion_errors
    assert exit_code == 0, {**{key: result.get(key) for key in ("status", "reason", "blockers", "task_agent_outcome")},
        "completion_errors": completion_errors}
    assert result["status"] == "awaiting_approval", result
    assert result["task_agent_outcome"]["status"] == "candidate_ready"
    execution = json.loads(Path(result["task_agent_execution_ref"]["path"]).read_bytes())
    assert len(prefix_bindings) == len(task_turns)
    assert all(row["model_identity"]["digest"] == execution["expected_model_identity"]["digest"]
        for row in prefix_bindings)
    assert execution.get("candidate_mode", "baseline") == candidate_mode
    manifest = json.loads((Path(result["run_root"]) / "run_manifest.json").read_bytes())
    assert manifest["technical_execution"].get("candidate_mode", "baseline") == candidate_mode
    assert result["patch_execution_receipts"] == []
    assert result["task_agent_execution_ref"] and len(task_turns) == (
        ((8 if search_replan else 7 if replan else 6) if candidate_mode == "current" else 3)
        + int(malformed_first))
    if malformed_first:
        loop = json.loads(Path(execution["loop_result_ref"]["path"]).read_bytes())
        assert loop["tool_argument_correction_used"] is True
        assert loop["call_records"]["entry-task-1"]["status"] == "not_dispatched"
        plan = json.loads(Path(result["artifacts"]["implementation_plan"]).read_bytes())["data"]
        assert plan["constraint_context"]["model_preservation_hints"]
        assert derived_hint in [row["text"] for row in plan["constraint_context"]["model_preservation_hints"]]
    if candidate_mode == "current":
        checks = [row for row in execution["completed_tool_records"] if row["role"] == "candidate_check"]
        assert len(checks) == 2 and checks[0]["candidate_ref"] != checks[1]["candidate_ref"]
    if replan:
        assert execution["initial_context"] != execution["final_context"]
        transfer = next(row for row in execution["completed_tool_records"] if row["role"] == "upstream")
        assert transfer["check_ref"] is None
        transferred = json.loads(Path(transfer["candidate_ref"]["path"]).read_bytes())
        assert transferred["model_origin"]["kind"] == "derived_from_context_transition"
        assert transferred["model_origin"]["parent_candidate_ref"] == checks[0]["candidate_ref"]
        assert "RecordWrongMarker" in json.dumps(revision_state["planning_request"], ensure_ascii=False)
        if search_replan:
            assert revision_state["search_evidence_id"] in json.dumps(
                revision_state["planning_request"], ensure_ascii=False)
        if malformed_first:
            planning = next(json.loads(row["content"]) for row in revision_state["planning_request"]["messages"]
                if row["role"] == "user" and row["content"].startswith("{")
                and "upstream_assessment" in row["content"])
            assert planning["constraint_context"]["projection_version"] == "source_priority_model_view_v2"
            assert all("text" not in row for row in planning["requirements"] if row["role"] == "preservation")
            retained_plan = planning["upstream_assessment"]["current_plan"]
            assert retained_plan["constraint_context"]["projection_version"] == "source_priority_model_view_v2"
            assert "model_preservation_hints" not in retained_plan["constraint_context"]
            assert derived_hint not in json.dumps(revision_state["planning_request"])
    assert result["approval_recorded"] is False and result["actual_write_performed"] is False
    assert (target / "Plugin.cs").read_bytes() == before
    assert result["model_call_budget"]["consumed"] >= 6
    if meter_only:
        assert result["model_call_budget"]["policy"] == "meter_only"
        assert result["model_call_budget"]["maximum"] is None
        assert result["model_call_budget"]["remaining"] is None
        assert sum(row["consumed"] for row in result["model_call_budget"]["by_stage"].values()) == result["model_call_budget"]["consumed"]
    assert "patch" not in roles and builds
