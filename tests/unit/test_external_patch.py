"""Owned synthetic C# source exercises the shared worker through Qwen HTTP."""
from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentTurnProviderError,
    AgentTurnRequest,
    QwenChatAgentAdapter,
)
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.external_patch import (
    PATCH_FEEDBACK_DATA_CLASSES,
    ExternalPatchError,
    ExternalQwenPatchAdapter,
    ExternalQwenPatchProfile,
    run_external_patch_continuation,
)
from src.mod_agent.local_entry import _restore_patch_agent_context, _sha256_json
from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
from src.mod_agent.product_executor import SOURCE_DISCLOSURE_DATA_CLASSES, _source_disclosure_record
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from tests.unit.test_agent_turn_adapters import _format, _tool
from tests.unit.test_fresh_planning_continuation import _patch_request, _source_hashes
from tests.unit.test_patch_repair_cli import _message_jsons, _public_case
from tests.unit.test_patch_upstream_cli import _UpstreamTransport


def _profile():
    return ExternalQwenPatchProfile(base_url="https://dashscope.example/compatible-mode/v1", model="qwen3.6-plus",
        model_catalog_url="https://dashscope.example/api/v1/models", context_window=49152,
        output_budget=4096, timeout_seconds=20)


def _consent(profile, *, feedback=True):
    result = _source_disclosure_record(provider="qwen_chat", model=profile.model, base_url=profile.base_url,
        approved_at="2026-09-07T00:00:00+00:00", approval_id="explicit-current-test-invocation",
        provider_adapter="ExternalQwenPatchAdapter", destination_url=profile.base_url + "/chat/completions")
    if feedback:
        result["approved_data_classes"].extend(PATCH_FEEDBACK_DATA_CLASSES)
    return result


class _QwenTransport:
    def __init__(self, *, direct=False, listed=True, wrong_model=False, missing_model=False, timeout=False, prose=False):
        self.calls, self.wires, self.reads, self.outputs = [], [], [], []
        self.chat_requests = []
        self.direct, self.listed, self.wrong_model, self.timeout = direct, listed, wrong_model, timeout
        self.missing_model = missing_model
        self.prose = prose

    def __call__(self, request):
        assert request.url.scheme == "https" and request.url.host == "dashscope.example"
        assert request.headers["Authorization"] == "Bearer unit-test-secret"
        self.calls.append((request.method, request.url.path))
        if request.method == "GET":
            assert request.url.path == "/api/v1/models"
            return httpx.Response(200, json={"output": {"models": [{"model": "qwen3.6-plus", "provider": "qwen"}]
                if self.listed else [], "total": 1 if self.listed else 0}}, request=request)
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        assert "/api/chat" not in request.url.path
        self.wires.append(request.content)
        if self.timeout:
            raise httpx.ReadTimeout("do not expose unit-test-secret", request=request)
        payload = json.loads(request.content)
        self.chat_requests.append((request.content, payload))
        assert payload["model"] == "qwen3.6-plus"
        assert payload["enable_thinking"] is False
        assert payload["tool_choice"] == "auto"
        assert "request_patch_upstream" in [row["function"]["name"] for row in payload["tools"]]
        number = len(self.wires)
        values = _message_jsons(payload)
        feedback = [row for _, row in values if row.get("message_type") == "patch_behavior_correction"]
        if not self.direct and number in (1, 3):
            if number == 3:
                assert feedback and feedback[-1]["validator_feedback"]["observations"]
                assert any(row["role"] == "assistant" and row.get("content") == self.outputs[0] for row in payload["messages"])
            tool = next(row["function"] for row in payload["tools"] if row["function"]["name"] == "inspect_plan_binding_source")
            selected = next(value for value in tool["parameters"]["properties"]["binding_id"]["enum"] if value not in self.reads)
            self.reads.append(selected)
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"source-read-{number}", "type": "function",
                "function": {"name": tool["name"], "arguments": json.dumps({"binding_id": selected})}}]}
        else:
            if not self.direct:
                tools = [row for row in payload["messages"] if row["role"] == "tool"]
                assert len(tools) == (1 if number == 2 else 2)
                assert all("tool_call_id" in row for row in tools)
            prompt = _restore_patch_agent_context(next(row for _, row in values if "required_change_units" in row))
            from tests.unit.test_fresh_planning_continuation import _fixture_visible_binding_text
            for unit in prompt["required_change_units"]:
                binding_id = unit["edit_binding_id"]
                prompt["edit_binding_contexts"][binding_id]["source_excerpt"] = _fixture_visible_binding_text(
                    prompt, binding_id, payload["messages"])
            assert "PreserveFeedback must remain unchanged." in json.dumps(payload)
            output = json.dumps(_UpstreamTransport._patch(self, prompt, passed=self.direct or number == 4))
            if self.prose:
                output = "Explanation preceding the requested JSON.\n" + output
            self.outputs.append(output)
            message = {"role": "assistant", "content": output}
        result = {"id": f"qwen-external-{number}", "model": "wrong-model" if self.wrong_model else "qwen3.6-plus",
            "choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 250, "completion_tokens": 60}}
        if self.missing_model:
            del result["model"]
        return httpx.Response(200, json=result, request=request)


def _install(monkeypatch, **options):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    handler = _QwenTransport(**options)
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: transport.handle_request(request))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("This acceptance must not call a real external model.")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)
    return handler


def _run(case, profile, consent, *, request=None, files=None):
    return run_external_patch_continuation(request or case.request, profile=profile,
        worker_configuration=PatchWorkerConfiguration.load(case.config_path), source_disclosure=consent,
        approved_source_files=_source_hashes(case.inputs.workspace) if files is None else files)


@pytest.mark.parametrize("output_format_policy", ["follow_request", "json_when_patch_submission_allowed"])
def test_external_qwen_reuses_source_feedback_revision_and_original_review_without_ollama_identity(tmp_path, monkeypatch, output_format_policy):
    case = _public_case(tmp_path, selected_policy=asdict(PatchAgentLoopPolicy(
        max_total_model_turns=5, max_tool_rounds=3,
        finalize_after_first_observation=False, require_source_inspection=False,
        enable_supporting_reads=True, allow_upstream_request=True, allow_batched_reads=True,
    )))
    profile = replace(_profile(), output_format_policy=output_format_policy)
    handler = _install(monkeypatch)
    actual_requests = []
    original_complete = QwenChatAgentAdapter.complete_turn
    def capture_request(client, request):
        actual_requests.append(request)
        return original_complete(client, request)
    monkeypatch.setattr(QwenChatAgentAdapter, "complete_turn", capture_request)
    before = _source_hashes(case.inputs.workspace)
    result = _run(case, profile, _consent(profile))
    assert result.status == "terminal", result.to_dict()
    assert len(handler.wires) == result.execution_counts["model"] == 4
    assert result.execution_counts["review"] == result.execution_counts["dry_run"] == 1
    assert result.execution_counts["apply"] == result.execution_counts["build"] == result.execution_counts["runtime"] == 0
    assert _source_hashes(case.inputs.workspace) == before
    journal = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))
    receipt = journal["factory_receipt"]
    assert receipt["adapter"] == "ExternalQwenPatchAdapter"
    assert receipt["model_identity"]["model_digest"] is None
    assert receipt["model_context_limit"] is None
    assert receipt["model_identity"]["availability"] == "catalogue_listing_not_generation_acceptance"
    assert receipt["profile"]["execution_locality"] == "external"
    assert receipt["profile"].get("output_format_policy", "follow_request") == output_format_policy
    assert all(set(PATCH_FEEDBACK_DATA_CLASSES) <= set(row["requested_data_classes"]) for row in receipt["external_provider_requests"])
    assert receipt["model_admission"]["requested_data_classes"] == receipt["source_disclosure"]["requested_data_classes"]
    for turn, request, wire, actual in zip(receipt["worker_turns"], receipt["external_provider_requests"], handler.wires, actual_requests, strict=True):
        assert turn["request_sha256"] == _sha256_json(asdict(actual))
        assert turn["provider_diagnostic"]["request_identity"] == request["request_identity"]
        assert request["request_identity"]["provider_body_sha256"] == hashlib.sha256(wire).hexdigest()
        assert request["request_identity"]["provider_body_utf8_bytes"] == len(wire)
        expected_format = {"type": "json_object"} if actual.response_format is not None else None
        expected_mode = "json_object" if expected_format else "provider_default"
        assert request["requested_response_format"] == json.loads(wire).get("response_format") == expected_format
        assert turn["provider_diagnostic"]["requested_output_mode"] == request["requested_output_mode"] == expected_mode
        assert request["output_format_policy"] == output_format_policy
    ordinary_output = json.loads(handler.wires[1])
    assert {"inspect_plan_binding_source", "request_patch_upstream"} == {
        row["function"]["name"] for row in ordinary_output["tools"]}
    assert receipt["worker_turns"][1]["tool_action"] == "allowed_but_not_requested"
    assert (actual_requests[1].response_format is not None) == (output_format_policy == "json_when_patch_submission_allowed")
    proposal = case.inputs.store.read(ArtifactRef(**result.patch_proposal_artifact))["data"]
    assert all(row["generation"]["provider"] == "qwen_chat" and row["generation"]["model_digest"] is None for row in proposal["diff_intent"])
    assert "unit-test-secret" not in json.dumps(journal)
    sent = len(handler.calls)
    replay = _run(case, profile, _consent(profile))
    assert replay.patch_proposal_artifact == result.patch_proposal_artifact and len(handler.calls) == sent
    changed = _run(case, replace(profile, model="qwen-another-explicit-model"), _consent(replace(profile, model="qwen-another-explicit-model")))
    assert changed.first_failure["code"] == "PATCH_CONTINUATION_ATTEMPT_IDENTITY_CONFLICT"
    assert len(handler.calls) == sent


def test_initial_external_patch_does_not_claim_unsent_feedback_classes(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = replace(_profile(), output_format_policy="json_when_patch_submission_allowed")
    handler = _install(monkeypatch, direct=True)
    request = replace(_patch_request(case.inputs), attempt_id="owned-initial-external")
    result = _run(case, profile, _consent(profile, feedback=False), request=request)
    assert result.status == "terminal", result.to_dict()
    assert len(handler.wires) == 1 and not handler.reads
    receipt = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))["factory_receipt"]
    assert receipt["external_provider_requests"][0]["requested_data_classes"] == list(SOURCE_DISCLOSURE_DATA_CLASSES)
    assert receipt["source_disclosure"]["requested_data_classes"] == list(SOURCE_DISCLOSURE_DATA_CLASSES)
    assert receipt["external_provider_requests"][0]["requested_response_format"] == {"type": "json_object"}


def test_baseline_compilation_feedback_requires_disclosure_before_external_send(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = _profile()
    handler = _install(monkeypatch)
    configuration = PatchWorkerConfiguration.load(case.config_path)
    adapter = ExternalQwenPatchAdapter(profile=profile, artifact_store=case.inputs.store,
        workspace_root=case.inputs.workspace, patch_agent_loop_policy=configuration.policy,
        source_disclosure=_consent(profile, feedback=False),
        approved_source_files=_source_hashes(case.inputs.workspace))
    recorded = []
    adapter.bind_attempt_recorder(recorded.append)
    evidence = case.inputs.store.read(case.request.evidence_pack_ref)["data"]
    adapter._admit_model_identity({}, {}, evidence)
    request = AgentTurnRequest(messages=(AgentMessage(role="user", content={
        "baseline_compilation_failure": {
            "status": "failed", "errors": [{"code": "CS0103", "message": "Owned synthetic missing name."}],
        },
    }),), tool_choice="none")

    with pytest.raises(AgentTurnProviderError, match="verification_feedback") as rejected:
        adapter._external_client(profile).complete_turn(request)

    assert rejected.value.diagnostic["failure_boundary"] == "external_patch_admission"
    assert rejected.value.diagnostic["provider_request_started"] is False
    assert handler.calls == [("GET", "/api/v1/models")] and not handler.wires
    receipt = adapter.patch_attempt_receipt()
    assert not receipt.get("external_provider_requests")
    assert receipt["source_disclosure"]["occurred"] is False
    assert receipt["source_disclosure"]["send_attempted"] is False


def test_json_mode_does_not_extract_patch_from_prose_and_retains_exact_response(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = replace(_profile(), output_format_policy="json_when_patch_submission_allowed")
    handler = _install(monkeypatch, direct=True, prose=True)

    result = _run(case, profile, _consent(profile))

    assert result.status != "terminal" and len(handler.wires) == 2
    assert result.execution_counts["review"] == result.execution_counts["dry_run"] == 0
    assert result.execution_counts["build"] == result.execution_counts["runtime"] == 0
    journal = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))
    receipt = journal["factory_receipt"]
    assert receipt["patch_agent_loop"]["termination"]["code"] == "PATCH_AGENT_REPEATED_INVALID_SUBMISSION"
    assert [row["status"] for row in receipt["patch_agent_loop"]["submission_trace"]] == ["rejected", "rejected"]
    request = receipt["external_provider_requests"][0]
    assert request["requested_response_format"] == {"type": "json_object"}
    assert receipt["model_response"] is None and not receipt["model_responses"]
    retained = json.loads(Path(receipt["worker_turns"][0]["response_ref"]["path"]).read_text(encoding="utf-8"))
    assert retained["content"] == handler.outputs[0]
    raw = Path(request["response_body_ref"]["path"]).read_bytes()
    assert json.loads(raw)["choices"][0]["message"]["content"] == handler.outputs[0]
    assert hashlib.sha256(raw).hexdigest() == request["response_body_sha256"]
    assert not list(case.reports.glob("report-*.json"))


def test_json_mode_bytes_are_admitted_before_send_without_mutating_caller_payload(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    request = AgentTurnRequest(messages=(AgentMessage(role="user", content="Return Patch JSON."),), tool_choice="none")
    output_budget = 32
    payloads = []
    def capture_payload(_url, payload, _headers):
        payloads.append(payload)
        return {"model": "qwen3.6-plus", "choices": [{"message": {"content": "{}"}}]}
    QwenChatAgentAdapter(api_key="unit-test-secret", base_url=_profile().base_url,
        model=_profile().model, max_completion_tokens=output_budget,
        request_sender=capture_payload).complete_turn(request)
    profile = replace(_profile(), output_format_policy="json_when_patch_submission_allowed",
        output_budget=output_budget, context_window=output_budget + len(
            httpx.Request("POST", _profile().base_url + "/chat/completions", json=payloads[0]).content))
    configuration = PatchWorkerConfiguration.load(case.config_path)
    handler = _install(monkeypatch)
    adapter = ExternalQwenPatchAdapter(profile=profile, artifact_store=case.inputs.store,
        workspace_root=case.inputs.workspace, patch_agent_loop_policy=configuration.policy,
        source_disclosure=_consent(profile), approved_source_files=_source_hashes(case.inputs.workspace))
    recorded = []
    adapter.bind_attempt_recorder(recorded.append)
    evidence = case.inputs.store.read(case.request.evidence_pack_ref)["data"]
    adapter._admit_model_identity({}, {}, evidence)

    with pytest.raises(AgentTurnProviderError, match="EXTERNAL_PATCH_REQUEST_CONTEXT_PROXY_EXCEEDED") as rejected:
        adapter._external_client(profile).complete_turn(adapter._request_for_output_policy(request))
    assert rejected.value.diagnostic["failure_boundary"] == "external_patch_admission"
    assert rejected.value.diagnostic["provider_request_started"] is False

    assert request.response_format is None
    assert "response_format" not in payloads[0]
    assert handler.calls == [("GET", "/api/v1/models")] and not handler.wires
    assert recorded and not adapter.patch_attempt_receipt().get("external_provider_requests")


@pytest.mark.parametrize("policy,choice,declared,requested", [
    ("follow_request", "required", False, False),
    ("follow_request", "auto", False, False),
    ("follow_request", "none", True, True),
    ("json_when_patch_submission_allowed", "required", False, False),
    ("json_when_patch_submission_allowed", "auto", False, True),
    ("json_when_patch_submission_allowed", "none", False, True),
    ("json_when_patch_submission_allowed", "required", True, True),
    ("json_when_patch_submission_allowed", "auto", True, True),
    ("json_when_patch_submission_allowed", "none", True, True),
])
def test_output_policy_preserves_declared_formats_and_pure_tool_turns(tmp_path, policy, choice, declared, requested):
    case = _public_case(tmp_path)
    profile = replace(_profile(), output_format_policy=policy)
    configuration = PatchWorkerConfiguration.load(case.config_path)
    adapter = ExternalQwenPatchAdapter(profile=profile, artifact_store=case.inputs.store,
        workspace_root=case.inputs.workspace, patch_agent_loop_policy=configuration.policy)
    original = AgentTurnRequest(messages=(AgentMessage(role="user", content="Return structured JSON or use the allowed tool."),),
        tools=() if choice == "none" else (_tool(),), tool_choice=choice,
        response_format=_format() if declared else None)

    selected = adapter._request_for_output_policy(original)

    if declared or policy == "follow_request" or choice == "required":
        assert selected is original
    if declared:
        assert selected.response_format is original.response_format
    else:
        assert original.response_format is None
    assert (selected.response_format is not None) == requested
    assert replace(selected, response_format=original.response_format) == original
    payloads = []
    def capture(_url, payload, _headers):
        payloads.append(payload)
        return {"model": profile.model, "choices": [{"message": {"content": "{}"}}]}
    QwenChatAgentAdapter(api_key="unused", base_url=profile.base_url, model=profile.model,
        max_completion_tokens=profile.output_budget, request_sender=capture).complete_turn(selected)
    assert payloads[0].get("response_format") == ({"type": "json_object"} if requested else None)
    assert payloads[0]["tool_choice"] == choice
    assert [row["function"]["name"] for row in payloads[0].get("tools", [])] == [tool.name for tool in original.tools]


def test_output_policy_is_explicit_validated_and_changes_profile_identity():
    original = _profile()
    selected = replace(original, output_format_policy="json_when_patch_submission_allowed")
    assert original.output_format_policy == "follow_request"
    assert "output_format_policy" not in original.to_dict()
    # This test profile's exact serialization at frozen b5b51d1.
    assert _sha256_json(original.to_dict()) == "37e8535f676a640ea30249aa7aa4e00fd1b6116737976a86a8ce956e7e033d67"
    assert selected.to_dict()["output_format_policy"] == "json_when_patch_submission_allowed"
    assert _sha256_json(original.to_dict()) != _sha256_json(selected.to_dict())
    for invalid in (None, [], "always_json", "FOLLOW_REQUEST"):
        with pytest.raises(ExternalPatchError, match="EXTERNAL_PATCH_OUTPUT_FORMAT_POLICY_INVALID"):
            replace(original, output_format_policy=invalid)


@pytest.mark.parametrize("invalid", ["no_consent", "wrong_destination", "feedback_not_approved", "wrong_manifest"])
def test_disclosure_and_exact_owned_source_scope_reject_before_catalog_or_model(tmp_path, monkeypatch, invalid):
    case = _public_case(tmp_path)
    profile = _profile()
    consent = _consent(profile)
    files = _source_hashes(case.inputs.workspace)
    if invalid == "no_consent":
        consent = None
    elif invalid == "wrong_destination":
        consent["destination"]["model"] = "another-model"
    elif invalid == "feedback_not_approved":
        consent = _consent(profile, feedback=False)
    else:
        files[next(iter(files))] = "0" * 64
    handler = _install(monkeypatch)
    result = _run(case, profile, consent, files=files)
    assert result.status != "terminal"
    assert result.execution_counts["model"] == 0 and handler.calls == []


def test_non_source_api_evidence_cannot_be_relabelled_by_an_owned_manifest(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = _profile()
    handler = _install(monkeypatch)
    configuration = PatchWorkerConfiguration.load(case.config_path)
    adapter = ExternalQwenPatchAdapter(profile=profile, artifact_store=case.inputs.store, workspace_root=case.inputs.workspace,
        patch_agent_loop_policy=configuration.policy, source_disclosure=_consent(profile),
        approved_source_files=_source_hashes(case.inputs.workspace))
    evidence = deepcopy(case.inputs.store.read(case.request.evidence_pack_ref)["data"])
    evidence["items"][0]["source_type"] = "decompiled_api"
    with pytest.raises(ExternalPatchError, match="INDEPENDENT_SOURCE_ONLY"):
        adapter._admit_model_identity({}, {}, evidence)
    assert handler.calls == []


def test_unlisted_model_stops_before_any_generation_without_fallback(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = _profile()
    handler = _install(monkeypatch, listed=False)
    result = _run(case, profile, _consent(profile))
    assert result.first_failure["code"] == "EXTERNAL_PATCH_MODEL_NOT_LISTED"
    assert handler.calls == [("GET", "/api/v1/models")] and not handler.wires


def test_external_timeout_is_retained_without_retry_or_secret_echo(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = _profile()
    handler = _install(monkeypatch, timeout=True)
    result = _run(case, profile, _consent(profile))
    assert result.status != "terminal" and len(handler.wires) == 1
    assert result.execution_counts["model"] == 1
    journal = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))
    assert "unit-test-secret" not in json.dumps(journal)
    assert journal["factory_receipt"]["source_disclosure"]["occurred"] is None


@pytest.mark.parametrize("identity", ["different", "missing"])
def test_wrong_response_model_is_retained_and_stops_before_source_tool_execution(tmp_path, monkeypatch, identity):
    case = _public_case(tmp_path)
    profile = _profile()
    handler = _install(monkeypatch, wrong_model=identity == "different", missing_model=identity == "missing")
    result = _run(case, profile, _consent(profile))
    assert result.first_failure["code"] == "EXTERNAL_PATCH_RESPONSE_MODEL_MISMATCH"
    assert len(handler.wires) == 1
    receipt = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))["factory_receipt"]
    first = receipt["worker_turns"][0]
    assert first["status"] == "responded"
    retained = json.loads(Path(first["response_ref"]["path"]).read_text(encoding="utf-8"))
    assert retained["raw_response"].get("model") != profile.model
    assert not receipt["patch_agent_loop"].get("observation_trace")


def test_external_profile_cannot_be_constructed_as_remote_ollama_or_replayed(tmp_path):
    with pytest.raises(ExternalPatchError, match="ENDPOINT"):
        replace(_profile(), base_url="http://127.0.0.1:11434")
    case = _public_case(tmp_path)
    configuration = PatchWorkerConfiguration.load(case.config_path)
    with pytest.raises(ExternalPatchError, match="REPLAY"):
        ExternalQwenPatchAdapter(profile=_profile(), artifact_store=case.inputs.store, workspace_root=case.inputs.workspace,
            patch_agent_loop_policy=configuration.policy, replay_attempt_ref={"path": "not-consumed", "sha256": "0" * 64})


def test_http_json_is_retained_when_agent_turn_parsing_fails(tmp_path, monkeypatch):
    case = _public_case(tmp_path)
    profile = _profile()
    original = _QwenTransport()
    body = {"model": profile.model, "choices": [], "diagnostic_marker": "unparsed-original-response"}
    sent = []

    def respond(request):
        if request.method == "GET":
            return original(request)
        sent.append(request.content)
        return httpx.Response(200, json=body, request=request)

    transport = httpx.MockTransport(respond)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: transport.handle_request(request))
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("No real network in this parser rejection test."))
    result = _run(case, profile, _consent(profile))
    assert result.status != "terminal" and len(sent) == 1
    receipt = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))["factory_receipt"]
    request = receipt["external_provider_requests"][0]
    assert request["response_received"] is True
    assert request["agent_turn_parse_status"] == "rejected"
    assert request["response_body_complete"] is True
    ref = request["response_body_ref"]
    raw = Path(ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"] == request["response_body_sha256"]
    assert json.loads(raw) == body
    assert receipt["worker_turns"][0]["status"] == "failed"
    assert "response_ref" not in receipt["worker_turns"][0]


@pytest.mark.parametrize("kind", ["invalid_json", "non_utf8", "oversized", "secret_echo", "escaped_secret_echo"])
def test_rejected_http_bodies_keep_bounded_truthful_evidence_without_retry(tmp_path, monkeypatch, kind):
    import src.mod_agent.external_patch as external

    case = _public_case(tmp_path)
    profile = _profile()
    original = _QwenTransport()
    if kind == "invalid_json":
        body = b'{"incomplete":'
    elif kind == "non_utf8":
        body = b"\xff\xfb\x80"
    elif kind == "oversized":
        body = b"x" * 2048
        monkeypatch.setattr(external, "MAX_RESPONSE_BYTES", 1024)
    elif kind == "secret_echo":
        body = b'{"error":"unit-test-secret"}'
    else:
        body = b'{"error":"\\u0075nit-test-secret"}'
    sent = []
    def respond(request):
        if request.method == "GET":
            return original(request)
        sent.append(request.content)
        return httpx.Response(200, content=body, request=request)
    transport = httpx.MockTransport(respond)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-secret")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: transport.handle_request(request))
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("No real network in response retention checks."))
    result = _run(case, profile, _consent(profile))
    assert result.status != "terminal" and len(sent) == 1
    assert result.execution_counts["model"] == 1
    receipt = json.loads(Path(result.attempt_receipt["path"]).read_text(encoding="utf-8"))["factory_receipt"]
    request = receipt["external_provider_requests"][0]
    assert request["response_received"] is True and request["http_status"] == 200
    assert request["agent_turn_parse_status"] == "not_attempted"
    if kind in ("invalid_json", "non_utf8"):
        assert request["json_parse_status"] == "rejected"
    assert receipt["source_disclosure"]["occurred"] is True
    ref = request["response_body_ref"]
    raw = Path(ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    decoded = base64.b64decode(raw) if request["response_body_encoding"] == "base64" else raw
    if kind == "oversized":
        assert decoded == body[:1024]
        assert request["response_body_complete"] is False
        assert request["response_body_sha256"] is None
        assert request["response_prefix_sha256"] == hashlib.sha256(decoded).hexdigest()
    else:
        assert request["response_body_complete"] is True
        assert request["response_body_sha256"] == hashlib.sha256(body).hexdigest()
        if kind.endswith("secret_echo"):
            assert request["response_body_redacted"] is True
            assert b"unit-test-secret" not in decoded
            assert b"[REDACTED_CREDENTIAL]" in decoded
        else:
            assert decoded == body
    assert "unit-test-secret" not in json.dumps(receipt)
    assert not any(b"unit-test-secret" in path.read_bytes() for path in (case.inputs.store.root / "cr").glob("*"))
