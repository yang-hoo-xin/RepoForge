"""Concrete external task boundaries; all providers and credentials are controlled.

为什么要有它：兼容adapter不能代替正常任务的真实身份、停止、披露与未知用量校验。
"""
from copy import deepcopy
from dataclasses import asdict, replace

import httpx
import pytest

from src.libs.llm.agent_turn import AgentTurnProviderError, AgentTurnResponse, AgentUsage
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.errors import LocalEntryError
from src.mod_agent.external_patch import (
    PATCH_FEEDBACK_DATA_CLASSES,
    ExternalOpenAIResponsesProfile,
    ExternalPatchError,
    RunRequestExternalAssembly,
    _validate_external_disclosure,
    validate_openai_experience_configuration,
)
from src.mod_agent.product_executor import SOURCE_DISCLOSURE_DATA_CLASSES, _source_disclosure_record
from src.mod_agent.task_agent_execution import _admitted_response
from src.mod_agent.task_agent_loop import measured_turn_usage
from tests.unit.test_task_agent_loop import Budget, Client, Owners, call, run


def profile(**changes):
    return ExternalOpenAIResponsesProfile(base_url="https://api.openai.com/v1", model="controlled-reference",
        model_catalog_url="https://api.openai.com/v1/models", context_window=65536, output_budget=4096,
        timeout_seconds=600, credential_env="REPOFORGE_CONTROLLED_OPENAI_KEY", **changes)


def disclosure(selected):
    return {"scope": "this_run_only", "approval_id": "controlled-current-run", "approved": True,
        "destination": {"provider": "openai_responses", "model": selected.model,
            "base_url": selected.base_url, "request_url": selected.base_url + "/responses"},
        "approved_data_classes": [*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES]}


def test_openai_catalog_is_once_per_invocation_and_disclosure_stays_exact(monkeypatch):
    selected = profile()
    monkeypatch.setenv(selected.credential_env, "controlled-test-value")
    requests = []

    def transport(_transport, request):
        requests.append(request)
        assert request.method == "GET" and str(request.url) == selected.model_catalog_url
        return httpx.Response(200, request=request, json={"data": [{"id": selected.model}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    grant = disclosure(selected)
    owner = RunRequestExternalAssembly(profile=selected, source_disclosure=grant)
    assert owner.read_identity(selected.base_url, selected.model) == owner.identity
    assert owner.read_identity(selected.base_url, selected.model) == owner.identity
    assert len(requests) == 1 and owner.identity["model_digest"] is None
    changed = deepcopy(grant)
    changed["destination"]["request_url"] = "https://api.openai.com/v1/chat/completions"
    with pytest.raises(ExternalPatchError, match="DISCLOSURE_DESTINATION_MISMATCH"):
        _validate_external_disclosure(selected, changed, SOURCE_DISCLOSURE_DATA_CLASSES)
    changed = deepcopy(grant)
    changed["approved_data_classes"].remove("verification_feedback")
    with pytest.raises(LocalEntryError, match="verification_feedback"):
        _validate_external_disclosure(selected, changed, PATCH_FEEDBACK_DATA_CLASSES)
    assert len(requests) == 1


def test_feedback_disclosure_is_recognized_but_not_implicitly_approved():
    selected = profile()
    arguments = {"provider": "openai_responses", "model": selected.model, "base_url": selected.base_url,
        "approved_at": "controlled", "approval_id": "controlled-run", "destination_url": selected.base_url + "/responses"}
    original = _source_disclosure_record(**arguments)
    assert tuple(original["approved_data_classes"]) == SOURCE_DISCLOSURE_DATA_CLASSES
    with pytest.raises(LocalEntryError, match="verification_feedback"):
        _validate_external_disclosure(selected, original, ("verification_feedback",))
    approved = _source_disclosure_record(**arguments,
        requested_data_classes=(*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES))
    _validate_external_disclosure(selected, approved, PATCH_FEEDBACK_DATA_CLASSES)
    with pytest.raises(LocalEntryError, match="unknown data classes"):
        _source_disclosure_record(**arguments, requested_data_classes=(*SOURCE_DISCLOSURE_DATA_CLASSES, "hidden_acceptance"))


@pytest.mark.parametrize("http_status", [401, 403])
def test_catalog_auth_failure_never_generates_or_repeats_secret_text(monkeypatch, http_status):
    selected = profile()
    secret = "controlled-test-credential-not-real"
    monkeypatch.setenv(selected.credential_env, secret)
    requests = []

    def transport(_transport, request):
        requests.append(request.method)
        return httpx.Response(http_status, request=request, json={"error": {"message": secret}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    with pytest.raises(ExternalPatchError) as error:
        RunRequestExternalAssembly(profile=selected, source_disclosure=disclosure(selected))
    assert str(error.value) == "OPENAI_TASK_MODEL_CATALOG_UNAVAILABLE" and secret not in str(error.value)
    assert requests == ["GET"]


@pytest.mark.parametrize("raw", [{"status": "completed"}, {"model": "controlled-reference"},
    {"model": "other", "status": "completed"}, {"model": "controlled-reference", "status": "failed"},
    {"model": "controlled-reference", "status": "incomplete"}])
def test_normal_admission_does_not_accept_adapter_identity_or_status_fallback(raw):
    response = AgentTurnResponse("openai_responses", "controlled-reference", None, stop_reason="completed",
        provider_state=[], raw_response=raw)
    with pytest.raises(ValueError, match="RESPONSE_NATIVE_IDENTITY_CHANGED"):
        _admitted_response(asdict(response), profile().to_dict())


@pytest.mark.parametrize("state", ["failed", "cancelled", "refusal", "incomplete"])
def test_stopped_native_turn_does_not_dispatch_and_keeps_measured_usage(state):
    measured = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18, "reasoning_tokens": 5, "cached_input_tokens": 3}
    response = AgentTurnResponse("openai_responses", "controlled-reference", "stopped",
        (call("source_read", {"binding_id": "one"}),), stop_reason=state,
        usage=AgentUsage(input_tokens=11, output_tokens=7, reasoning_tokens=5, cached_input_tokens=3),
        provider_state=[], provider_diagnostic={"measured_usage": measured, "usage_available": True})
    budget = Budget()
    owners, client = Owners(budget), Client([response])
    result = run(client, owners)
    assert len(client.requests) == 1 and not owners.effects and budget.consumed == 1
    assert result.usage_by_turn[0]["usage"] == measured and result.usage_by_turn[0]["usage_available"] is True


def test_missing_usage_is_unknown_but_malformed_return_keeps_known_cost():
    response = AgentTurnResponse("openai_responses", "controlled-reference", None, stop_reason="failed",
        provider_state=[], provider_diagnostic={"measured_usage": None, "usage_available": False})
    assert measured_turn_usage(response) is None
    assert measured_turn_usage(replace(response, provider_diagnostic=None)) is None
    owners = Owners(Budget())
    result = run(Client([response]), owners)
    assert result.usage_by_turn[0]["usage"] is None and result.usage_by_turn[0]["usage_available"] is False
    measured = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    result = run(Client([AgentTurnProviderError("controlled parse failure", diagnostic={
        "measured_usage": measured, "usage_available": True, "raw_response": {"status": "completed"}})]), Owners(Budget()))
    assert result.usage_by_turn[0]["usage"] == measured and result.usage_by_turn[0]["usage_available"] is True


@pytest.mark.parametrize("change", [{"snapshot_ref": {"fake": "frozen"}}, {"extract_on_terminal": True},
    {"extraction_episode_set_ref": {"fake": "set"}}])
def test_current_run_disclosure_does_not_include_cross_task_strategy(tmp_path, change):
    value = {"snapshot_ref": None, "extract_on_terminal": False, **change}
    with pytest.raises(ExternalPatchError, match="CROSS_TASK_EXPERIENCE_DISCLOSURE_UNSUPPORTED"):
        validate_openai_experience_configuration(_save(tmp_path / "configuration.json", value))


def test_local_episode_registration_without_external_learning_remains_allowed(tmp_path):
    validate_openai_experience_configuration(_save(tmp_path / "registration.json", {
        "snapshot_ref": None, "extract_on_terminal": False, "episode_catalog": {"catalog_id": "public-runs"}}))
