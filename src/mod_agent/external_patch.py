"""An explicitly authorized Qwen endpoint consuming the existing Patch worker.

No endpoint, credential, license label or source manifest grants disclosure.
The caller must supply the current run's existing disclosure record. This first
external profile supports independent source-only tasks, not decompiled APIs.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.libs.llm.agent_turn import (
    AgentResponseFormat,
    AgentTurnProviderError,
    OpenAIResponsesAgentAdapter,
    QwenChatAgentAdapter,
)
from src.mod_agent.fresh_planning_continuation import (
    _canonical_sha256,
    run_fresh_patch_continuation,
)
from src.mod_agent.local_entry import (
    LocalEntryError,
    LocalOllamaInitialPatchAdapter,
    _verify_source_disclosure,
    _write_local_patch_response_sidecar,
    make_exact_no_write_dry_run_factory,
)
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.product_executor import (
    SOURCE_DISCLOSURE_DATA_CLASSES,
    _external_model_admission_record,
)

EXTERNAL_PATCH_PROFILE_VERSION = "external_qwen_patch_profile_v1"
OPENAI_TASK_PROFILE_VERSION = "external_openai_responses_profile_v1"
PATCH_FEEDBACK_DATA_CLASSES = ("generated_patch", "verification_feedback")
PATCH_OUTPUT_FORMAT_POLICIES = ("follow_request", "json_when_patch_submission_allowed")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ExternalPatchError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code
        self.stage = "external_model_admission"


def _https_url(value: str):
    if not isinstance(value, str):
        raise ExternalPatchError("EXTERNAL_PATCH_ENDPOINT_INVALID")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or value.endswith("/")):
        raise ExternalPatchError("EXTERNAL_PATCH_ENDPOINT_INVALID")
    return parsed


@dataclass(frozen=True)
class ExternalQwenPatchProfile:
    base_url: str
    model: str
    model_catalog_url: str
    context_window: int
    output_budget: int
    timeout_seconds: float
    credential_env: str = "DASHSCOPE_API_KEY"
    profile_version: str = EXTERNAL_PATCH_PROFILE_VERSION
    output_format_policy: str = "follow_request"

    def __post_init__(self):
        endpoint, catalog = _https_url(self.base_url), _https_url(self.model_catalog_url)
        if (endpoint.scheme, endpoint.hostname, endpoint.port) != (catalog.scheme, catalog.hostname, catalog.port):
            raise ExternalPatchError("EXTERNAL_PATCH_CATALOG_ORIGIN_MISMATCH")
        if self.profile_version != EXTERNAL_PATCH_PROFILE_VERSION or not isinstance(self.model, str) or not self.model.strip():
            raise ExternalPatchError("EXTERNAL_PATCH_PROFILE_INVALID")
        if not isinstance(self.output_format_policy, str) or self.output_format_policy not in PATCH_OUTPUT_FORMAT_POLICIES:
            raise ExternalPatchError("EXTERNAL_PATCH_OUTPUT_FORMAT_POLICY_INVALID")
        if (type(self.context_window) is not int or type(self.output_budget) is not int
                or not 0 < self.output_budget < self.context_window
                or type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0 or not isinstance(self.credential_env, str) or not self.credential_env.isidentifier()):
            raise ExternalPatchError("EXTERNAL_PATCH_PROFILE_INVALID")

    def to_dict(self):
        result = {"profile_version": self.profile_version, "provider": "qwen_chat", "execution_locality": "external",
                "base_url": self.base_url, "model": self.model, "model_catalog_url": self.model_catalog_url,
                "context_window": self.context_window, "context_limit_source": "operator_budget_not_provider_capacity",
                "output_budget": self.output_budget, "timeout_seconds": float(self.timeout_seconds),
                "credential_env": self.credential_env, "temperature": 0.0,
                "native_json_schema": False, "local_schema_admission": True,
                "automatic_provider_fallback": False, "automatic_retry": False}
        # Existing profiles retain their serialized identity unless an operator
        # explicitly selects a different Patch output-format policy.
        if self.output_format_policy != "follow_request":
            result["output_format_policy"] = self.output_format_policy
        return result


@dataclass(frozen=True)
class ExternalOpenAIResponsesProfile:
    """Concrete Responses configuration for the existing v2 task owner.

    为什么要有它：提供方配置固定目的地、模型和资源条件，不能通过更换client绕过原执行与披露边界。
    """
    base_url: str
    model: str
    model_catalog_url: str
    context_window: int
    output_budget: int
    timeout_seconds: float
    credential_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = "medium"
    proxy_url: str | None = None
    profile_version: str = OPENAI_TASK_PROFILE_VERSION

    def __post_init__(self):
        endpoint, catalog = _https_url(self.base_url), _https_url(self.model_catalog_url)
        if (endpoint.hostname != "api.openai.com" or endpoint.port not in (None, 443)
                or endpoint.path != "/v1" or self.model_catalog_url != self.base_url + "/models"
                or catalog.hostname != endpoint.hostname or self.profile_version != OPENAI_TASK_PROFILE_VERSION
                or not isinstance(self.model, str) or not self.model.strip()
                or type(self.context_window) is not int or type(self.output_budget) is not int
                or not 0 < self.output_budget < self.context_window
                or type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0 or not isinstance(self.credential_env, str) or not self.credential_env.isidentifier()
                or self.reasoning_effort not in {"low", "medium", "high", "xhigh"}):
            raise ExternalPatchError("OPENAI_TASK_PROFILE_INVALID")
        if self.proxy_url is not None:
            proxy = urlsplit(self.proxy_url)
            if (proxy.scheme != "http" or proxy.hostname not in {"127.0.0.1", "::1"} or proxy.port is None
                    or proxy.username or proxy.password or proxy.query or proxy.fragment or proxy.path not in {"", "/"}):
                raise ExternalPatchError("OPENAI_TASK_PROXY_INVALID")

    def to_dict(self):
        return {"profile_version": self.profile_version, "provider": "openai_responses", "execution_locality": "external",
            "base_url": self.base_url, "model": self.model, "model_catalog_url": self.model_catalog_url,
            "context_window": self.context_window, "context_limit_source": "operator_budget_not_provider_capacity",
            "output_budget": self.output_budget, "timeout_seconds": float(self.timeout_seconds),
            "credential_env": self.credential_env, "reasoning_effort": self.reasoning_effort, "proxy_url": self.proxy_url,
            "native_json_schema": False, "local_schema_admission": True, "strict_tools": False, "store": False,
            "automatic_provider_fallback": False, "automatic_retry": False}


def external_profile_from_document(document):
    cls = ExternalOpenAIResponsesProfile if document.get("profile_version") == OPENAI_TASK_PROFILE_VERSION else ExternalQwenPatchProfile
    return cls(**document)


def validate_openai_experience_configuration(reference):
    if reference is None:
        return
    value = read_bound_json(reference)
    # 为什么要有它：当前run源码披露不授权跨任务策略/经历外发；仅本地封存登记不产生这类外部请求。
    if (value.get("snapshot_ref") is not None or value.get("extract_on_terminal") is not False
            or value.get("extraction_episode_set_ref") is not None):
        raise ExternalPatchError("OPENAI_TASK_CROSS_TASK_EXPERIENCE_DISCLOSURE_UNSUPPORTED")


def validate_openai_task_identity(profile, observed, expected=None):
    required = {"provider": "openai_responses", "model": profile.model, "base_url": profile.base_url,
        "catalog_model_id": profile.model, "catalogue_url": profile.model_catalog_url}
    if (not isinstance(observed, dict) or any(observed.get(key) != value for key, value in required.items())
            or not isinstance(observed.get("catalog_model_sha256"), str)
            or expected is not None and any(observed.get(key) != expected.get(key)
                for key in (*required, "catalog_model_sha256"))):
        raise ExternalPatchError("OPENAI_TASK_MODEL_IDENTITY_CHANGED")


def read_external_openai_model_identity(profile):
    """Read only the exact configured official catalogue; never retry or generate."""
    key = os.environ.get(profile.credential_env)
    if not key:
        raise ExternalPatchError("EXTERNAL_PATCH_CREDENTIAL_UNAVAILABLE")
    try:
        with httpx.Client(timeout=profile.timeout_seconds, proxy=profile.proxy_url,
                trust_env=False, follow_redirects=False) as client:
            response = client.get(profile.model_catalog_url, headers={"Authorization": "Bearer " + key})
        if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
            raise ExternalPatchError("OPENAI_TASK_MODEL_CATALOG_UNAVAILABLE")
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ExternalPatchError("OPENAI_TASK_MODEL_CATALOG_INVALID")
        matches = [row for row in rows if isinstance(row, dict) and row.get("id") == profile.model]
        if len(matches) != 1:
            raise ExternalPatchError("OPENAI_TASK_MODEL_NOT_UNIQUELY_LISTED")
        return {"provider": "openai_responses", "base_url": profile.base_url, "model": profile.model,
            "model_digest": None, "catalog_model_id": profile.model,
            "catalog_model_sha256": _canonical_sha256(matches[0]), "catalogue_url": profile.model_catalog_url,
            "availability": "catalogue_listing_not_generation_acceptance", "execution_locality": "external",
            "observed_at": datetime.now(timezone.utc).isoformat()}
    except (httpx.HTTPError, ValueError) as error:
        if isinstance(error, ExternalPatchError):
            raise
        raise ExternalPatchError("OPENAI_TASK_MODEL_CATALOG_UNAVAILABLE") from None


def _validate_external_disclosure(profile, record, requested):
    if not isinstance(record, dict) or record.get("scope") != "this_run_only" or not record.get("approval_id"):
        raise ExternalPatchError("EXTERNAL_PATCH_CURRENT_RUN_DISCLOSURE_REQUIRED")
    destination = record.get("destination") or {}
    openai = isinstance(profile, ExternalOpenAIResponsesProfile)
    if (destination.get("provider") != ("openai_responses" if openai else "qwen_chat") or destination.get("model") != profile.model
            or destination.get("base_url") != profile.base_url
            or destination.get("request_url") != profile.base_url + ("/responses" if openai else "/chat/completions")):
        raise ExternalPatchError("EXTERNAL_PATCH_DISCLOSURE_DESTINATION_MISMATCH")
    allowed = (*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES)
    classes = record.get("approved_data_classes")
    if not isinstance(classes, list) or any(value not in allowed for value in classes) or len(set(classes)) != len(classes):
        raise ExternalPatchError("EXTERNAL_PATCH_DISCLOSURE_CLASSES_INVALID")
    _verify_source_disclosure(record, requested_data_classes=requested)


def read_external_qwen_model_identity(profile):
    """Read the existing exact Qwen catalogue contract without source material."""
    key = os.environ.get(profile.credential_env)
    if not key:
        raise ExternalPatchError("EXTERNAL_PATCH_CREDENTIAL_UNAVAILABLE")
    for page in range(1, 4):
        try:
            with httpx.Client(timeout=profile.timeout_seconds, trust_env=False, follow_redirects=False) as client:
                response = client.get(profile.model_catalog_url,
                    params={"page_no": page, "page_size": 100}, headers={"Authorization": "Bearer " + key})
            if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                raise ExternalPatchError("EXTERNAL_PATCH_MODEL_CATALOG_UNAVAILABLE")
            payload = response.json()
            output = payload.get("output") if isinstance(payload, dict) else None
            models = output.get("models") if isinstance(output, dict) else None
            if not isinstance(models, list):
                raise ExternalPatchError("EXTERNAL_PATCH_MODEL_CATALOG_INVALID")
            matched = [row for row in models if isinstance(row, dict) and row.get("model") == profile.model]
            if len(matched) == 1:
                return {"provider": "qwen_chat", "model": profile.model, "model_digest": None,
                    "availability": "catalogue_listing_not_generation_acceptance", "catalog_model_id": matched[0]["model"],
                    "catalog_response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "catalogue_url": profile.model_catalog_url,
                    "observed_at": datetime.now(timezone.utc).isoformat(), "execution_locality": "external"}
            if matched:
                raise ExternalPatchError("EXTERNAL_PATCH_MODEL_CATALOG_AMBIGUOUS")
            if len(models) < 100:
                break
        except (httpx.HTTPError, ValueError) as exc:
            if isinstance(exc, ExternalPatchError):
                raise
            raise ExternalPatchError("EXTERNAL_PATCH_MODEL_CATALOG_UNAVAILABLE") from None
    raise ExternalPatchError("EXTERNAL_PATCH_MODEL_NOT_LISTED")


class ExternalQwenPatchAdapter(LocalOllamaInitialPatchAdapter):
    """Reuse the same tool/generation/feedback implementation through provider hooks."""

    def __init__(self, *, profile: ExternalQwenPatchProfile, artifact_store, workspace_root,
                 patch_agent_loop_policy, behavior_evaluator=None, enable_recovery_loop=True,
                 source_disclosure: Mapping[str, Any] | None = None,
                 approved_source_files: Mapping[str, str] | None = None,
                 replay_attempt_ref=None, **unsupported):
        if not isinstance(profile, ExternalQwenPatchProfile) or patch_agent_loop_policy is None:
            raise ExternalPatchError("EXTERNAL_PATCH_EXPLICIT_PROFILE_AND_LOOP_REQUIRED")
        if replay_attempt_ref is not None or unsupported:
            raise ExternalPatchError("EXTERNAL_PATCH_REPLAY_OR_UNKNOWN_OPTION_UNSUPPORTED")
        super().__init__(profile=profile, artifact_store=artifact_store, workspace_root=workspace_root,
            patch_agent_loop_policy=patch_agent_loop_policy, behavior_evaluator=behavior_evaluator,
            enable_recovery_loop=enable_recovery_loop, llm_factory=lambda _profile: None,
            agent_turn_client_factory=self._external_client)
        self._disclosure = deepcopy(dict(source_disclosure)) if source_disclosure is not None else None
        self._source_files = deepcopy(dict(approved_source_files or {}))
        self._external_identity = None
        self._active_evidence = None
        self._last_wire_identity = None
        self._receipt["adapter"] = "ExternalQwenPatchAdapter"
        self._receipt["provider_adapter_identity"] = {
            "path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "external_provider_boundary_in_addition_to_shared_worker_identity"}
        self._receipt["source_disclosure"] = {"approved": False, "scope": "this_run_only", "occurred": False}

    def _validate_source_scope(self):
        root = Path(self.workspace_root).resolve()
        if not self._source_files or any(type(path) is not str or type(digest) is not str
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                for path, digest in self._source_files.items()):
            raise ExternalPatchError("EXTERNAL_PATCH_EXACT_SOURCE_MANIFEST_REQUIRED")
        actual = {}
        for path in root.rglob("*.cs"):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ExternalPatchError("EXTERNAL_PATCH_SOURCE_SCOPE_ESCAPE")
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self._source_files:
            raise ExternalPatchError("EXTERNAL_PATCH_SOURCE_SCOPE_CHANGED")
        for item in self._active_evidence.get("items", []):
            if (item.get("source_type") != "target_source" or item.get("source_path") not in actual
                    or item.get("source_hash") != actual[item["source_path"]]):
                raise ExternalPatchError("EXTERNAL_PATCH_INDEPENDENT_SOURCE_ONLY_REQUIRED")

    def _validate_disclosure(self, requested):
        _validate_external_disclosure(self.profile, self._disclosure, requested)

    def _credential(self):
        key = os.environ.get(self.profile.credential_env)
        if not key:
            raise ExternalPatchError("EXTERNAL_PATCH_CREDENTIAL_UNAVAILABLE")
        return key

    def _catalog(self):
        return read_external_qwen_model_identity(self.profile)

    def _admit_model_identity(self, state, implementation, evidence_pack):
        self._active_evidence = deepcopy(evidence_pack)
        requested = (*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES) if self._repair_context else SOURCE_DISCLOSURE_DATA_CLASSES
        self._validate_disclosure(requested)
        self._validate_source_scope()
        self._external_identity = self._catalog()
        self._receipt["source_disclosure"] = {**deepcopy(self._disclosure), "requested_data_classes": [],
            "occurred": False, "send_attempted": False, "source_scope_sha256": _canonical_sha256(self._source_files)}
        self._receipt["model_admission"] = {**_external_model_admission_record(self._disclosure),
            "external_disclosure_occurred": False, "api_key_read": True, "availability": "catalogue_listing_only"}
        self._emit("external_patch_model_admitted")
        return deepcopy(self._external_identity), None

    def _admit_response_model_identity(self, response, identity):
        if response.model != self.profile.model:
            raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_MODEL_MISMATCH")
        self._validate_source_scope()
        return {**identity, "response_model": response.model, "model_digest": None}

    def _generation_provider_metadata(self, identity):
        return {"provider": "qwen_chat", "model_digest": None,
                "provider_identity_kind": "external_model_id_without_binary_digest"}

    def _requested_wire_classes(self, payload):
        extra = set()
        for message in payload.get("messages", []):
            content = message.get("content")
            try:
                value = json.loads(content) if isinstance(content, str) else content
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            if message.get("role") == "assistant" and isinstance(value.get("changes"), list):
                extra.add("generated_patch")
            if isinstance(value.get("validator_feedback"), dict) or isinstance(value.get("baseline_compilation_failure"), dict):
                extra.add("verification_feedback")
            if message.get("role") == "user" and value.get("message_type") == "patch_submission_feedback":
                # An unadmitted generated response is returned for format repair;
                # these classes do not assert it is valid Patch or behavior evidence.
                extra.update(PATCH_FEEDBACK_DATA_CLASSES)
        return (*SOURCE_DISCLOSURE_DATA_CLASSES, *(name for name in PATCH_FEEDBACK_DATA_CLASSES if name in extra))

    def _retain_http_response(self, record, body, *, complete, bytes_seen, credential):
        """Keep bounded response bytes before the existing AgentTurn parser runs."""
        if self._response_sidecar_path is None:
            raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_SIDECAR_UNAVAILABLE")
        received_hash = hashlib.sha256(body).hexdigest()
        secret = credential.encode("utf-8")
        redacted = bool(secret and secret in body)
        retained = body.replace(secret, b"[REDACTED_CREDENTIAL]") if redacted else body
        if secret and not redacted:
            try:
                decoded_json = json.dumps(json.loads(body), ensure_ascii=False)
            except ValueError:
                decoded_json = ""
            if credential in decoded_json:
                retained = decoded_json.replace(credential, "[REDACTED_CREDENTIAL]").encode("utf-8")
                redacted = True
        try:
            text = retained.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = base64.b64encode(retained).decode("ascii")
            encoding = "base64"
        path = self._response_sidecar_path.with_name(
            self._response_sidecar_path.name + f".external-http-{len(self._receipt['external_provider_requests']):03d}.txt")
        retained_ref = _write_local_patch_response_sidecar(path, text)
        record.update(
            response_body_ref={"path": retained_ref["path"], "sha256": retained_ref["output_sha256"]},
            response_body_encoding=encoding, response_body_redacted=redacted,
            response_body_complete=complete,
            response_body_sha256=received_hash if complete else None,
            response_prefix_sha256=received_hash if not complete else None,
            response_body_bytes=len(body) if complete else None,
            response_bytes_seen=bytes_seen, response_retained_prefix_bytes=len(body),
            response_hash_basis="httpx_decoded_response_body_not_network_packet",
            agent_turn_parse_status="not_attempted",
        )
        self._receipt["source_disclosure"]["occurred"] = True
        self._receipt["model_admission"]["external_disclosure_occurred"] = True
        self._emit("external_patch_http_response_retained")

    def _send(self, url, payload, headers):
        if url != self.profile.base_url + "/chat/completions":
            raise ExternalPatchError("EXTERNAL_PATCH_SEND_ENDPOINT_MISMATCH")
        self._validate_source_scope()
        requested = self._requested_wire_classes(payload)
        self._validate_disclosure(requested)
        wire = httpx.Request("POST", url, json=payload).content
        if len(wire) + self.profile.output_budget > self.profile.context_window:
            raise ExternalPatchError("EXTERNAL_PATCH_REQUEST_CONTEXT_PROXY_EXCEEDED")
        identity = {"hash_basis": "exact_httpx_json_body_utf8", "provider_body_sha256": hashlib.sha256(wire).hexdigest(),
                    "provider_body_utf8_bytes": len(wire)}
        self._last_wire_identity = identity
        self._receipt["source_disclosure"].update(requested_data_classes=list(requested), send_attempted=True)
        self._receipt["model_admission"]["requested_data_classes"] = list(requested)
        self._receipt.setdefault("external_provider_requests", []).append({"request_identity": identity,
            "requested_data_classes": list(requested), "source_scope_sha256": _canonical_sha256(self._source_files),
            "requested_response_format": deepcopy(payload.get("response_format")),
            "requested_output_mode": "json_object" if payload.get("response_format") == {"type": "json_object"} else "provider_default",
            "output_format_policy": self.profile.output_format_policy,
            "response_received": False})
        record = self._receipt["external_provider_requests"][-1]
        self._emit("external_patch_request_started")
        parts = []
        bytes_seen = 0
        complete = False
        credential = headers["Authorization"].removeprefix("Bearer ")
        try:
            with httpx.Client(timeout=self.profile.timeout_seconds, trust_env=False, follow_redirects=False) as client:
                from src.libs.llm.transport_observation import (
                    transport_response_received,
                    transport_started,
                )
                transport_started()
                with client.stream("POST", url, content=wire, headers=headers) as response:
                    record.update(response_received=True, http_status=response.status_code)
                    transport_response_received(http_status=response.status_code, response_complete=False)
                    for part in response.iter_bytes(chunk_size=65536):
                        remaining = MAX_RESPONSE_BYTES - bytes_seen
                        bytes_seen += len(part)
                        if remaining > 0:
                            parts.append(part[:remaining])
                        if bytes_seen > MAX_RESPONSE_BYTES:
                            break
                    else:
                        complete = True
            body = b"".join(parts)
            transport_response_received(http_status=record["http_status"], response_complete=complete)
            self._retain_http_response(record, body, complete=complete, bytes_seen=bytes_seen, credential=credential)
            if not complete:
                raise ExternalPatchError("EXTERNAL_PATCH_PROVIDER_RESPONSE_TOO_LARGE")
            if record["response_body_redacted"]:
                raise ExternalPatchError("EXTERNAL_PATCH_PROVIDER_SECRET_ECHO")
            if record["http_status"] != 200:
                raise ExternalPatchError("EXTERNAL_PATCH_PROVIDER_HTTP_" + str(record["http_status"]))
            try:
                result = json.loads(body)
            except ValueError:
                record["json_parse_status"] = "rejected"
                raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_JSON_INVALID") from None
            if not isinstance(result, dict):
                record["json_parse_status"] = "rejected"
                raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_INVALID")
            record["json_parse_status"] = "decoded_not_agent_turn_admitted"
        except (httpx.HTTPError, ValueError) as exc:
            if record["response_received"] and "response_body_ref" not in record:
                self._retain_http_response(record, b"".join(parts), complete=False, bytes_seen=bytes_seen, credential=credential)
            if not record["response_received"]:
                self._receipt["source_disclosure"]["occurred"] = None
                self._receipt["model_admission"]["external_disclosure_occurred"] = None
            raise AgentTurnProviderError(getattr(exc, "code", "EXTERNAL_PATCH_PROVIDER_TRANSPORT_FAILED"),
                diagnostic={"request_identity": identity, "response_body_ref": record.get("response_body_ref")}) from None
        self._emit("external_patch_response_received")
        return result

    def _external_client(self, profile):
        client = QwenChatAgentAdapter(api_key=self._credential(), base_url=profile.base_url,
            model=profile.model, max_completion_tokens=profile.output_budget, request_sender=self._send)

        def complete_turn(request):
            before = len(self._receipt.get("external_provider_requests", []))
            try:
                response = client.complete_turn(request)
            except (ExternalPatchError, LocalEntryError) as exc:
                # Admission failure is not a model formatting error. Normalize
                # the existing provider boundary so the loop can retain its
                # feedback state before stopping without another send.
                raise AgentTurnProviderError(str(exc), diagnostic={
                    "failure_boundary": "external_patch_admission",
                    "admission_error_type": type(exc).__name__,
                    "admission_error_code": getattr(exc, "code", "EXTERNAL_PATCH_DISCLOSURE_OR_SOURCE_ADMISSION_FAILED"),
                    "provider_request_started": len(self._receipt.get("external_provider_requests", [])) > before,
                }) from exc
            except Exception:
                if len(self._receipt.get("external_provider_requests", [])) > before:
                    failed = self._receipt["external_provider_requests"][-1]
                    if failed.get("json_parse_status") == "decoded_not_agent_turn_admitted":
                        failed["agent_turn_parse_status"] = "rejected"
                        self._emit("external_patch_agent_turn_rejected")
                    else:
                        self._emit("external_patch_provider_turn_failed")
                raise
            retained = self._receipt["external_provider_requests"][-1]
            retained["agent_turn_parse_status"] = "parsed_not_identity_admitted"
            return replace(response, provider_diagnostic={**dict(response.provider_diagnostic or {}),
                "request_identity": deepcopy(self._last_wire_identity),
                "response_body_ref": deepcopy(retained.get("response_body_ref")),
                "local_schema_admission": True,
                "requested_response_format": deepcopy(retained["requested_response_format"]),
                "requested_output_mode": retained["requested_output_mode"]})

        return SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete_turn)

    def _request_for_output_policy(self, request):
        if (self.profile.output_format_policy == "follow_request"
                or request.response_format is not None or request.tool_choice == "required"):
            return request
        # Patch can finish on an ordinary mixed-action turn. This profile asks
        # for an object without replacing the worker's full final-output schema.
        return replace(request, response_format=AgentResponseFormat(
            name="patch_json_object", description="Request a JSON object when submitting Patch output; local Patch validation still applies.",
            schema={"type": "object"}, strict=False))

    def _recording_agent_client(self, client, run_id):
        recorded = super()._recording_agent_client(client, run_id)

        def complete_turn(request):
            request = self._request_for_output_policy(request)
            # Preserve the original response before rejecting identity drift;
            # no tool result may be disclosed to an unexpectedly reported model.
            response = recorded.complete_turn(request)
            if (response.provider != "qwen_chat" or response.model != self.profile.model
                    or not isinstance(response.raw_response, dict)
                    or response.raw_response.get("model") != self.profile.model):
                raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_MODEL_MISMATCH")
            return response

        return SimpleNamespace(provider=recorded.provider, model=recorded.model, complete_turn=complete_turn)


def validate_run_request_external_profile(profile, *, model, base_url, credential_env):
    """Admit operator-supplied facts before run-request can call any upstream model."""
    if not isinstance(profile, (ExternalQwenPatchProfile, ExternalOpenAIResponsesProfile)):
        raise ExternalPatchError("EXTERNAL_PATCH_RUN_REQUEST_PROFILE_REQUIRED")
    if (profile.model != model or profile.base_url != base_url or profile.credential_env != credential_env):
        raise ExternalPatchError("EXTERNAL_PATCH_RUN_REQUEST_PROFILE_MISMATCH")
    return profile


class RunRequestExternalAssembly:
    """Provider-only assembly for the original Planning and shared Patch owners."""

    def __init__(self, *, profile, source_disclosure):
        self.profile = profile
        self.disclosure = deepcopy(source_disclosure)
        _validate_external_disclosure(profile, self.disclosure, SOURCE_DISCLOSURE_DATA_CLASSES)
        # This reads catalogue metadata only, before any source-bearing upstream
        # request. Listing is never reported as generation acceptance.
        self.identity = (read_external_openai_model_identity(profile) if isinstance(profile, ExternalOpenAIResponsesProfile)
                         else read_external_qwen_model_identity(profile))

    def read_identity(self, base_url, model):
        if (base_url, model) != (self.profile.base_url, self.profile.model):
            raise ExternalPatchError("EXTERNAL_PATCH_RUN_REQUEST_PROFILE_MISMATCH")
        return (deepcopy(self.identity) if isinstance(self.profile, ExternalOpenAIResponsesProfile)
                else read_external_qwen_model_identity(self.profile))

    def task_agent_client(self, profile):
        if not isinstance(profile, ExternalOpenAIResponsesProfile) or profile != self.profile:
            raise ExternalPatchError("OPENAI_TASK_PROFILE_CHANGED")
        key = os.environ.get(profile.credential_env)
        if not key:
            raise ExternalPatchError("EXTERNAL_PATCH_CREDENTIAL_UNAVAILABLE")
        client = OpenAIResponsesAgentAdapter(api_key=key, base_url=profile.base_url, model=profile.model,
            max_output_tokens=profile.output_budget, context_window=profile.context_window,
            timeout_seconds=profile.timeout_seconds, reasoning_effort=profile.reasoning_effort,
            strict_tools=False, store=False, proxy_url=profile.proxy_url)
        native_send = client._send

        def send(url, payload, headers):
            if url != profile.base_url + "/responses" or payload.get("model") != profile.model:
                raise ExternalPatchError("EXTERNAL_PATCH_SEND_ENDPOINT_MISMATCH")
            _validate_external_disclosure(profile, self.disclosure, (*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES))
            return native_send(url, payload, headers)

        client._send = send
        return client

    def make_worker(self, *, profile, artifact_store, workspace_root, patch_agent_loop_policy,
                    identity_reader, behavior_evaluator, enable_recovery_loop):
        if profile != self.profile:
            raise ExternalPatchError("EXTERNAL_PATCH_RUN_REQUEST_PROFILE_MISMATCH")
        if isinstance(profile, ExternalOpenAIResponsesProfile):
            raise ExternalPatchError("OPENAI_RESPONSES_REQUIRES_TASK_AGENT_V2")
        root = Path(workspace_root).resolve(strict=True)
        sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in root.rglob("*.cs")}
        return ExternalQwenPatchAdapter(profile=profile, artifact_store=artifact_store, workspace_root=root,
            patch_agent_loop_policy=patch_agent_loop_policy, behavior_evaluator=behavior_evaluator,
            enable_recovery_loop=enable_recovery_loop, source_disclosure=self.disclosure,
            approved_source_files=sources)

    def planning_client(self):
        profile = self.profile
        if isinstance(profile, ExternalOpenAIResponsesProfile):
            return self.task_agent_client(profile)
        key = os.environ.get(profile.credential_env)
        if not key:
            raise ExternalPatchError("EXTERNAL_PATCH_CREDENTIAL_UNAVAILABLE")

        def send(url, payload, headers):
            if url != profile.base_url + "/chat/completions" or payload.get("model") != profile.model:
                raise ExternalPatchError("EXTERNAL_PATCH_SEND_ENDPOINT_MISMATCH")
            _validate_external_disclosure(profile, self.disclosure, SOURCE_DISCLOSURE_DATA_CLASSES)
            wire = httpx.Request("POST", url, json=payload).content
            if len(wire) + profile.output_budget > profile.context_window:
                raise ExternalPatchError("EXTERNAL_PATCH_REQUEST_CONTEXT_PROXY_EXCEEDED")
            try:
                with httpx.Client(timeout=profile.timeout_seconds, trust_env=False, follow_redirects=False) as client:
                    response = client.post(url, content=wire, headers=headers)
                if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                    raise ExternalPatchError("EXTERNAL_PATCH_PLANNING_RESPONSE_UNAVAILABLE")
                return response.json()
            except httpx.HTTPError:
                raise ExternalPatchError("EXTERNAL_PATCH_PLANNING_TRANSPORT_FAILED") from None

        client = QwenChatAgentAdapter(api_key=key, base_url=profile.base_url, model=profile.model,
            max_completion_tokens=profile.output_budget, request_sender=send)

        def complete_turn(request):
            response = client.complete_turn(request)
            if (response.model != profile.model or response.provider != "qwen_chat"
                    or not isinstance(response.raw_response, dict) or response.raw_response.get("model") != profile.model):
                raise ExternalPatchError("EXTERNAL_PATCH_RESPONSE_MODEL_MISMATCH")
            return response

        return SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete_turn)


def run_external_patch_continuation(request, *, profile, worker_configuration, source_disclosure=None,
                                    approved_source_files=None, execution_trace_configuration=None):
    """Use the original continuation/Graph and bind its existing profile identity slots."""
    if request.patch_upstream_configuration_ref is not None or request.qualified_model_routing_ref is not None:
        raise ExternalPatchError("EXTERNAL_PATCH_LOCAL_ROUTE_OR_QUALIFICATION_UNSUPPORTED")
    if not worker_configuration.enable_recovery_loop:
        raise ExternalPatchError("EXTERNAL_PATCH_RECOVERY_WORKER_REQUIRED")
    bound = replace(request, worker_configuration_ref=worker_configuration.reference,
                    worker_profile_sha256=_canonical_sha256(profile.to_dict()))
    from src.mod_agent.artifacts import ArtifactStore
    adapter = ExternalQwenPatchAdapter(profile=profile, artifact_store=ArtifactStore(request.artifact_store_root),
        workspace_root=request.workspace_root, patch_agent_loop_policy=worker_configuration.policy,
        behavior_evaluator=worker_configuration.behavior_evaluator(), source_disclosure=source_disclosure,
        approved_source_files=approved_source_files)
    return run_fresh_patch_continuation(bound, patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(request.workspace_root),
        max_model_calls=worker_configuration.policy.max_total_model_turns,
        execution_trace_configuration=execution_trace_configuration)
