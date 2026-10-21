"""LLM boundary for evidence-bound exact-text patch generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from src.libs.llm import BaseLLM, ChatResponse, Message

PATCH_RESPONSE_CONTRACT_ID = "patch_generation"
PATCH_RESPONSE_CONTRACT_VERSION = "3"

_LITERAL_VALIDATION_MARKER_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
_DETERMINISTICALLY_INHERITED_VALIDATION = frozenset(
    {"compile or static validation"}
)
_NATIVE_PLAN_CHANGE_TYPE = "modify_behavior"
_NATIVE_PLAN_OPERATION_TO_PATCH_OPERATION = {
    "minimal_evidence_bound_change": "exact_text_replace",
}
_LEGACY_GENERATED_PATCH_RISK_TAGS = ("medium",)


class PatchGenerationError(RuntimeError):
    """Raised when model output cannot become a controlled patch proposal."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_content: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_content = diagnostic_content
        self.diagnostics = dict(diagnostics or {})


class PatchSubmissionValidationError(PatchGenerationError):
    """A complete submission has a bounded, model-correctable text defect.

    No-op replacements and the existing C# header-preservation check use this
    type. Authority, source identity,
    schema and provider-admission failures retain their original hard errors.
    """


@dataclass(frozen=True)
class GeneratedPatch:
    changes: tuple[dict[str, Any], ...]
    model: str
    usage: dict[str, int] | None
    raw_text: str


@dataclass(frozen=True)
class PreparedNativePatchContract:
    """Provider-neutral native Patch prompt and deterministic materializer."""

    messages: tuple[Message, ...]
    response_schema: dict[str, Any]
    materialization_context: dict[str, Any]
    contract_id: str
    contract_version: str
    schema_sha256: str


class _NativePatchContractPrepared(RuntimeError):
    def __init__(
        self,
        messages: list[Message],
        response_schema: Mapping[str, Any],
    ) -> None:
        super().__init__("Native Patch contract prepared without a provider call.")
        self.messages = tuple(messages)
        self.response_schema = dict(response_schema)


class _NativePatchContractCaptureLLM(BaseLLM):
    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace
        response_schema = kwargs.get("response_format")
        if not isinstance(response_schema, Mapping):
            raise PatchGenerationError(
                "Native Patch preparation requires a response schema."
            )
        raise _NativePatchContractPrepared(messages, response_schema)


class _FrozenProviderRequestPrepared(RuntimeError):
    def __init__(self, provider_request: Mapping[str, Any]) -> None:
        super().__init__("Native Patch provider request frozen.")
        self.provider_request = dict(provider_request)


class _FreezeOnlyLLM(BaseLLM):
    """Adapt the existing generator producer to a no-network freeze boundary."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace
        prepare = getattr(self.llm, "prepare_structured_chat", None)
        if not callable(prepare):
            raise PatchGenerationError(
                "The selected provider does not implement structured request freeze."
            )
        provider_request: dict[str, Any] = {}
        prepare(
            messages,
            **{
                **kwargs,
                "provider_request_recorder": provider_request.update,
            },
        )
        raise _FrozenProviderRequestPrepared(provider_request)


class LLMExactPatchGenerator:
    """Generate exact replacements without allowing the model to select paths."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def prepare_native_contract(
        self,
        **generation_inputs: Any,
    ) -> PreparedNativePatchContract:
        """Build the exact native prompt/schema without invoking a provider."""

        required_validation = _normalize_required_validation_mapping(
            generation_inputs.get("required_validation_by_binding") or {}
        )
        inputs = dict(generation_inputs)
        inputs["require_native_schema"] = True
        inputs["provider_request_recorder"] = lambda _: None
        try:
            LLMExactPatchGenerator(_NativePatchContractCaptureLLM()).generate(
                **inputs
            )
        except _NativePatchContractPrepared as prepared:
            provider_contract = _native_provider_contract(
                prepared.messages,
                prepared.response_schema,
            )
            return PreparedNativePatchContract(
                messages=prepared.messages,
                response_schema=prepared.response_schema,
                materialization_context=(
                    _native_materialization_context_from_frozen(
                        provider_contract,
                        required_validation=required_validation,
                    )
                ),
                contract_id=PATCH_RESPONSE_CONTRACT_ID,
                contract_version=PATCH_RESPONSE_CONTRACT_VERSION,
                schema_sha256=_canonical_json_sha256(
                    prepared.response_schema
                ),
            )
        raise PatchGenerationError(
            "Native Patch preparation did not reach the contract boundary."
        )

    def materialize_prepared_native(
        self,
        prepared: PreparedNativePatchContract,
        response: ChatResponse,
        *,
        post_response_admission: Callable[[ChatResponse], None] | None = None,
    ) -> GeneratedPatch:
        """Validate one already obtained response against a prepared contract."""

        if (
            not isinstance(prepared, PreparedNativePatchContract)
            or prepared.contract_id != PATCH_RESPONSE_CONTRACT_ID
            or prepared.contract_version not in {"2", PATCH_RESPONSE_CONTRACT_VERSION}
            or prepared.schema_sha256
            != _canonical_json_sha256(prepared.response_schema)
        ):
            raise PatchGenerationError(
                "Prepared native Patch contract identity is invalid."
            )
        contract_kind = _frozen_native_response_contract(_native_provider_contract(prepared.messages, prepared.response_schema))
        if (prepared.contract_version == "3") != (contract_kind == "binding_local_v3"):
            raise PatchGenerationError("Prepared native Patch version and schema disagree.")
        if post_response_admission is not None:
            post_response_admission(response)
        return _materialize_frozen_native_response(
            response,
            context=prepared.materialization_context,
            provider_request=_native_provider_contract(
                prepared.messages,
                prepared.response_schema,
            ),
        )

    def prepare_frozen_native(
        self,
        *,
        resume_identity: Mapping[str, Any],
        frozen_attempt_recorder: Callable[[dict[str, Any]], None],
        **generation_inputs: Any,
    ) -> dict[str, Any]:
        """Persist a paused native Patch attempt without provider generation."""

        if not callable(frozen_attempt_recorder):
            raise ValueError("Frozen Patch attempt recorder must be callable.")
        if not isinstance(resume_identity, Mapping) or not resume_identity:
            raise ValueError("Frozen Patch resume identity must be non-empty.")
        required_validation = _normalize_required_validation_mapping(
            generation_inputs.get("required_validation_by_binding") or {}
        )
        inputs = dict(generation_inputs)
        inputs["require_native_schema"] = True
        inputs["provider_request_recorder"] = lambda _: None
        try:
            LLMExactPatchGenerator(_FreezeOnlyLLM(self.llm)).generate(**inputs)
        except _FrozenProviderRequestPrepared as prepared:
            materialization_context = _native_materialization_context_from_frozen(
                prepared.provider_request,
                required_validation=required_validation,
            )
            frozen_identity = {
                "resume_identity": dict(resume_identity),
                "provider_request": dict(prepared.provider_request),
                "materialization_context": materialization_context,
            }
            attempt = {
                "schema_version": "frozen_native_patch_attempt_v1",
                "status": "paused_frozen",
                "consumed": False,
                **frozen_identity,
                "frozen_identity_sha256": _canonical_json_sha256(frozen_identity),
                "terminal": None,
            }
            frozen_attempt_recorder(dict(attempt))
            return attempt
        raise PatchGenerationError(
            "Native Patch freeze did not reach the provider request boundary."
        )

    def resume_frozen_native(
        self,
        *,
        frozen_attempt_loader: Callable[[], Mapping[str, Any]],
        frozen_attempt_recorder: Callable[[dict[str, Any]], None],
        current_resume_identity: Mapping[str, Any],
        post_response_admission: Callable[[ChatResponse], None] | None = None,
    ) -> GeneratedPatch:
        """Consume one persisted frozen attempt without rebuilding provider input."""

        if not callable(frozen_attempt_loader) or not callable(
            frozen_attempt_recorder
        ):
            raise ValueError("Frozen Patch attempt persistence callbacks are required.")
        attempt = self.validate_frozen_native(
            frozen_attempt_loader(),
            current_resume_identity=current_resume_identity,
        )
        consuming = {
            **attempt,
            "status": "resume_in_progress",
            "consumed": True,
            "terminal": None,
        }
        frozen_attempt_recorder(dict(consuming))
        resume = getattr(self.llm, "resume_structured_chat", None)
        if not callable(resume):
            error = PatchGenerationError(
                "The selected provider does not implement structured request resume."
            )
            _record_frozen_terminal(
                consuming,
                recorder=frozen_attempt_recorder,
                status="terminal_failed",
                error=error,
                failure_boundary="provider_resume",
            )
            raise error
        try:
            response = resume(dict(attempt["provider_request"]))
            if not isinstance(response, ChatResponse):
                raise PatchGenerationError(
                    "Frozen Patch provider response is invalid."
                )
            if post_response_admission is not None:
                post_response_admission(response)
            generated = _materialize_frozen_native_response(
                response,
                context=attempt["materialization_context"],
                provider_request=attempt["provider_request"],
            )
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            boundary = (
                diagnostics.get("failure_boundary")
                if isinstance(diagnostics, Mapping)
                else None
            )
            _record_frozen_terminal(
                consuming,
                recorder=frozen_attempt_recorder,
                status="terminal_failed",
                error=exc,
                failure_boundary=str(boundary or "frozen_patch_resume"),
            )
            raise
        _record_frozen_terminal(
            consuming,
            recorder=frozen_attempt_recorder,
            status="terminal_succeeded",
            error=None,
            failure_boundary=None,
        )
        return generated

    def validate_frozen_native(
        self,
        frozen_attempt: Mapping[str, Any],
        *,
        current_resume_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Perform the complete no-side-effect frozen Patch validation."""

        attempt = _validate_frozen_native_attempt(
            frozen_attempt,
            current_resume_identity=current_resume_identity,
        )
        validate_provider = getattr(
            self.llm,
            "validate_frozen_structured_chat",
            None,
        )
        if callable(validate_provider):
            validate_provider(attempt["provider_request"])
        _validate_frozen_materialization_context(attempt)
        return attempt

    def generate(
        self,
        *,
        user_request: str,
        evidence_items: list[dict[str, Any]],
        binding_registry: dict[str, Any],
        constraints: list[str],
        target_behavior: str | None = None,
        acceptance_criteria: list[str] | tuple[str, ...] | None = None,
        runtime_verification_plan: list[str] | tuple[str, ...] | None = None,
        required_change_units: list[dict[str, Any]]
        | tuple[dict[str, Any], ...]
        | None = None,
        required_validation_by_binding: Mapping[str, list[str] | tuple[str, ...]]
        | None = None,
        require_native_schema: bool = False,
        provider_request_recorder: Callable[[dict[str, Any]], None] | None = None,
        post_response_admission: Callable[[ChatResponse], None] | None = None,
        constraint_context: Mapping[str, Any] | None = None,
    ) -> GeneratedPatch:
        edit_bindings_by_id = _index_bindings(
            binding_registry.get("edit_bindings") or [],
            binding_kind="edit",
        )
        supporting_bindings_by_id = _index_bindings(
            binding_registry.get("supporting_bindings") or [],
            binding_kind="supporting",
        )
        allowed_bindings = set(edit_bindings_by_id)
        if not allowed_bindings:
            raise PatchGenerationError("No editable binding is available for generation.")
        if require_native_schema and required_change_units is None:
            raise _projection_error(
                "Native patch materialization requires Plan-bound change units.",
                failure_classification=(
                    "native_patch_contract_requires_plan_bound_changes"
                ),
            )

        evidence_by_id = _index_evidence_items(evidence_items)
        binding_contexts: dict[str, dict[str, Any]] = {}
        binding_evidence_excerpts: dict[str, str] = {}
        for binding_id, binding in edit_bindings_by_id.items():
            evidence = evidence_by_id.get(str(binding.get("evidence_id") or "")) or {}
            binding_contexts[binding_id] = _edit_binding_context(
                binding=binding,
                evidence=evidence,
            )
            excerpt = evidence.get("excerpt")
            if isinstance(excerpt, str) and excerpt:
                binding_evidence_excerpts[binding_id] = excerpt

        required_validation = _normalize_required_validation_mapping(
            required_validation_by_binding or {}
        )
        model_required_validation = {
            binding_id: tuple(
                item
                for item in values
                if item not in _DETERMINISTICALLY_INHERITED_VALIDATION
            )
            for binding_id, values in required_validation.items()
        }
        normalized_constraints = _normalize_semantic_obligation_list(
            constraints,
            field_name="constraints",
            supplied=True,
            allow_empty=True,
        )
        normalized_target_behavior = _normalize_semantic_obligation_string(
            target_behavior,
            field_name="target_behavior",
            supplied=target_behavior is not None,
        )
        normalized_acceptance_criteria = _normalize_semantic_obligation_list(
            acceptance_criteria,
            field_name="acceptance_criteria",
            supplied=acceptance_criteria is not None,
            allow_empty=False,
        )
        normalized_runtime_verification_plan = (
            _normalize_semantic_obligation_list(
                runtime_verification_plan,
                field_name="runtime_verification_plan",
                supplied=runtime_verification_plan is not None,
                allow_empty=False,
            )
        )
        constraint_instruction, constraint_payload = "", {}
        if constraint_context is not None:
            from src.mod_agent.planning.constraint_context import (
                CONSTRAINT_INSTRUCTION,
                constraint_model_view,
                validate_constraint_context,
            )

            # Scope: experimental context must agree with the actual unmodified feature contract.
            # 为什么要有它：不能丢掉用户原文/验收字段后仍声称完成来源分层，旧A调用不添加任何字段或提示。
            if any(value is None for value in (target_behavior, acceptance_criteria, runtime_verification_plan)):
                raise ValueError("CONSTRAINT_CONTEXT_USER_CONTRACT_REQUIRED")
            checked_context = validate_constraint_context(constraint_context, user_request=user_request,
                constraints=normalized_constraints, target_behavior=normalized_target_behavior,
                acceptance_criteria=normalized_acceptance_criteria, runtime_verification_plan=normalized_runtime_verification_plan)
            constraint_instruction = " " + CONSTRAINT_INSTRUCTION
            constraint_payload = {"constraint_context": constraint_model_view(checked_context)}
        normalized_required_changes = _normalize_required_change_units(
            required_change_units,
            allowed_bindings=allowed_bindings,
            supporting_bindings=set(supporting_bindings_by_id),
            require_native_contract=require_native_schema,
        )
        plan_bound_response = required_change_units is not None
        if require_native_schema:
            for item in normalized_required_changes:
                binding_id = str(item["edit_binding_id"])
                if (
                    binding_id in required_validation
                    and "expected_validation" in item
                    and tuple(item["expected_validation"])
                    != required_validation[binding_id]
                ):
                    raise _projection_error(
                        "Native required change validation does not match the Plan-bound "
                        "validation for its edit binding.",
                        failure_classification=(
                            "required_change_unit_expected_validation_mismatch"
                        ),
                        change_id=item["change_id"],
                        binding_id=binding_id,
                    )
        unknown_required_bindings = set(required_validation) - allowed_bindings
        if unknown_required_bindings:
            raise _projection_error(
                "Required validation references an unknown edit binding id.",
                failure_classification="required_validation_binding_unknown",
                binding_ids=sorted(unknown_required_bindings),
            )
        _validate_required_change_input_coverage(
            normalized_required_changes,
            allowed_bindings=allowed_bindings,
            required_change_units_supplied=required_change_units is not None,
        )
        model_input_projection = _build_patch_model_input_projection(
            evidence_items=evidence_items,
            evidence_by_id=evidence_by_id,
            edit_bindings_by_id=edit_bindings_by_id,
            supporting_bindings_by_id=supporting_bindings_by_id,
            binding_contexts=binding_contexts,
            binding_evidence_excerpts=binding_evidence_excerpts,
            required_change_units=normalized_required_changes,
            strict_required_projection=required_change_units is not None,
        )

        response_schema = _patch_response_schema(
            allowed_bindings,
            required_change_count=len(normalized_required_changes) or None,
        )
        evidence_excerpts = tuple(
            excerpt
            for item in evidence_items
            if isinstance(item, dict)
            and isinstance((excerpt := item.get("excerpt")), str)
            and excerpt
        )
        required_output_change = {
            "change_id": "string",
            "change_type": "modify_behavior",
            "edit_binding_id": "allowed id",
            "operation": "exact_text_replace",
            "rationale": "string",
            **(
                {}
                if plan_bound_response
                else {"expected_validation": ["dotnet build"]}
            ),
            "rollback_hint": "string",
            "patch": {
                "kind": "exact_text_replace",
                "before_text": "verbatim source text",
                "after_text": "replacement text",
                "expected_occurrences": 1,
            },
        }
        required_output = {"changes": [required_output_change]}
        output_contract = (
            {"required_output_schema": response_schema}
            if require_native_schema
            else {"required_output": required_output}
        )
        chat_kwargs: dict[str, Any] = {"temperature": 0.0}
        if require_native_schema:
            if provider_request_recorder is None:
                raise PatchGenerationError(
                    "Native patch schema requires a durable provider request recorder."
                )
            chat_kwargs.update(
                {
                    "response_format": response_schema,
                    "provider_request_recorder": provider_request_recorder,
                }
            )

        response = self.llm.chat(
            [
                Message(
                    role="system",
                    content=_patch_system_prompt(
                        require_native_schema=require_native_schema,
                        plan_bound_response=plan_bound_response,
                    ) + constraint_instruction,
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "request": user_request,
                            "target_behavior": normalized_target_behavior,
                            "constraints": list(normalized_constraints),
                            **constraint_payload,
                            "acceptance_criteria": list(
                                normalized_acceptance_criteria
                            ),
                            "runtime_verification_plan": list(
                                normalized_runtime_verification_plan
                            ),
                            "allowed_edit_binding_ids": sorted(allowed_bindings),
                            **model_input_projection,
                            "required_change_units": [
                                {
                                    key: value
                                    for key, value in item.items()
                                    if key not in {
                                        "rationale",
                                        "expected_validation",
                                    }
                                }
                                for item in normalized_required_changes
                            ],
                            "required_validation_by_binding": {
                                binding_id: list(values)
                                for binding_id, values in sorted(
                                    model_required_validation.items()
                                )
                            },
                            **output_contract,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ],
            **chat_kwargs,
        )
        try:
            if post_response_admission is not None:
                post_response_admission(response)
            if require_native_schema:
                raw_changes = _parse_native_response_changes(response.content)
                changes = _materialize_native_changes(
                    raw_changes,
                    allowed_bindings=allowed_bindings,
                    binding_evidence_excerpts=binding_evidence_excerpts,
                    normalized_required_changes=normalized_required_changes,
                    required_validation=required_validation,
                    binding_contexts=binding_contexts,
                )
            else:
                payload = _parse_json_object(response.content)
                raw_changes = payload.get("changes")
                if not isinstance(raw_changes, list) or not raw_changes:
                    raise PatchGenerationError(
                        "Model output must contain a non-empty changes list."
                    )
                required_changes_by_identity = (
                    {
                        (
                            str(item["change_id"]),
                            str(item["edit_binding_id"]),
                        ): item
                        for item in normalized_required_changes
                    }
                    if plan_bound_response
                    else None
                )
                changes = tuple(
                    _validate_change(
                        item,
                        allowed_bindings=allowed_bindings,
                        evidence_excerpts=evidence_excerpts,
                        binding_evidence_excerpts=binding_evidence_excerpts,
                        required_validation=required_validation,
                        required_changes_by_identity=required_changes_by_identity,
                        binding_contexts=binding_contexts,
                        index=index,
                    )
                    for index, item in enumerate(raw_changes, start=1)
                )
            _validate_required_change_coverage(
                changes,
                required_change_units=normalized_required_changes,
            )
            _validate_submission_text(
                changes, binding_contexts=binding_contexts, native=require_native_schema,
                binding_evidence_excerpts=binding_evidence_excerpts,
                native_v3_raw_changes=raw_changes if require_native_schema else None,
            )
        except Exception as exc:
            diagnostics = _response_diagnostics(response)
            if isinstance(exc, PatchGenerationError):
                diagnostics.update(exc.diagnostics)
                error_type = (
                    PatchSubmissionValidationError
                    if isinstance(exc, PatchSubmissionValidationError)
                    else PatchGenerationError
                )
                raise error_type(
                    str(exc),
                    diagnostic_content=response.content,
                    diagnostics=diagnostics,
                ) from exc
            setattr(exc, "diagnostic_content", response.content)
            existing = getattr(exc, "diagnostics", None)
            if isinstance(existing, Mapping):
                diagnostics.update(existing)
            setattr(exc, "diagnostics", diagnostics)
            raise
        return GeneratedPatch(
            changes=changes,
            model=response.model,
            usage=response.usage,
            raw_text=response.content,
        )


def _normalize_required_validation_mapping(
    values_by_binding: Mapping[str, list[str] | tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        str(binding_id): _normalize_semantic_obligation_list(
            values,
            field_name="required_validation",
            supplied=True,
            allow_empty=True,
            binding_id=str(binding_id),
        )
        for binding_id, values in values_by_binding.items()
    }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _native_provider_contract(
    messages: tuple[Message, ...] | list[Message],
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        "format": dict(response_schema),
    }
    return {
        "body_utf8": json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "response_format_sha256": _canonical_json_sha256(response_schema),
    }


def _native_materialization_context_from_frozen(
    provider_request: Mapping[str, Any],
    *,
    required_validation: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    body_utf8 = provider_request.get("body_utf8")
    if not isinstance(body_utf8, str):
        raise PatchGenerationError("Frozen Patch provider body is missing.")
    try:
        body = json.loads(body_utf8)
    except json.JSONDecodeError as exc:
        raise PatchGenerationError("Frozen Patch provider body is invalid JSON.") from exc
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or len(messages) != 2:
        raise PatchGenerationError("Frozen Patch provider messages are invalid.")
    user_message = messages[1]
    if not isinstance(user_message, dict) or user_message.get("role") != "user":
        raise PatchGenerationError("Frozen Patch user message is invalid.")
    content = user_message.get("content")
    if not isinstance(content, str):
        raise PatchGenerationError("Frozen Patch user message content is invalid.")
    try:
        prompt_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PatchGenerationError("Frozen Patch prompt payload is invalid JSON.") from exc
    if not isinstance(prompt_payload, dict):
        raise PatchGenerationError("Frozen Patch prompt payload root is invalid.")
    allowed_bindings = prompt_payload.get("allowed_edit_binding_ids")
    binding_contexts = prompt_payload.get("edit_binding_contexts")
    required_changes = prompt_payload.get("required_change_units")
    if (
        not isinstance(allowed_bindings, list)
        or not allowed_bindings
        or not all(isinstance(item, str) and item for item in allowed_bindings)
        or not isinstance(binding_contexts, dict)
        or set(binding_contexts) != set(allowed_bindings)
        or not isinstance(required_changes, list)
        or not required_changes
    ):
        raise PatchGenerationError(
            "Frozen Patch materialization identity is incomplete."
        )
    excerpts: dict[str, str] = {}
    for binding_id, context in binding_contexts.items():
        if not isinstance(context, dict):
            raise PatchGenerationError("Frozen Patch binding context is invalid.")
        excerpt = context.get("source_excerpt")
        if isinstance(excerpt, str) and excerpt:
            excerpts[str(binding_id)] = excerpt
    return {
        "allowed_bindings": sorted(allowed_bindings),
        "binding_contexts": binding_contexts,
        "binding_evidence_excerpts": excerpts,
        "required_change_units": required_changes,
        "required_validation_by_binding": {
            binding_id: list(values)
            for binding_id, values in sorted(required_validation.items())
        },
    }


def _frozen_native_response_contract(
    provider_request: Mapping[str, Any],
) -> str:
    """Identify the hash-bound response schema without changing v1 bytes."""

    body_utf8 = provider_request.get("body_utf8")
    if not isinstance(body_utf8, str):
        raise _native_validation_error(
            "Frozen Patch provider body is missing.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    try:
        body = json.loads(body_utf8)
    except json.JSONDecodeError as exc:
        raise _native_validation_error(
            "Frozen Patch provider body is invalid JSON.",
            failure_classification="frozen_patch_response_schema_invalid",
        ) from exc
    messages = body.get("messages") if isinstance(body, dict) else None
    body_schema = body.get("format") if isinstance(body, dict) else None
    if not isinstance(messages, list) or len(messages) != 2:
        raise _native_validation_error(
            "Frozen Patch provider messages are invalid.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    user_message = messages[1]
    if not isinstance(user_message, dict) or user_message.get("role") != "user":
        raise _native_validation_error(
            "Frozen Patch user message is invalid.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    content = user_message.get("content")
    if not isinstance(content, str):
        raise _native_validation_error(
            "Frozen Patch user message content is invalid.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    try:
        prompt_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise _native_validation_error(
            "Frozen Patch prompt payload is invalid JSON.",
            failure_classification="frozen_patch_response_schema_invalid",
        ) from exc
    prompt_schema = (
        prompt_payload.get("required_output_schema")
        if isinstance(prompt_payload, dict)
        else None
    )
    if not isinstance(body_schema, dict) or not isinstance(prompt_schema, dict):
        raise _native_validation_error(
            "Frozen Patch native response schema is missing.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    if body_schema != prompt_schema:
        raise _native_validation_error(
            "Frozen Patch provider and prompt response schemas disagree.",
            failure_classification="frozen_patch_response_schema_mismatch",
        )
    allowed_bindings = prompt_payload.get("allowed_edit_binding_ids")
    required_changes = prompt_payload.get("required_change_units")
    if (
        not isinstance(allowed_bindings, list)
        or not allowed_bindings
        or len(set(allowed_bindings)) != len(allowed_bindings)
        or not all(isinstance(item, str) and item for item in allowed_bindings)
        or not isinstance(required_changes, list)
        or not required_changes
    ):
        raise _native_validation_error(
            "Frozen Patch response schema inputs are invalid.",
            failure_classification="frozen_patch_response_schema_invalid",
        )
    allowed = set(allowed_bindings)
    required_count = len(required_changes)
    if body_schema == _legacy_pair_patch_response_schema(
        allowed,
        required_change_count=required_count,
    ):
        return "legacy_pair"
    if body_schema == _patch_response_schema(allowed, required_change_count=required_count, allow_local_before=False):
        return "binding_only"
    if body_schema == _patch_response_schema(allowed, required_change_count=required_count):
        return "binding_local_v3"
    raise _native_validation_error(
        "Frozen Patch native response schema is not recognized.",
        failure_classification="frozen_patch_response_schema_unknown",
    )


def _validate_frozen_native_attempt(
    raw_attempt: Mapping[str, Any],
    *,
    current_resume_identity: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "consumed",
        "resume_identity",
        "provider_request",
        "materialization_context",
        "frozen_identity_sha256",
        "terminal",
    }
    if not isinstance(raw_attempt, Mapping) or set(raw_attempt) != expected_fields:
        raise PatchGenerationError(
            "Frozen Patch attempt fields are invalid.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": "frozen_patch_attempt_fields_invalid",
            },
        )
    attempt = dict(raw_attempt)
    if (
        attempt.get("schema_version") != "frozen_native_patch_attempt_v1"
        or attempt.get("status") != "paused_frozen"
        or attempt.get("consumed") is not False
        or attempt.get("terminal") is not None
    ):
        raise PatchGenerationError(
            "Frozen Patch attempt is not eligible for resume.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": "frozen_patch_attempt_not_resumable",
            },
        )
    frozen_identity = attempt.get("resume_identity")
    if (
        not isinstance(frozen_identity, Mapping)
        or dict(frozen_identity) != dict(current_resume_identity)
    ):
        raise PatchGenerationError(
            "Frozen Patch resume identity has drifted.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": "frozen_patch_resume_identity_mismatch",
            },
        )
    if not isinstance(attempt.get("provider_request"), Mapping) or not isinstance(
        attempt.get("materialization_context"), Mapping
    ):
        raise PatchGenerationError(
            "Frozen Patch attempt payload is invalid.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": "frozen_patch_attempt_payload_invalid",
            },
        )
    immutable_identity = {
        "resume_identity": dict(frozen_identity),
        "provider_request": dict(attempt["provider_request"]),
        "materialization_context": dict(attempt["materialization_context"]),
    }
    if attempt.get("frozen_identity_sha256") != _canonical_json_sha256(
        immutable_identity
    ):
        raise PatchGenerationError(
            "Frozen Patch attempt hash is invalid.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": "frozen_patch_attempt_hash_mismatch",
            },
        )
    return attempt


def _validate_frozen_materialization_context(
    attempt: Mapping[str, Any],
) -> None:
    context = attempt["materialization_context"]
    expected_context_fields = {
        "allowed_bindings",
        "binding_contexts",
        "binding_evidence_excerpts",
        "required_change_units",
        "required_validation_by_binding",
    }
    if not isinstance(context, Mapping) or set(context) != expected_context_fields:
        raise PatchGenerationError(
            "Frozen Patch materialization context fields are invalid.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": (
                    "frozen_patch_materialization_context_invalid"
                ),
            },
        )
    required_validation_raw = context.get("required_validation_by_binding")
    if not isinstance(required_validation_raw, Mapping) or not all(
        isinstance(binding_id, str)
        and binding_id
        and isinstance(values, list)
        and all(isinstance(value, str) and value for value in values)
        for binding_id, values in required_validation_raw.items()
    ):
        raise PatchGenerationError(
            "Frozen Patch required validation context is invalid.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": (
                    "frozen_patch_materialization_context_invalid"
                ),
            },
        )
    required_validation = {
        str(binding_id): tuple(values)
        for binding_id, values in required_validation_raw.items()
    }
    derived = _native_materialization_context_from_frozen(
        attempt["provider_request"],
        required_validation=required_validation,
    )
    if derived != dict(context):
        raise PatchGenerationError(
            "Frozen Patch materialization context does not match its provider body.",
            diagnostics={
                "failure_boundary": "frozen_patch_identity_revalidation",
                "failure_classification": (
                    "frozen_patch_materialization_context_mismatch"
                ),
            },
        )
    _frozen_native_response_contract(attempt["provider_request"])


def _record_frozen_terminal(
    attempt: Mapping[str, Any],
    *,
    recorder: Callable[[dict[str, Any]], None],
    status: str,
    error: Exception | None,
    failure_boundary: str | None,
) -> None:
    terminal = {
        "outcome": "succeeded" if error is None else "failed",
        "failure_boundary": failure_boundary,
        "failure_type": None if error is None else type(error).__name__,
        "failure_message": None if error is None else str(error),
        "diagnostics": (
            {}
            if error is None
            else dict(getattr(error, "diagnostics", {}) or {})
        ),
        "automatic_retry": False,
        "automatic_correction": False,
        "automatic_provider_fallback": False,
        "model_switching": False,
    }
    recorder(
        {
            **dict(attempt),
            "status": status,
            "consumed": True,
            "terminal": terminal,
        }
    )


def _parse_native_response_changes(content: str) -> list[Any]:
    payload = _parse_json_object(content)
    if set(payload) != {"changes"}:
        raise _native_validation_error(
            "Native Patch response root does not match its schema.",
            failure_classification="native_response_fields_invalid",
            missing_fields=sorted({"changes"} - set(payload)),
            unexpected_fields=sorted(set(payload) - {"changes"}),
        )
    raw_changes = payload.get("changes")
    if not isinstance(raw_changes, list) or not raw_changes:
        raise PatchGenerationError(
            "Model output must contain a non-empty changes list."
        )
    return raw_changes


def _materialize_native_changes(
    raw_changes: list[Any],
    *,
    allowed_bindings: set[str],
    binding_evidence_excerpts: Mapping[str, str],
    normalized_required_changes: tuple[dict[str, Any], ...],
    required_validation: Mapping[str, tuple[str, ...]],
    binding_contexts: Mapping[str, Mapping[str, Any]],
    allow_local_before: bool = True,
) -> tuple[dict[str, Any], ...]:
    required_changes_by_binding = {
        str(item["edit_binding_id"]): item
        for item in normalized_required_changes
    }
    if len(required_changes_by_binding) != len(normalized_required_changes):
        raise _native_validation_error(
            "Native Plan change units do not have unique edit bindings.",
            failure_classification="required_change_unit_binding_duplicate",
        )
    changes = tuple(
        _materialize_native_change(
            item,
            allowed_bindings=allowed_bindings,
            binding_evidence_excerpts=binding_evidence_excerpts,
            required_changes_by_binding=required_changes_by_binding,
            required_validation=required_validation,
            binding_contexts=binding_contexts,
            index=index,
            allow_local_before=allow_local_before,
        )
        for index, item in enumerate(raw_changes, start=1)
    )

    if allow_local_before and any(isinstance(row, dict) and "before_text" in row for row in raw_changes):
        _validate_native_local_ranges(changes, binding_contexts, binding_evidence_excerpts)
    return changes


def _materialize_legacy_native_changes(
    raw_changes: list[Any],
    *,
    allowed_bindings: set[str],
    binding_evidence_excerpts: Mapping[str, str],
    normalized_required_changes: tuple[dict[str, Any], ...],
    required_validation: Mapping[str, tuple[str, ...]],
    binding_contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    required_changes_by_identity = {
        (str(item["change_id"]), str(item["edit_binding_id"])): item
        for item in normalized_required_changes
    }
    return tuple(
        _materialize_native_change(
            item,
            allowed_bindings=allowed_bindings,
            binding_evidence_excerpts=binding_evidence_excerpts,
            required_changes_by_binding={},
            legacy_required_changes_by_identity=required_changes_by_identity,
            required_validation=required_validation,
            binding_contexts=binding_contexts,
            index=index,
        )
        for index, item in enumerate(raw_changes, start=1)
    )


def _materialize_frozen_native_response(
    response: ChatResponse,
    *,
    context: Mapping[str, Any],
    provider_request: Mapping[str, Any],
) -> GeneratedPatch:
    expected_context_fields = {
        "allowed_bindings",
        "binding_contexts",
        "binding_evidence_excerpts",
        "required_change_units",
        "required_validation_by_binding",
    }
    if set(context) != expected_context_fields:
        raise PatchGenerationError(
            "Frozen Patch materialization context fields are invalid."
        )
    allowed_bindings = set(context["allowed_bindings"])
    binding_contexts = dict(context["binding_contexts"])
    binding_evidence_excerpts = dict(context["binding_evidence_excerpts"])
    normalized_required_changes = tuple(
        dict(item) for item in context["required_change_units"]
    )
    required_validation = {
        str(binding_id): tuple(values)
        for binding_id, values in dict(
            context["required_validation_by_binding"]
        ).items()
    }
    try:
        raw_changes = _parse_native_response_changes(response.content)
        response_contract = _frozen_native_response_contract(provider_request)
        materializer = (
            _materialize_legacy_native_changes
            if response_contract == "legacy_pair"
            else _materialize_native_changes
        )
        changes = materializer(
            raw_changes,
            allowed_bindings=allowed_bindings,
            binding_evidence_excerpts=binding_evidence_excerpts,
            normalized_required_changes=normalized_required_changes,
            required_validation=required_validation,
            binding_contexts=binding_contexts,
            **({"allow_local_before": response_contract == "binding_local_v3"}
               if response_contract != "legacy_pair" else {}),
        )
        _validate_required_change_coverage(
            changes,
            required_change_units=normalized_required_changes,
        )
        _validate_submission_text(changes, binding_contexts=binding_contexts, native=True,
                                  binding_evidence_excerpts=binding_evidence_excerpts,
                                  native_v3_raw_changes=raw_changes if response_contract == "binding_local_v3" else None)
    except Exception as exc:
        diagnostics = _response_diagnostics(response)
        if isinstance(exc, PatchGenerationError):
            diagnostics.update(exc.diagnostics)
            error_type = (
                PatchSubmissionValidationError
                if isinstance(exc, PatchSubmissionValidationError)
                else PatchGenerationError
            )
            raise error_type(
                str(exc),
                diagnostic_content=response.content,
                diagnostics=diagnostics,
            ) from exc
        setattr(exc, "diagnostic_content", response.content)
        existing = getattr(exc, "diagnostics", None)
        if isinstance(existing, Mapping):
            diagnostics.update(existing)
        setattr(exc, "diagnostics", diagnostics)
        raise
    return GeneratedPatch(
        changes=changes,
        model=response.model,
        usage=response.usage,
        raw_text=response.content,
    )


def _projection_error(
    message: str,
    *,
    failure_classification: str,
    **diagnostics: Any,
) -> PatchGenerationError:
    return PatchGenerationError(
        message,
        diagnostics={
            "failure_boundary": "patch_model_input_projection",
            "failure_classification": failure_classification,
            **diagnostics,
        },
    )


def _index_bindings(
    raw_bindings: Any,
    *,
    binding_kind: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_bindings, (list, tuple)):
        return indexed
    for raw in raw_bindings:
        if not isinstance(raw, dict) or not raw.get("binding_id"):
            continue
        binding_id = str(raw["binding_id"])
        if binding_id in indexed:
            raise _projection_error(
                f"Duplicate {binding_kind} binding id is not admissible.",
                failure_classification="duplicate_binding_id",
                binding_kind=binding_kind,
                binding_id=binding_id,
            )
        indexed[binding_id] = raw
    return indexed


def _index_evidence_items(
    evidence_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in evidence_items:
        if not isinstance(raw, dict) or not raw.get("evidence_id"):
            continue
        evidence_id = str(raw["evidence_id"])
        if evidence_id in indexed:
            raise _projection_error(
                "Duplicate evidence_id is not admissible for patch generation.",
                failure_classification="duplicate_evidence_id",
                evidence_id=evidence_id,
            )
        indexed[evidence_id] = raw
    return indexed


def _binding_or_evidence_value(
    binding: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    binding_field: str,
    evidence_field: str,
) -> Any:
    binding_value = binding.get(binding_field)
    return binding_value if binding_value is not None else evidence.get(evidence_field)


def _edit_binding_context(
    *,
    binding: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    binding_id = str(binding.get("binding_id") or "")
    context: dict[str, Any] = {"binding_id": binding_id}
    for binding_field, evidence_field in (
        ("evidence_id", "evidence_id"),
        ("source_hash", "source_hash"),
        ("canonical_target_path", "source_path"),
        ("source_type", "source_type"),
        ("target_symbol", "symbol"),
        ("target_span", "span"),
        ("declaring_type", "declaring_type"),
        ("method_name", "method_name"),
        ("parameter_types", "parameter_types"),
        ("return_type", "return_type"),
        ("method_kind", "method_kind"),
        ("declared_member_types", "declared_member_types"),
        ("declaring_type_full_name", "declaring_type_full_name"),
        ("declared_members", "declared_members"),
        ("verified_member_accesses", "verified_member_accesses"),
        ("usage", "usage"),
    ):
        value = _binding_or_evidence_value(
            binding,
            evidence,
            binding_field=binding_field,
            evidence_field=evidence_field,
        )
        if value is not None:
            context[binding_field] = value
    return context


def _evidence_source_hash(evidence: Mapping[str, Any]) -> Any:
    return (
        evidence.get("source_hash")
        or evidence.get("file_hash")
        or evidence.get("doc_hash")
    )


def _validate_required_edit_projection(
    *,
    binding_id: str,
    binding: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    evidence_id = binding.get("evidence_id")
    canonical_target_path = binding.get("canonical_target_path")
    source_hash = binding.get("source_hash")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise _projection_error(
            "Required edit binding is missing its evidence identity.",
            failure_classification="required_edit_binding_identity_incomplete",
            binding_id=binding_id,
            missing_field="evidence_id",
        )
    if not isinstance(canonical_target_path, str) or not canonical_target_path:
        raise _projection_error(
            "Required edit binding is missing its canonical target path.",
            failure_classification="required_edit_binding_identity_incomplete",
            binding_id=binding_id,
            evidence_id=evidence_id,
            missing_field="canonical_target_path",
        )
    if not isinstance(source_hash, str) or not source_hash:
        raise _projection_error(
            "Required edit binding is missing its source hash.",
            failure_classification="required_edit_binding_source_hash_missing",
            binding_id=binding_id,
            evidence_id=evidence_id,
        )
    evidence = evidence_by_id.get(evidence_id)
    if evidence is None:
        raise _projection_error(
            "Required edit binding evidence is missing.",
            failure_classification="required_edit_evidence_missing",
            binding_id=binding_id,
            evidence_id=evidence_id,
        )
    excerpt = evidence.get("excerpt")
    if not isinstance(excerpt, str) or not excerpt:
        raise _projection_error(
            "Required edit binding evidence excerpt is missing.",
            failure_classification="required_edit_evidence_excerpt_missing",
            binding_id=binding_id,
            evidence_id=evidence_id,
        )
    identity_checks = (
        ("canonical_target_path", "source_path"),
        ("target_symbol", "symbol"),
        ("target_span", "span"),
    )
    for binding_field, evidence_field in identity_checks:
        if binding.get(binding_field) != evidence.get(evidence_field):
            raise _projection_error(
                "Required edit binding identity does not match its evidence.",
                failure_classification="required_edit_binding_evidence_drift",
                binding_id=binding_id,
                evidence_id=evidence_id,
                binding_field=binding_field,
                evidence_field=evidence_field,
            )
    evidence_source_hash = _evidence_source_hash(evidence)
    if not isinstance(evidence_source_hash, str) or not evidence_source_hash:
        raise _projection_error(
            "Required edit evidence is missing its source hash.",
            failure_classification="required_edit_evidence_source_hash_missing",
            binding_id=binding_id,
            evidence_id=evidence_id,
        )
    if source_hash != evidence_source_hash:
        raise _projection_error(
            "Required edit binding source hash does not match its evidence.",
            failure_classification="required_edit_binding_evidence_drift",
            binding_id=binding_id,
            evidence_id=evidence_id,
            binding_field="source_hash",
            evidence_field="source_hash|file_hash|doc_hash",
        )
    return evidence


def _normalized_identity_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_identity_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _normalized_identity_value(nested))
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
    return value


def _validate_supporting_projection(
    *,
    binding_id: str,
    binding: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    evidence_id = str(binding.get("evidence_id") or "")
    for binding_field, evidence_field in (
        ("source_path", "source_path"),
        ("target_symbol", "symbol"),
        ("target_span", "span"),
        ("source_type", "source_type"),
        ("declaring_type", "declaring_type"),
        ("method_name", "method_name"),
        ("parameter_types", "parameter_types"),
        ("return_type", "return_type"),
        ("method_kind", "method_kind"),
    ):
        if _normalized_identity_value(
            binding.get(binding_field)
        ) != _normalized_identity_value(evidence.get(evidence_field)):
            raise _projection_error(
                "Supporting binding identity does not match its evidence.",
                failure_classification="supporting_binding_evidence_drift",
                binding_id=binding_id,
                evidence_id=evidence_id,
                binding_field=binding_field,
                evidence_field=evidence_field,
            )


def _required_supporting_binding_ids(
    required_change_units: tuple[dict[str, Any], ...],
    *,
    supporting_bindings: set[str],
) -> set[str]:
    required: set[str] = set()
    for unit in required_change_units:
        required.update(
            binding_id
            for binding_id in unit.get("supporting_binding_ids") or []
            if binding_id in supporting_bindings
        )
        hook_target_binding_id = unit.get("hook_target_binding_id")
        if hook_target_binding_id in supporting_bindings:
            required.add(str(hook_target_binding_id))
    return required


def _supporting_evidence_view(
    *,
    binding_id: str,
    binding: Mapping[str, Any],
    evidence: Mapping[str, Any],
    include_member_facts: bool,
) -> dict[str, Any]:
    view: dict[str, Any] = {"binding_id": binding_id}
    for field in (
        "evidence_id",
        "source_path",
        "source_type",
        "target_symbol",
        "target_span",
        "declaring_type",
        "method_name",
        "parameter_types",
        "return_type",
        "method_kind",
        "claim_supported",
        "usage",
    ):
        value = binding.get(field)
        if value is not None:
            view[field] = value
    source_hash = _evidence_source_hash(evidence)
    if source_hash is not None:
        view["source_hash"] = source_hash
    view["excerpt"] = evidence["excerpt"]
    if evidence.get("document_provenance") is not None:
        from src.mod_agent.evidence.model_view import api_evidence_model_fields
        view.update(api_evidence_model_fields(evidence))
    if include_member_facts:
        for field in (
            "declared_member_types",
            "declared_members",
            "verified_member_accesses",
        ):
            value = evidence.get(field)
            if value:
                view[field] = value
    return view


def _build_patch_model_input_projection(
    *,
    evidence_items: list[dict[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    edit_bindings_by_id: Mapping[str, Mapping[str, Any]],
    supporting_bindings_by_id: Mapping[str, Mapping[str, Any]],
    binding_contexts: Mapping[str, Mapping[str, Any]],
    binding_evidence_excerpts: Mapping[str, str],
    required_change_units: tuple[dict[str, Any], ...],
    strict_required_projection: bool,
) -> dict[str, Any]:
    required_edit_binding_ids = {
        str(item["edit_binding_id"]) for item in required_change_units
    }
    selected_edit_binding_ids = (
        required_edit_binding_ids
        if strict_required_projection
        else set(edit_bindings_by_id)
    )
    projected_edit_contexts: dict[str, dict[str, Any]] = {}
    consumed_evidence_ids: set[str] = set()
    for binding_id in sorted(selected_edit_binding_ids):
        binding = edit_bindings_by_id[binding_id]
        evidence: Mapping[str, Any] = {}
        if strict_required_projection:
            evidence = _validate_required_edit_projection(
                binding_id=binding_id,
                binding=binding,
                evidence_by_id=evidence_by_id,
            )
        else:
            evidence = evidence_by_id.get(
                str(binding.get("evidence_id") or ""),
                {},
            )
        context = dict(binding_contexts[binding_id])
        excerpt = binding_evidence_excerpts.get(binding_id)
        if excerpt is not None:
            context["source_excerpt"] = excerpt
        projected_edit_contexts[binding_id] = context
        evidence_id = evidence.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            consumed_evidence_ids.add(evidence_id)

    required_supporting_ids = _required_supporting_binding_ids(
        required_change_units,
        supporting_bindings=set(supporting_bindings_by_id),
    )
    supporting_views: list[dict[str, Any]] = []
    for binding_id in sorted(supporting_bindings_by_id):
        binding = supporting_bindings_by_id[binding_id]
        evidence_id = binding.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            if binding_id in required_supporting_ids:
                raise _projection_error(
                    "Required supporting binding is missing its evidence identity.",
                    failure_classification=(
                        "required_supporting_binding_identity_incomplete"
                    ),
                    binding_id=binding_id,
                )
            continue
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            if binding_id in required_supporting_ids:
                raise _projection_error(
                    "Required supporting evidence is missing.",
                    failure_classification="required_supporting_evidence_missing",
                    binding_id=binding_id,
                    evidence_id=evidence_id,
                )
            continue
        excerpt = evidence.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt:
            if strict_required_projection or binding_id in required_supporting_ids:
                raise _projection_error(
                    "Supporting evidence excerpt is missing.",
                    failure_classification="supporting_evidence_excerpt_missing",
                    binding_id=binding_id,
                    evidence_id=evidence_id,
                )
            continue
        if strict_required_projection or binding_id in required_supporting_ids:
            _validate_supporting_projection(
                binding_id=binding_id,
                binding=binding,
                evidence=evidence,
            )
            source_hash = _evidence_source_hash(evidence)
            if not isinstance(source_hash, str) or not source_hash:
                raise _projection_error(
                    "Supporting evidence is missing its source hash.",
                    failure_classification="supporting_evidence_source_hash_missing",
                    binding_id=binding_id,
                    evidence_id=evidence_id,
                )
        supporting_views.append(
            _supporting_evidence_view(
                binding_id=binding_id,
                binding=binding,
                evidence=evidence,
                include_member_facts=binding_id in required_supporting_ids,
            )
        )
        consumed_evidence_ids.add(evidence_id)

    projection: dict[str, Any] = {
        "edit_binding_contexts": projected_edit_contexts,
        "supporting_evidence": supporting_views,
    }
    if not strict_required_projection:
        unbound_excerpts = [
            excerpt
            for item in evidence_items
            if isinstance(item, dict)
            and str(item.get("evidence_id") or "") not in consumed_evidence_ids
            and isinstance((excerpt := item.get("excerpt")), str)
            and excerpt
        ]
        if unbound_excerpts:
            projection["unbound_evidence_excerpts"] = unbound_excerpts
    return projection


def _validate_required_change_input_coverage(
    required_change_units: tuple[dict[str, Any], ...],
    *,
    allowed_bindings: set[str],
    required_change_units_supplied: bool,
) -> None:
    if not required_change_units_supplied:
        return
    covered_bindings = {
        str(item["edit_binding_id"]) for item in required_change_units
    }
    missing_bindings = sorted(allowed_bindings - covered_bindings)
    if not required_change_units or missing_bindings:
        raise _projection_error(
            "Required change units do not cover every editable binding.",
            failure_classification="required_change_input_coverage_incomplete",
            missing_edit_binding_ids=missing_bindings,
        )


def _patch_system_prompt(
    *,
    require_native_schema: bool,
    plan_bound_response: bool,
) -> str:
    implementation_scope = (
        "Implement the complete request with exact-text edits. Minimal means avoiding unrelated changes, "
        "not omitting required behavior or necessary state. Do not fabricate existing files, binding ids, "
        "existing symbols, or external dependency APIs. You may add internal declarations, state and logic "
        "inside authorized edit bindings; preserve public interfaces required by the user. "
        "This grants no additional paths or edit authority. "
        "为什么要有它：防止虚构依赖和越权，不应禁止实现完整需求所需的内部代码。 "
    )
    if require_native_schema:
        return (
            implementation_scope
            + "Use verified evidence; strict schema-only JSON, no fences. "
            "Do not add trailing spaces or tabs. One change per required_change_units item, solely exact "
            "edit_binding_id. Omit Plan/binding-derived "
            "change_id/paths/hashes/kind/occurrence count/change type/operation/validation. "
            "Native v3: optional before_text selects one exact nonempty unique fragment of matching source_excerpt; after_text replaces only it. Prefer a small unique exact fragment for a local change; include enough original context to disambiguate. Omit before_text to replace the full excerpt. Preserve all unrelated text and CRLF/BOM. 为什么要有它：全文阅读不应强迫重写全文；局部锚点仍须受原binding和源码前像约束。 "
            "target_behavior/constraints/acceptance_criteria/runtime_verification_plan form one feature contract. "
            "Units carry only identity/operation/binding refs, not per-change obligations. "
            "supporting_evidence is read-only. Non-Boolean receivers as Boolean operands require exact "
            "verified_member_accesses. Uppercase required_validation_by_binding identifiers must appear literally in the resulting bound source (preserved original text plus replacement)."
        )
    identity_contract = (
        "Return exactly one change for every required_change_units entry and "
        "identify it only by its exact edit_binding_id. Do not return change_id; "
        "deterministic code derives it from the unique Plan binding. "
        if require_native_schema
        else (
            "Return exactly one change for every required_change_units entry and "
            "preserve its exact change_id and edit_binding_id. "
        )
    )
    shared = (
        implementation_scope
        + "Use verified evidence. Return strict JSON only. "
        "supporting_evidence is read-only context and never "
        "authorizes an edit target. "
        + identity_contract
        + "target_behavior, constraints, acceptance_criteria, "
        "and runtime_verification_plan form one feature-level generation contract. "
        "required_change_units owns only the displayed change identity, operation, "
        "and binding references; it does not assign a feature obligation to an "
        "individual change. Combine the feature-level generation contract with "
        "each matching binding source to implement the feature. Any "
        "non-Boolean bound receiver may be used as a "
        "Boolean operand only through an exact verified_member_accesses entry. "
        "Do not fabricate external dependency API members. Any required_validation_by_binding entry "
        "formatted as an uppercase identifier is a literal marker and must "
        "appear in after_text. Do not add trailing spaces or tabs at line ends. "
        "Do not include markdown fences. "
    )
    if require_native_schema:
        return shared + (
            "Native v3 allows optional before_text selecting one exact nonempty unique fragment of the matching source_excerpt. "
            "after_text replaces that fragment only; omitted before_text replaces the entire source_excerpt. Preserve all "
            "unrelated text. Return only the schema fields. Do not return or copy "
            "a file path, a source hash, patch kind, occurrence count, "
            "change type, operation, or expected validation; deterministic code "
            "materializes those authoritative fields from the verified binding and "
            "Plan."
        )
    if plan_bound_response:
        return shared + (
            "Each change must use one allowed edit_binding_id and an "
            "exact_text_replace patch whose before_text is copied verbatim from the "
            "source_excerpt in the matching edit_binding_context, including its exact "
            "line endings. Do not return expected_validation; deterministic code "
            "materializes it from the exact Plan change_id and edit_binding_id."
        )
    return shared + (
        "Each change must use one allowed edit_binding_id and an "
        "exact_text_replace patch whose before_text is copied verbatim from the "
        "source_excerpt in the matching edit_binding_context, including its exact "
        "line endings. Any required_validation_by_binding entries are authoritative "
        "upstream requirements and must all be copied into the matching change's "
        "expected_validation list."
    )


def _patch_response_schema(
    allowed_bindings: set[str],
    *,
    required_change_count: int | None = None,
    allow_local_before: bool = True,
) -> dict[str, Any]:
    """Return the single schema used by the local prompt and native format."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["changes"],
        "properties": {
            "changes": {
                "type": "array",
                "minItems": required_change_count or 1,
                **(
                    {"maxItems": required_change_count}
                    if required_change_count is not None
                    else {}
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "edit_binding_id",
                        "rationale",
                        "rollback_hint",
                        "after_text",
                    ],
                    "properties": {
                        "edit_binding_id": {
                            "type": "string",
                            "enum": sorted(allowed_bindings),
                        },
                        "rationale": {"type": "string", "minLength": 1},
                        "rollback_hint": {"type": "string", "minLength": 1},
                        "after_text": {"type": "string", "minLength": 1},
                        **({"before_text": {"type": "string", "minLength": 1}}
                           if allow_local_before else {}),
                    },
                },
            }
        },
    }


def _legacy_pair_patch_response_schema(
    allowed_bindings: set[str],
    *,
    required_change_count: int | None = None,
) -> dict[str, Any]:
    """Return the exact five-field schema already frozen in v1 attempts."""

    schema = _patch_response_schema(
        allowed_bindings,
        required_change_count=required_change_count,
        allow_local_before=False,
    )
    item_schema = schema["properties"]["changes"]["items"]
    item_schema["required"] = [
        "change_id",
        "edit_binding_id",
        "rationale",
        "rollback_hint",
        "after_text",
    ]
    item_schema["properties"] = {
        "change_id": {"type": "string", "minLength": 1},
        **item_schema["properties"],
    }
    return schema


def _normalize_semantic_obligation_list(
    raw: Any,
    *,
    field_name: str,
    supplied: bool,
    allow_empty: bool,
    required: bool = False,
    **diagnostics: Any,
) -> tuple[str, ...]:
    if not supplied:
        if required:
            raise _projection_error(
                f"Required change unit {field_name} is missing.",
                failure_classification=(
                    f"required_change_unit_{field_name}_invalid"
                ),
                **diagnostics,
            )
        return ()
    if not isinstance(raw, (list, tuple)) or (not allow_empty and not raw):
        raise _projection_error(
            f"{field_name} must be a"
            + (" list." if allow_empty else " non-empty list."),
            failure_classification=(
                f"required_change_unit_{field_name}_invalid"
                if diagnostics
                else f"{field_name}_invalid"
            ),
            **diagnostics,
        )
    if any(not isinstance(item, str) or not item.strip() for item in raw):
        raise _projection_error(
            f"{field_name} must contain only non-empty strings.",
            failure_classification=(
                f"required_change_unit_{field_name}_invalid"
                if diagnostics
                else f"{field_name}_invalid"
            ),
            **diagnostics,
        )
    return tuple(raw)


def _normalize_semantic_obligation_string(
    raw: Any,
    *,
    field_name: str,
    supplied: bool,
) -> str:
    if not supplied:
        return ""
    if not isinstance(raw, str) or not raw.strip():
        raise _projection_error(
            f"{field_name} must be a non-empty string.",
            failure_classification=f"{field_name}_invalid",
        )
    return raw


def _normalize_required_change_units(
    raw_units: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    allowed_bindings: set[str],
    supporting_bindings: set[str],
    require_native_contract: bool = False,
) -> tuple[dict[str, Any], ...]:
    if raw_units is None:
        return ()
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_binding_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_units, start=1):
        if not isinstance(raw, dict):
            raise _projection_error(
                f"Required change unit {index} must be an object.",
                failure_classification="required_change_unit_invalid",
                change_index=index,
            )
        change_id = str(raw.get("change_id") or "")
        binding_id = str(raw.get("edit_binding_id") or "")
        if not change_id or not binding_id:
            raise _projection_error(
                f"Required change unit {index} requires change_id and edit_binding_id.",
                failure_classification="required_change_unit_identity_incomplete",
                change_index=index,
            )
        if binding_id not in allowed_bindings:
            raise _projection_error(
                f"Required change unit {index} uses an unknown edit binding id.",
                failure_classification="required_change_unit_binding_unknown",
                change_index=index,
                binding_id=binding_id,
            )
        change_type = raw.get("change_type")
        if require_native_contract and change_type != _NATIVE_PLAN_CHANGE_TYPE:
            raise _projection_error(
                "Native required change unit has an invalid Plan change_type.",
                failure_classification="required_change_unit_change_type_invalid",
                change_index=index,
                change_id=change_id,
                expected_change_type=_NATIVE_PLAN_CHANGE_TYPE,
                observed_change_type=change_type,
            )
        operation = raw.get("operation")
        if (
            require_native_contract
            and operation not in _NATIVE_PLAN_OPERATION_TO_PATCH_OPERATION
        ):
            raise _projection_error(
                "Native required change unit has an invalid Plan operation.",
                failure_classification="required_change_unit_operation_invalid",
                change_index=index,
                change_id=change_id,
                allowed_operations=sorted(
                    _NATIVE_PLAN_OPERATION_TO_PATCH_OPERATION
                ),
                observed_operation=operation,
            )
        rationale = raw.get("rationale")
        if "rationale" in raw and (
            not isinstance(rationale, str) or not rationale.strip()
        ):
            raise _projection_error(
                "Required change unit rationale must be a non-empty string when supplied.",
                failure_classification="required_change_unit_rationale_invalid",
                change_index=index,
                change_id=change_id,
            )
        expected_validation = _normalize_semantic_obligation_list(
            raw.get("expected_validation"),
            field_name="expected_validation",
            supplied="expected_validation" in raw,
            allow_empty=True,
            change_index=index,
            change_id=change_id,
        )
        pair = (change_id, binding_id)
        if require_native_contract and binding_id in seen_binding_ids:
            raise _projection_error(
                "Native required change units must have unique edit_binding_id values.",
                failure_classification="required_change_unit_binding_duplicate",
                change_index=index,
                change_id=change_id,
                binding_id=binding_id,
            )
        if change_id in seen_ids or pair in seen_pairs:
            raise _projection_error(
                "Required change units must have unique change_id values.",
                failure_classification="required_change_unit_identity_duplicate",
                change_index=index,
                change_id=change_id,
            )
        raw_supporting_ids = raw.get("supporting_binding_ids")
        if raw_supporting_ids is None:
            normalized_supporting_ids: list[str] = []
        elif isinstance(raw_supporting_ids, (list, tuple)):
            normalized_supporting_ids = [str(item) for item in raw_supporting_ids]
        else:
            raise _projection_error(
                "Required change unit supporting_binding_ids must be a list.",
                failure_classification="required_supporting_binding_ids_invalid",
                change_index=index,
                change_id=change_id,
            )
        if any(not item for item in normalized_supporting_ids) or len(
            set(normalized_supporting_ids)
        ) != len(normalized_supporting_ids):
            raise _projection_error(
                "Required change unit supporting binding ids must be non-empty and unique.",
                failure_classification="required_supporting_binding_ids_invalid",
                change_index=index,
                change_id=change_id,
            )
        unknown_supporting_ids = sorted(
            set(normalized_supporting_ids) - allowed_bindings - supporting_bindings
        )
        if unknown_supporting_ids:
            raise _projection_error(
                "Required change unit references an unknown supporting binding id.",
                failure_classification="required_supporting_binding_unknown",
                change_index=index,
                change_id=change_id,
                binding_ids=unknown_supporting_ids,
            )
        raw_hook_target = raw.get("hook_target_binding_id")
        hook_target_binding_id = (
            str(raw_hook_target) if raw_hook_target is not None else None
        )
        if hook_target_binding_id == "" or (
            hook_target_binding_id is not None
            and hook_target_binding_id not in supporting_bindings
        ):
            raise _projection_error(
                "Required change unit references an unknown hook target binding id.",
                failure_classification="required_hook_target_binding_unknown",
                change_index=index,
                change_id=change_id,
                binding_id=hook_target_binding_id,
            )
        seen_ids.add(change_id)
        seen_binding_ids.add(binding_id)
        seen_pairs.add(pair)
        normalized.append(
            {
                "change_id": raw.get("change_id"),
                "change_type": change_type,
                "edit_binding_id": raw.get("edit_binding_id"),
                "hook_target_binding_id": hook_target_binding_id,
                "supporting_binding_ids": normalized_supporting_ids,
                "operation": operation,
                **(
                    {"rationale": rationale}
                    if "rationale" in raw
                    else {}
                ),
                **(
                    {"expected_validation": list(expected_validation)}
                    if "expected_validation" in raw
                    else {}
                ),
            }
        )
    return tuple(normalized)


def _validate_required_change_coverage(
    changes: tuple[dict[str, Any], ...],
    *,
    required_change_units: tuple[dict[str, Any], ...],
) -> None:
    if not required_change_units:
        return
    expected = [
        (str(item["change_id"]), str(item["edit_binding_id"]))
        for item in required_change_units
    ]
    actual = [
        (str(item.get("change_id") or ""), str(item.get("edit_binding_id") or ""))
        for item in changes
    ]
    missing = [pair for pair in expected if pair not in actual]
    unexpected = [pair for pair in actual if pair not in expected]
    duplicates = sorted({pair for pair in actual if actual.count(pair) > 1})
    if missing or unexpected or duplicates or len(actual) != len(expected):
        raise PatchGenerationError(
            "Model output does not exactly cover required implementation-plan changes.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "plan_change_coverage_incomplete",
                "missing_plan_changes": [
                    {"change_id": change_id, "edit_binding_id": binding_id}
                    for change_id, binding_id in missing
                ],
                "unexpected_plan_changes": [
                    {"change_id": change_id, "edit_binding_id": binding_id}
                    for change_id, binding_id in unexpected
                ],
                "duplicate_plan_changes": [
                    {"change_id": change_id, "edit_binding_id": binding_id}
                    for change_id, binding_id in duplicates
                ],
            },
        )


def _response_diagnostics(response: ChatResponse) -> dict[str, Any]:
    raw = response.raw_response if isinstance(response.raw_response, Mapping) else {}
    return {
        "failure_boundary": "deterministic_patch_validation",
        "response_model": response.model,
        "done": raw.get("done"),
        "done_reason": raw.get("done_reason"),
        "usage": dict(response.usage) if response.usage is not None else None,
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PatchGenerationError(
            f"Model output is not strict JSON: {exc}",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "malformed_structured_output",
                "json_error_line": exc.lineno,
                "json_error_column": exc.colno,
                "json_error_character_index": exc.pos,
            },
        ) from exc
    if not isinstance(value, dict):
        raise PatchGenerationError("Model output root must be an object.")
    return value


def _submission_diagnostic_path(path: Any) -> list[str | int]:
    """Keep feedback bounded even when an output contains unusually long keys."""

    return [part if isinstance(part, int) else str(part)[:96] for part in list(path)[:32]]


def _patch_submission_validator(schema: Mapping[str, Any]) -> Any:
    if not isinstance(schema, Mapping):
        raise ValueError("Patch submission schema must be an object.")
    if "$schema" in schema and not isinstance(schema["$schema"], str):
        raise ValueError("Patch submission schema dialect must be a string.")
    validator_class = validator_for(dict(schema), default=None)
    if validator_class is None:
        # The production schema omits $schema; use the library's current default.
        if "$schema" in schema:
            raise ValueError("Patch submission schema declares an unsupported dialect.")
        validator_class = validator_for(dict(schema))
    try:
        validator_class.check_schema(dict(schema))
    except SchemaError as exc:
        path = _submission_diagnostic_path(exc.absolute_path)
        raise ValueError(f"Invalid Patch submission schema at {path}.") from exc
    # This boundary is deterministic and cannot fetch a remote schema.
    return validator_class(dict(schema), registry=Registry())


def validate_patch_submission_schema(schema: Mapping[str, Any]) -> None:
    """Reject configuration defects before a model call, without defining a schema."""

    _patch_submission_validator(schema)


def validate_patch_submission(
    content: str, schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one complete submission, without extracting or materializing code.

    Model format errors use PatchGenerationError. Invalid or unresolvable schemas
    are configuration errors (ValueError), never instructions to repair code.
    """

    validator = _patch_submission_validator(schema)
    try:
        value = _parse_json_object(content)
    except PatchGenerationError as exc:
        if not exc.diagnostics:
            exc.diagnostics.update(
                failure_boundary="deterministic_patch_validation",
                failure_classification="malformed_structured_output",
                field_path=[],
                schema_constraint="type",
            )
        raise
    try:
        error = next(validator.iter_errors(value), None)
    except Unresolvable as exc:
        raise ValueError("Patch submission schema contains an unresolved reference.") from exc
    if error is None:
        return value
    path = _submission_diagnostic_path(error.absolute_path)
    constraint = str(error.validator)
    details = ""
    if constraint == "required" and isinstance(error.instance, dict):
        missing = [key[:96] for key in error.validator_value if key not in error.instance][:8]
        details = f"; missing required fields: {json.dumps(missing)}"
    raise PatchGenerationError(
        f"Patch submission at {path} violates schema constraint {constraint}{details}.",
        diagnostics={
            "failure_boundary": "deterministic_patch_validation",
            "failure_classification": "patch_submission_schema_violation",
            "field_path": path,
            "schema_path": _submission_diagnostic_path(error.absolute_schema_path),
            "schema_constraint": constraint,
        },
    )


def _native_validation_error(
    message: str,
    *,
    failure_classification: str,
    **diagnostics: Any,
) -> PatchGenerationError:
    return PatchGenerationError(
        message,
        diagnostics={
            "failure_boundary": "deterministic_patch_validation",
            "failure_classification": failure_classification,
            **diagnostics,
        },
    )


def _materialize_native_change(
    raw: Any,
    *,
    allowed_bindings: set[str],
    binding_evidence_excerpts: Mapping[str, str],
    required_changes_by_binding: Mapping[str, Mapping[str, Any]],
    legacy_required_changes_by_identity: Mapping[
        tuple[str, str], Mapping[str, Any]
    ]
    | None = None,
    required_validation: Mapping[str, tuple[str, ...]],
    binding_contexts: Mapping[str, Mapping[str, Any]],
    index: int,
    allow_local_before: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _native_validation_error(
            f"Change {index} must be an object.",
            failure_classification="native_response_fields_invalid",
            change_index=index,
        )
    legacy_pair_identity = legacy_required_changes_by_identity is not None
    required_fields = {
        "edit_binding_id",
        "rationale",
        "rollback_hint",
        "after_text",
    }
    if legacy_pair_identity:
        required_fields.add("change_id")
    actual_fields = set(raw)
    local = "before_text" in raw
    if local and allow_local_before and not legacy_pair_identity:
        required_fields.add("before_text")
    if actual_fields != required_fields:
        raise _native_validation_error(
            f"Change {index} does not match the native Patch response contract.",
            failure_classification="native_response_fields_invalid",
            change_index=index,
            missing_fields=sorted(required_fields - actual_fields),
            unexpected_fields=sorted(actual_fields - required_fields),
        )

    binding_id = raw["edit_binding_id"]
    rationale = raw["rationale"]
    rollback_hint = raw["rollback_hint"]
    after = raw["after_text"]
    for field, value in (
        ("edit_binding_id", binding_id),
        ("rationale", rationale),
        ("rollback_hint", rollback_hint),
    ):
        if not isinstance(value, str) or not value:
            raise _native_validation_error(
                f"Change {index} {field} must be a non-empty string.",
                failure_classification="native_response_fields_invalid",
                change_index=index,
                field=field,
            )
    if binding_id not in allowed_bindings:
        raise _native_validation_error(
            f"Change {index} uses an unknown edit binding id.",
            failure_classification="native_change_binding_unknown",
            change_index=index,
            binding_id=binding_id,
        )
    if legacy_pair_identity:
        change_id = raw["change_id"]
        if not isinstance(change_id, str) or not change_id:
            raise _native_validation_error(
                f"Change {index} change_id must be a non-empty string.",
                failure_classification="native_response_fields_invalid",
                change_index=index,
                field="change_id",
            )
        required_change = legacy_required_changes_by_identity.get(
            (change_id, binding_id)
        )
        if required_change is None:
            raise _native_validation_error(
                f"Change {index} is not a required Plan change identity.",
                failure_classification="native_change_identity_unknown",
                change_index=index,
                change_id=change_id,
                binding_id=binding_id,
            )
    else:
        required_change = required_changes_by_binding.get(binding_id)
        if required_change is None:
            raise _native_validation_error(
                f"Change {index} binding is not a required Plan change.",
                failure_classification="native_change_binding_not_required",
                change_index=index,
                binding_id=binding_id,
            )
        change_id = str(required_change["change_id"])
    if not isinstance(after, str) or not after:
        raise _native_validation_error(
            f"Change {index} after_text must be non-empty.",
            failure_classification="native_after_text_invalid",
            change_index=index,
            change_id=change_id,
            binding_id=binding_id,
        )
    full_before = binding_evidence_excerpts.get(binding_id)
    before = full_before
    if not isinstance(before, str) or not before:
        raise _native_validation_error(
            f"Change {index} has no exact authoritative source excerpt.",
            failure_classification="native_binding_excerpt_missing",
            change_index=index,
            change_id=change_id,
            binding_id=binding_id,
        )
    if local:
        before = raw["before_text"]
        if (not isinstance(before, str) or not before or _overlapping_count(full_before, before) != 1):
            raise _native_validation_error("Native local before_text must select exactly one bound source fragment.",
                failure_classification="native_before_text_not_unique_in_binding", binding_id=binding_id)
        if before.startswith("\ufeff") != after.startswith("\ufeff"):
            raise _native_validation_error("Native local replacement must preserve the selected BOM bytes.",
                failure_classification="native_local_bom_changed", binding_id=binding_id)
    if not local and before.startswith("\ufeff"):
        if not after.startswith("\ufeff"):
            after = "\ufeff" + after
    elif not local and after.startswith("\ufeff"):
        raise _native_validation_error(
            f"Change {index} must not introduce a UTF-8 BOM.",
            failure_classification="native_after_text_bom_invalid",
            change_index=index,
            change_id=change_id,
            binding_id=binding_id,
        )
    # 为什么要有它：局部raw输出保持原样，语言和验证义务仍针对重建后的完整绑定上下文检查。
    complete_after = full_before.replace(before, after, 1) if local else after
    _validate_bound_language_semantics(
        after=complete_after,
        binding_id=binding_id,
        binding_contexts=binding_contexts,
        index=index,
    )
    expected_validation = list(
        required_change.get(
            "expected_validation",
            required_validation.get(binding_id, ()),
        )
    )
    missing_markers = [
        marker
        for marker in expected_validation
        if _LITERAL_VALIDATION_MARKER_RE.fullmatch(marker) and marker not in complete_after
    ]
    if missing_markers:
        raise _native_validation_error(
            f"Change {index} after_text is missing Plan validation marker(s): "
            + ", ".join(missing_markers),
            failure_classification="declared_validation_marker_missing",
            change_index=index,
            change_id=change_id,
            binding_id=binding_id,
            missing_validation_markers=missing_markers,
        )

    plan_operation = required_change["operation"]
    patch_operation = _NATIVE_PLAN_OPERATION_TO_PATCH_OPERATION[plan_operation]
    return {
        "change_id": change_id,
        "change_type": required_change["change_type"],
        "edit_binding_id": binding_id,
        "supporting_binding_ids": list(
            required_change.get("supporting_binding_ids") or []
        ),
        "operation": patch_operation,
        "rationale": rationale,
        # Compatibility metadata only. local_entry preserves authoritative Plan
        # risk_tags when it overlays this generated patch onto the planned change.
        "risk_tags": list(_LEGACY_GENERATED_PATCH_RISK_TAGS),
        "expected_validation": expected_validation,
        "rollback_hint": rollback_hint,
        "patch": {
            "kind": "exact_text_replace",
            "before_text": before,
            "after_text": after,
            "expected_occurrences": 1,
        },
    }


def _overlapping_count(text: str, selected: str) -> int:
    if not selected:
        return 0
    count, offset = 0, 0
    while (offset := text.find(selected, offset)) >= 0:
        count += 1
        offset += 1
    return count


def _validate_native_local_ranges(changes, contexts, excerpts):
    """Compare original-source coordinates, not positions shifted by earlier replacements."""
    occupied = {}
    for change in changes:
        binding_id = change["edit_binding_id"]
        context = contexts[binding_id]
        span = context.get("target_span")
        full = excerpts[binding_id]
        before = change["patch"]["before_text"]
        if (not isinstance(span, Mapping) or type(span.get("start_line")) is not int
                or type(span.get("end_line")) is not int or span["start_line"] < 1
                or span["end_line"] < span["start_line"] or _overlapping_count(full, before) != 1):
            raise _native_validation_error("Native local patch requires a unique original span.",
                failure_classification="native_local_span_invalid", binding_id=binding_id)
        start = full.index(before)
        def coordinate(position):
            prefix = full[:position]
            return (span["start_line"] + prefix.count("\n"),
                    len(prefix.rsplit("\n", 1)[-1].encode("utf-8")))
        interval = coordinate(start), coordinate(start + len(before))
        path = context.get("canonical_target_path")
        last_line = coordinate(start + len(before.rstrip("\r\n")))[0]
        if not isinstance(path, str) or not path or last_line > span["end_line"]:
            raise _native_validation_error("Native local patch escapes its Plan span.",
                failure_classification="native_local_span_invalid", binding_id=binding_id)
        # 为什么要有它：全部片段必须对同一B0定位，重叠选择不能依赖先前生成文本或执行顺序取得授权。
        for previous in occupied.setdefault(path, []):
            if max(interval[0], previous[0]) < min(interval[1], previous[1]):
                raise _native_validation_error("Native local patch selections overlap.",
                    failure_classification="native_local_selections_overlap", binding_id=binding_id)
        occupied[path].append(interval)


def _validate_submission_text(
    changes: tuple[dict[str, Any], ...],
    *,
    binding_contexts: Mapping[str, Mapping[str, Any]],
    native: bool,
    binding_evidence_excerpts: Mapping[str, str] | None = None,
    native_v3_raw_changes: list[Any] | None = None,
) -> None:
    """Check repairable text only after every change's hard validation passes."""
    for index, change in enumerate(changes, start=1):
        binding_id = change["edit_binding_id"]
        before = change["patch"]["before_text"]
        after = change["patch"]["after_text"]
        if before == after:
            from .noop_submission_feedback import whole_noop_facts

            raise PatchSubmissionValidationError(
                f"Change {index} after_text must be a real replacement.",
                diagnostics={
                    "failure_boundary": "deterministic_patch_validation",
                    "failure_classification": (
                        "native_after_text_not_replacement"
                        if native else "patch_after_text_not_replacement"
                    ),
                    "change_index": index,
                    "change_id": change["change_id"],
                    "binding_id": binding_id,
                    **(whole_noop_facts(changes) if native else {}),
                },
            )
        if native:
            full_before = (binding_evidence_excerpts or {}).get(binding_id, before)
            full_after = full_before.replace(before, after, 1) if before != full_before else after
            try:
                _validate_csharp_header_preservation(
                    before=full_before, after=full_after,
                    binding_context=binding_contexts.get(binding_id) or {},
                    index=index, change_id=change["change_id"], binding_id=binding_id,
                )
            except PatchSubmissionValidationError as error:
                _add_native_v3_omitted_anchor_feedback(error, native_v3_raw_changes, index)
                raise


def _add_native_v3_omitted_anchor_feedback(error, raw_changes, index):
    """Explain an omitted v3 anchor without inferring or changing submitted bytes."""
    if (error.diagnostics.get("failure_classification") != "csharp_file_header_usings_removed"
            or raw_changes is None or index > len(raw_changes)):
        return
    raw = raw_changes[index - 1]
    if not isinstance(raw, dict) or "before_text" in raw:
        return
    # 为什么要有它：v3 片段漏填锚点会被当作完整绑定替换；纠错需说明协议原因，不能自动猜锚点或放宽原校验。
    error.diagnostics["native_v3_correction"] = (
        "This native v3 change omitted before_text, so after_text replaces the entire bound source_excerpt. "
        "For a local replacement, provide before_text copied verbatim from exactly one nonempty unique fragment "
        "of the original source_excerpt, and after_text for that fragment. Alternatively, omit before_text and "
        "submit the complete replacement source_excerpt as after_text, preserving unrelated code and using directives. "
        "为什么要有它：省略锚点的局部片段不能被当作完整源码替换；原始绑定与校验边界保持不变。"
    )


def _validate_csharp_header_preservation(
    *,
    before: str,
    after: str,
    binding_context: Mapping[str, Any],
    index: int,
    change_id: str,
    binding_id: str,
) -> None:
    path = str(binding_context.get("canonical_target_path") or "")
    span = binding_context.get("target_span")
    if (
        not path.lower().endswith(".cs")
        or not isinstance(span, Mapping)
        or span.get("start_line") != 1
    ):
        return
    before_usings = {
        line.strip().lstrip("\ufeff")
        for line in before.splitlines()
        if line.strip().lstrip("\ufeff").startswith("using ")
        and line.strip().endswith(";")
    }
    after_usings = {
        line.strip().lstrip("\ufeff")
        for line in after.splitlines()
        if line.strip().lstrip("\ufeff").startswith("using ")
        and line.strip().endswith(";")
    }
    if before_usings and not after_usings:
        raise PatchSubmissionValidationError(
            f"Change {index} removes every existing C# using directive from "
            "a file-header replacement.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "csharp_file_header_usings_removed",
                "change_index": index,
                "change_id": change_id,
                "binding_id": binding_id,
                "removed_using_count": len(before_usings),
            },
        )


def _validate_change(
    raw: Any,
    *,
    allowed_bindings: set[str],
    evidence_excerpts: tuple[str, ...],
    binding_evidence_excerpts: Mapping[str, str],
    required_validation: Mapping[str, tuple[str, ...]],
    required_changes_by_identity: Mapping[
        tuple[str, str], Mapping[str, Any]
    ]
    | None,
    binding_contexts: Mapping[str, Mapping[str, Any]],
    index: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PatchGenerationError(f"Change {index} must be an object.")
    required_change: Mapping[str, Any] | None = None
    if required_changes_by_identity is not None:
        if "expected_validation" in raw:
            raise _native_validation_error(
                f"Change {index} must not return Plan-owned expected_validation.",
                failure_classification=(
                    "plan_bound_non_native_response_fields_invalid"
                ),
                change_index=index,
                unexpected_fields=["expected_validation"],
            )
        raw_change_id = raw.get("change_id")
        raw_binding_id = raw.get("edit_binding_id")
        identity = (
            raw_change_id if isinstance(raw_change_id, str) else "",
            raw_binding_id if isinstance(raw_binding_id, str) else "",
        )
        required_change = required_changes_by_identity.get(identity)
        if required_change is None:
            raise _native_validation_error(
                f"Change {index} is not a required Plan change identity.",
                failure_classification=(
                    "plan_bound_non_native_change_identity_unknown"
                ),
                change_index=index,
                change_id=identity[0],
                binding_id=identity[1],
            )
    binding_id = str(raw.get("edit_binding_id") or "")
    if binding_id not in allowed_bindings:
        raise PatchGenerationError(f"Change {index} uses an unknown edit binding id.")
    if any(raw.get(key) is not None for key in ("target_file", "edit_target_path", "path")):
        raise PatchGenerationError(f"Change {index} must not copy or invent a path.")
    patch = raw.get("patch")
    if not isinstance(patch, dict) or patch.get("kind") != "exact_text_replace":
        raise PatchGenerationError(f"Change {index} requires an exact_text_replace patch.")
    before = patch.get("before_text")
    after = patch.get("after_text")
    if not isinstance(before, str) or not before:
        raise PatchGenerationError(f"Change {index} before_text must be non-empty.")
    binding_excerpt = binding_evidence_excerpts.get(binding_id)
    if binding_excerpt is not None and before not in binding_excerpt:
        raise PatchGenerationError(
            f"Change {index} before_text is outside its edit binding evidence.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "before_text_outside_edit_binding",
                "binding_id": binding_id,
            },
        )
    if binding_excerpt is None and not any(
        before in excerpt for excerpt in evidence_excerpts
    ):
        raise PatchGenerationError(
            f"Change {index} before_text is not verbatim current evidence.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "before_text_not_verbatim_evidence",
            },
        )
    if not isinstance(after, str) or not after:
        raise PatchGenerationError(f"Change {index} after_text must be a real replacement.")
    _validate_bound_language_semantics(
        after=after,
        binding_id=binding_id,
        binding_contexts=binding_contexts,
        index=index,
    )
    if patch.get("expected_occurrences") != 1:
        raise PatchGenerationError(f"Change {index} expected_occurrences must equal 1.")
    if required_change is not None:
        expected_validation = list(
            required_change.get("expected_validation") or ()
        )
    else:
        model_expected_validation = [
            str(item)
            for item in raw.get("expected_validation") or ["dotnet build"]
        ]
        upstream_validation = required_validation.get(binding_id, ())
        missing_upstream_validation = [
            item
            for item in upstream_validation
            if item not in _DETERMINISTICALLY_INHERITED_VALIDATION
            and item not in model_expected_validation
        ]
        if missing_upstream_validation:
            raise PatchGenerationError(
                f"Change {index} omits upstream validation requirement(s): "
                + ", ".join(missing_upstream_validation),
                diagnostics={
                    "failure_boundary": "deterministic_patch_validation",
                    "failure_classification": "upstream_validation_omitted",
                    "missing_upstream_validation": missing_upstream_validation,
                },
            )
        expected_validation = list(model_expected_validation)
        for item in reversed(upstream_validation):
            if (
                item in _DETERMINISTICALLY_INHERITED_VALIDATION
                and item not in expected_validation
            ):
                expected_validation.insert(0, item)
    missing_markers = [
        marker
        for marker in expected_validation
        if _LITERAL_VALIDATION_MARKER_RE.fullmatch(marker) and marker not in after
    ]
    if missing_markers:
        raise PatchGenerationError(
            f"Change {index} after_text is missing declared validation marker(s): "
            + ", ".join(missing_markers),
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "declared_validation_marker_missing",
                "missing_validation_markers": missing_markers,
            },
        )
    return {
        "change_id": (
            str(required_change["change_id"])
            if required_change is not None
            else str(raw.get("change_id") or f"change_{index:03d}")
        ),
        "change_type": str(raw.get("change_type") or "modify_behavior"),
        "edit_binding_id": binding_id,
        "supporting_binding_ids": [],
        "operation": "exact_text_replace",
        "rationale": str(raw.get("rationale") or "Evidence-bound minimal change."),
        "risk_tags": ["medium"],
        "expected_validation": expected_validation,
        "rollback_hint": str(raw.get("rollback_hint") or "Restore before_text."),
        "patch": {
            "kind": "exact_text_replace",
            "before_text": before,
            "after_text": after,
            "expected_occurrences": 1,
        },
    }


def _validate_bound_language_semantics(
    *,
    after: str,
    binding_id: str,
    binding_contexts: Mapping[str, Mapping[str, Any]],
    index: int,
) -> None:
    context = binding_contexts.get(binding_id) or {}
    source_path = context.get("canonical_target_path")
    return_type = context.get("return_type")
    if not isinstance(source_path, str) or not source_path.lower().endswith(".cs"):
        return
    code_only = _strip_csharp_non_code(after)
    if return_type == "void":
        invalid_returns = [
            match.group(1).strip()
            for match in re.finditer(r"\breturn\b\s*([^;]*);", code_only)
            if match.group(1).strip()
        ]
        if invalid_returns:
            raise PatchGenerationError(
                f"Change {index} returns a value from bound void method.",
                diagnostics={
                    "failure_boundary": "deterministic_patch_validation",
                    "failure_classification": "csharp_return_value_in_void_method",
                    "binding_id": binding_id,
                    "method_name": context.get("method_name"),
                    "return_type": return_type,
                },
            )

    declared_member_types = context.get("declared_member_types")
    if not isinstance(declared_member_types, Mapping):
        return
    _validate_bound_member_shadowing(
        after=code_only,
        declared_member_types=declared_member_types,
        binding_id=binding_id,
        context=context,
        index=index,
    )
    for match in re.finditer(
        r"!(?!=)\s*(?:this\s*\.\s*)?(?P<member>[A-Za-z_][A-Za-z0-9_]*)"
        r"\b(?!\s*[.(])",
        code_only,
    ):
        member_name = match.group("member")
        declared_type = declared_member_types.get(member_name)
        if not isinstance(declared_type, str):
            continue
        normalized_type = re.sub(r"\s+", "", declared_type)
        if normalized_type in {"bool", "bool?", "System.Boolean", "System.Boolean?"}:
            continue
        raise PatchGenerationError(
            f"Change {index} applies unary ! to bound non-boolean member "
            f"{member_name} of type {declared_type}.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": (
                    "csharp_boolean_operator_operand_type_mismatch"
                ),
                "binding_id": binding_id,
                "method_name": context.get("method_name"),
                "operator": "!",
                "member_name": member_name,
                "declared_type": declared_type,
            },
        )
    for match in re.finditer(
        r"!(?!=)\s*(?:this\s*\.\s*)?(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\.\s*(?P<member>[A-Za-z_][A-Za-z0-9_]*)\b(?!\s*[.(])",
        code_only,
    ):
        receiver_name = match.group("receiver")
        receiver_type = declared_member_types.get(receiver_name)
        if not isinstance(receiver_type, str):
            continue
        normalized_type = re.sub(r"\s+", "", receiver_type)
        if normalized_type in {"bool", "bool?", "System.Boolean", "System.Boolean?"}:
            continue
        member_name = match.group("member")
        if _has_verified_boolean_member_access(
            context=context,
            binding_id=binding_id,
            receiver_name=receiver_name,
            receiver_type=receiver_type,
            member_name=member_name,
        ):
            continue
        raise PatchGenerationError(
            f"Change {index} uses unverified Boolean member access "
            f"{receiver_name}.{member_name} on bound type {receiver_type}.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": (
                    "csharp_unverified_boolean_member_operand"
                ),
                "binding_id": binding_id,
                "method_name": context.get("method_name"),
                "operator": "!",
                "receiver_member_name": receiver_name,
                "receiver_declared_type": receiver_type,
                "accessed_member_name": member_name,
            },
        )


def _has_verified_boolean_member_access(
    *,
    context: Mapping[str, Any],
    binding_id: str,
    receiver_name: str,
    receiver_type: str,
    member_name: str,
) -> bool:
    declared_members = context.get("declared_members")
    if not isinstance(declared_members, Mapping):
        return False
    declared_identity = declared_members.get(receiver_name)
    if not isinstance(declared_identity, Mapping):
        return False
    for fact in context.get("verified_member_accesses") or []:
        if not isinstance(fact, Mapping):
            continue
        source_receiver = fact.get("source_receiver")
        consumer = fact.get("consumer")
        type_reference = fact.get("type_reference")
        member_access = fact.get("member_access")
        if (
            not isinstance(source_receiver, Mapping)
            or not isinstance(consumer, Mapping)
            or not isinstance(type_reference, Mapping)
            or not isinstance(member_access, Mapping)
        ):
            continue
        compiled_receiver_type = consumer.get("receiver_closed_type")
        if (
            fact.get("edit_binding_id") == binding_id
            and source_receiver.get("evidence_id") == context.get("evidence_id")
            and source_receiver.get("source_path")
            == context.get("canonical_target_path")
            and source_receiver.get("source_hash") == context.get("source_hash")
            and source_receiver.get("declaring_type")
            == context.get("declaring_type_full_name")
            and source_receiver.get("member_name") == receiver_name
            and source_receiver.get("member_kind")
            == declared_identity.get("member_kind")
            and source_receiver.get("declaration_span")
            == declared_identity.get("declaration_span")
            and source_receiver.get("declared_type") == receiver_type
            and source_receiver.get("receiver_closed_type")
            == compiled_receiver_type
            and consumer.get("declaring_type")
            == context.get("declaring_type_full_name")
            and consumer.get("receiver_member") == receiver_name
            and consumer.get("member_kind")
            == declared_identity.get("member_kind")
            and consumer.get("is_static")
            is declared_identity.get("is_static")
            and member_access.get("member_name") == member_name
            and member_access.get("declaring_type")
            == type_reference.get("symbol")
            and member_access.get("member_kind") == "property"
            and member_access.get("access_kind") == "instance_property_get"
            and member_access.get("closed_result_type") == "System.Boolean"
        ):
            return True
    return False


def _validate_bound_member_shadowing(
    *,
    after: str,
    declared_member_types: Mapping[str, Any],
    binding_id: str,
    context: Mapping[str, Any],
    index: int,
) -> None:
    after = re.sub(r"(?m)^[ \t]*#[^\r\n]*$", "", after)
    # 为什么要有它：预处理指令不是C#类型声明；否则`#endif`后的字段实参会被误判为`endif Field`局部变量影子。
    csharp_type_token = (
        r"[A-Za-z_]"
        r"(?:[A-Za-z0-9_.<>,?\[\]]*[A-Za-z0-9_>?\]])?"
    )
    # A member use after a statement/operator keyword is not a declaration.
    # Contextual words (var, dynamic, record, yield, await...) remain eligible
    # identifiers/types, and escaped reserved identifiers remain eligible too.
    non_type_keywords = (
        "as", "base", "break", "case", "catch", "checked", "continue", "default", "do", "else",
        "false", "finally", "fixed", "for", "foreach", "goto", "if", "in", "is", "lock", "new",
        "null", "out", "ref", "return", "sizeof", "switch", "this", "throw", "true", "try",
        "typeof", "unchecked", "using", "while",
    )
    declaration_type = rf"(?:@{csharp_type_token}|(?!(?:{'|'.join(non_type_keywords)})\b)(?:var|{csharp_type_token}))"
    for member_name, declared_type in declared_member_types.items():
        if not isinstance(member_name, str) or not isinstance(declared_type, str):
            continue
        normalized_type = re.sub(r"\s+", "", declared_type)
        if normalized_type in {"bool", "bool?", "System.Boolean", "System.Boolean?"}:
            continue
        escaped = re.escape(member_name)
        shadow_patterns = (
            rf"(?<![A-Za-z0-9_@]){declaration_type}\s+{escaped}\b",
            rf"\bout\s+var\s+{escaped}\b",
            rf"\bis\s+{csharp_type_token}\s+{escaped}\b",
            rf"\bforeach\s*\([^)]*\b{escaped}\s+in\b",
        )
        if not any(re.search(pattern, after) for pattern in shadow_patterns):
            continue
        raise PatchGenerationError(
            f"Change {index} shadows bound member {member_name}.",
            diagnostics={
                "failure_boundary": "deterministic_patch_validation",
                "failure_classification": "csharp_bound_member_shadowed",
                "binding_id": binding_id,
                "method_name": context.get("method_name"),
                "member_name": member_name,
                "declared_type": declared_type,
            },
        )


def _strip_csharp_non_code(value: str) -> str:
    """Replace C# comments and string/character contents while preserving layout."""

    selected: list[str] = []
    index = 0
    state = "code"
    while index < len(value):
        current = value[index]
        following = value[index + 1] if index + 1 < len(value) else ""
        if state == "code":
            if current == "/" and following == "/":
                selected.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                selected.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if current == '"':
                selected.append(" ")
                index += 1
                state = "verbatim_string" if index >= 2 and value[index - 2] == "@" else "string"
                continue
            if current == "'":
                selected.append(" ")
                index += 1
                state = "character"
                continue
            selected.append(current)
            index += 1
            continue
        if state == "line_comment":
            selected.append(current if current in "\r\n" else " ")
            index += 1
            if current == "\n":
                state = "code"
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                selected.extend("  ")
                index += 2
                state = "code"
            else:
                selected.append(current if current in "\r\n" else " ")
                index += 1
            continue
        if state == "verbatim_string":
            if current == '"' and following == '"':
                selected.extend("  ")
                index += 2
            elif current == '"':
                selected.append(" ")
                index += 1
                state = "code"
            else:
                selected.append(current if current in "\r\n" else " ")
                index += 1
            continue
        if current == "\\":
            selected.append(" ")
            if following:
                selected.append(" ")
                index += 2
            else:
                index += 1
            continue
        if (state == "string" and current == '"') or (
            state == "character" and current == "'"
        ):
            selected.append(" ")
            index += 1
            state = "code"
            continue
        selected.append(current if current in "\r\n" else " ")
        index += 1
    return "".join(selected)
