"""Typed contracts and deterministic result projection for product execution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from src.mod_agent.action_identity import canonical_action_json, hash_canonical_value
from src.mod_agent.artifacts import ArtifactRef, ArtifactWriteResult


PRODUCT_EXECUTION_CONTRACT_VERSION = "1"
PRODUCT_EXECUTION_REQUEST_CONTRACT_VERSION = "1"
PRODUCT_EXECUTION_BINDING_CONTRACT_VERSION = "1"

ProductCompletionScope = Literal["compile_only", "runtime_workflow"]
ProductExecutionPhase = Literal[
    "planning",
    "evidence",
    "proposal",
    "application",
    "build",
    "repair",
    "runtime",
    "evaluation",
    "terminal",
]
ProductExecutionDisposition = Literal[
    "succeeded", "domain_blocked", "platform_blocked", "partial", "awaiting_delivery"
]

_BINDING_VERSIONS = {
    "source_revision_binding": "source_revision_binding_v1",
    "source_disclosure": "source_disclosure_v1",
    "model_admission": "model_admission_v1",
    "runtime_admission": "runtime_admission_v1",
}
_BINDING_ORDER = tuple(_BINDING_VERSIONS)
_BINDING_VERSIONS["repository_build_and_test"] = "repository_build_and_test_v1"
_PHASES = {
    "planning",
    "evidence",
    "proposal",
    "application",
    "build",
    "repair",
    "runtime",
    "evaluation",
    "terminal",
}
_VALIDATION_KINDS = (
    "output_readback",
    "canonical_source",
    "compiled_artifact",
    "source_change",
    "completion_classification",
)
_VALIDATION_STATUSES = {"not_applicable", "passed", "failed"}


class ProductExecutionContractError(ValueError):
    """Raised when public product execution facts violate the frozen contract."""

    def __init__(self, reason: str, *, side_effects_may_have_started: bool) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ProductExecutionContractError reason must be non-empty.")
        if type(side_effects_may_have_started) is not bool:
            raise ValueError(
                "ProductExecutionContractError side_effects_may_have_started must be bool."
            )
        super().__init__(reason)
        self.reason = reason
        self.side_effects_may_have_started = side_effects_may_have_started


def _require_exact_fields(value: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    if not isinstance(value, Mapping) or tuple(value) != fields:
        raise ProductExecutionContractError(
            f"{name} fields changed.", side_effects_may_have_started=False
        )


def _require_non_empty(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProductExecutionContractError(
            f"{name} must be a non-empty string.",
            side_effects_may_have_started=False,
        )
    return value


def _require_exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProductExecutionContractError(
            f"{name} must be an exact integer >= {minimum}.",
            side_effects_may_have_started=False,
        )
    return value


def _require_sha256(value: Any, name: str) -> str:
    selected = _require_non_empty(value, name)
    body = selected[7:] if selected.startswith("sha256:") else selected
    if len(body) != 64 or any(ch not in "0123456789abcdef" for ch in body):
        raise ProductExecutionContractError(
            f"{name} must be a lowercase SHA256 value.",
            side_effects_may_have_started=False,
        )
    return selected


def _validate_json(value: Any, name: str) -> Any:
    def visit(item: Any) -> None:
        if item is None or isinstance(item, (str, bool)):
            return
        if type(item) is int:
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child)
            return
        raise ProductExecutionContractError(
            f"{name} contains a non-canonical JSON value.",
            side_effects_may_have_started=False,
        )

    visit(value)
    canonical_action_json(value)
    return json.loads(canonical_action_json(value).decode("utf-8"))


def _ref_to_dict(ref: ArtifactRef | None) -> dict[str, str] | None:
    if ref is None:
        return None
    return {
        "feature_id": ref.feature_id,
        "artifact_type": ref.artifact_type,
        "path": ref.path,
        "artifact_hash": ref.artifact_hash,
    }


def _ref_from_dict(value: Any, name: str) -> ArtifactRef | None:
    if value is None:
        return None
    _require_exact_fields(
        value,
        ("feature_id", "artifact_type", "path", "artifact_hash"),
        name,
    )
    return ArtifactRef(
        feature_id=_require_non_empty(value["feature_id"], f"{name}.feature_id"),
        artifact_type=_require_non_empty(
            value["artifact_type"], f"{name}.artifact_type"
        ),
        path=_require_non_empty(value["path"], f"{name}.path"),
        artifact_hash=_require_non_empty(
            value["artifact_hash"], f"{name}.artifact_hash"
        ),
    )


@dataclass(frozen=True)
class ProductExecutionInputBindingV1:
    contract_version: Literal["1"]
    binding_kind: str
    binding_contract_version: str
    canonical_payload: Any
    canonical_payload_sha256: str

    def __post_init__(self) -> None:
        if self.contract_version != "1":
            raise ProductExecutionContractError(
                "Product execution input binding version is unsupported.",
                side_effects_may_have_started=False,
            )
        expected_version = _BINDING_VERSIONS.get(self.binding_kind)
        if expected_version is None or self.binding_contract_version != expected_version:
            raise ProductExecutionContractError(
                "Product execution input binding kind/version is unsupported.",
                side_effects_may_have_started=False,
            )
        canonical = _validate_json(self.canonical_payload, "canonical_payload")
        if self.binding_kind != "source_revision_binding" and type(canonical) is not dict:
            raise ProductExecutionContractError(
                f"{self.binding_kind} payload must be an object.",
                side_effects_may_have_started=False,
            )
        if self.binding_kind == "source_revision_binding" and canonical is not None and type(canonical) is not dict:
            raise ProductExecutionContractError(
                "source_revision_binding payload must be an object or null.",
                side_effects_may_have_started=False,
            )
        if self.canonical_payload_sha256 != hash_canonical_value(canonical):
            raise ProductExecutionContractError(
                "Product execution input binding payload hash changed.",
                side_effects_may_have_started=False,
            )

    @classmethod
    def create(cls, binding_kind: str, payload: Any) -> "ProductExecutionInputBindingV1":
        canonical = _validate_json(payload, "canonical_payload")
        version = _BINDING_VERSIONS.get(binding_kind)
        if version is None:
            raise ProductExecutionContractError(
                "Product execution input binding kind is unsupported.",
                side_effects_may_have_started=False,
            )
        return cls(
            contract_version="1",
            binding_kind=binding_kind,
            binding_contract_version=version,
            canonical_payload=canonical,
            canonical_payload_sha256=hash_canonical_value(canonical),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "binding_kind": self.binding_kind,
            "binding_contract_version": self.binding_contract_version,
            "canonical_payload": self.canonical_payload,
            "canonical_payload_sha256": self.canonical_payload_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionInputBindingV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "binding_kind",
                "binding_contract_version",
                "canonical_payload",
                "canonical_payload_sha256",
            ),
            "ProductExecutionInputBindingV1",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductExecutionRequestV1:
    contract_version: Literal["1"]
    run_id: str
    request_id: str
    user_request: str
    target_domain: str
    target_mod: str
    requested_completion_scope: ProductCompletionScope
    repair_budget: Literal[2]
    isolated_workspace_root: str
    isolated_target_root: str
    source_root: str
    artifact_store_root: str
    source_snapshot_sha256: str
    isolated_source_snapshot_sha256: str
    input_bindings: tuple[ProductExecutionInputBindingV1, ...]
    completion_classification: tuple[tuple[str, bool], ...]
    confirmed_isolated_write: Literal[True]
    allow_build: Literal[True]

    def __post_init__(self) -> None:
        if self.contract_version != "1":
            raise ProductExecutionContractError(
                "Product execution request version is unsupported.",
                side_effects_may_have_started=False,
            )
        for field in ("run_id", "request_id", "user_request", "target_domain", "target_mod"):
            _require_non_empty(getattr(self, field), field)
        if self.requested_completion_scope not in {"compile_only", "runtime_workflow"}:
            raise ProductExecutionContractError(
                "Product completion scope is unsupported.",
                side_effects_may_have_started=False,
            )
        if type(self.repair_budget) is not int or self.repair_budget != 2:
            raise ProductExecutionContractError(
                "Product repair_budget must be exact int 2.",
                side_effects_may_have_started=False,
            )
        for field in (
            "isolated_workspace_root",
            "isolated_target_root",
            "source_root",
            "artifact_store_root",
        ):
            selected = _require_non_empty(getattr(self, field), field)
            path = Path(selected)
            if not path.is_absolute() or str(path.resolve()) != selected:
                raise ProductExecutionContractError(
                    f"{field} must be a normalized absolute path.",
                    side_effects_may_have_started=False,
                )
        _require_sha256(self.source_snapshot_sha256, "source_snapshot_sha256")
        _require_sha256(
            self.isolated_source_snapshot_sha256,
            "isolated_source_snapshot_sha256",
        )
        allowed_orders = {_BINDING_ORDER}
        if self.target_domain == "csharp_repository" and self.requested_completion_scope == "compile_only":
            allowed_orders.add((*_BINDING_ORDER, "repository_build_and_test"))
        if type(self.input_bindings) is not tuple or tuple(
            item.binding_kind for item in self.input_bindings
        ) not in allowed_orders:
            raise ProductExecutionContractError(
                "Product input bindings must contain the exact ordered binding kinds for this domain.",
                side_effects_may_have_started=False,
            )
        if any(not isinstance(item, ProductExecutionInputBindingV1) for item in self.input_bindings):
            raise ProductExecutionContractError(
                "Product input bindings must use ProductExecutionInputBindingV1.",
                side_effects_may_have_started=False,
            )
        if type(self.completion_classification) is not tuple:
            raise ProductExecutionContractError(
                "completion_classification must be a tuple.",
                side_effects_may_have_started=False,
            )
        keys: list[str] = []
        for item in self.completion_classification:
            if type(item) is not tuple or len(item) != 2:
                raise ProductExecutionContractError(
                    "completion_classification entries must be exact pairs.",
                    side_effects_may_have_started=False,
                )
            key, flag = item
            keys.append(_require_non_empty(key, "completion_classification key"))
            if type(flag) is not bool:
                raise ProductExecutionContractError(
                    "completion_classification values must be bool.",
                    side_effects_may_have_started=False,
                )
        if keys != sorted(keys, key=lambda item: item.encode("utf-8")) or len(keys) != len(set(keys)):
            raise ProductExecutionContractError(
                "completion_classification must use unique UTF-8 sorted keys.",
                side_effects_may_have_started=False,
            )
        if self.confirmed_isolated_write is not True or self.allow_build is not True:
            raise ProductExecutionContractError(
                "Product execution requires isolated-write and Build authorization.",
                side_effects_may_have_started=False,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "user_request": self.user_request,
            "target_domain": self.target_domain,
            "target_mod": self.target_mod,
            "requested_completion_scope": self.requested_completion_scope,
            "repair_budget": self.repair_budget,
            "isolated_workspace_root": self.isolated_workspace_root,
            "isolated_target_root": self.isolated_target_root,
            "source_root": self.source_root,
            "artifact_store_root": self.artifact_store_root,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "isolated_source_snapshot_sha256": self.isolated_source_snapshot_sha256,
            "input_bindings": [item.to_dict() for item in self.input_bindings],
            "completion_classification": [list(item) for item in self.completion_classification],
            "confirmed_isolated_write": self.confirmed_isolated_write,
            "allow_build": self.allow_build,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionRequestV1":
        fields = (
            "contract_version",
            "run_id",
            "request_id",
            "user_request",
            "target_domain",
            "target_mod",
            "requested_completion_scope",
            "repair_budget",
            "isolated_workspace_root",
            "isolated_target_root",
            "source_root",
            "artifact_store_root",
            "source_snapshot_sha256",
            "isolated_source_snapshot_sha256",
            "input_bindings",
            "completion_classification",
            "confirmed_isolated_write",
            "allow_build",
        )
        _require_exact_fields(value, fields, "ProductExecutionRequestV1")
        if type(value["input_bindings"]) is not list or type(value["completion_classification"]) is not list:
            raise ProductExecutionContractError(
                "Product request tuple fields must serialize as arrays.",
                side_effects_may_have_started=False,
            )
        selected = dict(value)
        selected["input_bindings"] = tuple(
            ProductExecutionInputBindingV1.from_dict(item)
            for item in value["input_bindings"]
        )
        selected["completion_classification"] = tuple(
            tuple(item) if type(item) is list and len(item) == 2 else ()
            for item in value["completion_classification"]
        )
        return cls(**selected)


@dataclass(frozen=True)
class ProductExecutionGraphModeV1:
    contract_version: Literal["1"]
    request: ProductExecutionRequestV1
    interrupted_patch_selection: Mapping[str, Any] | None = None
    pretransport_patch_selection: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.contract_version != "1" or not isinstance(
            self.request, ProductExecutionRequestV1
        ):
            raise ProductExecutionContractError(
                "Product execution Graph mode is invalid.",
                side_effects_may_have_started=False,
            )
        if self.interrupted_patch_selection is not None:
            _validate_interrupted_selection(self.interrupted_patch_selection)
        if self.pretransport_patch_selection is not None:
            from src.mod_agent.pretransport_patch_admission import validate_selection
            if self.interrupted_patch_selection is not None:
                raise ProductExecutionContractError("Conflicting Patch recovery selections.", side_effects_may_have_started=False)
            validate_selection(self.pretransport_patch_selection)



def _validate_interrupted_selection(selection: Mapping[str, Any]) -> None:
    """Accept only historical exact references, never caller-supplied state."""
    # 为什么要有它：续接入口只选择原事实，不能借可选参数注入审批、重试次数或伪造工作流状态。
    if not isinstance(selection, Mapping) or set(selection) != {"source_attempt_ref", "parent_manifest_ref", "execution_stop_ref"}:
        raise ProductExecutionContractError("Interrupted Patch selection must contain three exact references.",
                                            side_effects_may_have_started=False)
    for ref in selection.values():
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ProductExecutionContractError("Interrupted Patch selection reference is invalid.",
                                                side_effects_may_have_started=False)
        _require_non_empty(ref["path"], "interrupted selection path")
        _require_sha256(ref["sha256"], "interrupted selection sha256")


@dataclass(frozen=True)
class ProductExecutionBlockerV1:
    contract_version: Literal["1"]
    owner: Literal["domain", "platform"]
    stage: str
    code: str
    message: str
    source_node: str

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.owner not in {"domain", "platform"}:
            raise ProductExecutionContractError(
                "Product blocker contract is invalid.",
                side_effects_may_have_started=False,
            )
        for field in ("stage", "code", "message", "source_node"):
            _require_non_empty(getattr(self, field), f"blocker.{field}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "owner": self.owner,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionBlockerV1":
        _require_exact_fields(
            value,
            ("contract_version", "owner", "stage", "code", "message", "source_node"),
            "ProductExecutionBlockerV1",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductExecutionStoreAttemptV1:
    contract_version: Literal["1"]
    sequence: int
    phase: ProductExecutionPhase
    repair_attempt: int
    node_name: str
    artifact_role: str
    store_result: ArtifactWriteResult

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.phase not in _PHASES:
            raise ProductExecutionContractError(
                "Product Store attempt contract is invalid.",
                side_effects_may_have_started=True,
            )
        _require_exact_int(self.sequence, "store attempt sequence")
        attempt = _require_exact_int(self.repair_attempt, "repair attempt")
        if attempt > 2:
            raise ProductExecutionContractError(
                "repair attempt exceeds 2.", side_effects_may_have_started=True
            )
        _require_non_empty(self.node_name, "store attempt node_name")
        _require_non_empty(self.artifact_role, "store attempt artifact_role")
        if not isinstance(self.store_result, ArtifactWriteResult):
            raise ProductExecutionContractError(
                "store_result must be Store-owned ArtifactWriteResult.",
                side_effects_may_have_started=True,
            )
        if self.store_result.artifact_type != self.artifact_role:
            raise ProductExecutionContractError(
                "Store result artifact type disagrees with its product role.",
                side_effects_may_have_started=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "sequence": self.sequence,
            "phase": self.phase,
            "repair_attempt": self.repair_attempt,
            "node_name": self.node_name,
            "artifact_role": self.artifact_role,
            "store_result": self.store_result.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionStoreAttemptV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "sequence",
                "phase",
                "repair_attempt",
                "node_name",
                "artifact_role",
                "store_result",
            ),
            "ProductExecutionStoreAttemptV1",
        )
        selected = dict(value)
        if not isinstance(selected["store_result"], Mapping):
            raise ProductExecutionContractError(
                "store_result must be an object.",
                side_effects_may_have_started=True,
            )
        selected["store_result"] = ArtifactWriteResult.from_dict(
            selected["store_result"]
        )
        return cls(**selected)


@dataclass(frozen=True)
class ProductExecutionOutputV1:
    contract_version: Literal["1"]
    sequence: int
    phase: ProductExecutionPhase
    repair_attempt: int
    node_name: str
    artifact_role: str
    artifact_ref: ArtifactRef
    origin: Literal["generated", "reused_input"] = "generated"
    reuse_selection: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.phase not in _PHASES:
            raise ProductExecutionContractError(
                "Product output contract is invalid.",
                side_effects_may_have_started=True,
            )
        _require_exact_int(self.sequence, "output sequence")
        if _require_exact_int(self.repair_attempt, "output repair attempt") > 2:
            raise ProductExecutionContractError(
                "output repair attempt exceeds 2.",
                side_effects_may_have_started=True,
            )
        _require_non_empty(self.node_name, "output node_name")
        _require_non_empty(self.artifact_role, "output artifact_role")
        if not isinstance(self.artifact_ref, ArtifactRef) or self.artifact_ref.artifact_type != self.artifact_role:
            raise ProductExecutionContractError(
                "Product output ArtifactRef is invalid.",
                side_effects_may_have_started=True,
            )
        if self.origin == "reused_input":
            if self.node_name == "pretransport_patch_admission":
                from src.mod_agent.pretransport_patch_admission import validate_selection
                validate_selection(self.reuse_selection)
            else:
                _validate_interrupted_selection(self.reuse_selection)
            if (self.node_name not in {"interrupted_patch_admission", "pretransport_patch_admission"} or self.repair_attempt != 0
                    or self.artifact_role not in {"feature_spec", "evidence_pack", "implementation_plan"}):
                raise ProductExecutionContractError("Reused product input identity is invalid.",
                                                    side_effects_may_have_started=False)
        elif self.origin != "generated" or self.reuse_selection is not None:
            raise ProductExecutionContractError("Product output origin is invalid.",
                                                side_effects_may_have_started=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "sequence": self.sequence,
            "phase": self.phase,
            "repair_attempt": self.repair_attempt,
            "node_name": self.node_name,
            "artifact_role": self.artifact_role,
            "artifact_ref": _ref_to_dict(self.artifact_ref),
            **({"origin": self.origin, "reuse_selection": self.reuse_selection}
               if self.origin == "reused_input" else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionOutputV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "sequence",
                "phase",
                "repair_attempt",
                "node_name",
                "artifact_role",
                "artifact_ref",
            ) + (("origin", "reuse_selection") if "origin" in value else ()),
            "ProductExecutionOutputV1",
        )
        selected = dict(value)
        selected["artifact_ref"] = _ref_from_dict(
            selected["artifact_ref"], "output.artifact_ref"
        )
        return cls(**selected)


@dataclass(frozen=True)
class ProductExecutionEffectEventV1:
    contract_version: Literal["1"]
    sequence: int
    phase: ProductExecutionPhase
    repair_attempt: int
    effect_role: Literal[
        "evidence_retrieval",
        "patch_generation",
        "dry_run",
        "isolated_apply",
        "build",
        "runtime",
    ]
    event_status: Literal["started", "completed", "failed"]
    failure_class: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.phase not in _PHASES:
            raise ProductExecutionContractError(
                "Product effect event contract is invalid.",
                side_effects_may_have_started=True,
            )
        _require_exact_int(self.sequence, "effect sequence")
        if _require_exact_int(self.repair_attempt, "effect repair attempt") > 2:
            raise ProductExecutionContractError(
                "effect repair attempt exceeds 2.",
                side_effects_may_have_started=True,
            )
        if self.effect_role not in {
            "evidence_retrieval",
            "patch_generation",
            "dry_run",
            "isolated_apply",
            "build",
            "runtime",
        } or self.event_status not in {"started", "completed", "failed"}:
            raise ProductExecutionContractError(
                "Product effect role/status is unsupported.",
                side_effects_may_have_started=True,
            )
        if self.event_status == "failed":
            _require_non_empty(self.failure_class, "effect failure_class")
            _require_non_empty(self.failure_message, "effect failure_message")
        elif self.failure_class is not None or self.failure_message is not None:
            raise ProductExecutionContractError(
                "Successful effect events cannot carry failure facts.",
                side_effects_may_have_started=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "sequence": self.sequence,
            "phase": self.phase,
            "repair_attempt": self.repair_attempt,
            "effect_role": self.effect_role,
            "event_status": self.event_status,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionEffectEventV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "sequence",
                "phase",
                "repair_attempt",
                "effect_role",
                "event_status",
                "failure_class",
                "failure_message",
            ),
            "ProductExecutionEffectEventV1",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductExecutionValidationDetailV1:
    contract_version: Literal["1"]
    detail_kind: str
    payload: Any

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.detail_kind not in _VALIDATION_KINDS:
            raise ProductExecutionContractError(
                "Product validation detail is invalid.",
                side_effects_may_have_started=True,
            )
        _validate_json(self.payload, "validation detail payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "detail_kind": self.detail_kind,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionValidationDetailV1":
        _require_exact_fields(
            value,
            ("contract_version", "detail_kind", "payload"),
            "ProductExecutionValidationDetailV1",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductExecutionValidationFactsV1:
    contract_version: Literal["1"]
    output_readback_status: str
    canonical_source_status: str
    compiled_artifact_status: str
    source_change_status: str
    completion_classification_status: str
    failure_class: str | None
    failure_message: str | None
    details: tuple[ProductExecutionValidationDetailV1, ...]

    def __post_init__(self) -> None:
        statuses = (
            self.output_readback_status,
            self.canonical_source_status,
            self.compiled_artifact_status,
            self.source_change_status,
            self.completion_classification_status,
        )
        if self.contract_version != "1" or any(item not in _VALIDATION_STATUSES for item in statuses):
            raise ProductExecutionContractError(
                "Product validation status is invalid.",
                side_effects_may_have_started=True,
            )
        if type(self.details) is not tuple or tuple(item.detail_kind for item in self.details) != _VALIDATION_KINDS:
            raise ProductExecutionContractError(
                "Product validation details changed order or shape.",
                side_effects_may_have_started=True,
            )
        failed = "failed" in statuses
        if failed:
            _require_non_empty(self.failure_class, "validation failure_class")
            _require_non_empty(self.failure_message, "validation failure_message")
        elif self.failure_class is not None or self.failure_message is not None:
            raise ProductExecutionContractError(
                "Passing validation facts cannot carry failure fields.",
                side_effects_may_have_started=True,
            )
        for status, detail in zip(statuses, self.details, strict=True):
            if status == "not_applicable" and detail.payload is not None:
                raise ProductExecutionContractError(
                    "Not-applicable validation detail must be null.",
                    side_effects_may_have_started=True,
                )
            if status == "passed" and detail.payload is None:
                raise ProductExecutionContractError(
                    "Passing validation detail must preserve its payload.",
                    side_effects_may_have_started=True,
                )

    @property
    def failed(self) -> bool:
        return "failed" in (
            self.output_readback_status,
            self.canonical_source_status,
            self.compiled_artifact_status,
            self.source_change_status,
            self.completion_classification_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "output_readback_status": self.output_readback_status,
            "canonical_source_status": self.canonical_source_status,
            "compiled_artifact_status": self.compiled_artifact_status,
            "source_change_status": self.source_change_status,
            "completion_classification_status": self.completion_classification_status,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
            "details": [item.to_dict() for item in self.details],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionValidationFactsV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "output_readback_status",
                "canonical_source_status",
                "compiled_artifact_status",
                "source_change_status",
                "completion_classification_status",
                "failure_class",
                "failure_message",
                "details",
            ),
            "ProductExecutionValidationFactsV1",
        )
        if type(value["details"]) is not list:
            raise ProductExecutionContractError(
                "Product validation details must serialize as an array.",
                side_effects_may_have_started=True,
            )
        selected = dict(value)
        selected["details"] = tuple(
            ProductExecutionValidationDetailV1.from_dict(item)
            for item in value["details"]
        )
        return cls(**selected)


def empty_product_validation_facts() -> ProductExecutionValidationFactsV1:
    return ProductExecutionValidationFactsV1(
        contract_version="1",
        output_readback_status="not_applicable",
        canonical_source_status="not_applicable",
        compiled_artifact_status="not_applicable",
        source_change_status="not_applicable",
        completion_classification_status="not_applicable",
        failure_class=None,
        failure_message=None,
        details=tuple(
            ProductExecutionValidationDetailV1(
                contract_version="1", detail_kind=kind, payload=None
            )
            for kind in _VALIDATION_KINDS
        ),
    )


@dataclass(frozen=True)
class ProductBuildCompletionV1:
    contract_version: Literal["1"]
    build_result_ref: ArtifactRef | None
    workflow_checkpoint_ref: ArtifactRef | None
    build_status: Literal["not_reached", "unavailable", "success", "failed", "blocked"]
    claims_real_build_success: bool | None

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.build_status not in {
            "not_reached", "unavailable", "success", "failed", "blocked"
        }:
            raise ProductExecutionContractError(
                "Product Build completion is invalid.",
                side_effects_may_have_started=True,
            )
        if self.build_status == "not_reached" and (
            self.build_result_ref is not None
            or self.workflow_checkpoint_ref is not None
            or self.claims_real_build_success is not None
        ):
            raise ProductExecutionContractError(
                "not_reached Build completion carries Build facts.",
                side_effects_may_have_started=True,
            )
        if self.build_status == "unavailable" and (
            self.build_result_ref is None or self.claims_real_build_success is not None
        ):
            raise ProductExecutionContractError(
                "unavailable Build completion is inconsistent.",
                side_effects_may_have_started=True,
            )
        if self.build_status in {"success", "failed", "blocked"}:
            if self.build_result_ref is None or type(self.claims_real_build_success) is not bool:
                raise ProductExecutionContractError(
                    "Reached Build completion lacks exact Build facts.",
                    side_effects_may_have_started=True,
                )
            if (self.build_status == "success") is not self.claims_real_build_success:
                raise ProductExecutionContractError(
                    "Build status and real-success claim disagree.",
                    side_effects_may_have_started=True,
                )
            if self.build_status != "success" and self.workflow_checkpoint_ref is not None:
                raise ProductExecutionContractError(
                    "Non-successful Build completion cannot carry a handoff checkpoint.",
                    side_effects_may_have_started=True,
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "build_result_ref": _ref_to_dict(self.build_result_ref),
            "workflow_checkpoint_ref": _ref_to_dict(self.workflow_checkpoint_ref),
            "build_status": self.build_status,
            "claims_real_build_success": self.claims_real_build_success,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductBuildCompletionV1":
        _require_exact_fields(
            value,
            (
                "contract_version",
                "build_result_ref",
                "workflow_checkpoint_ref",
                "build_status",
                "claims_real_build_success",
            ),
            "ProductBuildCompletionV1",
        )
        selected = dict(value)
        selected["build_result_ref"] = _ref_from_dict(
            selected["build_result_ref"], "build_result_ref"
        )
        selected["workflow_checkpoint_ref"] = _ref_from_dict(
            selected["workflow_checkpoint_ref"], "workflow_checkpoint_ref"
        )
        return cls(**selected)


@dataclass(frozen=True)
class ProductRuntimeCompletionV1:
    contract_version: Literal["1"]
    build_result_ref: ArtifactRef | None
    runtime_evidence_ref: ArtifactRef | None
    eval_report_ref: ArtifactRef | None
    workflow_summary_ref: ArtifactRef | None
    blocked_report_ref: ArtifactRef | None
    build_status: Literal["not_reached", "unavailable", "success", "failed", "blocked"]
    runtime_status: Literal["not_reached", "unavailable", "accepted", "not_accepted", "blocked"]
    eval_verdict: Literal["not_reached", "unavailable", "accepted", "blocked", "inconclusive"]
    final_status: Literal["not_reached", "unavailable", "accepted", "blocked", "inconclusive"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "build_result_ref": _ref_to_dict(self.build_result_ref),
            "runtime_evidence_ref": _ref_to_dict(self.runtime_evidence_ref),
            "eval_report_ref": _ref_to_dict(self.eval_report_ref),
            "workflow_summary_ref": _ref_to_dict(self.workflow_summary_ref),
            "blocked_report_ref": _ref_to_dict(self.blocked_report_ref),
            "build_status": self.build_status,
            "runtime_status": self.runtime_status,
            "eval_verdict": self.eval_verdict,
            "final_status": self.final_status,
        }

    def __post_init__(self) -> None:
        if self.contract_version != "1":
            raise ProductExecutionContractError(
                "Product Runtime completion version is unsupported.",
                side_effects_may_have_started=True,
            )
        if self.build_status not in {"not_reached", "unavailable", "success", "failed", "blocked"}:
            raise ProductExecutionContractError("Runtime completion Build status is invalid.", side_effects_may_have_started=True)
        if self.runtime_status not in {"not_reached", "unavailable", "accepted", "not_accepted", "blocked"}:
            raise ProductExecutionContractError("Runtime status is invalid.", side_effects_may_have_started=True)
        if self.eval_verdict not in {"not_reached", "unavailable", "accepted", "blocked", "inconclusive"}:
            raise ProductExecutionContractError("Eval verdict is invalid.", side_effects_may_have_started=True)
        if self.final_status not in {"not_reached", "unavailable", "accepted", "blocked", "inconclusive"}:
            raise ProductExecutionContractError("Final Runtime status is invalid.", side_effects_may_have_started=True)
        if (self.build_result_ref is None) is not (self.build_status == "not_reached"):
            if not (self.build_result_ref is not None and self.build_status == "unavailable"):
                raise ProductExecutionContractError(
                    "Runtime completion Build ref/status disagree.",
                    side_effects_may_have_started=True,
                )
        if (self.runtime_evidence_ref is None) is not (self.runtime_status == "not_reached"):
            if not (self.runtime_evidence_ref is not None and self.runtime_status == "unavailable"):
                raise ProductExecutionContractError(
                    "RuntimeEvidence ref/status disagree.",
                    side_effects_may_have_started=True,
                )
        if (self.eval_report_ref is None) is not (self.eval_verdict == "not_reached"):
            if not (self.eval_report_ref is not None and self.eval_verdict == "unavailable"):
                raise ProductExecutionContractError(
                    "EvalReport ref/verdict disagree.",
                    side_effects_may_have_started=True,
                )
        if self.runtime_evidence_ref is not None and self.build_status != "success":
            raise ProductExecutionContractError(
                "RuntimeEvidence requires a successful Build.",
                side_effects_may_have_started=True,
            )
        if self.eval_report_ref is not None and self.runtime_status != "accepted":
            raise ProductExecutionContractError(
                "EvalReport requires accepted RuntimeEvidence.",
                side_effects_may_have_started=True,
            )
        if self.final_status == "not_reached":
            if self.workflow_summary_ref is not None or self.blocked_report_ref is not None:
                raise ProductExecutionContractError(
                    "not_reached terminal status carries a terminal ref.",
                    side_effects_may_have_started=True,
                )
        elif self.final_status == "unavailable":
            if self.workflow_summary_ref is None or self.blocked_report_ref is not None:
                raise ProductExecutionContractError(
                    "unavailable terminal status lacks its WorkflowSummary ref.",
                    side_effects_may_have_started=True,
                )
        elif self.final_status == "blocked":
            if self.blocked_report_ref is None or self.workflow_summary_ref is not None:
                raise ProductExecutionContractError(
                    "Blocked terminal status requires only a BlockedReport ref.",
                    side_effects_may_have_started=True,
                )
        elif self.workflow_summary_ref is None or self.blocked_report_ref is not None:
            raise ProductExecutionContractError(
                "Accepted/inconclusive terminal status requires only WorkflowSummary.",
                side_effects_may_have_started=True,
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductRuntimeCompletionV1":
        fields = (
            "contract_version",
            "build_result_ref",
            "runtime_evidence_ref",
            "eval_report_ref",
            "workflow_summary_ref",
            "blocked_report_ref",
            "build_status",
            "runtime_status",
            "eval_verdict",
            "final_status",
        )
        _require_exact_fields(value, fields, "ProductRuntimeCompletionV1")
        selected = dict(value)
        for field in fields[1:6]:
            selected[field] = _ref_from_dict(selected[field], field)
        return cls(**selected)


@dataclass(frozen=True)
class ProductExecutionLifecycleGrammarV1:
    contract_version: Literal["1"]
    initial_prefix: tuple[str, ...]
    allowed_successors: tuple[tuple[str, tuple[str, ...]], ...]
    repair_transition_sources: tuple[str, ...]
    max_repair_attempt: Literal[2]
    terminal_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != "1" or self.max_repair_attempt != 2:
            raise ProductExecutionContractError(
                "Product lifecycle grammar version or repair budget changed.",
                side_effects_may_have_started=False,
            )
        if not self.initial_prefix or not self.allowed_successors or not self.terminal_nodes:
            raise ProductExecutionContractError(
                "Product lifecycle grammar cannot be empty.",
                side_effects_may_have_started=False,
            )
        successor_nodes = [node for node, _ in self.allowed_successors]
        if len(successor_nodes) != len(set(successor_nodes)) or any(
            type(node) is not str or not node or not successors
            for node, successors in self.allowed_successors
        ):
            raise ProductExecutionContractError(
                "Product lifecycle successor grammar is invalid.",
                side_effects_may_have_started=False,
            )


@dataclass(frozen=True)
class ProductExecutionDeclarationV1:
    contract_version: Literal["1"]
    node_store_roles: tuple[tuple[str, str], ...]
    node_phases: tuple[tuple[str, str], ...]
    lifecycle_grammar: ProductExecutionLifecycleGrammarV1

    def __post_init__(self) -> None:
        if self.contract_version != "1":
            raise ProductExecutionContractError(
                "Product execution declaration version is unsupported.",
                side_effects_may_have_started=False,
            )
        nodes = {node for node, _ in self.node_phases}
        if len(nodes) != len(self.node_phases):
            raise ProductExecutionContractError(
                "Product declaration repeats a node phase.",
                side_effects_may_have_started=False,
            )
        if any(phase not in _PHASES for _, phase in self.node_phases):
            raise ProductExecutionContractError(
                "Product declaration contains an unsupported phase.",
                side_effects_may_have_started=False,
            )
        role_pairs = set(self.node_store_roles)
        if len(role_pairs) != len(self.node_store_roles) or any(
            node not in nodes or type(role) is not str or not role
            for node, role in self.node_store_roles
        ):
            raise ProductExecutionContractError(
                "Product declaration node/role registry is invalid.",
                side_effects_may_have_started=False,
            )
        grammar_nodes = set(self.lifecycle_grammar.initial_prefix)
        grammar_nodes.update(node for node, _ in self.lifecycle_grammar.allowed_successors)
        grammar_nodes.update(
            successor
            for _, successors in self.lifecycle_grammar.allowed_successors
            for successor in successors
        )
        grammar_nodes.update(self.lifecycle_grammar.terminal_nodes)
        if not grammar_nodes.issubset(nodes):
            raise ProductExecutionContractError(
                "Product lifecycle grammar references an undeclared node.",
                side_effects_may_have_started=False,
            )

    @property
    def role_by_node(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {}
        for node, role in self.node_store_roles:
            result.setdefault(node, []).append(role)
        return {node: tuple(roles) for node, roles in result.items()}

    @property
    def phase_by_node(self) -> dict[str, str]:
        return dict(self.node_phases)


PRODUCT_EXECUTION_DECLARATION = ProductExecutionDeclarationV1(
    contract_version="1",
    node_store_roles=(
        ("plan_request", "request_planning_report"),
        ("planning_brief", "mod_planning_brief"),
        ("draft_feature_spec", "feature_spec"),
        ("retrieve_evidence", "evidence_pack"),
        ("create_implementation_plan", "implementation_plan"),
        ("generate_patch_proposal", "patch_proposal"),
        ("review_proposal", "critic_assessment"),
        ("review_proposal", "proposal_review"),
        ("dry_run_apply", "dry_run_result"),
        ("isolated_apply", "isolated_apply_result"),
        ("build", "build_result"),
        ("build", "workflow_checkpoint"),
        ("create_repair_plan", "repair_plan"),
        ("create_repair_plan", "evidence_pack"),
        ("create_repair_plan", "implementation_plan"),
        ("resolve_runtime_binding", "binding_resolution"),
        ("collect_runtime_evidence", "runtime_evidence"),
        ("evaluate", "eval_report"),
        ("done", "workflow_summary"),
        ("blocked", "blocked_report"),
    ),
    node_phases=(
        ("plan_request", "planning"),
        ("planning_brief", "planning"),
        ("draft_feature_spec", "planning"),
        ("retrieve_evidence", "evidence"),
        ("create_implementation_plan", "planning"),
        ("generate_patch_proposal", "proposal"),
        ("review_proposal", "proposal"),
        ("dry_run_apply", "application"),
        ("isolated_apply", "application"),
        ("build", "build"),
        ("create_repair_plan", "repair"),
        ("resolve_runtime_binding", "repair"),
        ("collect_runtime_evidence", "runtime"),
        ("evaluate", "evaluation"),
        ("done", "terminal"),
        ("blocked", "terminal"),
    ),
    lifecycle_grammar=ProductExecutionLifecycleGrammarV1(
        contract_version="1",
        initial_prefix=(
            "plan_request",
            "planning_brief",
            "draft_feature_spec",
            "retrieve_evidence",
            "create_implementation_plan",
            "generate_patch_proposal",
            "review_proposal",
        ),
        allowed_successors=(
            ("generate_patch_proposal", ("review_proposal",)),
            ("review_proposal", ("dry_run_apply", "create_repair_plan", "blocked")),
            ("dry_run_apply", ("isolated_apply", "blocked")),
            ("isolated_apply", ("build", "blocked")),
            ("build", ("collect_runtime_evidence", "create_repair_plan", "blocked")),
            ("collect_runtime_evidence", ("evaluate", "resolve_runtime_binding", "create_repair_plan", "blocked")),
            ("resolve_runtime_binding", ("create_repair_plan", "blocked")),
            ("create_repair_plan", ("generate_patch_proposal", "blocked")),
            ("evaluate", ("done", "blocked")),
        ),
        repair_transition_sources=(
            "review_proposal", "build", "collect_runtime_evidence", "resolve_runtime_binding"
        ),
        max_repair_attempt=2,
        terminal_nodes=("build", "done", "blocked"),
    ),
)


def _validate_store_trace(
    attempts: tuple[ProductExecutionStoreAttemptV1, ...],
    outputs: tuple[ProductExecutionOutputV1, ...],
    *, upstream_sequences: frozenset[int] = frozenset(), reused_input_count: int = 0,
    task_agent_sequences: frozenset[int] = frozenset(),
) -> None:
    declaration = PRODUCT_EXECUTION_DECLARATION
    roles = declaration.role_by_node
    phases = declaration.phase_by_node
    if tuple(item.sequence for item in attempts) != tuple(range(reused_input_count, reused_input_count + len(attempts))):
        raise ProductExecutionContractError(
            "Product Store attempt sequence changed.",
            side_effects_may_have_started=True,
        )
    committed = tuple(item for item in attempts if item.store_result.outcome == "committed")
    noncommitted = tuple(item for item in attempts if item.store_result.outcome != "committed")
    if len(noncommitted) > 1 or (noncommitted and attempts[-1] is not noncommitted[0]):
        raise ProductExecutionContractError(
            "A noncommitted Store attempt must be the unique final attempt.",
            side_effects_may_have_started=True,
        )
    if len(outputs) != len(committed):
        raise ProductExecutionContractError(
            "Each committed Store attempt must have one product output.",
            side_effects_may_have_started=True,
        )
    for attempt, output in zip(committed, outputs, strict=True):
        ref = attempt.store_result.artifact_ref
        if ref is None or (
            output.sequence,
            output.phase,
            output.repair_attempt,
            output.node_name,
            output.artifact_role,
            output.artifact_ref,
        ) != (
            attempt.sequence,
            attempt.phase,
            attempt.repair_attempt,
            attempt.node_name,
            attempt.artifact_role,
            ref,
        ):
            raise ProductExecutionContractError(
                "Product output disagrees with its committed Store attempt.",
                side_effects_may_have_started=True,
            )
    for attempt in attempts:
        if (attempt.sequence not in task_agent_sequences
                and (attempt.node_name not in roles or attempt.artifact_role not in roles[attempt.node_name])):
            raise ProductExecutionContractError(
                "Product Store node/role is not declared.",
                side_effects_may_have_started=True,
            )
        if phases.get(attempt.node_name) != attempt.phase:
            raise ProductExecutionContractError(
                "Product Store node/phase is not declared.",
                side_effects_may_have_started=True,
            )
    # Store-role order within a node is exact, including critic before review and
    # BuildResult before WorkflowCheckpoint.
    for index, item in enumerate(attempts):
        if (index and attempts[index - 1].node_name == item.node_name
                and item.sequence not in upstream_sequences | task_agent_sequences
                and attempts[index - 1].sequence not in task_agent_sequences):
            declared = roles[item.node_name]
            previous = attempts[index - 1].artifact_role
            if declared.index(item.artifact_role) <= declared.index(previous):
                raise ProductExecutionContractError(
                    "Product Store roles changed order within one Graph node.",
                    side_effects_may_have_started=True,
                )


def _validate_effect_trace(events: tuple[ProductExecutionEffectEventV1, ...]) -> None:
    if tuple(item.sequence for item in events) != tuple(range(len(events))):
        raise ProductExecutionContractError(
            "Product effect sequence changed.", side_effects_may_have_started=True
        )
    pending: tuple[str, str, int] | None = None
    for event in events:
        key = (event.effect_role, event.phase, event.repair_attempt)
        if event.event_status == "started":
            if pending is not None:
                raise ProductExecutionContractError(
                    "Product effects cannot overlap.", side_effects_may_have_started=True
                )
            pending = key
        elif pending != key:
            raise ProductExecutionContractError(
                "Product effect terminal event lacks its exact start.",
                side_effects_may_have_started=True,
            )
        else:
            pending = None
    if pending is not None:
        raise ProductExecutionContractError(
            "Product result cannot freeze an unterminated effect.",
            side_effects_may_have_started=True,
        )


def _validate_lifecycle(attempts: tuple[ProductExecutionStoreAttemptV1, ...], *, interrupted_patch: bool = False,
        task_agent_v2: bool = False) -> None:
    if not attempts:
        return
    nodes: list[tuple[str, int]] = []
    for item in attempts:
        key = (item.node_name, item.repair_attempt)
        if not nodes or nodes[-1] != key:
            nodes.append(key)
    grammar = PRODUCT_EXECUTION_DECLARATION.lifecycle_grammar
    initial = grammar.initial_prefix[5:] if interrupted_patch else grammar.initial_prefix
    if task_agent_v2:
        # 为什么要有它：显式v2跳过技术Evidence/Plan，旧协议完整顺序仍受原有grammar约束。
        initial = tuple(node for node in initial if node not in {"retrieve_evidence", "create_implementation_plan"})
    observed_initial_nodes = [node for node, attempt in nodes if attempt == 0]
    observed_without_terminal = list(observed_initial_nodes)
    if observed_without_terminal and observed_without_terminal[-1] == "blocked":
        observed_without_terminal.pop()
    initial_count = min(len(observed_without_terminal), len(initial))
    if tuple(observed_without_terminal[:initial_count]) != initial[:initial_count]:
        raise ProductExecutionContractError(
            "Product lifecycle changed its initial ordered prefix.",
            side_effects_may_have_started=True,
        )
    if len(observed_without_terminal) < len(initial):
        if any(node != "blocked" for node in observed_initial_nodes[initial_count:]):
            raise ProductExecutionContractError(
                "Product lifecycle skipped an initial node.",
                side_effects_may_have_started=True,
            )
        return
    if any(attempt > 0 and node in grammar.initial_prefix[:5] for node, attempt in nodes):
        raise ProductExecutionContractError(
            "Planning/evidence nodes cannot repeat during repair.",
            side_effects_may_have_started=True,
        )
    allowed = dict(grammar.allowed_successors)
    for index, ((node, attempt), (next_node, next_attempt)) in enumerate(
        zip(nodes, nodes[1:])
    ):
        if index < len(initial) - 1:
            continue
        if next_node == "blocked":
            if index + 1 != len(nodes) - 1 or next_attempt != attempt:
                raise ProductExecutionContractError(
                    "Product blocked terminal changed attempt or has a successor.",
                    side_effects_may_have_started=True,
                )
            continue
        if next_node not in allowed.get(node, ()):
            raise ProductExecutionContractError(
                f"Product lifecycle transition {node}->{next_node} is not declared.",
                side_effects_may_have_started=True,
            )
        if next_node == "create_repair_plan":
            same_attempt_terminal = (
                next_attempt == attempt
                and (
                    node in {"collect_runtime_evidence", "resolve_runtime_binding"}
                    or (
                        node in {"build", "review_proposal"}
                        and attempt == grammar.max_repair_attempt
                    )
                )
            )
            if node not in grammar.repair_transition_sources or not (
                next_attempt == attempt + 1 or same_attempt_terminal
            ):
                raise ProductExecutionContractError(
                    "RepairPlan transition changed its source or attempt number.",
                    side_effects_may_have_started=True,
                )
        elif next_attempt != attempt:
            raise ProductExecutionContractError(
                "Product lifecycle changed attempt outside RepairPlan.",
                side_effects_may_have_started=True,
            )
    # A Build must be preceded in the same attempt by exact dry-run and isolated apply.
    for index, (node, attempt) in enumerate(nodes):
        if node == "build":
            prior = [name for name, selected_attempt in nodes[:index] if selected_attempt == attempt]
            if len(prior) < 2 or prior[-2:] != ["dry_run_apply", "isolated_apply"]:
                raise ProductExecutionContractError(
                    "Product Build skipped exact dry-run or isolated apply.",
                    side_effects_may_have_started=True,
                )
    seen_attempts = sorted({attempt for _, attempt in nodes})
    if (
        seen_attempts
        and (
            seen_attempts != list(range(max(seen_attempts) + 1))
            or max(seen_attempts) > grammar.max_repair_attempt
        )
    ):
        raise ProductExecutionContractError(
            "Product repair attempts must be contiguous and within budget.",
            side_effects_may_have_started=True,
        )
    for attempt in seen_attempts[1:]:
        previous = ("create_repair_plan", attempt)
        if previous not in nodes:
            raise ProductExecutionContractError(
                "A repair attempt lacks its committed RepairPlan transition.",
                side_effects_may_have_started=True,
            )


def _output_readback_envelopes(
    facts: "ProductExecutionFactsV1",
) -> dict[int, Mapping[str, Any]] | None:
    status = facts.validation_facts.output_readback_status
    if status == "failed":
        return None
    committed_outputs = facts.outputs
    if status == "not_applicable":
        if committed_outputs:
            raise ProductExecutionContractError(
                "Committed product outputs lack output readback facts.",
                side_effects_may_have_started=True,
            )
        return {}
    if status != "passed":
        raise ProductExecutionContractError(
            "Product output readback status cannot validate committed outputs.",
            side_effects_may_have_started=True,
        )
    detail = next(
        (
            item
            for item in facts.validation_facts.details
            if item.detail_kind == "output_readback"
        ),
        None,
    )
    payload = detail.payload if detail is not None else None
    if type(payload) is not list or len(payload) != len(committed_outputs):
        raise ProductExecutionContractError(
            "Product output readback does not match committed output order.",
            side_effects_may_have_started=True,
        )
    envelopes: dict[int, Mapping[str, Any]] = {}
    for output, envelope in zip(committed_outputs, payload, strict=True):
        if not isinstance(envelope, Mapping) or (
            envelope.get("feature_id") != output.artifact_ref.feature_id
            or envelope.get("artifact_type") != output.artifact_ref.artifact_type
            or envelope.get("artifact_hash") != output.artifact_ref.artifact_hash
        ):
            raise ProductExecutionContractError(
                "Product output readback changed Artifact identity or order.",
                side_effects_may_have_started=True,
            )
        envelopes[output.sequence] = envelope
    return envelopes


def _validate_repair_plan_lifecycle_predicates(
    facts: "ProductExecutionFactsV1",
    *, upstream_sequences: frozenset[int] = frozenset(),
) -> None:
    envelopes = _output_readback_envelopes(facts)
    if envelopes is None:
        if (
            facts.terminal_node == "blocked"
            and facts.store_attempts
            and all(
                item.store_result.outcome == "committed"
                for item in facts.store_attempts
            )
            and facts.store_attempts[-1].node_name == "create_repair_plan"
            and facts.failure_stage in {None, "post_execution_validation"}
        ):
            raise ProductExecutionContractError(
                "Unreadable terminal RepairPlan lacks its committed BlockedReport.",
                side_effects_may_have_started=True,
            )
        return
    attempts = facts.store_attempts
    grammar = PRODUCT_EXECUTION_DECLARATION.lifecycle_grammar
    for index, item in enumerate(attempts):
        if (item.node_name != "create_repair_plan" or item.artifact_role != "repair_plan"
                or item.store_result.outcome != "committed" or item.sequence in upstream_sequences):
            continue
        envelope = envelopes.get(item.sequence)
        if envelope is None:
            raise ProductExecutionContractError(
                "Committed RepairPlan lacks exact output readback.",
                side_effects_may_have_started=True,
            )
        prior = next(
            (
                previous
                for previous in reversed(attempts[:index])
                if previous.node_name != "create_repair_plan"
            ),
            None,
        )
        if prior is None or prior.node_name not in grammar.repair_transition_sources:
            raise ProductExecutionContractError(
                "RepairPlan readback lacks its declared lifecycle source.",
                side_effects_may_have_started=True,
            )
        later = next(
            (
                following
                for following in attempts[index + 1 :]
                if following.node_name != "create_repair_plan"
            ),
            None,
        )
        data = envelope.get("data")
        payload_terminal = envelope.get("status") == "blocked"
        terminal = later is not None and later.node_name == "blocked"
        physical_source_blocked = False
        if terminal and prior.node_name == "build":
            blocked_envelope = envelopes.get(later.sequence, {})
            physical_source_blocked = any(
                isinstance(blocker, Mapping)
                and blocker.get("code") == "PRODUCT_REPAIR_SOURCE_REANCHORING_REQUIRED"
                and blocker.get("stage") == "repair_plan"
                and blocker.get("required_next_owner") == "evidence"
                for blocker in blocked_envelope.get("data", {}).get("blockers", [])
            )
            if (not physical_source_blocked and later == attempts[-1]
                    and later.artifact_role == "blocked_report"
                    and later.store_result.outcome != "committed"
                    and facts.failure_stage == "blocked_report_store"):
                # A failed report write has no committed readback. The existing
                # typed blocker facts retain the exact reason submitted by the
                # Graph before C5 failed; they do not stand in for an Artifact.
                physical_source_blocked = any(
                    blocker.owner == "domain" and blocker.source_node == "blocked"
                    and blocker.stage == "repair_plan"
                    and blocker.code == "PRODUCT_REPAIR_SOURCE_REANCHORING_REQUIRED"
                    for blocker in facts.blockers
                )
        if later is None and payload_terminal:
            terminal = facts.failure_stage == "blocked_execution"
        source_attempt = prior.repair_attempt
        expected_attempt = (
            source_attempt
            if terminal
            and (
                prior.node_name in {"collect_runtime_evidence", "resolve_runtime_binding"}
                or source_attempt == grammar.max_repair_attempt
            )
            else source_attempt + 1
        )
        if item.repair_attempt != expected_attempt:
            raise ProductExecutionContractError(
                "RepairPlan readback disagrees with its declared attempt transition.",
                side_effects_may_have_started=True,
            )
        blockers = data.get("blockers") if isinstance(data, Mapping) else None
        expected_exhausted = terminal and source_attempt == grammar.max_repair_attempt
        allowed_exhausted = (
            {False, True}
            if expected_exhausted
            and prior.node_name
            in {"collect_runtime_evidence", "resolve_runtime_binding"}
            else {expected_exhausted}
        )
        semantic_terminal = terminal and not physical_source_blocked
        expected_status = "blocked" if semantic_terminal else "success"
        expected_next_stage = None if semantic_terminal else "generate_patch_proposal"
        if (
            not isinstance(data, Mapping)
            or envelope.get("status") != expected_status
            or type(data.get("retry_count")) is not int
            or data.get("retry_count") != expected_attempt
            or type(data.get("retry_budget")) is not int
            or data.get("retry_budget") != grammar.max_repair_attempt
            or type(data.get("retry_budget_remaining")) is not int
            or data.get("retry_budget_remaining")
            != grammar.max_repair_attempt - source_attempt
            or type(data.get("retry_exhausted")) is not bool
            or data.get("retry_exhausted") not in allowed_exhausted
            or data.get("allowed_next_stage") != expected_next_stage
            or type(blockers) is not list
            or (semantic_terminal and not blockers)
            or (not semantic_terminal and blockers)
        ):
            raise ProductExecutionContractError(
                "RepairPlan readback violates its declaration-owned lifecycle predicate.",
                side_effects_may_have_started=True,
            )


def _validated_product_upstream_sequences(facts: ProductExecutionFactsV1, *,
        task_agent_sequences: frozenset[int] = frozenset()) -> frozenset[int]:
    """Recognize exact assessment insertions before applying the original grammar.

    Assessment can precede the first Proposal, so it has no invented failure
    ancestor. Its stored selected inputs and semantic boundary identify the
    insertion; ordinary RepairPlan and approval/Apply ordering stay unchanged.
    """
    envelopes = _output_readback_envelopes(facts)
    if envelopes is None:
        return frozenset()
    excluded: set[int] = set()
    attempts = facts.store_attempts

    def require(condition):
        if not condition:
            raise ProductExecutionContractError(
                "Product upstream assessment violates its exact input or lifecycle boundary.",
                side_effects_may_have_started=True)

    for index, item in enumerate(attempts):
        if item.sequence in task_agent_sequences:
            continue
        envelope = envelopes.get(item.sequence, {})
        data = envelope.get("data", {})
        if (item.artifact_role != "repair_plan" or not isinstance(data, Mapping)
                or data.get("failure_class") != "patch_upstream_assessment"):
            continue
        retained_plan = next((output for output in facts.outputs
                              if output.origin == "reused_input" and output.artifact_role == "implementation_plan"), None)
        require((index > 0 or retained_plan is not None) and item.node_name == "create_repair_plan")
        context = data.get("repair_context") or {}
        upstream = context.get("upstream_request") or {}
        decision = data.get("semantic_decision") or {}
        owner = upstream.get("required_owner")
        previous = attempts[index - 1] if index else retained_plan
        require(previous.node_name in {"create_implementation_plan", "create_repair_plan", "interrupted_patch_admission", "pretransport_patch_admission"}
                and previous.repair_attempt == item.repair_attempt
                and envelope.get("status") == "success"
                and owner in {"evidence", "planning"}
                and upstream.get("origin") == "model"
                and decision.get("decision") == "upstream_assessment_ready"
                and decision.get("required_next_owner") == owner
                and decision.get("authority_granted") is False
                and context.get("approval_inherited") is False
                and data.get("scope_expansion_allowed") is False
                and data.get("minimal_changes") == [] and data.get("blockers") == []
                and data.get("allowed_next_stage") is None and data.get("source_proposal") == {}
                and data.get("retry_count") == item.repair_attempt
                and data.get("retry_budget") == PRODUCT_EXECUTION_DECLARATION.lifecycle_grammar.max_repair_attempt
                and data.get("retry_budget_remaining") == data["retry_budget"] - item.repair_attempt
                and data.get("retry_exhausted") is False)
        _require_sha256(context.get("product_context_sha256"), "product_context_sha256")
        _require_sha256(context.get("source_snapshot_sha256"), "source_snapshot_sha256")
        _require_sha256(context.get("workspace_snapshot_sha256"), "workspace_snapshot_sha256")
        response_ref = upstream.get("source_worker_response_ref") or {}
        _require_non_empty(response_ref.get("path"), "source_worker_response_ref.path")
        _require_sha256(response_ref.get("sha256"), "source_worker_response_ref.sha256")
        expected = {}
        for kind in ("feature_spec", "evidence_pack", "implementation_plan"):
            prior = next((row for row in reversed(facts.outputs)
                          if row.sequence < item.sequence and row.artifact_role == kind), None)
            require(prior is not None)
            ref = prior.artifact_ref
            expected[kind] = {"feature_id": ref.feature_id, "artifact_type": ref.artifact_type,
                              "path": ref.path, "artifact_hash": ref.artifact_hash}
        require(context.get("selected_artifacts") == expected
                and envelope.get("inputs", {}).get("upstream_artifacts") == [
                    {"artifact_type": kind, "path": ref["path"], "artifact_hash": ref["artifact_hash"],
                     "require_latest": False} for kind, ref in expected.items()])
        failure = context.get("failure_report")
        if context.get("current_failure_available") is True:
            require(isinstance(failure, Mapping) and failure.get("execution_origin") == "executed"
                    and failure.get("status") in {"behavior_failed", "needs_evidence"}
                    and data.get("failure_source") == "behavior_report"
                    and data.get("failure_findings") == failure.get("observations")
                    and failure.get("source_snapshot_sha256") == context.get("source_snapshot_sha256")
                    and failure.get("generated_changes_sha256") == context.get("generated_changes_sha256")
                    and upstream.get("failure_report_ref") == context.get("failure_report_ref"))
            for field in ("failed_response_ref", "failure_report_ref"):
                ref = context.get(field) or {}
                _require_non_empty(ref.get("path"), field + ".path")
                _require_sha256(ref.get("sha256"), field + ".sha256")
        else:
            require(context.get("current_failure_available") is False and failure is None
                    and data.get("failure_source") == "patch_upstream_request"
                    and data.get("failure_findings") == [] and data.get("failure_evidence") == []
                    and all(context.get(key) is None for key in (
                        "failed_response_ref", "failure_report_ref", "generated_changes_sha256"))
                    and upstream.get("failure_report_ref") is None)
        excluded.add(item.sequence)
        # A retained Plan produces no new Plan Artifact. Evidence assessment must
        # still publish Evidence before a new Plan, or stop at its first failure.
        expected_nodes = (["retrieve_evidence", "create_implementation_plan"]
                          if owner == "evidence" else ["create_implementation_plan"])
        following = index + 1
        for expected_node in expected_nodes:
            if following >= len(attempts) or attempts[following].node_name in {"blocked", "generate_patch_proposal"}:
                require(owner == "planning" or following >= len(attempts)
                        or attempts[following].node_name == "blocked")
                break
            row = attempts[following]
            require(row.node_name == expected_node and row.repair_attempt == item.repair_attempt)
            excluded.add(row.sequence)
            following += 1
        if following < len(attempts):
            next_attempt = attempts[following]
            next_data = envelopes.get(next_attempt.sequence, {}).get("data", {})
            require(next_attempt.node_name in {"generate_patch_proposal", "blocked"}
                    or next_attempt.node_name == "create_repair_plan"
                    and next_data.get("failure_class") == "patch_upstream_assessment")
    return frozenset(excluded)


def _validated_initial_findings_reassessment_sequences(
    facts: "ProductExecutionFactsV1",
) -> frozenset[int]:
    """Admit one second Evidence revision inside the initial retrieval owner."""
    from src.mod_agent.evidence.models import EvidencePack
    from src.mod_agent.evidence.reassessment import initial_findings_reassessment_allowed

    envelopes = _output_readback_envelopes(facts)
    admitted: set[int] = set()
    attempts = facts.store_attempts
    initial_prefix = PRODUCT_EXECUTION_DECLARATION.lifecycle_grammar.initial_prefix[:4]

    def require(condition):
        if not condition:
            raise ProductExecutionContractError(
                "Initial Findings reassessment changed its retained Evidence or requirement boundary.",
                side_effects_may_have_started=True)

    def read_pack(payload):
        try:
            return EvidencePack.from_artifact_payload(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise ProductExecutionContractError(
                "Initial Findings reassessment Evidence readback is invalid.",
                side_effects_may_have_started=True) from exc

    for index, item in enumerate(attempts):
        if (index == 0 or item.node_name != "retrieve_evidence"
                or attempts[index - 1].node_name != "retrieve_evidence"):
            continue
        previous = attempts[index - 1]
        require(not admitted and tuple(row.node_name for row in attempts[:index]) == initial_prefix
                and item.artifact_role == previous.artifact_role == "evidence_pack"
                and item.repair_attempt == previous.repair_attempt == 0
                and previous.store_result.outcome == "committed")
        if envelopes is None:
            # Failed readback can retain a partial execution, never validate a
            # successful reassessment or let this unreadable result continue.
            require(facts.failure_stage == "post_execution_validation")
            admitted.add(item.sequence)
            continue
        prior_envelope = envelopes.get(previous.sequence, {})
        prior_payload = prior_envelope.get("data", {})
        prior_pack = read_pack(prior_payload)
        feature = next((output for output in facts.outputs
                        if output.sequence < previous.sequence and output.artifact_role == "feature_spec"), None)
        require(feature is not None)
        feature_data = envelopes[feature.sequence]["data"]
        require(prior_envelope.get("status") == "blocked"
                and initial_findings_reassessment_allowed(prior_payload, feature_data)
                and prior_envelope.get("inputs", {}).get("upstream_artifacts") == [{
                    "artifact_type": "feature_spec", "path": feature.artifact_ref.path,
                    "artifact_hash": feature.artifact_ref.artifact_hash, "require_latest": True}])
        if item.store_result.outcome != "committed":
            # There is no second output to read back after a failed Store write.
            # Keep the first negative revision and the exact terminal Store fact.
            require(index == len(attempts) - 1 and item.sequence not in envelopes)
            admitted.add(item.sequence)
            continue
        envelope = envelopes.get(item.sequence, {})
        payload = envelope.get("data", {})
        pack = read_pack(payload)
        metadata = pack.budget_summary.get("initial_findings_reassessment") or {}
        prior_ref = previous.store_result.artifact_ref
        require(isinstance(metadata, Mapping) and prior_ref is not None
                and metadata.get("prior_evidence_ref") == {
                    "feature_id": prior_ref.feature_id, "artifact_type": prior_ref.artifact_type,
                    "path": prior_ref.path, "artifact_hash": prior_ref.artifact_hash}
                and envelope.get("feature_id") == prior_envelope.get("feature_id")
                and envelope.get("inputs") == prior_envelope.get("inputs")
                and f"{prior_ref.path}#sha256={prior_ref.artifact_hash}" in envelope.get("trace_refs", [])
                and pack.query == prior_pack.query and isinstance(pack.requirement_coverage, Mapping)
                and all(pack.requirement_coverage.get(key) == prior_pack.requirement_coverage.get(key) for key in (
                    "schema_version", "feature_spec_identity", "request_identity", "projection_identity",
                    "input_obligations", "input_dispositions", "requirements", "relations", "queries"))
                and envelope.get("status") == ("success" if pack.can_generate_patch else "blocked"))
        admitted.add(item.sequence)
    return frozenset(admitted)


def _validate_terminal_effect_boundary(facts: "ProductExecutionFactsV1") -> None:
    if not facts.store_attempts or facts.store_attempts[-1].node_name != "blocked":
        return
    terminal_attempt = facts.store_attempts[-1].repair_attempt
    if any(
        event.repair_attempt > terminal_attempt for event in facts.effect_events
    ):
        raise ProductExecutionContractError(
            "Product effect trace continues after its blocked repair terminal.",
            side_effects_may_have_started=True,
        )


@dataclass(frozen=True)
class _ProductExecutionResultIdentityV1:
    run_id: str
    request_id: str
    requested_completion_scope: ProductCompletionScope


@dataclass(frozen=True)
class ProductTaskAgentExecutionV1:
    """One trusted technical execution, retaining its real internal Store attempts.

    Scope: optional task-agent owner only; no extra outer workflow stage or permission.
    为什么要有它：内部调查可以产生真实Evidence/Plan修订，但不能伪装旧Graph节点或丢掉失败时已经发生的写入。
    """
    contract_version: Literal["1"]
    run_id: str
    request_id: str
    repair_attempt: int
    store_start_sequence: int
    initial_context: Mapping[str, Any]
    outcome: Mapping[str, Any]
    store_sequences: tuple[int, ...]

    def __post_init__(self):
        if self.contract_version != "1":
            raise ProductExecutionContractError("Task execution record version changed.", side_effects_may_have_started=True)
        _require_non_empty(self.run_id, "task run_id")
        _require_non_empty(self.request_id, "task request_id")
        if _require_exact_int(self.repair_attempt, "task outer repair attempt") > 2:
            raise ProductExecutionContractError("Task outer repair attempt changed.", side_effects_may_have_started=True)
        _require_exact_int(self.store_start_sequence, "task Store starting sequence")
        if (not isinstance(self.initial_context, Mapping) or not isinstance(self.outcome, Mapping)
                or any(type(value) is not int or value < 0 for value in self.store_sequences)
                or tuple(sorted(set(self.store_sequences))) != self.store_sequences):
            raise ProductExecutionContractError("Task execution record is invalid.", side_effects_may_have_started=True)

    def to_dict(self):
        return {"contract_version": self.contract_version, "run_id": self.run_id, "request_id": self.request_id,
            "repair_attempt": self.repair_attempt, "store_start_sequence": self.store_start_sequence,
            "initial_context": dict(self.initial_context),
            "outcome": dict(self.outcome), "store_sequences": list(self.store_sequences)}

    @classmethod
    def from_dict(cls, value):
        _require_exact_fields(value, ("contract_version", "run_id", "request_id", "repair_attempt", "store_start_sequence",
            "initial_context", "outcome", "store_sequences"), "ProductTaskAgentExecutionV1")
        return cls(**{**value, "store_sequences": tuple(value["store_sequences"])})


@dataclass(frozen=True)
class ProductExecutionFactsV1:
    request: ProductExecutionRequestV1 | _ProductExecutionResultIdentityV1
    feature_id: str | None
    terminal_node: str | None
    store_attempts: tuple[ProductExecutionStoreAttemptV1, ...]
    outputs: tuple[ProductExecutionOutputV1, ...]
    effect_events: tuple[ProductExecutionEffectEventV1, ...]
    validation_facts: ProductExecutionValidationFactsV1
    blockers: tuple[ProductExecutionBlockerV1, ...]
    variant_result: ProductBuildCompletionV1 | ProductRuntimeCompletionV1
    failure_stage: str | None
    task_agent_executions: tuple[ProductTaskAgentExecutionV1, ...] = ()
    v2_delivery_checkpoint_ref: str | None = None


def _validated_task_agent_sequences(facts: ProductExecutionFactsV1) -> frozenset[int]:
    """Exclude only exact, receipt-bound internal writes from the outer grammar.

    为什么要有它：局部接入新技术循环不能放宽旧Store顺序、审批、Apply或正式Build边界。
    """
    from src.mod_agent.model_qualification import read_bound_json

    selected: set[int] = set()
    attempts = {row.sequence: row for row in facts.store_attempts}
    def require(condition):
        if not condition:
            raise ProductExecutionContractError("Task-agent internal Store binding changed.", side_effects_may_have_started=True)
    for execution in facts.task_agent_executions:
        require(isinstance(execution, ProductTaskAgentExecutionV1)
            and execution.run_id == facts.request.run_id and execution.request_id == facts.request.request_id)
        outcome = execution.outcome
        require(execution.store_sequences == tuple(range(execution.store_start_sequence,
            execution.store_start_sequence + len(execution.store_sequences))))
        require(outcome.get("initial_context") == execution.initial_context
            and outcome.get("whole_task_delivered") is False
            and outcome.get("status") in {"candidate_ready", "blocked", "failed", "unknown"})
        rows = []
        for sequence in execution.store_sequences:
            require(sequence not in selected and sequence in attempts)
            row = attempts[sequence]
            require(row.node_name == "generate_patch_proposal"
                and row.phase == PRODUCT_EXECUTION_DECLARATION.phase_by_node["generate_patch_proposal"]
                and row.repair_attempt == execution.repair_attempt
                and row.artifact_role in {"repair_plan", "evidence_pack", "implementation_plan"})
            rows.append(row.to_dict())
            selected.add(sequence)
        try:
            if outcome.get("execution_ref") is not None:
                retained = read_bound_json(outcome["execution_ref"])
                require(retained.get("run_id") == execution.run_id and retained.get("request_id") == execution.request_id
                    and retained.get("product_store_attempts") == rows
                    and all(retained.get(key) == value for key, value in outcome.items() if key != "execution_ref"))
            else:
                # Scope: failed final-file persistence only; never a delivery-capable result.
                # 为什么要有它：落盘失败不能清除已发生的Store操作，但未绑定最终结果也绝不具备交付资格。
                require(outcome.get("status") in {"failed", "unknown"} and facts.failure_stage is not None)
                if outcome.get("claim_ref") is None:
                    require(not rows)
                else:
                    claim = read_bound_json(outcome["claim_ref"])
                    require(claim.get("run_id") == execution.run_id and claim.get("request_id") == execution.request_id
                        and claim.get("initial_context") == execution.initial_context)
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, ProductExecutionContractError):
                raise
            raise ProductExecutionContractError("Task-agent retained execution evidence is unavailable.",
                side_effects_may_have_started=True) from error
        initial, final = execution.initial_context, outcome.get("final_context")
        expected_context = ({"feature_spec_ref"} if outcome.get("schema_version") == "task-agent-product-execution.v2"
            else {"feature_spec_ref", "evidence_pack_ref", "implementation_plan_ref"})
        require(isinstance(final, Mapping) and set(initial) == set(final)
            == expected_context
            and initial["feature_spec_ref"] == final["feature_spec_ref"])
        for context, sequence_limit in ((initial, execution.store_start_sequence),
                (final, execution.store_start_sequence + len(execution.store_sequences))):
            for name, reference in context.items():
                prior = next((output for output in reversed(facts.outputs)
                    if output.sequence < sequence_limit and output.artifact_role == name.removesuffix("_ref")), None)
                current_required = context is initial or outcome.get("status") == "candidate_ready"
                # Scope: a failed internal Plan publication may leave a newer Evidence head.
                # 为什么要有它：失败回执必须保留半途真实写入；旧成对上下文可作为失败位置，绝不能作为可交付最新状态。
                bound = (prior is not None and _ref_to_dict(prior.artifact_ref) == reference) if current_required else any(
                    output.sequence < sequence_limit and _ref_to_dict(output.artifact_ref) == reference for output in facts.outputs)
                require(isinstance(reference, Mapping) and reference.get("artifact_type") == name.removesuffix("_ref")
                    and reference.get("feature_id") == facts.feature_id
                    and bound)
        require(any(event.effect_role == "patch_generation" and event.event_status == "started"
            and event.repair_attempt == execution.repair_attempt for event in facts.effect_events))
    return frozenset(selected)


def _validated_reused_product_inputs(facts: ProductExecutionFactsV1) -> tuple[ProductExecutionOutputV1, ...]:
    reused = tuple(output for output in facts.outputs if output.origin == "reused_input")
    if not reused:
        return ()
    roles = ("feature_spec", "evidence_pack", "implementation_plan")
    def require(condition):
        if not condition:
            raise ProductExecutionContractError("Interrupted Patch reused inputs changed their exact lineage.",
                                                side_effects_may_have_started=True)
    # 为什么要有它：复用标记本身不能放宽生命周期，三件只读输入必须保持顺序、同一选择来源及原始依赖链。
    require(reused == facts.outputs[:3] and len(reused) == 3)
    require(tuple(row.artifact_role for row in reused) == roles
            and tuple(row.sequence for row in reused) == (0, 1, 2)
            and tuple(row.phase for row in reused) == ("planning", "evidence", "planning")
            and all(row.artifact_ref.feature_id == facts.feature_id
                    and row.reuse_selection == reused[0].reuse_selection and row.node_name == reused[0].node_name for row in reused))
    from src.mod_agent.saved_patch_admission import _json
    try:
        journal = _json(reused[0].reuse_selection["source_attempt_ref"])
        manifest = _json(reused[0].reuse_selection["parent_manifest_ref"])
        pretransport = reused[0].node_name == "pretransport_patch_admission"
        if pretransport:
            terminal = _json(reused[0].reuse_selection["parent_result_ref"])
            require(manifest.get("terminal_record") == reused[0].reuse_selection["parent_result_ref"]
                and journal.get("model_call_count") == 0 and journal.get("proposal_admitted") is False
                and terminal.get("product_execution_result", {}).get("execution_disposition") == "partial"
                and terminal.get("model_call_budget") == journal.get("shared_model_budget"))
        else:
            stop = _json(reused[0].reuse_selection["execution_stop_ref"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise ProductExecutionContractError("Interrupted Patch reused inputs lost their exact original records.",
                                            side_effects_may_have_started=True) from exc
    require(journal.get("selected_artifacts") == {row.artifact_role: _ref_to_dict(row.artifact_ref) for row in reused}
            and journal.get("feature_id") == facts.feature_id
            and journal.get("request_id") == manifest.get("request_id") == facts.request.request_id
            and journal.get("parent_run_id") == manifest.get("run_id") == facts.request.run_id
            and (pretransport or (all(stop.get(key) == reused[0].reuse_selection[key] for key in ("source_attempt_ref", "parent_manifest_ref"))
            and stop.get("original_execution_stopped") is True and stop.get("origin") == "host_controller")))
    envelopes = _output_readback_envelopes(facts)
    if envelopes is None:
        require(facts.failure_stage == "post_execution_validation")
        return reused
    for index, output in enumerate(reused):
        envelope = envelopes[output.sequence]
        candidate = None
        if output.artifact_role == "evidence_pack" and envelope.get("status") == "blocked":
            from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope
            candidate = compilation_candidate_scope(envelope["data"], envelopes[reused[0].sequence]["data"])
        require(envelope.get("status") == "success" or candidate is not None)
        if index:
            previous = reused[index - 1].artifact_ref
            upstream = envelope.get("inputs", {}).get("upstream_artifacts", [])
            require(any(row.get("artifact_type") == previous.artifact_type
                        and row.get("path") == previous.path
                        and row.get("artifact_hash") == previous.artifact_hash for row in upstream))
    return reused


def _validate_product_execution_result_facts(
    facts: ProductExecutionFactsV1,
) -> None:
    reused = _validated_reused_product_inputs(facts)
    task_agent_sequences = _validated_task_agent_sequences(facts)
    upstream_sequences = _validated_product_upstream_sequences(facts, task_agent_sequences=task_agent_sequences)
    initial_reassessment_sequences = _validated_initial_findings_reassessment_sequences(facts)
    _validate_store_trace(facts.store_attempts, tuple(row for row in facts.outputs if row.origin == "generated"),
                         upstream_sequences=upstream_sequences | initial_reassessment_sequences,
                         reused_input_count=len(reused), task_agent_sequences=task_agent_sequences)
    _validate_effect_trace(facts.effect_events)
    _validate_lifecycle(tuple(item for item in facts.store_attempts if item.sequence not in upstream_sequences | task_agent_sequences),
                        interrupted_patch=bool(reused), task_agent_v2=any(
                            row.outcome.get("schema_version") == "task-agent-product-execution.v2"
                            for row in facts.task_agent_executions))
    _validate_repair_plan_lifecycle_predicates(facts, upstream_sequences=upstream_sequences | task_agent_sequences)
    _validate_terminal_effect_boundary(facts)
    if facts.feature_id is not None:
        _require_non_empty(facts.feature_id, "feature_id")
    if facts.terminal_node not in {None, "build", "done", "blocked", "dry_run_apply"}:
        raise ProductExecutionContractError(
            "Product terminal node is unsupported.",
            side_effects_may_have_started=True,
        )
    if facts.request.requested_completion_scope == "compile_only" and not isinstance(
        facts.variant_result, ProductBuildCompletionV1
    ):
        raise ProductExecutionContractError(
            "Compile product result requires ProductBuildCompletionV1.",
            side_effects_may_have_started=True,
        )
    if facts.request.requested_completion_scope == "runtime_workflow" and not isinstance(
        facts.variant_result, ProductRuntimeCompletionV1
    ):
        raise ProductExecutionContractError(
            "Runtime product result requires ProductRuntimeCompletionV1.",
            side_effects_may_have_started=True,
        )
    allowed_stages = {"graph_start", "post_execution_validation"}
    allowed_stages.update(f"{node}_execution" for node, _ in PRODUCT_EXECUTION_DECLARATION.node_phases)
    allowed_stages.update(f"{role}_store" for _, role in PRODUCT_EXECUTION_DECLARATION.node_store_roles)
    if facts.failure_stage is not None and facts.failure_stage not in allowed_stages:
        raise ProductExecutionContractError(
            "Product failure stage is not declared.",
            side_effects_may_have_started=True,
        )


def validate_product_execution_facts(facts: ProductExecutionFactsV1) -> None:
    if not isinstance(facts, ProductExecutionFactsV1):
        raise ProductExecutionContractError(
            "Product execution facts are invalid.",
            side_effects_may_have_started=True,
        )
    if not isinstance(facts.request, ProductExecutionRequestV1):
        raise ProductExecutionContractError(
            "Live product facts require an exact ProductExecutionRequestV1.",
            side_effects_may_have_started=True,
        )
    _validate_product_execution_result_facts(facts)


@dataclass(frozen=True)
class ProductExecutionResultV1:
    contract_version: Literal["1"]
    run_id: str
    request_id: str
    feature_id: str | None
    requested_completion_scope: ProductCompletionScope
    platform_status: Literal["completed", "blocked", "partial"]
    execution_disposition: ProductExecutionDisposition
    failure_stage: str | None
    side_effects_may_have_started: bool
    terminal_node: Literal["build", "done", "blocked", "dry_run_apply"] | None
    store_attempts: tuple[ProductExecutionStoreAttemptV1, ...]
    outputs: tuple[ProductExecutionOutputV1, ...]
    effect_events: tuple[ProductExecutionEffectEventV1, ...]
    validation_facts: ProductExecutionValidationFactsV1
    blockers: tuple[ProductExecutionBlockerV1, ...]
    variant_result: ProductBuildCompletionV1 | ProductRuntimeCompletionV1
    task_agent_executions: tuple[ProductTaskAgentExecutionV1, ...] = ()
    v2_delivery_checkpoint_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "feature_id": self.feature_id,
            "requested_completion_scope": self.requested_completion_scope,
            "platform_status": self.platform_status,
            "execution_disposition": self.execution_disposition,
            "failure_stage": self.failure_stage,
            "side_effects_may_have_started": self.side_effects_may_have_started,
            "terminal_node": self.terminal_node,
            "store_attempts": [item.to_dict() for item in self.store_attempts],
            "outputs": [item.to_dict() for item in self.outputs],
            "effect_events": [item.to_dict() for item in self.effect_events],
            "validation_facts": self.validation_facts.to_dict(),
            "blockers": [item.to_dict() for item in self.blockers],
            "variant_result": self.variant_result.to_dict(),
            **({"task_agent_executions": [item.to_dict() for item in self.task_agent_executions]}
               if self.task_agent_executions else {}),
            **({"v2_delivery_checkpoint_ref": self.v2_delivery_checkpoint_ref} if self.v2_delivery_checkpoint_ref else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductExecutionResultV1":
        fields = (
            "contract_version",
            "run_id",
            "request_id",
            "feature_id",
            "requested_completion_scope",
            "platform_status",
            "execution_disposition",
            "failure_stage",
            "side_effects_may_have_started",
            "terminal_node",
            "store_attempts",
            "outputs",
            "effect_events",
            "validation_facts",
            "blockers",
            "variant_result",
        )
        if "task_agent_executions" in value:
            fields = (*fields, "task_agent_executions")
        if "v2_delivery_checkpoint_ref" in value:
            fields = (*fields, "v2_delivery_checkpoint_ref")
        _require_exact_fields(value, fields, "ProductExecutionResultV1")
        if value["contract_version"] != "1":
            raise ProductExecutionContractError(
                "Product execution result version is unsupported.",
                side_effects_may_have_started=True,
            )
        _require_non_empty(value["run_id"], "run_id")
        _require_non_empty(value["request_id"], "request_id")
        if value["feature_id"] is not None:
            _require_non_empty(value["feature_id"], "feature_id")
        if value["requested_completion_scope"] not in {
            "compile_only",
            "runtime_workflow",
        }:
            raise ProductExecutionContractError(
                "Product result completion scope is unsupported.",
                side_effects_may_have_started=True,
            )
        if type(value["side_effects_may_have_started"]) is not bool:
            raise ProductExecutionContractError(
                "Product result side-effect fact must be bool.",
                side_effects_may_have_started=True,
            )
        for field in ("store_attempts", "outputs", "effect_events", "blockers"):
            if type(value[field]) is not list:
                raise ProductExecutionContractError(
                    f"{field} must serialize as an array.",
                    side_effects_may_have_started=True,
                )
        scope = value["requested_completion_scope"]
        variant = (
            ProductBuildCompletionV1.from_dict(value["variant_result"])
            if scope == "compile_only"
            else ProductRuntimeCompletionV1.from_dict(value["variant_result"])
        )
        parsed = cls(
            contract_version=value["contract_version"],
            run_id=value["run_id"],
            request_id=value["request_id"],
            feature_id=value["feature_id"],
            requested_completion_scope=scope,
            platform_status=value["platform_status"],
            execution_disposition=value["execution_disposition"],
            failure_stage=value["failure_stage"],
            side_effects_may_have_started=value["side_effects_may_have_started"],
            terminal_node=value["terminal_node"],
            store_attempts=tuple(ProductExecutionStoreAttemptV1.from_dict(item) for item in value["store_attempts"]),
            outputs=tuple(ProductExecutionOutputV1.from_dict(item) for item in value["outputs"]),
            effect_events=tuple(ProductExecutionEffectEventV1.from_dict(item) for item in value["effect_events"]),
            validation_facts=ProductExecutionValidationFactsV1.from_dict(value["validation_facts"]),
            blockers=tuple(ProductExecutionBlockerV1.from_dict(item) for item in value["blockers"]),
            variant_result=variant,
            task_agent_executions=tuple(ProductTaskAgentExecutionV1.from_dict(item)
                for item in value.get("task_agent_executions", ())),
            v2_delivery_checkpoint_ref=value.get("v2_delivery_checkpoint_ref"),
        )
        request = _ProductExecutionResultIdentityV1(
            run_id=parsed.run_id,
            request_id=parsed.request_id,
            requested_completion_scope=parsed.requested_completion_scope,
        )
        facts = ProductExecutionFactsV1(
            request=request,
            feature_id=parsed.feature_id,
            terminal_node=parsed.terminal_node,
            store_attempts=parsed.store_attempts,
            outputs=parsed.outputs,
            effect_events=parsed.effect_events,
            validation_facts=parsed.validation_facts,
            blockers=parsed.blockers,
            variant_result=parsed.variant_result,
            failure_stage=parsed.failure_stage,
            task_agent_executions=parsed.task_agent_executions,
            v2_delivery_checkpoint_ref=parsed.v2_delivery_checkpoint_ref,
        )
        projected = project_product_execution_result(facts, validate_request=False)
        if projected.to_dict() != parsed.to_dict():
            raise ProductExecutionContractError(
                "Product execution result changed during strict replay.",
                side_effects_may_have_started=parsed.side_effects_may_have_started,
            )
        return parsed


def project_product_execution_result(
    facts: ProductExecutionFactsV1,
    *,
    validate_request: bool = True,
) -> ProductExecutionResultV1:
    if validate_request:
        validate_product_execution_facts(facts)
    else:
        _validate_product_execution_result_facts(facts)
    effects_started = any(item.event_status == "started" for item in facts.effect_events)
    last = facts.store_attempts[-1].store_result if facts.store_attempts else None
    domain_blockers = tuple(item for item in facts.blockers if item.owner == "domain")
    platform_blockers = tuple(item for item in facts.blockers if item.owner == "platform")
    failure_stage = facts.failure_stage
    if last is not None and last.outcome != "committed":
        failure_stage = f"{facts.store_attempts[-1].artifact_role}_store"
        is_partial = bool(facts.outputs) or effects_started or last.outcome == "partial"
        platform_status = "partial" if is_partial else "blocked"
        disposition: ProductExecutionDisposition = (
            "partial" if is_partial else "platform_blocked"
        )
    elif failure_stage is not None:
        is_partial = bool(facts.outputs) or effects_started
        platform_status = "partial" if is_partial else "blocked"
        disposition = "partial" if is_partial else "platform_blocked"
    elif facts.validation_facts.failed:
        failure_stage = "post_execution_validation"
        platform_status = "partial"
        disposition = "partial"
    elif facts.terminal_node == "dry_run_apply":
        # 为什么要有它：U1真实接管完成但尚未Apply，必须明确待交付而不是虚报成功或伪造平台故障。
        executions = [row for row in facts.task_agent_executions
            if row.outcome.get("schema_version") == "task-agent-product-execution.v2"]
        if (len(executions) != 1 or not facts.outputs or facts.outputs[-1].artifact_role != "dry_run_result"
                or any(event.effect_role in {"isolated_apply", "build", "runtime"} for event in facts.effect_events)
                or facts.blockers):
            raise ProductExecutionContractError("V2 pending delivery boundary changed.", side_effects_may_have_started=effects_started)
        from src.mod_agent.artifacts import ArtifactStore
        reference = facts.outputs[-1].artifact_ref
        envelope = json.loads(Path(reference.path).read_text(encoding="utf-8"))
        payload = envelope["data"]
        if (ArtifactStore._stable_hash(ArtifactStore._envelope_content_for_hash(envelope)) != reference.artifact_hash
                or payload.get("task_agent_protocol") != "v2" or payload.get("delivery_status") != "pending"
                or payload.get("actual_write_performed") is not False or payload.get("whole_task_delivered") is not False
                or payload.get("candidate_ref") != executions[0].outcome.get("candidate_ref")):
            raise ProductExecutionContractError("V2 pending delivery artifact changed.", side_effects_may_have_started=effects_started)
        platform_status, disposition = "completed", "awaiting_delivery"
        if facts.v2_delivery_checkpoint_ref is not None:
            from src.mod_agent.task_agent_v2_state import load_v2_artifact
            candidates = [parent for parent in Path(reference.path).resolve().parents
                if (parent / facts.v2_delivery_checkpoint_ref).is_file()]
            if len(candidates) != 1:
                raise ProductExecutionContractError("V2 delivery checkpoint unavailable.", side_effects_may_have_started=effects_started)
            checkpoint = load_v2_artifact(candidates[0], facts.v2_delivery_checkpoint_ref)
            if (checkpoint.get("schema") != "task-agent-delivery-checkpoint.v2"
                    or checkpoint.get("run_id") != facts.request.run_id or checkpoint.get("request_id") != facts.request.request_id
                    or checkpoint.get("artifact_refs", {}).get("dry_run_result") != _ref_to_dict(reference)
                    or checkpoint.get("candidate_source_id") != payload.get("candidate_source_id")):
                raise ProductExecutionContractError("V2 delivery checkpoint binding changed.", side_effects_may_have_started=effects_started)
    elif facts.terminal_node == "blocked":
        if not facts.outputs or facts.outputs[-1].artifact_role != "blocked_report":
            raise ProductExecutionContractError(
                "Domain-blocked product result lacks committed BlockedReport.",
                side_effects_may_have_started=effects_started,
            )
        platform_status = "completed"
        disposition = "domain_blocked"
    elif facts.terminal_node in {"build", "done"}:
        if facts.terminal_node == "build":
            if (
                facts.request.requested_completion_scope != "compile_only"
                or not isinstance(facts.variant_result, ProductBuildCompletionV1)
                or facts.variant_result.build_status != "success"
                or facts.variant_result.claims_real_build_success is not True
                or facts.variant_result.build_result_ref is None
                or facts.variant_result.workflow_checkpoint_ref is None
            ):
                raise ProductExecutionContractError(
                    "Compile product success lacks the exact Build/checkpoint chain.",
                    side_effects_may_have_started=effects_started,
                )
        if facts.terminal_node == "done":
            variant = facts.variant_result
            if (
                facts.request.requested_completion_scope != "runtime_workflow"
                or not isinstance(variant, ProductRuntimeCompletionV1)
                or variant.build_status != "success"
                or variant.runtime_status != "accepted"
                or variant.eval_verdict != "accepted"
                or variant.final_status != "accepted"
                or any(
                    ref is None
                    for ref in (
                        variant.build_result_ref,
                        variant.runtime_evidence_ref,
                        variant.eval_report_ref,
                        variant.workflow_summary_ref,
                    )
                )
                or variant.blocked_report_ref is not None
            ):
                raise ProductExecutionContractError(
                    "Runtime product success lacks the exact accepted terminal chain.",
                    side_effects_may_have_started=effects_started,
                )
        platform_status = "completed"
        disposition = "succeeded"
    else:
        raise ProductExecutionContractError(
            "Product execution lacks a legal terminal or platform failure.",
            side_effects_may_have_started=effects_started,
        )
    if facts.v2_delivery_checkpoint_ref is not None and disposition != "awaiting_delivery":
        raise ProductExecutionContractError("V2 delivery checkpoint requires pending delivery.", side_effects_may_have_started=effects_started)
    if platform_status == "completed":
        failure_stage = None
        if facts.feature_id is None:
            raise ProductExecutionContractError(
                "Completed product result requires an exact feature_id.",
                side_effects_may_have_started=effects_started,
            )
        if platform_blockers:
            raise ProductExecutionContractError(
                "Completed product result cannot carry platform blockers.",
                side_effects_may_have_started=effects_started,
            )
        if disposition == "succeeded" and facts.blockers:
            raise ProductExecutionContractError(
                "Succeeded product result cannot carry blockers.",
                side_effects_may_have_started=effects_started,
            )
        if disposition == "domain_blocked" and not domain_blockers:
            raise ProductExecutionContractError(
                "Domain-blocked product result requires a domain blocker.",
                side_effects_may_have_started=effects_started,
            )
    elif not platform_blockers:
        raise ProductExecutionContractError(
            "Platform-blocked or partial product result requires a platform blocker.",
            side_effects_may_have_started=effects_started,
        )
    return ProductExecutionResultV1(
        contract_version="1",
        run_id=facts.request.run_id,
        request_id=facts.request.request_id,
        feature_id=facts.feature_id,
        requested_completion_scope=facts.request.requested_completion_scope,
        platform_status=platform_status,
        execution_disposition=disposition,
        failure_stage=failure_stage,
        side_effects_may_have_started=effects_started,
        terminal_node=facts.terminal_node,
        store_attempts=facts.store_attempts,
        outputs=facts.outputs,
        effect_events=facts.effect_events,
        validation_facts=facts.validation_facts,
        blockers=facts.blockers,
        variant_result=facts.variant_result,
        task_agent_executions=facts.task_agent_executions,
        v2_delivery_checkpoint_ref=facts.v2_delivery_checkpoint_ref,
    )


@dataclass(frozen=True)
class ProductExecutionPreflightResultV1:
    contract_version: Literal["1"]
    preflight_status: Literal["ready", "blocked"]
    request: ProductExecutionRequestV1
    failure_stage: Literal["authorization", "admission", "execution_constraint"] | None
    blockers: tuple[ProductExecutionBlockerV1, ...]

    def __post_init__(self) -> None:
        if self.contract_version != "1" or not isinstance(
            self.request, ProductExecutionRequestV1
        ):
            raise ProductExecutionContractError(
                "Product preflight contract is invalid.",
                side_effects_may_have_started=False,
            )
        if self.preflight_status == "ready":
            if self.failure_stage is not None or self.blockers:
                raise ProductExecutionContractError(
                    "Ready product preflight cannot carry failure facts.",
                    side_effects_may_have_started=False,
                )
        elif self.preflight_status == "blocked":
            if self.failure_stage not in {"authorization", "admission", "execution_constraint"} or not self.blockers:
                raise ProductExecutionContractError(
                    "Blocked product preflight lacks exact failure facts.",
                    side_effects_may_have_started=False,
                )
        else:
            raise ProductExecutionContractError(
                "Product preflight status is unsupported.",
                side_effects_may_have_started=False,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "preflight_status": self.preflight_status,
            "request": self.request.to_dict(),
            "failure_stage": self.failure_stage,
            "blockers": [item.to_dict() for item in self.blockers],
        }
