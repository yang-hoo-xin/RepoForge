"""Public same-invocation product executor for the existing StateGraph."""

from __future__ import annotations

import hashlib
import copy
import json
import subprocess
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore, ArtifactWriteResult
from src.mod_agent.domain import DomainProfile, GTFOAdapter, UnsupportedDomainAdapter
from src.mod_agent.errors import LocalEntryError as LocalEntryError
from src.mod_agent.execution import ExactDiffApprovalRejectedError
from src.mod_agent.git.admission import GitAdmissionError, StrictGitAdmission
from src.mod_agent.model_admission import (
    local_model_admission_record,
    local_no_external_disclosure_record,
)
from src.mod_agent.product_execution import (
    PRODUCT_EXECUTION_DECLARATION,
    ProductBuildCompletionV1,
    ProductExecutionBlockerV1,
    ProductExecutionContractError,
    ProductExecutionEffectEventV1,
    ProductExecutionFactsV1,
    ProductExecutionGraphModeV1,
    ProductExecutionInputBindingV1,
    ProductExecutionOutputV1,
    ProductExecutionPreflightResultV1,
    ProductExecutionRequestV1,
    ProductExecutionResultV1,
    ProductExecutionStoreAttemptV1,
    ProductTaskAgentExecutionV1,
    ProductExecutionValidationDetailV1,
    ProductExecutionValidationFactsV1,
    ProductRuntimeCompletionV1,
    project_product_execution_result,
)
from src.mod_agent.runtime import (
    GTFORuntimeAutomationError,
    build_gtfo_runtime_capability_report,
    build_gtfo_stagger_acceptance_report,
    profile_from_settings,
    stagger_profile_from_settings,
)
from src.mod_agent.workflow.graph import build_graph
from src.mod_agent.workflow.state import create_initial_state

SOURCE_DISCLOSURE_DATA_CLASSES = (
    "natural_language_request",
    "retrieved_local_source_excerpts",
    "run_local_api_member_metadata_facts",
    "binding_registry",
    "side_effect_boundaries",
    "feature_target_behavior",
    "feature_acceptance_criteria",
    "feature_runtime_verification_plan",
)

# 为什么要有它：识别公开补丁和反馈类型不自动批准它们；默认披露请求仍是原来的SOURCE集合。
SOURCE_DISCLOSURE_ALLOWED_DATA_CLASSES = (*SOURCE_DISCLOSURE_DATA_CLASSES, "generated_patch", "verification_feedback")


PATCH_PROVIDER_PAYLOAD_DISCLOSURE_BINDINGS = (
    (("request",), "request", ("natural_language_request",)),
    (
        ("evidence", "binding_registry"),
        "edit_binding_contexts",
        (
            "retrieved_local_source_excerpts",
            "run_local_api_member_metadata_facts",
            "binding_registry",
        ),
    ),
    (
        ("evidence", "binding_registry", "required_change_units"),
        "supporting_evidence",
        (
            "retrieved_local_source_excerpts",
            "run_local_api_member_metadata_facts",
            "binding_registry",
        ),
    ),
    (("binding_registry",), "allowed_edit_binding_ids", ("binding_registry",)),
    (
        ("required_change_units",),
        "required_change_units",
        ("binding_registry",),
    ),
    (
        ("required_change_units",),
        "required_validation_by_binding",
        ("binding_registry",),
    ),
    (
        ("side_effect_boundaries",),
        "constraints",
        ("side_effect_boundaries",),
    ),
    (("target_behavior",), "target_behavior", ("feature_target_behavior",)),
    (
        ("acceptance_criteria",),
        "acceptance_criteria",
        ("feature_acceptance_criteria",),
    ),
    (
        ("runtime_verification_plan",),
        "runtime_verification_plan",
        ("feature_runtime_verification_plan",),
    ),
)


class _ProductExecutionAdmissionError(ProductExecutionContractError):
    """A well-formed product request failed semantic admission."""


def _sanitized_endpoint_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or parsed.hostname is None:
        raise LocalEntryError("Configured LLM base_url is not a valid absolute URL.")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    try:
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    except ValueError as exc:
        raise LocalEntryError("Configured LLM base_url has an invalid port.") from exc
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _verify_source_revision_binding(
    *,
    source: Path,
    snapshot: list[dict[str, Any]],
    snapshot_sha256: str,
    binding: Mapping[str, Any] | None,
    strict_git_admission: StrictGitAdmission | None = None,
    repository_profile_ref: Mapping[str, str] | None = None,
) -> None:
    if binding is None:
        projects = [
            path
            for path in source.rglob("*.csproj")
            if not any(part in {"bin", "obj", ".git"} for part in path.parts)
        ]
        if len(projects) > 1 and repository_profile_ref is not None:
            from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
            selected = CSharpRepositoryProfile(repository_profile_ref)
            selected.bind(source)
            declared = selected.data["project_path"]
            if declared not in {row["path"] for row in snapshot}:
                raise LocalEntryError("Declared repository project is absent from the source snapshot.")
        elif len(projects) > 1:
            raise LocalEntryError(
                "Multiple source projects require an explicit source revision binding."
            )
        return
    if binding.get("binding_version") == "product_source_revision_binding_v2":
        if strict_git_admission is not None:
            try:
                strict_git_admission.verify_source_revision_binding(
                    source=source,
                    snapshot=snapshot,
                    snapshot_sha256=snapshot_sha256,
                    binding=binding,
                )
            except GitAdmissionError as exc:
                raise LocalEntryError(str(exc)) from exc
            return
        _verify_generic_source_revision_binding(
            source=source,
            snapshot=snapshot,
            snapshot_sha256=snapshot_sha256,
            binding=binding,
        )
        return
    if binding.get("complete_snapshot_sha256") != snapshot_sha256:
        raise LocalEntryError(
            "Canonical source complete snapshot SHA256 does not match binding."
        )
    plugin_entry = next(
        (item for item in snapshot if item.get("path") == "Plugin.cs"), None
    )
    if plugin_entry is None:
        raise LocalEntryError("Canonical source binding requires Plugin.cs.")
    if binding.get("plugin_cs_sha256") != plugin_entry.get("sha256"):
        raise LocalEntryError(
            "Canonical source Plugin.cs SHA256 does not match binding."
        )
    if not binding.get("repository_head") or not binding.get("subtree_tree"):
        raise LocalEntryError(
            "Canonical source binding requires repository_head and subtree_tree."
        )


def _relative_binding_path(value: object, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LocalEntryError(f"Source binding {field} must be a non-empty POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise LocalEntryError(f"Source binding {field} escapes the repository.")
    if value != "." and path.as_posix() != value:
        raise LocalEntryError(f"Source binding {field} is not normalized.")
    return path


def _git_text(
    repository: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> str:
    resolved_repository = repository.resolve()
    try:
        completed = subprocess.run(
            (
                "git",
                "-c",
                f"safe.directory={resolved_repository.as_posix()}",
                "-C",
                str(resolved_repository),
                *arguments,
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalEntryError("Source revision Git query could not complete.") from exc
    if completed.returncode != 0:
        raise LocalEntryError(
            "Source revision Git query failed: " + str(completed.stderr or "").strip()
        )
    value = str(completed.stdout or "").strip()
    if not value and not allow_empty:
        raise LocalEntryError("Source revision Git query returned no identity.")
    return value


def _repository_root_for_source(source: Path) -> Path:
    resolved_source = source.resolve()
    for selected in (resolved_source, *resolved_source.parents):
        if (selected / ".git").exists():
            repository_root = selected.resolve()
            discovered_root = Path(
                _git_text(repository_root, "rev-parse", "--show-toplevel")
            ).resolve()
            if discovered_root != repository_root:
                raise LocalEntryError(
                    "Source revision Git root does not match its repository marker."
                )
            return repository_root
    raise LocalEntryError("Source revision binding repository root is missing.")


def _verify_generic_source_revision_binding(
    *,
    source: Path,
    snapshot: list[dict[str, Any]],
    snapshot_sha256: str,
    binding: Mapping[str, Any],
) -> None:
    required_fields = {
        "binding_version",
        "complete_snapshot_sha256",
        "repository_head",
        "repository_tree",
        "project_root",
        "project_file",
        "project_tree",
        "primary_sources",
    }
    if set(binding) != required_fields:
        raise LocalEntryError("Generic source revision binding shape is invalid.")
    if binding.get("complete_snapshot_sha256") != snapshot_sha256:
        raise LocalEntryError(
            "Canonical source complete snapshot SHA256 does not match binding."
        )
    resolved_source = source.resolve()
    repository_root = _repository_root_for_source(resolved_source)
    project_root = _relative_binding_path(binding.get("project_root"), "project_root")
    project_path = (
        repository_root
        if project_root.as_posix() == "."
        else (repository_root / Path(*project_root.parts)).resolve()
    )
    if project_path != resolved_source:
        raise LocalEntryError(
            "Source binding project_root does not equal the selected source root."
        )
    project_file = _relative_binding_path(binding.get("project_file"), "project_file")
    project_file_path = (repository_root / Path(*project_file.parts)).resolve()
    try:
        project_file_relative = project_file_path.relative_to(resolved_source).as_posix()
    except ValueError as exc:
        raise LocalEntryError("Source binding project_file escapes project_root.") from exc
    snapshot_by_path = {
        str(item.get("path")): item
        for item in snapshot
        if isinstance(item, dict) and item.get("path")
    }
    if (
        project_file_relative not in snapshot_by_path
        or project_file_path.suffix.casefold() != ".csproj"
        or not project_file_path.is_file()
    ):
        raise LocalEntryError("Source binding project_file is absent from snapshot.")

    repository_head = _git_text(repository_root, "rev-parse", "HEAD")
    repository_tree = _git_text(repository_root, "rev-parse", "HEAD^{tree}")
    project_tree = (
        repository_tree
        if project_root.as_posix() == "."
        else _git_text(repository_root, "rev-parse", f"HEAD:{project_root.as_posix()}")
    )
    if binding.get("repository_head") != repository_head:
        raise LocalEntryError("Source binding repository head drifted.")
    if binding.get("repository_tree") != repository_tree:
        raise LocalEntryError("Source binding repository tree drifted.")
    if binding.get("project_tree") != project_tree:
        raise LocalEntryError("Source binding project tree drifted.")
    if _git_text(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        project_root.as_posix(),
        allow_empty=True,
    ):
        raise LocalEntryError("Source binding project subtree is not clean.")

    tracked = set(
        _git_text(repository_root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )
    if project_file.as_posix() not in tracked:
        raise LocalEntryError("Source binding project_file is not tracked at HEAD.")
    primary_sources = binding.get("primary_sources")
    if not isinstance(primary_sources, list) or not primary_sources:
        raise LocalEntryError("Source binding requires at least one primary source.")
    seen_sources: set[str] = set()
    for item in primary_sources:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise LocalEntryError("Source binding primary source shape is invalid.")
        selected = _relative_binding_path(item.get("path"), "primary_sources.path")
        selected_text = selected.as_posix()
        if selected_text in seen_sources:
            raise LocalEntryError("Source binding primary source path is duplicated.")
        seen_sources.add(selected_text)
        if selected_text not in tracked:
            raise LocalEntryError("Source binding primary source is not tracked at HEAD.")
        selected_path = (repository_root / Path(*selected.parts)).resolve()
        try:
            selected_relative = selected_path.relative_to(resolved_source).as_posix()
        except ValueError as exc:
            raise LocalEntryError("Source binding primary source escapes project_root.") from exc
        snapshot_entry = snapshot_by_path.get(selected_relative)
        if snapshot_entry is None:
            raise LocalEntryError("Source binding primary source is absent from snapshot.")
        if item.get("sha256") != snapshot_entry.get("sha256"):
            raise LocalEntryError("Source binding primary source SHA256 changed.")


def _source_disclosure_record(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    approved_at: str,
    approval_id: str | None = None,
    provider_adapter: str | None = None,
    destination_url: str | None = None,
    requested_data_classes: tuple[str, ...] = SOURCE_DISCLOSURE_DATA_CLASSES,
    approved_data_classes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    requested = _validated_disclosure_data_classes(
        requested_data_classes,
        field_name="requested_data_classes",
    )
    approved = _validated_disclosure_data_classes(
        requested if approved_data_classes is None else approved_data_classes,
        field_name="approved_data_classes",
    )
    if not set(approved).issubset(requested):
        raise LocalEntryError(
            "Source disclosure approved_data_classes must be a subset of "
            "requested_data_classes."
        )
    return {
        "approved": True,
        "approved_at": approved_at,
        "approval_id": approval_id,
        "approval_actor_type": "local_cli_invoker",
        "consent_mechanism": "--allow-source-disclosure",
        "provider_adapter": provider_adapter,
        "destination": {
            "provider": provider,
            "model": model,
            "base_url": _sanitized_endpoint_url(base_url),
            "request_url": _sanitized_endpoint_url(destination_url),
        },
        "approved_data_classes": list(approved),
        "requested_data_classes": list(requested),
        "scope": "this_run_only",
    }


def _validated_disclosure_data_classes(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise LocalEntryError(f"Source disclosure {field_name} must be a tuple.")
    if any(not isinstance(item, str) for item in values):
        raise LocalEntryError(
            f"Source disclosure {field_name} must contain only strings."
        )
    if len(set(values)) != len(values):
        raise LocalEntryError(
            f"Source disclosure {field_name} contains duplicate data classes."
        )
    unknown = [
        item for item in values if item not in SOURCE_DISCLOSURE_ALLOWED_DATA_CLASSES
    ]
    if unknown:
        raise LocalEntryError(
            f"Source disclosure {field_name} contains unknown data classes: "
            + ", ".join(unknown)
        )
    canonical = tuple(
        item for item in SOURCE_DISCLOSURE_ALLOWED_DATA_CLASSES if item in values
    )
    if values != canonical:
        raise LocalEntryError(
            f"Source disclosure {field_name} is not in canonical order."
        )
    return values


def _local_no_external_disclosure_record() -> dict[str, Any]:
    return local_no_external_disclosure_record()


def _local_model_admission_record(
    *, model_identity: Mapping[str, Any], provider_adapter: str
) -> dict[str, Any]:
    return local_model_admission_record(
        model_identity=model_identity,
        provider_adapter=provider_adapter,
    )


def _external_model_admission_record(
    source_disclosure: Mapping[str, Any],
) -> dict[str, Any]:
    destination = source_disclosure.get("destination")
    destination_data = destination if isinstance(destination, dict) else {}
    return {
        "provider": destination_data.get("provider"),
        "model": destination_data.get("model"),
        "model_digest": None,
        "provider_adapter": source_disclosure.get("provider_adapter"),
        "execution_locality": "external",
        "base_url": destination_data.get("base_url"),
        "destination": destination_data.get("request_url"),
        "external_disclosure_occurred": True,
        "requested_data_classes": list(
            source_disclosure.get("requested_data_classes") or []
        ),
        "approved_data_classes": list(
            source_disclosure.get("approved_data_classes") or []
        ),
        "api_key_required": True,
        "api_key_read": True,
        "api_key_persisted": False,
        "automatic_provider_fallback": False,
        "automatic_external_retry": False,
        "data_processing_location": "external_provider",
    }


def _runtime_admission_record(
    settings: Any,
    *,
    run_gtfo_runtime: bool,
    allow_game_launch: bool,
    runtime_plugin_destination: str | Path | None,
    allow_game_directory_write: bool,
    acceptance_criteria: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Bind one run's requested Runtime boundary before any mutable action."""

    runtime = getattr(settings, "runtime", None)
    gtfo = getattr(runtime, "gtfo", None)
    settings_enabled = bool(gtfo is not None and getattr(gtfo, "enabled", False))
    if allow_game_launch and not run_gtfo_runtime:
        raise LocalEntryError(
            "--allow-game-launch requires --run-gtfo-runtime for the same run."
        )
    if (runtime_plugin_destination is not None or allow_game_directory_write) and not run_gtfo_runtime:
        raise LocalEntryError(
            "Runtime plugin deployment requires --run-gtfo-runtime for the same run."
        )
    if not run_gtfo_runtime:
        return {
            "requested": False,
            "evidence_source": None,
            "allow_game_launch": False,
            "authorization_scope": None,
            "settings_enabled": settings_enabled,
            "profile": None,
            "capability_report": None,
            "plugin_deployment": {
                "requested": False,
                "destination_path": None,
                "allow_game_directory_write": False,
                "restore_after_runtime": False,
                "performed_by_run_request": False,
                "verified_deployed_artifact_hash": False,
            },
        }
    if not settings_enabled:
        raise LocalEntryError(
            "--run-gtfo-runtime requires runtime.gtfo.enabled=true in the selected settings file."
        )
    if runtime_plugin_destination is None:
        raise LocalEntryError(
            "--run-gtfo-runtime requires --runtime-plugin-destination so the physical DLL loaded by GTFO can be bound to this run's Build."
        )
    if not allow_game_directory_write:
        raise LocalEntryError(
            "--run-gtfo-runtime requires --allow-game-directory-write for run-scoped plugin deployment and restoration."
        )
    try:
        profile = profile_from_settings(settings)
        capability_report = build_gtfo_runtime_capability_report(
            profile,
            allow_game_launch=allow_game_launch,
            acceptance_criteria=acceptance_criteria,
        )
    except GTFORuntimeAutomationError as exc:
        raise LocalEntryError(f"GTFO Runtime admission failed: {exc}") from exc
    if capability_report["status"] != "ready":
        raise LocalEntryError(
            "GTFO Runtime admission blocked: "
            + "; ".join(str(item) for item in capability_report["blockers"])
        )
    destination_input = Path(runtime_plugin_destination)
    if not destination_input.is_absolute():
        raise LocalEntryError(
            "--runtime-plugin-destination must be an absolute .dll path."
        )
    destination = destination_input.resolve()
    working_directory = Path(profile.working_directory).resolve()
    try:
        destination.relative_to(working_directory)
    except ValueError as exc:
        raise LocalEntryError(
            "--runtime-plugin-destination must remain inside the configured runtime.gtfo.working_directory."
        ) from exc
    if destination.suffix.lower() != ".dll":
        raise LocalEntryError(
            "--runtime-plugin-destination must identify one .dll file."
        )
    if not destination.parent.is_dir():
        raise LocalEntryError(
            f"Runtime plugin destination directory does not exist: {destination.parent}"
        )
    if profile.terminate_after_capture is not True:
        raise LocalEntryError(
            "Run-scoped plugin deployment requires runtime.gtfo.terminate_after_capture=true before launch."
        )
    stagger_profile = stagger_profile_from_settings(settings)
    stagger_acceptance = None
    if stagger_profile is not None:
        if profile.operator_completion_required is not True:
            raise LocalEntryError(
                "Two-process stagger Runtime requires runtime.gtfo.operator_completion_required=true."
            )
        stagger_acceptance = build_gtfo_stagger_acceptance_report(
            stagger_profile,
            working_directory=profile.working_directory,
            acceptance_criteria=acceptance_criteria,
        )
        if stagger_acceptance["status"] != "ready":
            raise LocalEntryError(
                "GTFO stagger Runtime admission blocked: "
                + "; ".join(str(item) for item in stagger_acceptance["blockers"])
            )
    return {
        "requested": True,
        "evidence_source": "gtfo_real_runtime_automation",
        "capability_scope": "runtime_harness_preflight",
        "allow_game_launch": allow_game_launch,
        "authorization_scope": "this_run_only" if allow_game_launch else None,
        "settings_enabled": True,
        "profile": {
            "profile_id": profile.profile_id,
            "game_executable": profile.game_executable,
            "working_directory": profile.working_directory,
            "launch_args": list(profile.launch_args),
            "log_paths": list(profile.log_paths),
            "screenshot_directory": profile.screenshot_directory,
            "screenshot_timeout_seconds": profile.screenshot_timeout_seconds,
            "startup_markers": list(profile.startup_markers),
            "acceptance_markers": dict(profile.acceptance_markers),
            "timeout_seconds": profile.timeout_seconds,
            "poll_interval_seconds": profile.poll_interval_seconds,
            "output_limit_chars": profile.output_limit_chars,
            "terminate_after_capture": profile.terminate_after_capture,
            "operator_completion_required": profile.operator_completion_required,
            "operator_completion_timeout_seconds": profile.operator_completion_timeout_seconds,
        },
        "capability_report": capability_report,
        "stagger_acceptance": stagger_acceptance,
        "plugin_deployment": {
            "requested": True,
            "destination_path": str(destination),
            "allow_game_directory_write": True,
            "restore_after_runtime": True,
            "performed_by_run_request": True,
            "verified_deployed_artifact_hash": False,
            "destination_existed_at_admission": destination.is_file(),
            "destination_sha256_at_admission": (
                hashlib.sha256(destination.read_bytes()).hexdigest()
                if destination.is_file()
                else None
            ),
        },
    }


@dataclass(frozen=True)
class ProductExecutionAdaptersV1:
    """Ephemeral effect adapters; callables never enter product identity or result."""

    evidence_pack_factory: Callable[..., Any]
    patch_change_factory: Callable[..., Any]
    dry_run_factory: Callable[..., Any]
    isolated_apply_factory: Callable[..., Any]
    build_result_factory: Callable[..., Any]
    runtime_evidence_factory: Callable[..., Any] | None
    compiled_artifact_verifier: Callable[..., Any] | None
    source_change_verifier: Callable[..., Any] | None
    proposal_review_context_factory: Callable[..., Any] | None = None
    planning_target_selection_factory: Callable[..., Any] | None = None
    product_patch_upstream_execution: Any = None
    patch_upstream_evidence_pack_factory: Callable[..., Any] | None = None
    patch_upstream_planning_target_selection_factory: Callable[..., Any] | None = None
    initial_findings_reassessment_factory: Callable[..., Any] | None = None
    domain_adapter: Any = None
    task_agent_execution: Any = None

    def validate_for(self, request: ProductExecutionRequestV1) -> None:
        if self.task_agent_execution is not None:
            from src.mod_agent.task_agent_execution import TaskAgentExecution
            from src.mod_agent.model_admission import ModelAdmissionError, validate_ollama_model_identity, validate_ollama_context_window

            if (not isinstance(self.task_agent_execution, TaskAgentExecution)
                    or self.task_agent_execution is not self.patch_change_factory
                    or self.product_patch_upstream_execution is not None):
                raise ProductExecutionContractError("Task execution must be the explicit trusted Patch owner, not a legacy worker.",
                    side_effects_may_have_started=False)
            owner = self.task_agent_execution
            try:
                workspace = Path(request.isolated_workspace_root).resolve(strict=True)
                target = Path(request.isolated_target_root).resolve(strict=True)
                if (owner.workspace_root.resolve(strict=True) != workspace
                        or owner.store.root.resolve() != Path(request.artifact_store_root).resolve()
                        or not target.is_relative_to(workspace) or not owner.allowed_read_roots or not owner.allowed_edit_paths):
                    raise ValueError("Task owner does not match the isolated Product workspace and Store.")
                for root in owner.allowed_read_roots:
                    selected = Path(root).resolve(strict=True)
                    if not selected.is_dir() or not selected.is_relative_to(target):
                        raise ValueError("Task read scope escapes the isolated target.")
                for relative in owner.allowed_edit_paths:
                    selected = PurePosixPath(relative)
                    if selected.is_absolute() or ".." in selected.parts or "\\" in relative:
                        raise ValueError("Task edit scope is not an exact relative path.")
                    resolved = (workspace / relative).resolve(strict=True)
                    if not resolved.is_file() or not resolved.is_relative_to(target):
                        raise ValueError("Task edit scope escapes the isolated target.")
                binding = next(ProductExecutionInputBindingV1.from_dict(item.to_dict()).canonical_payload
                    for item in request.input_bindings if item.binding_kind == "model_admission")
                from src.mod_agent.external_patch import (
                    PATCH_FEEDBACK_DATA_CLASSES,
                    ExternalOpenAIResponsesProfile,
                    _validate_external_disclosure,
                    validate_openai_task_identity,
                )
                if isinstance(owner.profile, ExternalOpenAIResponsesProfile):
                    if getattr(owner, "protocol", None) != "v2":
                        raise ValueError("External task execution requires the admitted v2 owner.")
                    validate_openai_task_identity(owner.profile, owner.expected_model_identity)
                    disclosure = next(ProductExecutionInputBindingV1.from_dict(item.to_dict()).canonical_payload
                        for item in request.input_bindings if item.binding_kind == "source_disclosure")
                    if disclosure != owner.source_disclosure or binding != _external_model_admission_record(disclosure):
                        raise ValueError("Task model or disclosure differs from the admitted Product input.")
                    _validate_external_disclosure(owner.profile, disclosure,
                        (*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES))
                else:
                    validate_ollama_model_identity(owner.expected_model_identity,
                        expected_base_url=owner.profile.base_url, expected_model=owner.profile.model)
                    validate_ollama_context_window(owner.expected_model_identity, requested_context_window=owner.profile.context_window)
                    if binding != local_model_admission_record(model_identity=owner.expected_model_identity,
                            provider_adapter=binding.get("provider_adapter", "")):
                        raise ValueError("Task model differs from the admitted Product model.")
            except (ValueError, TypeError, KeyError, OSError, AttributeError, ModelAdmissionError) as error:
                raise ProductExecutionContractError("Task execution input admission failed: " + str(error),
                    side_effects_may_have_started=False) from error
        if self.domain_adapter is not None:
            # Scope: trusted in-process injection, never a domain name or model-authored callback payload.
            # 为什么要有它：共享C#接入必须显式绑定领域、源码范围和验证语义，默认GTFO/unsupported路径不能被悄悄替换。
            if (getattr(self.domain_adapter, "domain_name", None) != request.target_domain
                    or any(not callable(getattr(self.domain_adapter, name, None)) for name in
                           ("identify_project", "build_feature_spec", "validate_product_request"))):
                raise ProductExecutionContractError("Explicit product domain adapter is invalid.", side_effects_may_have_started=False)
            try:
                self.domain_adapter.validate_product_request(request)
                profile = self.domain_adapter.identify_project(request.isolated_workspace_root)
                if not isinstance(profile, DomainProfile) or profile.project_root != request.isolated_workspace_root:
                    raise ValueError("Domain profile does not identify the admitted workspace.")
            except (ValueError, TypeError) as error:
                raise ProductExecutionContractError("Explicit product domain admission failed: " + str(error),
                    side_effects_may_have_started=False) from error
        required = (
            "evidence_pack_factory",
            "patch_change_factory",
            "dry_run_factory",
            "isolated_apply_factory",
            "build_result_factory",
        )
        for field in required:
            if not callable(getattr(self, field)):
                raise ProductExecutionContractError(
                    f"Product adapter {field} must be callable.",
                    side_effects_may_have_started=False,
                )
        for field in ("compiled_artifact_verifier", "source_change_verifier"):
            selected = getattr(self, field)
            if selected is not None and not callable(selected):
                raise ProductExecutionContractError(
                    f"Product adapter {field} must be callable or null.",
                    side_effects_may_have_started=False,
                )
        if self.proposal_review_context_factory is not None and not callable(
            self.proposal_review_context_factory
        ):
            raise ProductExecutionContractError(
                "Product adapter proposal_review_context_factory must be callable "
                "or null.",
                side_effects_may_have_started=False,
            )
        if self.planning_target_selection_factory is not None and not callable(
            self.planning_target_selection_factory
        ):
            raise ProductExecutionContractError(
                "Product adapter planning_target_selection_factory must be "
                "callable or null.",
                side_effects_may_have_started=False,
            )
        for field in ("patch_upstream_evidence_pack_factory", "patch_upstream_planning_target_selection_factory",
                      "initial_findings_reassessment_factory"):
            selected = getattr(self, field)
            if selected is not None and not callable(selected):
                raise ProductExecutionContractError(
                    f"Product adapter {field} must be callable or null.",
                    side_effects_may_have_started=False,
                )
        if self.product_patch_upstream_execution is not None:
            from src.mod_agent.request_patch_execution import RequestPatchExecution

            if (not isinstance(self.product_patch_upstream_execution, RequestPatchExecution)
                    or self.product_patch_upstream_execution is not self.patch_change_factory):
                raise ProductExecutionContractError(
                    "Product upstream execution must be the current product Patch worker owner.",
                    side_effects_may_have_started=False,
                )
        runtime_required = request.requested_completion_scope == "runtime_workflow"
        if runtime_required is not callable(self.runtime_evidence_factory):
            raise _ProductExecutionAdmissionError(
                "Product Runtime adapter topology disagrees with completion scope.",
                side_effects_may_have_started=False,
            )

    def topology(self) -> tuple[tuple[str, bool], ...]:
        original = tuple(
            (field, callable(getattr(self, field)))
            for field in (
                "evidence_pack_factory",
                "planning_target_selection_factory",
                "product_patch_upstream_execution",
                "patch_upstream_evidence_pack_factory",
                "patch_upstream_planning_target_selection_factory",
                "initial_findings_reassessment_factory",
                "patch_change_factory",
                "dry_run_factory",
                "isolated_apply_factory",
                "build_result_factory",
                "runtime_evidence_factory",
                "compiled_artifact_verifier",
                "source_change_verifier",
            )
        )
        return original + ((("domain_adapter", True),) if self.domain_adapter is not None else ()) + (
            (("task_agent_execution", True),) if self.task_agent_execution is not None else ())


@dataclass(frozen=True)
class _OwnerBoundRuntimeEvidenceFactoryV1:
    factory: Callable[..., Any]
    settings: Any
    allow_game_launch: bool
    runtime_plugin_destination: str | Path
    allow_game_directory_write: bool
    expected_runtime_admission: Mapping[str, Any]

    def validate_runtime_admission(self, observed: Mapping[str, Any]) -> None:
        expected = _runtime_admission_record(
            self.settings,
            run_gtfo_runtime=True,
            allow_game_launch=self.allow_game_launch,
            runtime_plugin_destination=self.runtime_plugin_destination,
            allow_game_directory_write=self.allow_game_directory_write,
        )
        if expected != dict(self.expected_runtime_admission) or observed != expected:
            raise LocalEntryError(
                "Runtime admission changed after its owner produced it."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_runtime_admission(self.expected_runtime_admission)
        return self.factory(*args, **kwargs)


def _bind_owner_validated_runtime_evidence_factory(
    factory: Callable[..., Any],
    *,
    settings: Any,
    allow_game_launch: bool,
    runtime_plugin_destination: str | Path,
    allow_game_directory_write: bool,
    expected_runtime_admission: Mapping[str, Any],
) -> Callable[..., Any]:
    """Bind the Runtime effect to the same owner inputs used by run-request."""

    return _OwnerBoundRuntimeEvidenceFactoryV1(
        factory=factory,
        settings=settings,
        allow_game_launch=allow_game_launch,
        runtime_plugin_destination=runtime_plugin_destination,
        allow_game_directory_write=allow_game_directory_write,
        expected_runtime_admission=dict(expected_runtime_admission),
    )


class ProductExecutionInputBindingResolverV1:
    """Validate wrapper integrity without copying delegated record schemas."""

    def validate(
        self,
        request: ProductExecutionRequestV1,
        adapters: ProductExecutionAdaptersV1,
        *,
        strict_git_admission: StrictGitAdmission | None = None,
    ) -> None:
        if not isinstance(request, ProductExecutionRequestV1):
            raise ProductExecutionContractError(
                "Product request must be ProductExecutionRequestV1.",
                side_effects_may_have_started=False,
            )
        bindings = {
            item.binding_kind: ProductExecutionInputBindingV1.from_dict(
                item.to_dict()
            ).canonical_payload
            for item in request.input_bindings
        }
        try:
            source_snapshot_value = source_snapshot(Path(request.source_root))
            _verify_source_revision_binding(
                source=Path(request.source_root),
                snapshot=source_snapshot_value,
                snapshot_sha256=source_snapshot_sha256(source_snapshot_value),
                binding=bindings["source_revision_binding"],
                strict_git_admission=strict_git_admission,
                repository_profile_ref=(bindings.get("repository_build_and_test") or {}).get("configuration_ref"),
            )

            disclosure = bindings["source_disclosure"]
            model = bindings["model_admission"]
            if not isinstance(disclosure, Mapping) or not isinstance(model, Mapping):
                raise LocalEntryError(
                    "Product disclosure/model bindings must be owner-produced records."
                )
            if disclosure.get("occurred") is False:
                expected_disclosure = _local_no_external_disclosure_record()
            else:
                destination = disclosure.get("destination")
                if not isinstance(destination, Mapping):
                    raise LocalEntryError(
                        "External source disclosure destination is missing."
                    )
                expected_disclosure = _source_disclosure_record(
                    provider=str(destination.get("provider") or ""),
                    model=str(destination.get("model") or ""),
                    base_url=destination.get("base_url"),
                    approved_at=str(disclosure.get("approved_at") or ""),
                    approval_id=disclosure.get("approval_id"),
                    provider_adapter=disclosure.get("provider_adapter"),
                    destination_url=destination.get("request_url"),
                    requested_data_classes=tuple(
                        disclosure.get("requested_data_classes") or ()
                    ),
                    approved_data_classes=tuple(
                        disclosure.get("approved_data_classes") or ()
                    ),
                )
            if disclosure != expected_disclosure:
                raise LocalEntryError(
                    "Source disclosure record changed after its owner produced it."
                )

            if model.get("execution_locality") == "local":
                expected_model = _local_model_admission_record(
                    model_identity={
                        "name": model.get("model"),
                        "digest": model.get("model_digest"),
                        "base_url": model.get("base_url"),
                    },
                    provider_adapter=str(model.get("provider_adapter") or ""),
                )
            else:
                expected_model = _external_model_admission_record(disclosure)
            if model != expected_model:
                raise LocalEntryError(
                    "Model admission record changed after its owner produced it."
                )

            runtime = bindings["runtime_admission"]
            if not isinstance(runtime, Mapping):
                raise LocalEntryError(
                    "Runtime admission binding must be an owner-produced record."
                )
            runtime_requested = request.requested_completion_scope == "runtime_workflow"
            if runtime_requested:
                validator = getattr(
                    adapters.runtime_evidence_factory,
                    "validate_runtime_admission",
                    None,
                )
                if runtime.get("requested") is not True or not callable(validator):
                    raise LocalEntryError(
                        "Runtime admission disagrees with product Runtime scope."
                    )
                validator(runtime)
            else:
                if runtime.get("requested") is not False or adapters.runtime_evidence_factory is not None:
                    raise LocalEntryError(
                        "No-Runtime admission disagrees with compile-only scope."
                    )
            if "repository_build_and_test" in bindings:
                from src.mod_agent.domain.csharp_repository_delivery import (
                    validate_repository_binding,
                )
                validate_repository_binding(bindings["repository_build_and_test"], request.isolated_workspace_root)
        except (KeyError, TypeError, ValueError, LocalEntryError) as exc:
            raise _ProductExecutionAdmissionError(
                str(exc) or type(exc).__name__,
                side_effects_may_have_started=False,
            ) from exc


class ProductExecutionPresentationError(RuntimeError):
    """Presentation failed after the product machine result was frozen."""

    def __init__(
        self,
        *,
        product_execution_result: ProductExecutionResultV1,
        failure_class: str,
        failure_message: str,
        sink: str,
    ) -> None:
        super().__init__(failure_message)
        self.product_execution_result = product_execution_result
        self.failure_class = failure_class
        self.failure_message = failure_message
        self.sink = sink


class ProductApprovalDeferred(RuntimeError):  # noqa: N818 - intentional control pause, not failure
    """An intentional approval pause, not a failed or rejected typed terminal."""

    def __init__(self, *, request, checkpoint_ref, dry_run_ref, dry_run, outputs, effect_events):
        super().__init__("The saved exact diff awaits human semantic approval.")
        self.request = request
        self.checkpoint_ref = checkpoint_ref
        self.dry_run_ref = dry_run_ref
        self.outputs = outputs
        self.effect_events = effect_events
        self._dry_run = dry_run

    def to_dict(self):
        def reference(ref):
            return {name: getattr(ref, name) for name in ("feature_id", "artifact_type", "path", "artifact_hash")}

        return {
            "status": "awaiting_approval", "run_id": self.request.run_id,
            "request_id": self.request.request_id, "feature_id": self.checkpoint_ref.feature_id,
            "isolated_workspace": self.request.isolated_workspace_root,
            "workflow_checkpoint": reference(self.checkpoint_ref), "dry_run_result": reference(self.dry_run_ref),
            "diff_hash": self._dry_run["diff_hash"], "unified_diff": self._dry_run["unified_diff"],
            "unified_diff_sha256": self._dry_run["unified_diff_sha256"],
            "human_semantic_decision_binding": self._dry_run["human_semantic_decision_binding"],
            "product_source_writes_performed": False, "builds_performed": 0,
            "runtime_executions_performed": 0, "approval_recorded": False, "stopped_before": "isolated_apply",
        }


class _ProductStoreStopped(RuntimeError):  # noqa: N818 - existing control-flow signal
    pass


class _ProductStoreRecorder:
    """Convert one Graph invocation's writes to authoritative C5 attempts."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self.attempts: list[ProductExecutionStoreAttemptV1] = []
        self.outputs: list[ProductExecutionOutputV1] = []
        self.effect_events: list[ProductExecutionEffectEventV1] = []
        self._repair_attempt = 0
        self._role_owner: dict[str, str] = {}
        for node, role in PRODUCT_EXECUTION_DECLARATION.node_store_roles:
            self._role_owner.setdefault(role, node)
        self._phase_by_node = PRODUCT_EXECUTION_DECLARATION.phase_by_node
        self._repair_source_scope: ArtifactRef | None = None
        self._repair_source_roles: list[str] = []
        self.uncommitted_blocked_causes: tuple[ProductExecutionBlockerV1, ...] = ()
        self.task_agent_executions: list[ProductTaskAgentExecutionV1] = []
        self._task_agent_owner = None
        self._task_agent_start = None

    def task_agent_store_attempts(self, owner):
        if owner is not self._task_agent_owner or self._task_agent_start is None:
            raise ProductExecutionContractError("Task Store observations require the active trusted owner.",
                side_effects_may_have_started=True)
        return [row.to_dict() for row in self.attempts[self._task_agent_start:]]

    @contextmanager
    def task_agent_scope(self, owner, state):
        from src.mod_agent.task_agent_execution import TaskAgentExecution

        if (not isinstance(owner, TaskAgentExecution) or self._task_agent_owner is not None
                or self._repair_source_scope is not None):
            raise ProductExecutionContractError("Task Store scope requires one trusted, nonnested execution.",
                side_effects_may_have_started=False)
        initial = {role + "_ref": {"feature_id": state["feature_id"], "artifact_type": role,
            "path": state["artifacts"][role], "artifact_hash": state["artifact_hashes"][role]}
            for role in (("feature_spec",) if getattr(owner, "protocol", None) == "v2"
                else ("feature_spec", "evidence_pack", "implementation_plan"))}
        self._task_agent_owner, self._task_agent_start = owner, len(self.attempts)
        start_sequence = len(self.attempts) + sum(row.origin == "reused_input" for row in self.outputs)
        outer_attempt = self._repair_attempt
        try:
            yield
        finally:
            try:
                outcome = owner.execution_receipt()
                sequences = tuple(row.sequence for row in self.attempts[self._task_agent_start:])
                if outcome is not None:
                    self.task_agent_executions.append(ProductTaskAgentExecutionV1(
                        contract_version="1", run_id=state["run_id"], request_id=state["request_id"],
                        repair_attempt=outer_attempt, store_start_sequence=start_sequence, initial_context=initial,
                        outcome=copy.deepcopy(outcome.to_dict()), store_sequences=sequences))
                elif sequences:
                    raise ProductExecutionContractError("Task Store effects have no retained execution owner.",
                        side_effects_may_have_started=True)
            finally:
                self._task_agent_owner, self._task_agent_start = None, None

    def _validate_task_agent_write(self, kwargs):
        if self._task_agent_owner is not None and kwargs.get("artifact_type") not in {
                "repair_plan", "evidence_pack", "implementation_plan"}:
            raise ProductExecutionContractError("Task technical scope cannot write delivery artifacts.",
                side_effects_may_have_started=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def reuse_interrupted_inputs(self, admitted: Mapping[str, Any], selection: Mapping[str, Any], *, pretransport=False) -> None:
        """Retain prior identities without inventing Store writes or effects."""
        # 为什么要有它：续接结果需要原输入身份，但本次没有重新生成它们，不能污染调用量和副作用记录。
        if self.outputs or self.attempts or self.effect_events:
            raise ProductExecutionContractError("Reused inputs must precede current execution.",
                                                side_effects_may_have_started=False)
        for role, phase in (("feature_spec", "planning"), ("evidence_pack", "evidence"),
                            ("implementation_plan", "planning")):
            self.outputs.append(ProductExecutionOutputV1(contract_version="1", sequence=len(self.outputs),
                phase=phase, repair_attempt=0, node_name=("pretransport_patch_admission" if pretransport else "interrupted_patch_admission"), artifact_role=role,
                artifact_ref=ArtifactRef(**admitted["selected_artifacts"][role]), origin="reused_input",
                reuse_selection=copy.deepcopy(selection)))

    @property
    def current_repair_attempt(self) -> int:
        return self._repair_attempt

    @contextmanager
    def repair_source_revision_scope(self, repair_ref: ArtifactRef):
        """Record physical revisions written by the existing Repair Graph node."""
        if (self._repair_source_scope is not None or not self.attempts
                or self.attempts[-1].node_name != "create_repair_plan"
                or self.attempts[-1].artifact_role != "repair_plan"
                or self.attempts[-1].store_result.artifact_ref != repair_ref
                or self.attempts[-1].store_result.outcome != "committed"
                or self._repair_attempt < 1):
            raise ProductExecutionContractError("Repair source revisions require the just-committed Repair owner.",
                                                side_effects_may_have_started=True)
        envelope = self._store.semantic_read_immutable(repair_ref)
        repair = envelope["data"]
        if (envelope["status"] != "success" or repair.get("failure_source") != "build"
                or repair.get("blockers") or repair.get("allowed_next_stage") != "generate_patch_proposal"):
            raise ProductExecutionContractError("Repair source revision scope has no admitted Build repair.",
                                                side_effects_may_have_started=True)
        self._repair_source_scope, self._repair_source_roles = repair_ref, []
        try:
            yield
            if self._repair_source_roles != ["evidence_pack", "implementation_plan"]:
                raise ProductExecutionContractError("Repair source revision scope must publish Evidence then Plan once.",
                                                    side_effects_may_have_started=True)
        finally:
            self._repair_source_scope, self._repair_source_roles = None, []

    def _validate_repair_source_write(self, kwargs: Mapping[str, Any]) -> None:
        ref = self._repair_source_scope
        if ref is None:
            return
        expected = ("evidence_pack", "implementation_plan")
        index = len(self._repair_source_roles)
        role = kwargs.get("artifact_type")
        if (index >= len(expected) or role != expected[index] or kwargs.get("feature_id") != ref.feature_id
                or f"{ref.path}#sha256={ref.artifact_hash}" not in kwargs.get("trace_refs", [])):
            raise ProductExecutionContractError("Repair source Store roles or exact owner binding changed.",
                                                side_effects_may_have_started=True)
        if role == "evidence_pack":
            provenance = kwargs["payload"].get("provenance_validation", {}).get("repair_source_reanchoring", {})
            if (provenance.get("repair_plan_ref") != {"path": ref.path, "artifact_hash": ref.artifact_hash}
                    or provenance.get("findings_reexecuted") is not False
                    or provenance.get("planning_model_reexecuted") is not False
                    or provenance.get("approval_inherited") is not False):
                raise ProductExecutionContractError("Physical repair revision must not claim a new semantic model decision.",
                                                    side_effects_may_have_started=True)
        else:
            evidence_ref = self.attempts[-1].store_result.artifact_ref
            upstream_refs = kwargs.get("upstream_refs", [])
            if evidence_ref is None or evidence_ref not in upstream_refs:
                raise ProductExecutionContractError("Repair Plan must bind its just-observed Evidence revision.",
                                                    side_effects_may_have_started=True)

    def _record(self, result: ArtifactWriteResult, *, payload: Mapping[str, Any]) -> None:
        role = result.artifact_type
        node = ("generate_patch_proposal" if self._task_agent_owner is not None else
                "create_repair_plan" if self._repair_source_scope is not None else self._role_owner.get(role))
        if node is None:
            raise ProductExecutionContractError(
                f"Product Graph attempted undeclared Store role {role}.",
                side_effects_may_have_started=True,
            )
        attempt_number = self._repair_attempt
        if role == "repair_plan" and self._task_agent_owner is None:
            payload_attempt = payload.get("retry_count")
            attempt_number = (
                payload_attempt
                if type(payload_attempt) is int
                else self._repair_attempt + 1
            )
        attempt = ProductExecutionStoreAttemptV1(
            contract_version="1",
            sequence=len(self.attempts) + sum(row.origin == "reused_input" for row in self.outputs),
            phase=self._phase_by_node[node],
            repair_attempt=attempt_number,
            node_name=node,
            artifact_role=role,
            store_result=result,
        )
        self.attempts.append(attempt)
        if self._repair_source_scope is not None:
            self._repair_source_roles.append(role)
        if result.outcome == "committed":
            if result.artifact_ref is None:
                raise ProductExecutionContractError(
                    "Committed Store result is missing ArtifactRef.",
                    side_effects_may_have_started=True,
                )
            self.outputs.append(
                ProductExecutionOutputV1(
                    contract_version="1",
                    sequence=attempt.sequence,
                    phase=attempt.phase,
                    repair_attempt=attempt.repair_attempt,
                    node_name=attempt.node_name,
                    artifact_role=attempt.artifact_role,
                    artifact_ref=result.artifact_ref,
                )
            )
            if role == "repair_plan" and self._task_agent_owner is None:
                self._repair_attempt = attempt_number
        else:
            if role == "blocked_report" and isinstance(payload.get("blockers"), list) and payload["blockers"]:
                # Preserve the Graph's actual submitted reasons even when C5
                # cannot commit its report. These are owner facts, not an output
                # Artifact or a claim that the failed Store write succeeded.
                self.uncommitted_blocked_causes = _domain_blockers(payload)
            raise _ProductStoreStopped(
                f"{role} did not commit: {result.outcome}"
            )

    def write(self, **kwargs: Any) -> ArtifactRef:
        self._validate_task_agent_write(kwargs)
        self._validate_repair_source_write(kwargs)
        result = self._store.write_with_result(**kwargs)
        self._record(result, payload=kwargs["payload"])
        if result.artifact_ref is None:
            raise ProductExecutionContractError(
                "Committed product Store result is missing ArtifactRef.",
                side_effects_may_have_started=True,
            )
        return result.artifact_ref

    def write_with_result(self, **kwargs: Any) -> ArtifactWriteResult:
        self._validate_task_agent_write(kwargs)
        self._validate_repair_source_write(kwargs)
        result = self._store.write_with_result(**kwargs)
        self._record(result, payload=kwargs["payload"])
        return result

    def wrap_effect(
        self,
        *,
        effect_role: str,
        node_name: str,
        factory: Callable[..., Any] | None,
    ) -> Callable[..., Any] | None:
        if factory is None:
            return None

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            phase = self._phase_by_node[node_name]
            start_sequence = len(self.effect_events)
            self.effect_events.append(
                ProductExecutionEffectEventV1(
                    contract_version="1",
                    sequence=start_sequence,
                    phase=phase,
                    repair_attempt=self._repair_attempt,
                    effect_role=effect_role,
                    event_status="started",
                    failure_class=None,
                    failure_message=None,
                )
            )
            try:
                result = factory(*args, **kwargs)
            except ExactDiffApprovalRejectedError:
                # The approval reader rejected before isolated apply began. It is
                # a domain decision, not an effect start/failure fact.
                if effect_role == "isolated_apply":
                    self.effect_events.pop()
                raise
            except Exception as exc:
                self.effect_events.append(
                    ProductExecutionEffectEventV1(
                        contract_version="1",
                        sequence=len(self.effect_events),
                        phase=phase,
                        repair_attempt=self._repair_attempt,
                        effect_role=effect_role,
                        event_status="failed",
                        failure_class=type(exc).__name__,
                        failure_message=str(exc) or type(exc).__name__,
                    )
                )
                raise
            self.effect_events.append(
                ProductExecutionEffectEventV1(
                    contract_version="1",
                    sequence=len(self.effect_events),
                    phase=phase,
                    repair_attempt=self._repair_attempt,
                    effect_role=effect_role,
                    event_status="completed",
                    failure_class=None,
                    failure_message=None,
                )
            )
            return result

        # Preserve the existing worker's reporting/control hooks across effect
        # accounting. They do not grant a route or create another executor.
        for name in ("bind_upstream_effects", "bind_attempt_recorder", "patch_attempt_receipt"):
            hook = getattr(factory, name, None)
            if hook is not None:
                if not callable(hook):
                    raise ProductExecutionContractError(
                        f"Product adapter {name} must be callable.",
                        side_effects_may_have_started=False,
                    )
                setattr(wrapped, name, hook)
        return wrapped


def source_snapshot(root: Path) -> list[dict[str, Any]]:
    """Return the deterministic source snapshot used by the product boundary."""

    selected_root = root.resolve()
    result: list[dict[str, Any]] = []
    for path in sorted(selected_root.rglob("*")):
        if not path.is_file() or any(part in {"bin", "obj"} for part in path.parts):
            continue
        result.append(
            {
                "path": path.relative_to(selected_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return result


def source_snapshot_sha256(snapshot: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def preflight_product_execution(
    request: ProductExecutionRequestV1,
    adapters: ProductExecutionAdaptersV1,
    artifact_store: ArtifactStore,
    *,
    strict_git_admission: StrictGitAdmission | None = None,
) -> ProductExecutionPreflightResultV1:
    """Perform a no-Graph, no-Store, no-effect product admission check."""

    resolver = ProductExecutionInputBindingResolverV1()
    try:
        adapters.validate_for(request)
        resolver.validate(
            request,
            adapters,
            strict_git_admission=strict_git_admission,
        )
    except _ProductExecutionAdmissionError as exc:
        return ProductExecutionPreflightResultV1(
            contract_version="1",
            preflight_status="blocked",
            request=request,
            failure_stage="admission",
            blockers=(
                ProductExecutionBlockerV1(
                    contract_version="1",
                    owner="platform",
                    stage="admission",
                    code="PRODUCT_INPUT_BINDING_ADMISSION_FAILED",
                    message=exc.reason,
                    source_node="product_preflight",
                ),
            ),
        )
    blockers: list[ProductExecutionBlockerV1] = []
    checks = (
        (
            Path(request.artifact_store_root).resolve() == artifact_store.root,
            "PRODUCT_STORE_ROOT_MISMATCH",
            "ArtifactStore root changed after product request construction.",
        ),
        (
            Path(request.source_root).is_dir(),
            "PRODUCT_SOURCE_ROOT_MISSING",
            "Canonical source root is missing.",
        ),
        (
            Path(request.isolated_target_root).is_dir(),
            "PRODUCT_ISOLATED_ROOT_MISSING",
            "Isolated target root is missing.",
        ),
    )
    for passed, code, message in checks:
        if not passed:
            blockers.append(
                ProductExecutionBlockerV1(
                    contract_version="1",
                    owner="platform",
                    stage="execution_constraint",
                    code=code,
                    message=message,
                    source_node="product_preflight",
                )
            )
    if not blockers:
        current_source_hash = source_snapshot_sha256(
            source_snapshot(Path(request.source_root))
        )
        current_isolated_hash = source_snapshot_sha256(
            source_snapshot(Path(request.isolated_target_root))
        )
        if current_source_hash != request.source_snapshot_sha256:
            blockers.append(
                ProductExecutionBlockerV1(
                    contract_version="1",
                    owner="platform",
                    stage="execution_constraint",
                    code="PRODUCT_SOURCE_SNAPSHOT_DRIFT",
                    message="Canonical source snapshot changed before Graph START.",
                    source_node="product_preflight",
                )
            )
        if current_isolated_hash != request.isolated_source_snapshot_sha256:
            blockers.append(
                ProductExecutionBlockerV1(
                    contract_version="1",
                    owner="platform",
                    stage="execution_constraint",
                    code="PRODUCT_ISOLATED_SNAPSHOT_DRIFT",
                    message="Isolated source snapshot changed before Graph START.",
                    source_node="product_preflight",
                )
            )
    return ProductExecutionPreflightResultV1(
        contract_version="1",
        preflight_status="blocked" if blockers else "ready",
        request=request,
        failure_stage="execution_constraint" if blockers else None,
        blockers=tuple(blockers),
    )


def _platform_blocker(stage: str, code: str, message: str, node: str) -> ProductExecutionBlockerV1:
    return ProductExecutionBlockerV1(
        contract_version="1",
        owner="platform",
        stage=stage,
        code=code,
        message=message,
        source_node=node,
    )


def _domain_blockers(state: Mapping[str, Any]) -> tuple[ProductExecutionBlockerV1, ...]:
    selected: list[ProductExecutionBlockerV1] = []
    for raw in state.get("blockers") or ():
        if not isinstance(raw, Mapping):
            continue
        selected.append(
            ProductExecutionBlockerV1(
                contract_version="1",
                owner="domain",
                stage=str(raw.get("stage") or state.get("current_stage") or "blocked"),
                code=str(raw.get("code") or raw.get("reason") or "PRODUCT_DOMAIN_BLOCKED"),
                message=str(raw.get("message") or raw.get("reason") or "Product Graph selected a blocked terminal."),
                source_node="blocked",
            )
        )
    if not selected:
        selected.append(
            ProductExecutionBlockerV1(
                contract_version="1",
                owner="domain",
                stage="blocked",
                code="PRODUCT_DOMAIN_BLOCKED",
                message="Product Graph selected a blocked terminal.",
                source_node="blocked",
            )
        )
    return tuple(selected)


def _output_by_role(
    outputs: tuple[ProductExecutionOutputV1, ...], role: str
) -> ProductExecutionOutputV1 | None:
    matches = [item for item in outputs if item.artifact_role == role]
    return matches[-1] if matches else None


def _read_payload(
    store: ArtifactStore,
    output: ProductExecutionOutputV1 | None,
) -> dict[str, Any] | None:
    if output is None:
        return None
    return store.read(output.artifact_ref)["data"]


def _build_completion(
    request: ProductExecutionRequestV1,
    store: ArtifactStore,
    outputs: tuple[ProductExecutionOutputV1, ...],
) -> ProductBuildCompletionV1:
    build_output = _output_by_role(outputs, "build_result")
    checkpoint_output = _output_by_role(outputs, "workflow_checkpoint")
    if build_output is None:
        return ProductBuildCompletionV1(
            contract_version="1",
            build_result_ref=None,
            workflow_checkpoint_ref=None,
            build_status="not_reached",
            claims_real_build_success=None,
        )
    try:
        payload = _read_payload(store, build_output)
    except Exception:
        return ProductBuildCompletionV1(
            contract_version="1",
            build_result_ref=build_output.artifact_ref,
            workflow_checkpoint_ref=(checkpoint_output.artifact_ref if checkpoint_output else None),
            build_status="unavailable",
            claims_real_build_success=None,
        )
    claim = payload.get("claims_real_build_success")
    status = payload.get("status")
    exact_status = "success" if status == "success" and claim is True else (
        "blocked" if status == "blocked" else "failed"
    )
    return ProductBuildCompletionV1(
        contract_version="1",
        build_result_ref=build_output.artifact_ref,
        workflow_checkpoint_ref=(checkpoint_output.artifact_ref if checkpoint_output else None),
        build_status=exact_status,
        claims_real_build_success=(claim if type(claim) is bool else None),
    )


def _runtime_completion(
    store: ArtifactStore,
    outputs: tuple[ProductExecutionOutputV1, ...],
) -> ProductRuntimeCompletionV1:
    by_role = {role: _output_by_role(outputs, role) for role in (
        "build_result", "runtime_evidence", "eval_report", "workflow_summary", "blocked_report"
    )}
    payloads: dict[str, dict[str, Any] | None] = {}
    for role, output in by_role.items():
        try:
            payloads[role] = _read_payload(store, output)
        except Exception:
            payloads[role] = None
    build = payloads["build_result"]
    runtime = payloads["runtime_evidence"]
    evaluation = payloads["eval_report"]
    summary = payloads["workflow_summary"]
    build_status = (
        "not_reached" if by_role["build_result"] is None else
        "unavailable" if build is None else
        "success" if build.get("claims_real_build_success") is True else
        "blocked" if build.get("status") == "blocked" else "failed"
    )
    runtime_status = (
        "not_reached" if by_role["runtime_evidence"] is None else
        "unavailable" if runtime is None else
        "accepted" if runtime.get("acceptance_verified") is True else
        "blocked" if runtime.get("verification_status") == "blocked" else "not_accepted"
    )
    eval_verdict = (
        "not_reached" if by_role["eval_report"] is None else
        "unavailable" if evaluation is None else
        evaluation.get("verdict") if evaluation.get("verdict") in {"accepted", "blocked", "inconclusive"} else "inconclusive"
    )
    final_status = (
        "blocked" if by_role["blocked_report"] is not None else
        "not_reached" if by_role["workflow_summary"] is None else
        "unavailable" if summary is None else
        summary.get("final_status") if summary.get("final_status") in {"accepted", "blocked", "inconclusive"} else "inconclusive"
    )
    return ProductRuntimeCompletionV1(
        contract_version="1",
        build_result_ref=by_role["build_result"].artifact_ref if by_role["build_result"] else None,
        runtime_evidence_ref=by_role["runtime_evidence"].artifact_ref if by_role["runtime_evidence"] else None,
        eval_report_ref=by_role["eval_report"].artifact_ref if by_role["eval_report"] else None,
        workflow_summary_ref=by_role["workflow_summary"].artifact_ref if by_role["workflow_summary"] else None,
        blocked_report_ref=by_role["blocked_report"].artifact_ref if by_role["blocked_report"] else None,
        build_status=build_status,
        runtime_status=runtime_status,
        eval_verdict=eval_verdict,
        final_status=final_status,
    )


def validate_product_execution_outputs(
    *,
    request: ProductExecutionRequestV1,
    store: ArtifactStore,
    outputs: tuple[ProductExecutionOutputV1, ...],
    adapters: ProductExecutionAdaptersV1,
    initial_isolated_snapshot: list[dict[str, Any]],
    successful_terminal: bool,
) -> ProductExecutionValidationFactsV1:
    statuses = {kind: "not_applicable" for kind in (
        "output_readback", "canonical_source", "compiled_artifact", "source_change", "completion_classification"
    )}
    payloads: dict[str, Any] = {kind: None for kind in statuses}
    failure: Exception | None = None
    ordered = tuple(statuses)
    try:
        if outputs:
            payloads["output_readback"] = [store.read(item.artifact_ref) for item in outputs]
            statuses["output_readback"] = "passed"
        if not successful_terminal:
            return ProductExecutionValidationFactsV1(
                contract_version="1",
                output_readback_status=statuses["output_readback"],
                canonical_source_status="not_applicable",
                compiled_artifact_status="not_applicable",
                source_change_status="not_applicable",
                completion_classification_status="not_applicable",
                failure_class=None,
                failure_message=None,
                details=tuple(
                    ProductExecutionValidationDetailV1(
                        contract_version="1",
                        detail_kind=kind,
                        payload=payloads[kind],
                    )
                    for kind in ordered
                ),
            )
        canonical = source_snapshot(Path(request.source_root))
        canonical_hash = source_snapshot_sha256(canonical)
        if canonical_hash != request.source_snapshot_sha256:
            raise RuntimeError("Canonical source changed during product execution.")
        statuses["canonical_source"] = "passed"
        payloads["canonical_source"] = {"source_snapshot_sha256": canonical_hash}
        build_output = _output_by_role(outputs, "build_result")
        build_payload = _read_payload(store, build_output)
        build_succeeded = bool(build_payload and build_payload.get("claims_real_build_success") is True)
        if build_succeeded and adapters.compiled_artifact_verifier is not None:
            verified = [
                item for item in build_payload.get("verified_artifacts") or ()
                if isinstance(item, Mapping) and item.get("path")
            ]
            if len(verified) != 1:
                raise RuntimeError("Successful Build lacks one exact verified artifact.")
            payloads["compiled_artifact"] = adapters.compiled_artifact_verifier(
                Path(request.isolated_target_root) / str(verified[0]["path"]),
                "final_build",
            )
            statuses["compiled_artifact"] = "passed"
        if build_succeeded and adapters.source_change_verifier is not None:
            final_isolated = source_snapshot(Path(request.isolated_target_root))
            source_change = adapters.source_change_verifier(
                initial_isolated_snapshot,
                final_isolated,
            )
            payloads["source_change"] = {
                **dict(source_change),
                "initial_snapshot": initial_isolated_snapshot,
                "final_snapshot": final_isolated,
            }
            statuses["source_change"] = "passed"
        if request.completion_classification:
            if not build_succeeded:
                raise RuntimeError("Completion classification requires a successful Build.")
            if statuses["compiled_artifact"] != "passed" or statuses["source_change"] != "passed":
                raise RuntimeError("Completion classification requires both fixed verifiers.")
            payloads["completion_classification"] = dict(request.completion_classification)
            statuses["completion_classification"] = "passed"
    except Exception as exc:
        failure = exc
        failed_kind = next(
            kind for kind in ordered
            if statuses[kind] == "not_applicable" and (
                kind == "output_readback" and bool(outputs)
                or kind == "canonical_source"
                or kind == "compiled_artifact" and adapters.compiled_artifact_verifier is not None
                or kind == "source_change" and adapters.source_change_verifier is not None
                or kind == "completion_classification" and bool(request.completion_classification)
            )
        )
        statuses[failed_kind] = "failed"
    return ProductExecutionValidationFactsV1(
        contract_version="1",
        output_readback_status=statuses["output_readback"],
        canonical_source_status=statuses["canonical_source"],
        compiled_artifact_status=statuses["compiled_artifact"],
        source_change_status=statuses["source_change"],
        completion_classification_status=statuses["completion_classification"],
        failure_class=type(failure).__name__ if failure else None,
        failure_message=(str(failure) or type(failure).__name__) if failure else None,
        details=tuple(
            ProductExecutionValidationDetailV1(
                contract_version="1", detail_kind=kind, payload=payloads[kind]
            )
            for kind in ordered
        ),
    )


def execute_product_execution(
    preflight: ProductExecutionPreflightResultV1,
    adapters: ProductExecutionAdaptersV1,
    artifact_store: ArtifactStore,
    *,
    strict_git_admission: StrictGitAdmission | None = None,
    defer_approval: bool = False,
    interrupted_patch_selection: Mapping[str, Any] | None = None,
    pretransport_patch_selection: Mapping[str, Any] | None = None,
) -> ProductExecutionResultV1:
    """Execute the existing StateGraph exactly once and freeze one machine result."""

    if type(defer_approval) is not bool:
        raise ValueError("defer_approval must be boolean.")
    if not isinstance(preflight, ProductExecutionPreflightResultV1) or preflight.preflight_status != "ready":
        raise ProductExecutionContractError(
            "Product execute requires a ready ProductExecutionPreflightResultV1.",
            side_effects_may_have_started=False,
        )
    request = preflight.request
    repeated = preflight_product_execution(
        request,
        adapters,
        artifact_store,
        strict_git_admission=strict_git_admission,
    )
    if repeated.preflight_status != "ready" or repeated.request.to_dict() != request.to_dict():
        raise ProductExecutionContractError(
            "Product execution admission changed before Graph START.",
            side_effects_may_have_started=False,
        )
    initial_isolated_snapshot = source_snapshot(Path(request.isolated_target_root))
    recorder = _ProductStoreRecorder(artifact_store)
    admitted_interruption = None
    if pretransport_patch_selection is not None:
        from src.mod_agent.pretransport_patch_admission import admit_pretransport_patch
        if interrupted_patch_selection is not None:
            raise ProductExecutionContractError("Conflicting Patch recovery selections.", side_effects_may_have_started=False)
        admitted_interruption = admit_pretransport_patch(pretransport_patch_selection,
            workspace_root=Path(request.isolated_workspace_root), artifact_store=artifact_store)
        if (admitted_interruption["request_id"] != request.request_id
                or admitted_interruption["parent_run_id"] != request.run_id
                or admitted_interruption["user_request"] != request.user_request):
            raise ProductExecutionContractError("Pretransport Patch request identity changed.", side_effects_may_have_started=False)
        recorder.reuse_interrupted_inputs(admitted_interruption, pretransport_patch_selection, pretransport=True)
    if interrupted_patch_selection is not None:
        from src.mod_agent.interrupted_patch_admission import admit_interrupted_patch
        admitted_interruption = admit_interrupted_patch(interrupted_patch_selection,
            workspace_root=Path(request.isolated_workspace_root), artifact_store=artifact_store)
        if (admitted_interruption["request_id"] != request.request_id
                or admitted_interruption["parent_run_id"] != request.run_id
                or admitted_interruption["user_request"] != request.user_request):
            raise ProductExecutionContractError("Interrupted Patch selection changed product request identity.",
                                                side_effects_may_have_started=False)
        recorder.reuse_interrupted_inputs(admitted_interruption, interrupted_patch_selection)

    def await_exact_approval(_proposal, _implementation, _review):
        from langgraph.types import interrupt

        from src.mod_agent.workflow.interrupts import InterruptRequest

        dry_output = _output_by_role(tuple(recorder.outputs), "dry_run_result")
        if dry_output is None:
            raise ProductExecutionContractError("Approval pause requires the committed DryRunResult.",
                side_effects_may_have_started=True)
        dry = artifact_store.semantic_read_immutable(dry_output.artifact_ref)["data"]
        pending = InterruptRequest(interrupt_id="approval_" + request.run_id,
            run_id=request.run_id, feature_id=dry_output.artifact_ref.feature_id, stage="dry_run_apply",
            reason="The complete exact diff awaits human semantic approval before isolated Apply.",
            payload={"diff_hash": dry["diff_hash"], "unified_diff": dry["unified_diff"],
                "human_semantic_decision_binding": dry["human_semantic_decision_binding"]},
            resume_allowed_fields=("approvals.dry_run",),
            artifact_refs=tuple(item.artifact_ref.path for item in recorder.outputs))
        interrupt(pending.to_dict())
        raise ProductExecutionContractError(
            "Deferred approval resumes only through its persisted exact checkpoint and approved diff.",
            side_effects_may_have_started=False)

    upstream_evidence_factory = adapters.patch_upstream_evidence_pack_factory
    if upstream_evidence_factory is not None:
        from src.mod_agent.evidence.reassessment import RequirementCoverageReassessmentFactory

        if isinstance(upstream_evidence_factory, RequirementCoverageReassessmentFactory):
            recorded_reassessment = recorder.wrap_effect(effect_role="evidence_retrieval",
                node_name="retrieve_evidence", factory=upstream_evidence_factory)

            class RecordedReassessment(RequirementCoverageReassessmentFactory):
                def __init__(self):
                    # Provider identity is checked above; only effect accounting
                    # is added around this exact admitted factory.
                    pass

                def __call__(self, state, feature_data):
                    return recorded_reassessment(state, feature_data)

            upstream_evidence_factory = RecordedReassessment()

    selected_patch_factory = adapters.patch_change_factory
    if adapters.task_agent_execution is not None:
        task_owner = adapters.task_agent_execution
        task_owner.bind_product_store(recorder)

        def selected_patch_factory(state, implementation, evidence_pack):
            # Scope: technical owner only; existing Review/Apply/Build nodes remain outside.
            # 为什么要有它：新循环产生真实内部工件，但不应让内部候选构建冒充正式交付或绕过审批。
            with recorder.task_agent_scope(task_owner, state):
                return task_owner(state, implementation, evidence_pack)

    wrapped = ProductExecutionAdaptersV1(
        evidence_pack_factory=recorder.wrap_effect(effect_role="evidence_retrieval", node_name="retrieve_evidence", factory=adapters.evidence_pack_factory),
        patch_change_factory=recorder.wrap_effect(effect_role="patch_generation", node_name="generate_patch_proposal", factory=selected_patch_factory),
        dry_run_factory=recorder.wrap_effect(effect_role="dry_run", node_name="dry_run_apply", factory=adapters.dry_run_factory),
        isolated_apply_factory=(await_exact_approval if defer_approval else recorder.wrap_effect(
            effect_role="isolated_apply", node_name="isolated_apply", factory=adapters.isolated_apply_factory)),
        build_result_factory=recorder.wrap_effect(effect_role="build", node_name="build", factory=adapters.build_result_factory),
        runtime_evidence_factory=recorder.wrap_effect(effect_role="runtime", node_name="collect_runtime_evidence", factory=adapters.runtime_evidence_factory),
        compiled_artifact_verifier=adapters.compiled_artifact_verifier,
        source_change_verifier=adapters.source_change_verifier,
        proposal_review_context_factory=adapters.proposal_review_context_factory,
        planning_target_selection_factory=(
            adapters.planning_target_selection_factory
        ),
        product_patch_upstream_execution=adapters.product_patch_upstream_execution,
        patch_upstream_evidence_pack_factory=upstream_evidence_factory,
        patch_upstream_planning_target_selection_factory=adapters.patch_upstream_planning_target_selection_factory,
        initial_findings_reassessment_factory=recorder.wrap_effect(effect_role="evidence_retrieval",
            node_name="retrieve_evidence", factory=adapters.initial_findings_reassessment_factory),
        domain_adapter=adapters.domain_adapter,
        task_agent_execution=adapters.task_agent_execution,
    )
    if defer_approval:
        from src.mod_agent.workflow.checkpointing import WorkflowCheckpointStore

    graph = build_graph(
        recorder,
        evidence_pack_factory=wrapped.evidence_pack_factory,
        planning_target_selection_factory=(
            wrapped.planning_target_selection_factory
        ),
        patch_change_factory=wrapped.patch_change_factory,
        proposal_review_context_factory=wrapped.proposal_review_context_factory,
        dry_run_factory=wrapped.dry_run_factory,
        isolated_apply_factory=wrapped.isolated_apply_factory,
        build_result_factory=wrapped.build_result_factory,
        runtime_evidence_factory=wrapped.runtime_evidence_factory,
        domain_review_policy=(
            wrapped.domain_adapter if wrapped.domain_adapter is not None else (
                GTFOAdapter()
                if request.target_domain == "gtfo"
                else UnsupportedDomainAdapter()
            )
        ).identify_project(request.isolated_workspace_root).review_policy(),
        **({"domain_feature_spec_factory": wrapped.domain_adapter.build_feature_spec}
           if wrapped.domain_adapter is not None else {}),
        use_default_proposal_git_binding=False,
        product_execution_mode=ProductExecutionGraphModeV1(
            contract_version="1", request=request,
            interrupted_patch_selection=copy.deepcopy(interrupted_patch_selection),
            pretransport_patch_selection=copy.deepcopy(pretransport_patch_selection),
        ),
        enable_patch_upstream_replanning=adapters.product_patch_upstream_execution is not None,
        product_patch_upstream_execution=wrapped.product_patch_upstream_execution,
        task_agent_execution=wrapped.task_agent_execution,
        patch_upstream_evidence_pack_factory=wrapped.patch_upstream_evidence_pack_factory,
        patch_upstream_planning_target_selection_factory=wrapped.patch_upstream_planning_target_selection_factory,
        initial_findings_reassessment_factory=wrapped.initial_findings_reassessment_factory,
        **({"checkpointer": WorkflowCheckpointStore.langgraph_memory_checkpointer()} if defer_approval else {}),
    )
    state = create_initial_state(
        run_id=request.run_id,
        request_id=request.request_id,
        user_request=request.user_request,
        target_domain=request.target_domain,
        target_mod=request.target_mod,
        confirmed_isolated_write=True,
    )
    if admitted_interruption is not None:
        feature = admitted_interruption["feature_spec"]
        state.update(feature_id=admitted_interruption["feature_id"],
            risk_level=feature["risk_level"],
            current_stage="implementation_plan", next_stage="patch_proposal",
            artifacts={role: ref["path"] for role, ref in admitted_interruption["selected_artifacts"].items()},
            artifact_hashes={role: ref["artifact_hash"] for role, ref in admitted_interruption["selected_artifacts"].items()})
    graph_result: Mapping[str, Any] = state
    platform_blockers: tuple[ProductExecutionBlockerV1, ...] = ()
    failure_stage: str | None = None
    config = {"configurable": {"thread_id": request.run_id}} if defer_approval else None
    try:
        graph_result = graph.invoke(state, config) if defer_approval else graph.invoke(state)
    except _ProductStoreStopped:
        last = recorder.attempts[-1]
        failure_stage = f"{last.artifact_role}_store"
        platform_blockers = (
            _platform_blocker(
                failure_stage,
                "PRODUCT_STORE_NOT_COMMITTED",
                last.store_result.failure_message or "Product Store attempt did not commit.",
                last.node_name,
            ),
        )
    except ProductExecutionContractError:
        raise
    except Exception as exc:
        node = getattr(exc, "_mod_agent_graph_node", None)
        if node not in PRODUCT_EXECUTION_DECLARATION.phase_by_node:
            raise ProductExecutionContractError(
                "Graph exception lacks its exact current-node fact.",
                side_effects_may_have_started=bool(
                    recorder.outputs or recorder.effect_events
                ),
            ) from exc
        failure_stage = f"{node}_execution"
        platform_blockers = (
            _platform_blocker(
                failure_stage,
                "PRODUCT_NODE_EXECUTION_FAILED",
                f"{type(exc).__name__}: {exc}",
                node,
            ),
        )
    if defer_approval and graph_result.get("__interrupt__"):
        snapshot = graph.get_state(config)
        paused = dict(snapshot.values)
        if (paused.get("current_stage") != "dry_run_apply" or tuple(snapshot.next) != ("isolated_apply",)
                or "isolated_apply_result" in paused["artifacts"]
                or any(item.effect_role in {"isolated_apply", "build", "runtime"} for item in recorder.effect_events)):
            raise ProductExecutionContractError("Approval pause did not occur at the exact no-write boundary.",
                side_effects_may_have_started=True)
        if source_snapshot_sha256(source_snapshot(Path(request.source_root))) != request.source_snapshot_sha256:
            raise ProductExecutionContractError("Canonical source changed before approval pause.", side_effects_may_have_started=True)
        if source_snapshot_sha256(source_snapshot(Path(request.isolated_target_root))) != request.isolated_source_snapshot_sha256:
            raise ProductExecutionContractError("Isolated source changed before approval pause.", side_effects_may_have_started=True)
        # The existing typed decoder validates field order. Retain its own
        # serialized representation intact across the Store's sorted envelope.
        paused["runtime_summary"] = {**paused["runtime_summary"],
            "deferred_product_request_json": json.dumps(request.to_dict(), ensure_ascii=False, separators=(",", ":"))}
        checkpoint_ref = WorkflowCheckpointStore(artifact_store).create_checkpoint(paused,
            interrupt_reason="The saved exact diff awaits human semantic approval before isolated Apply.",
            resume_allowed_fields=("approvals.dry_run",))
        dry_ref = ArtifactRef(feature_id=paused["feature_id"], artifact_type="dry_run_result",
            path=paused["artifacts"]["dry_run_result"], artifact_hash=paused["artifact_hashes"]["dry_run_result"])
        raise ProductApprovalDeferred(request=request, checkpoint_ref=checkpoint_ref, dry_run_ref=dry_ref,
            dry_run=artifact_store.semantic_read_immutable(dry_ref)["data"],
            outputs=tuple(recorder.outputs), effect_events=tuple(recorder.effect_events))
    attempts = tuple(recorder.attempts)
    outputs = tuple(recorder.outputs)
    terminal = (
        graph_result.get("current_stage")
        if graph_result.get("current_stage") in ({"build", "done", "blocked", "dry_run_apply"}
            if getattr(adapters.task_agent_execution, "protocol", None) == "v2" else {"build", "done", "blocked"})
        and failure_stage is None
        else None
    )
    validation = validate_product_execution_outputs(
        request=request,
        store=artifact_store,
        outputs=outputs,
        adapters=adapters,
        initial_isolated_snapshot=initial_isolated_snapshot,
        successful_terminal=terminal in {"build", "done"},
    )
    blockers = (
        _domain_blockers(graph_result)
        if terminal == "blocked" and not platform_blockers
        else platform_blockers
    )
    blockers = (*blockers, *recorder.uncommitted_blocked_causes)
    if validation.failed and not platform_blockers:
        blockers = (
            _platform_blocker(
                "post_execution_validation",
                "PRODUCT_POST_EXECUTION_VALIDATION_FAILED",
                validation.failure_message or "Product output validation failed.",
                terminal or "product_executor",
            ),
        )
    variant = (
        _build_completion(request, artifact_store, outputs)
        if request.requested_completion_scope == "compile_only"
        else _runtime_completion(artifact_store, outputs)
    )
    return project_product_execution_result(
        ProductExecutionFactsV1(
            request=request,
            feature_id=(
                graph_result.get("feature_id")
                or (outputs[-1].artifact_ref.feature_id if outputs else None)
            ),
            terminal_node=terminal,
            store_attempts=attempts,
            outputs=outputs,
            effect_events=tuple(recorder.effect_events),
            validation_facts=validation,
            blockers=blockers,
            variant_result=variant,
            failure_stage=failure_stage,
            task_agent_executions=tuple(recorder.task_agent_executions),
            v2_delivery_checkpoint_ref=graph_result.get("runtime_summary", {}).get("v2_delivery_checkpoint_ref"),
        )
    )
