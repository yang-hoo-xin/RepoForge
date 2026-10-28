"""Run DSH's literal request through the existing local, deferred product CLI.

The administrator supplies a fixed JSON object containing ``arguments``. Only
the description comes from stdin. This bridge adds no execution state machine
and preserves the product owner's outcome and exit status, adding an inspection locator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.libs.llm import validate_loopback_http_base_url
from src.mod_agent import cli
from src.mod_agent.patch_worker_configuration import read_json_object

MAX_REQUEST_BYTES = 32768
MAX_RUN_RECORD_BYTES = 8 * 1024 * 1024

# Exact installed run-request options. Isolated preparation/build are explicit;
# deferred approval still prevents applying the final diff. An existing worker
# configuration is administrator-selected and validated by its original owner.
_SWITCHES = frozenset({
    "--local-ollama", "--defer-approval", "--allow-build", "--confirm-isolated-write",
    "--allow-source-disclosure",
    "--reuse-query-observations", "--no-reuse-query-observations",
    "--compress-findings-input", "--no-compress-findings-input",
})
_VALUES = frozenset({
    "--target-project", "--workspace-root", "--settings",
    "--game-folder-build-binding", "--ollama-base-url", "--ollama-model",
    "--ollama-timeout-seconds", "--ollama-max-output-tokens",
    "--ollama-context-window", "--source-retrieval", "--task-model-call-limit",
    "--stage-model-call-limits", "--model-call-budget-policy", "--trace-environment", "--patch-worker-config",
    "--run-id", "--request-id", "--dependency-api-config", "--knowledge-evidence-config",
    "--technical-executor", "--constraint-policy", "--task-agent-edit-paths", "--task-agent-candidate-mode",
    "--task-agent-protocol",
    "--task-agent-experience-config",
    "--task-agent-semantic-state",
    "--repository-profile",
    "--external-patch-profile", "--api-key-env", "--base-url-env", "--source-disclosure-approval-id",
})
_REQUIRED = {
    "--local-ollama", "--defer-approval", "--allow-build", "--confirm-isolated-write",
    "--target-project", "--workspace-root", "--settings", "--ollama-model",
    "--ollama-context-window", "--task-model-call-limit", "--stage-model-call-limits",
}
_PATHS = frozenset({
    "--task-agent-experience-config",
    "--target-project", "--workspace-root", "--settings",
    "--game-folder-build-binding", "--stage-model-call-limits", "--patch-worker-config",
    "--dependency-api-config", "--knowledge-evidence-config",
    "--repository-profile",
    "--external-patch-profile",
})
_STAGES = frozenset({
    "requirement_projection", "requirement_findings", "planning", "initial_patch", "repair",
})


@dataclass(frozen=True)
class OpenTaskRun:
    execution_config_ref: dict[str, str]
    intent: str


@dataclass(frozen=True)
class ResumeTaskRun:
    run_ref: dict[str, str]
    resume_ref: dict[str, str]


@dataclass(frozen=True)
class TaskRunSuspended:
    run_ref: dict[str, str]
    run_id: str
    request_id: str
    resume_ref: dict[str, str]
    progress_receipt_ref: dict[str, str]


@dataclass(frozen=True)
class TaskRunCompleted:
    run_ref: dict[str, str]
    run_id: str
    request_id: str
    status: str
    run_outcome_ref: dict[str, str]
    evidence_bundle_ref: dict[str, str]


TaskRunCommand = OpenTaskRun | ResumeTaskRun
TaskRunExecution = TaskRunSuspended | TaskRunCompleted


def _configuration_arguments(execution_config: str) -> tuple[list[str], dict[str, str | None]]:
    """Read the fixed administrator options without loading execution providers."""
    selected = Path(execution_config)
    if not selected.is_absolute():
        raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE")
    document = read_json_object(selected)
    if (not {"arguments"}.issubset(document) or set(document) - {"arguments", "worker_path_bindings", "new_project_binding", "followup_binding", "dependency_api_binding", "knowledge_evidence_binding", "repository_profile_binding", "external_profile_binding"}
            or not isinstance(document["arguments"], list)):
        raise ValueError("DSH_RUN_CONFIGURATION_SHAPE_INVALID")
    arguments = document["arguments"]
    if any(type(value) is not str or not value or "\0" in value for value in arguments):
        raise ValueError("DSH_RUN_CONFIGURATION_ARGUMENT_INVALID")
    seen: dict[str, str | None] = {}
    index = 0
    while index < len(arguments):
        option, equals, value = arguments[index].partition("=")
        if option not in _SWITCHES | _VALUES or option in seen:
            raise ValueError("DSH_RUN_CONFIGURATION_OPTION_NOT_ALLOWED: " + option)
        if option in _SWITCHES:
            if equals:
                raise ValueError("DSH_RUN_CONFIGURATION_SWITCH_VALUE_INVALID: " + option)
            seen[option] = None
        else:
            if not equals:
                index += 1
                if index >= len(arguments) or arguments[index].startswith("--"):
                    raise ValueError("DSH_RUN_CONFIGURATION_VALUE_MISSING: " + option)
                value = arguments[index]
            if not value:
                raise ValueError("DSH_RUN_CONFIGURATION_VALUE_MISSING: " + option)
            seen[option] = value
        index += 1
    # 为什么要有它：固定执行配置中的正反策略开关不能靠顺序覆盖，避免在线对照出现两种声明。
    for name in ("reuse-query-observations", "compress-findings-input"):
        if "--" + name in seen and "--no-" + name in seen:
            raise ValueError("DSH_RUN_CONFIGURATION_OPTION_CONFLICT: " + name)
    external_options = {"--external-patch-profile", "--allow-source-disclosure", "--api-key-env", "--base-url-env", "--source-disclosure-approval-id"} & seen.keys()
    if external_options and "--local-ollama" in seen:
        raise ValueError("DSH_RUN_CONFIGURATION_PROVIDER_CONFLICT")
    if external_options and "--external-patch-profile" not in seen:
        raise ValueError("DSH_RUN_CONFIGURATION_EXTERNAL_PROFILE_REQUIRED")
    # 为什么要有它：已分配身份的外部配置不能删除绑定后伪装成尚未冻结的声明输入。
    if ("--external-patch-profile" in seen and {"--run-id", "--request-id"} & seen.keys()
            and "external_profile_binding" not in document):
        raise ValueError("DSH_EXTERNAL_PROFILE_BINDING_REQUIRED")
    if "dependency_api_binding" in document:
        if document["dependency_api_binding"] != dependency_api_binding(seen):
            raise ValueError("DSH_DEPENDENCY_API_BINDING_CHANGED")
    if "knowledge_evidence_binding" in document and document["knowledge_evidence_binding"] != knowledge_evidence_binding(seen):
        raise ValueError("DSH_KNOWLEDGE_EVIDENCE_BINDING_CHANGED")
    if "repository_profile_binding" in document and document["repository_profile_binding"] != repository_profile_binding(seen):
        raise ValueError("DSH_REPOSITORY_PROFILE_BINDING_CHANGED")
    if "external_profile_binding" in document and document["external_profile_binding"] != external_profile_binding(seen):
        raise ValueError("DSH_EXTERNAL_PROFILE_BINDING_CHANGED")
    return arguments, seen


def dependency_api_binding(options):
    if not options.get("--dependency-api-config"):
        return None
    if not options.get("--game-folder-build-binding"):
        raise ValueError("DEPENDENCY_API_GAME_FOLDER_BINDING_REQUIRED")
    from src.mod_agent.dependency_api_config import DependencyAPIConfiguration

    config = DependencyAPIConfiguration.load(options["--dependency-api-config"])
    return {"configuration_ref": config.reference,
            "inspector_ref": {"path": str(config.inspector_path), "sha256": config.inspector_sha256}}


def knowledge_evidence_binding(options):
    if not options.get("--knowledge-evidence-config"):
        return None
    from src.mod_agent.knowledge_evidence_config import KnowledgeEvidenceConfiguration
    configuration = KnowledgeEvidenceConfiguration.load(options["--knowledge-evidence-config"])
    return configuration.boundary()


def repository_profile_binding(options):
    """Keep the selected public repository checks fixed across routing and dispatch.

    为什么要有它：普通入口不能在选路后悄悄换构建或测试合同；这不是新增资源审批。
    """
    if not options.get("--repository-profile"):
        return None
    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile

    profile = CSharpRepositoryProfile.load(options["--repository-profile"])
    return {"configuration_ref": profile.reference, "dotnet_ref": profile.data["dotnet_ref"]}


def validate_external_settings_credentials(settings):
    """Reject embedded credentials before any settings copy or provider access.

    为什么要有它：登记入口与冻结调用者应共用同一凭据边界；只校验已解析数据，不读ENV或回显字段值。
    """
    def embedded_secret(value):
        if isinstance(value, dict):
            return any((str(key).casefold() in {"api_key", "secret_key", "access_token", "refresh_token", "password", "authorization", "client_secret"}
                and item not in (None, "")) or embedded_secret(item) for key, item in value.items())
        return isinstance(value, list) and any(embedded_secret(item) for item in value)

    if embedded_secret(settings):
        raise ValueError("DSH_EXTERNAL_SETTINGS_EMBEDDED_CREDENTIAL_DENIED")


def external_profile_binding(options):
    """Bind the existing explicit OpenAI v2 profile without reading credentials or sending.

    为什么要有它：注册、选路、派发与查询须指向同一非秘密配置；ENV名称属于合同，值只由运行owner在内存读取。
    """
    if not options.get("--external-patch-profile"):
        return None
    import yaml

    from src.mod_agent.external_patch import (
        ExternalOpenAIResponsesProfile,
        external_profile_from_document,
    )
    from src.mod_agent.task_agent_v2_state import _no_links

    required = {"--settings", "--api-key-env", "--base-url-env", "--allow-source-disclosure",
        "--repository-profile", "--technical-executor", "--task-agent-protocol", "--constraint-policy"}
    if not required <= options.keys():
        raise ValueError("DSH_EXTERNAL_PROFILE_REQUIRED_OPTIONS_MISSING")
    if ("--local-ollama" in options or options["--technical-executor"] != "task_agent"
            or options["--task-agent-protocol"] != "v2" or options["--constraint-policy"] != "source_priority_v1"
            or options.get("--model-call-budget-policy", "task_total_only") not in {"task_total_only", "meter_only"}
            or any(key.startswith("--ollama-") for key in options)):
        raise ValueError("DSH_EXTERNAL_PROFILE_REQUIRES_OPENAI_V2")
    paths = [Path(options[name]) for name in ("--external-patch-profile", "--settings")]
    for path in paths:
        if not path.is_absolute():
            raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE")
        _no_links(path)
    try:
        profile = external_profile_from_document(read_json_object(str(paths[0])))
    except (TypeError, ValueError, KeyError):
        raise ValueError("DSH_EXTERNAL_PROFILE_INVALID") from None
    if not isinstance(profile, ExternalOpenAIResponsesProfile):
        raise ValueError("DSH_EXTERNAL_PROFILE_REQUIRES_OPENAI_V2")
    if (options["--api-key-env"] != profile.credential_env
            or not options["--api-key-env"].isidentifier() or not options["--base-url-env"].isidentifier()):
        raise ValueError("DSH_EXTERNAL_CREDENTIAL_ENV_MISMATCH")
    try:
        settings = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError):
        raise ValueError("DSH_EXTERNAL_SETTINGS_INVALID") from None

    validate_external_settings_credentials(settings)
    llm = settings.get("llm") if isinstance(settings, dict) else None
    if (not isinstance(llm, dict) or llm.get("provider") != "openai"
            or llm.get("model") != profile.model or llm.get("base_url") != profile.base_url):
        raise ValueError("DSH_EXTERNAL_SETTINGS_PROFILE_MISMATCH")
    return {"configuration_ref": _file_ref(paths[0]), "settings_ref": _file_ref(paths[1]),
        "profile": profile.to_dict(), "api_key_env": options["--api-key-env"], "base_url_env": options["--base-url-env"],
        "source_disclosure": {"allowed": True, "approval_id": options.get("--source-disclosure-approval-id")},
        "scope": "declared_configuration_only_not_model_or_account_admission", "credential_values_recorded": False}


def prepare_new_task(*, execution_config: str, description: str, project_name: str | None = None,
                     parent_run_id: str | None = None, delivery_root: str | None = None,
                     parent_execution_config: str | None = None) -> str:
    """Allocate exact invocation inputs; the public owner still owns the run directory.

    Evaluator paths are relocated only at administrator-declared JSON locations.
    No string replacement, path inference, policy change or budget increase occurs.
    """
    _, seen = _configuration_arguments(execution_config)
    document = read_json_object(execution_config)
    api_binding = dependency_api_binding(seen)
    knowledge_binding = knowledge_evidence_binding(seen)
    repository_binding = repository_profile_binding(seen)
    external_binding = external_profile_binding(seen)
    if repository_binding is not None and project_name is not None:
        raise ValueError("DSH_REPOSITORY_PROFILE_REQUIRES_EXISTING_REPOSITORY")
    if parent_run_id is not None and project_name is not None:
        raise ValueError("DSH_FOLLOWUP_CANNOT_INITIALIZE_PROJECT")
    if parent_execution_config is not None and parent_run_id is None:
        raise ValueError("DSH_FOLLOWUP_PARENT_RUN_REQUIRED")
    followup_parent = (_verified_delivered_source(parent_execution_config or execution_config, parent_run_id, delivery_root)
              if parent_run_id is not None else None)
    identity = uuid4().hex
    run_id, request_id = "dsh_" + identity, "request_" + identity
    workspace = Path(seen["--workspace-root"])
    if not workspace.is_absolute():
        raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE")
    workspace = workspace.resolve()
    run_root = workspace / run_id
    input_root = _contained_record_path(workspace / ".dsh-inputs" / run_id, workspace)
    worker = None
    bindings = document.get("worker_path_bindings", [])
    if "--patch-worker-config" in seen:
        worker = read_json_object(seen["--patch-worker-config"])
        if worker.get("behavior_evaluator") is not None and not bindings:
            raise ValueError("DSH_NEW_TASK_WORKER_PATH_BINDINGS_REQUIRED")
    if not isinstance(bindings, list) or (bindings and worker is None):
        raise ValueError("DSH_NEW_TASK_WORKER_PATH_BINDINGS_INVALID")
    locators = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"path", "relative_path"}:
            raise ValueError("DSH_NEW_TASK_WORKER_PATH_BINDINGS_INVALID")
        locator, relative = binding["path"], binding["relative_path"]
        if (not isinstance(locator, list) or len(locator) < 3
                or locator[:2] != ["behavior_evaluator", "kwargs"]
                or any(not isinstance(key, str) or not key for key in locator)
                or tuple(locator) in locators or not isinstance(relative, str)):
            raise ValueError("DSH_NEW_TASK_WORKER_PATH_BINDINGS_INVALID")
        locators.add(tuple(locator))
        destination = (run_root / relative).resolve()
        if Path(relative).is_absolute() or not destination.is_relative_to(run_root):
            raise ValueError("DSH_NEW_TASK_WORKER_PATH_OUTSIDE_RUN")
        parent = worker
        for key in locator[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                raise ValueError("DSH_NEW_TASK_WORKER_EXACT_PATH_MISSING")
            parent = parent[key]
        if not isinstance(parent, dict) or not isinstance(parent.get(locator[-1]), str):
            raise ValueError("DSH_NEW_TASK_WORKER_EXACT_PATH_MISSING")
        parent[locator[-1]] = str(destination)
    options = deepcopy(seen)
    options.update({"--run-id": run_id, "--request-id": request_id})
    if worker is not None:
        options["--patch-worker-config"] = str(input_root / "worker.json")
    arguments = [part for key, value in options.items() for part in ([key] if value is None else [key, value])]
    # Validate non-path invocation input before writing its durable configuration.
    if not isinstance(description, str) or not description.strip() or "\0" in description or len(description.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValueError("DSH_RUN_REQUEST_INVALID")
    input_root.mkdir(parents=True, exist_ok=False)
    project_binding = None
    followup_binding = None
    if followup_parent is not None:
        options["--target-project"] = followup_parent["target_project"]
        if worker is not None and worker.get("behavior_evaluator") is not None:
            from src.mod_agent.local_entry import _optional_project_text_property

            target = Path(followup_parent["target_project"])
            projects = list(target.glob("*.csproj"))
            if len(projects) != 1 or not seen.get("--game-folder-build-binding"):
                raise ValueError("DSH_FOLLOWUP_COMPILATION_INPUTS_REQUIRED")
            assembly = _optional_project_text_property(projects[0], property_name="AssemblyName") or projects[0].stem
            _bind_project_compilation_feedback(worker, {
                "target_project": str(target), "project_file": str(projects[0]),
                "identity": {"assembly_name": assembly},
                "dependency_binding": read_json_object(seen["--game-folder-build-binding"]),
            }, run_root)
        request_path = input_root / "followup_request.json"
        request_path.write_text(json.dumps({"run_id": run_id, "request_id": request_id,
            "description": description}, ensure_ascii=False), encoding="utf-8")
        followup_binding = {"parent": followup_parent, "base_profile_ref": _file_ref(Path(execution_config)),
                            "request_ref": _file_ref(request_path)}
        arguments = [part for key, value in options.items() for part in ([key] if value is None else [key, value])]
    if project_name is not None:
        from src.mod_agent.new_mod_project import (
            initialize_new_mod_project,
            prepare_new_mod_project,
        )

        dependency_path = seen.get("--game-folder-build-binding")
        if not dependency_path:
            raise ValueError("DSH_NEW_PROJECT_DEPENDENCY_BINDING_REQUIRED")
        prepared = prepare_new_mod_project(
            workspace_root=workspace, project_id=run_id, display_name=project_name,
            dependency_binding_ref=_file_ref(Path(dependency_path)),
        )
        receipt = initialize_new_mod_project(prepared)
        if worker is not None and worker.get("behavior_evaluator") is not None:
            _bind_project_compilation_feedback(worker, receipt, run_root)
        options["--target-project"] = receipt["target_project"]
        arguments = [part for key, value in options.items() for part in ([key] if value is None else [key, value])]
        request_path = input_root / "new_project_request.json"
        request_path.write_text(json.dumps({"run_id": run_id, "request_id": request_id,
            "description": description}, ensure_ascii=False), encoding="utf-8")
        project_binding = {
            "receipt_ref": _file_ref(Path(receipt["receipt_path"])),
            "scope_hash": receipt["scope_hash"],
            "base_profile_ref": _file_ref(Path(execution_config)),
            "request_ref": _file_ref(request_path),
        }
    if worker is not None:
        (input_root / "worker.json").write_text(json.dumps(worker, ensure_ascii=False, indent=2), encoding="utf-8")
        if project_binding is not None:
            project_binding["worker_ref"] = _file_ref(input_root / "worker.json")
        if followup_binding is not None:
            followup_binding["worker_ref"] = _file_ref(input_root / "worker.json")
    selected = input_root / "execution.json"
    selected.write_text(json.dumps({"arguments": arguments,
        **({"dependency_api_binding": api_binding} if api_binding is not None else {}),
        **({"knowledge_evidence_binding": knowledge_binding} if knowledge_binding is not None else {}),
        **({"repository_profile_binding": repository_binding} if repository_binding is not None else {}),
        **({"external_profile_binding": external_binding} if external_binding is not None else {}),
        **({"new_project_binding": project_binding} if project_binding is not None else {}),
        **({"followup_binding": followup_binding} if followup_binding is not None else {})},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if project_binding is not None:
        (input_root / "project_start.json").write_text(json.dumps({
            "execution_config_ref": _file_ref(selected),
            "base_profile_ref": project_binding["base_profile_ref"],
        }), encoding="utf-8")
    if followup_binding is not None:
        (input_root / "followup_start.json").write_text(json.dumps({
            "execution_config_ref": _file_ref(selected),
            "base_profile_ref": followup_binding["base_profile_ref"],
        }), encoding="utf-8")
    build_product_argv(execution_config=str(selected), description=description)
    return str(selected)


def prepare_followup_task(*, execution_config: str, parent_run_id: str, description: str,
                          delivery_root: str | None = None, parent_execution_config: str | None = None) -> str:
    """Allocate a new literal request from one verified delivered source revision.

    为什么要有它：新的需求需要新任务及原配置预算，不能把新正文丢给不接收正文的续跑入口。
    """
    return prepare_new_task(execution_config=execution_config, description=description,
                            parent_run_id=parent_run_id, delivery_root=delivery_root,
                            parent_execution_config=parent_execution_config)


def _verified_delivered_source(execution_config: str, run_id: str, delivery_root: str | None) -> dict:
    """Read original Apply/Build artifacts and their exact current source bytes.

    为什么要有它：成功显示文本或未 Apply 的候选不能成为已交付工程；当前字节必须匹配原验收版本。
    """
    from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
    from src.mod_agent.execution.task_delegation import has_verified_apply_authorization
    from src.mod_agent.product_execution import ProductExecutionRequestV1
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
    from src.mod_agent.reviewed_patch_continuation import _applied_source_snapshot

    selected = task_execution_config(execution_config=execution_config, run_id=run_id)
    _, options = _configuration_arguments(selected)
    workspace = Path(options["--workspace-root"]).resolve()
    root = _contained_record_path(workspace / run_id, workspace)
    store = ArtifactStore(root / "a")
    refs = [_file_ref(Path(selected))]
    if delivery_root is not None:
        from scripts.dsh_deliver_run import _inputs, _read_receipt

        case = _inputs(selected, delivery_root, run_id)
        receipt = _read_receipt(case)
        if not receipt or receipt.get("status") != "delivered":
            raise ValueError("DSH_FOLLOWUP_DELIVERED_SOURCE_REQUIRED")
        applied_view, built_view = receipt["apply_result"], receipt["build_result"]
        refs.append(_file_ref(case["receipt_path"]))
    else:
        paths = [root / "delivery" / name for name in ("result.json", "apply.json", "build.json")]
        for path in paths:
            _contained_record_path(path, root)
        if any(not path.is_file() for path in paths):
            raise ValueError("DSH_FOLLOWUP_DELIVERED_SOURCE_REQUIRED")
        delivered, applied_view, built_view = (read_json_object(path) for path in paths)
        if delivered.get("status") != "delivered" or delivered.get("delivery_complete") is not True:
            raise ValueError("DSH_FOLLOWUP_DELIVERED_SOURCE_REQUIRED")
        refs.extend(_file_ref(path) for path in paths)
        if not all(ref in delivered.get("execution_refs", []) for ref in refs[-2:]):
            raise ValueError("DSH_FOLLOWUP_DELIVERY_REFERENCES_CHANGED")
    if (applied_view.get("status") != "applied_in_isolation" or built_view.get("status") != "built"
            or applied_view.get("run_id") != run_id or built_view.get("run_id") != run_id
            or built_view.get("build_input_checkpoint") != applied_view.get("workflow_checkpoint")
            or built_view.get("isolated_apply_result") != applied_view.get("isolated_apply_result")):
        raise ValueError("DSH_FOLLOWUP_APPLY_BUILD_CHAIN_REQUIRED")

    def artifact(reference, kind):
        ref = ArtifactRef(**reference)
        if ref.artifact_type != kind:
            raise ValueError("DSH_FOLLOWUP_ARTIFACT_KIND_MISMATCH")
        _contained_record_path(Path(ref.path), store.root.resolve())
        refs.append(_file_ref(Path(ref.path)))
        return store.semantic_read_immutable(ref)["data"]

    checkpoint = artifact(applied_view["workflow_checkpoint"], "workflow_checkpoint")
    state = checkpoint["state_snapshot"]
    request = ProductExecutionRequestV1.from_dict(json.loads(state["runtime_summary"]["deferred_product_request_json"]))
    applied = artifact(applied_view["isolated_apply_result"], "isolated_apply_result")
    built = artifact(built_view["build_result"], "build_result")
    build_checkpoint = artifact(built_view["workflow_checkpoint"], "workflow_checkpoint")
    build_state = build_checkpoint["state_snapshot"]
    source = _contained_record_path(Path(request.isolated_target_root), root)
    if (request.run_id != run_id or request.request_id != options["--request-id"]
            or Path(request.source_root).resolve() != Path(options["--target-project"]).resolve()
            or Path(request.isolated_workspace_root).resolve() != root
            or checkpoint["run_id"] != run_id or build_checkpoint["run_id"] != run_id
            or checkpoint["request_id"] != request.request_id or build_checkpoint["request_id"] != request.request_id
            or checkpoint["feature_id"] != build_checkpoint["feature_id"]
            or checkpoint["checkpoint_stage"] != "isolated_apply"
            or build_checkpoint["checkpoint_stage"] != "build"
            or build_state["artifacts"].get("build_result") != built_view["build_result"]["path"]
            or build_state["artifact_hashes"].get("build_result") != built_view["build_result"]["artifact_hash"]
            or build_state["artifact_hashes"].get("isolated_apply_result") != applied_view["isolated_apply_result"]["artifact_hash"]
            or state["artifact_hashes"].get("isolated_apply_result") != applied_view["isolated_apply_result"]["artifact_hash"]
            or applied.get("actual_write_performed") is not True or applied.get("execution_mode") != "isolated_apply"
            or not has_verified_apply_authorization(applied, run_id=run_id, request_id=request.request_id)
            or built.get("status") != "success" or built.get("real_build_performed") is not True
            or built.get("claims_real_build_success") is not True or built.get("input_execution_mode") != "isolated_apply"
            or not built.get("verified_artifacts")
            or Path(built["build_profile"]["working_directory"]).resolve() != source):
        raise ValueError("DSH_FOLLOWUP_OWNER_EVIDENCE_MISMATCH")
    source_binding = state["runtime_summary"]["reviewed_apply"]
    if (Path(source_binding["applied_source_root"]).resolve() != source
            or source_binding["applied_source_snapshot"] != _applied_source_snapshot(source)
            or applied["approved_diff_hash"] != source_binding["displayed_binding"]["diff_hash"]):
        raise ValueError("DSH_FOLLOWUP_APPLIED_SOURCE_CHANGED")
    for item in built["verified_artifacts"]:
        path = _contained_record_path(Path(item["resolved_path"]), source)
        if _file_ref(path)["sha256"] != item["sha256"]:
            raise ValueError("DSH_FOLLOWUP_BUILT_ARTIFACT_CHANGED")
    return {"run_id": run_id, "execution_config": selected, "delivery_root": delivery_root,
            "target_project": str(source), "source_snapshot_sha256": source_snapshot_sha256(source_snapshot(source)),
            "evidence_refs": refs}


def _verify_followup_configuration(selected: Path, options: dict, *, base_profile: str | None = None) -> dict:
    """Revalidate parent output and invocation inputs before normal owner dispatch.

    为什么要有它：保存的路由不是授权，父交付源码、配置和新正文的漂移不能悄悄进入下一次执行。
    """
    binding = read_json_object(selected)["followup_binding"]
    profile = _verify_ref(binding["base_profile_ref"])
    if base_profile is not None and profile != Path(base_profile).resolve():
        raise ValueError("DSH_FOLLOWUP_PROFILE_MISMATCH")
    workspace = Path(options["--workspace-root"]).resolve()
    input_root = _contained_record_path(workspace / ".dsh-inputs" / options["--run-id"], workspace)
    checkpoint = read_json_object(input_root / "followup_start.json")
    if (selected.resolve() != input_root / "execution.json"
            or checkpoint["base_profile_ref"] != binding["base_profile_ref"]
            or _verify_ref(checkpoint["execution_config_ref"], root=input_root) != selected.resolve()):
        raise ValueError("DSH_FOLLOWUP_CONFIGURATION_CHANGED")
    request = read_json_object(_verify_ref(binding["request_ref"], root=input_root))
    if request["run_id"] != options["--run-id"] or request["request_id"] != options["--request-id"]:
        raise ValueError("DSH_FOLLOWUP_REQUEST_IDENTITY_MISMATCH")
    if "worker_ref" in binding and _verify_ref(binding["worker_ref"], root=input_root) != Path(options["--patch-worker-config"]).resolve():
        raise ValueError("DSH_FOLLOWUP_WORKER_CHANGED")
    parent = binding["parent"]
    # 为什么要有它：新需求可选择新的验收配置，父任务仍按其原始配置验证，不能改写旧profile来迁就新任务。
    if (_verified_delivered_source(parent["execution_config"], parent["run_id"], parent["delivery_root"]) != parent
            or Path(options["--target-project"]).resolve() != Path(parent["target_project"]).resolve()):
        raise ValueError("DSH_FOLLOWUP_PARENT_CHANGED")
    return {"request": request, "binding": binding}


def _file_ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bind_project_compilation_feedback(worker: dict, receipt: dict, run_root: Path) -> None:
    """Map the existing compiler contract to the exact initialized source identity."""
    from src.mod_agent.local_entry import _optional_project_text_property

    evaluator = worker["behavior_evaluator"]
    contracts = {"make_compilation_feedback": {"compilation_kwargs", "output_root"},
        "make_startup_notice_evaluator": {"compilation_kwargs", "output_root", "probe_configuration_ref"}}
    if evaluator.get("factory_name") not in contracts:
        raise ValueError("DSH_NEW_PROJECT_COMPILATION_FACTORY_REQUIRED")
    _verify_ref(evaluator["factory_ref"])
    kwargs = evaluator.get("kwargs")
    required_kwargs = contracts[evaluator["factory_name"]]
    optional_kwargs = {"suppress_blank_messages"} if evaluator["factory_name"] == "make_startup_notice_evaluator" else set()
    if (not isinstance(kwargs, dict) or not required_kwargs.issubset(kwargs)
            or set(kwargs) - required_kwargs - optional_kwargs):
        raise ValueError("DSH_NEW_PROJECT_COMPILATION_CONTRACT_INVALID")
    parameters = kwargs["compilation_kwargs"]
    required = {"source_root", "workspace_root", "artifact_store_root", "output_root",
                "project_relative_path", "expected_artifact", "dependency_binding", "nuget_config_ref"}
    if not isinstance(parameters, dict) or not required.issubset(parameters):
        raise ValueError("DSH_NEW_PROJECT_COMPILATION_CONTRACT_INVALID")
    project = Path(receipt["project_file"])
    target = Path(receipt["target_project"])
    framework = _optional_project_text_property(project, property_name="TargetFramework")
    assembly = _optional_project_text_property(project, property_name="AssemblyName") or project.stem
    if not framework or assembly != receipt["identity"]["assembly_name"]:
        raise ValueError("DSH_NEW_PROJECT_ASSEMBLY_IDENTITY_INVALID")
    append = _optional_project_text_property(project, property_name="AppendTargetFrameworkToOutputPath")
    parameters.update({
        "source_root": str(run_root / "generated_mods" / target.name),
        "workspace_root": str(run_root), "artifact_store_root": str(run_root / "a"),
        "project_relative_path": project.relative_to(target).as_posix(),
        "expected_artifact": f"bin/Debug/{framework}/{assembly}.dll" if append != "false"
            else f"bin/Debug/{assembly}.dll",
        "dependency_binding": receipt["dependency_binding"],
        "nuget_config_ref": _file_ref(target / "nuget.config"),
    })


def _verify_ref(ref: dict, *, root: Path | None = None) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("DSH_NEW_PROJECT_REFERENCE_INVALID")
    path = Path(ref["path"])
    if not path.is_absolute():
        raise ValueError("DSH_NEW_PROJECT_REFERENCE_INVALID")
    if root is not None:
        path = _contained_record_path(path, root)
    if _file_ref(path) != ref:
        raise ValueError("DSH_NEW_PROJECT_REFERENCE_CHANGED")
    return path


def _verify_project_configuration(selected: Path, options: dict, *, base_profile: str | None = None) -> dict:
    """Keep the dynamic target tied to its original neutral project and request."""
    from src.mod_agent.new_mod_project import validate_new_mod_project_receipt

    document = read_json_object(selected)
    binding = document.get("new_project_binding")
    required_binding = {"receipt_ref", "scope_hash", "base_profile_ref", "request_ref"}
    if (not isinstance(binding, dict) or not required_binding.issubset(binding)
            or set(binding) - required_binding - {"worker_ref"}):
        raise ValueError("DSH_NEW_PROJECT_BINDING_INVALID")
    workspace = Path(options["--workspace-root"]).resolve()
    run_id = options["--run-id"]
    input_root = _contained_record_path(workspace / ".dsh-inputs" / run_id, workspace)
    if selected.resolve() != input_root / "execution.json":
        raise ValueError("DSH_NEW_PROJECT_CONFIGURATION_LOCATION_INVALID")
    profile = _verify_ref(binding["base_profile_ref"])
    if base_profile is not None and profile != Path(base_profile).resolve():
        raise ValueError("DSH_NEW_PROJECT_PROFILE_MISMATCH")
    checkpoint_path = _contained_record_path(input_root / "project_start.json", workspace)
    checkpoint = read_json_object(checkpoint_path)
    if checkpoint.get("base_profile_ref") != binding["base_profile_ref"]:
        raise ValueError("DSH_NEW_PROJECT_PROFILE_MISMATCH")
    if _verify_ref(checkpoint.get("execution_config_ref"), root=input_root) != selected.resolve():
        raise ValueError("DSH_NEW_PROJECT_CONFIGURATION_CHANGED")
    request_path = _verify_ref(binding["request_ref"], root=input_root)
    if "worker_ref" in binding:
        worker_path = _verify_ref(binding["worker_ref"], root=input_root)
        if worker_path != Path(options["--patch-worker-config"]).resolve():
            raise ValueError("DSH_NEW_PROJECT_WORKER_CHANGED")
    request = read_json_object(request_path)
    if (request.get("run_id") != run_id or request.get("request_id") != options["--request-id"]
            or not isinstance(request.get("description"), str)):
        raise ValueError("DSH_NEW_PROJECT_REQUEST_IDENTITY_INVALID")
    _verify_ref(binding["receipt_ref"], root=workspace)
    receipt = validate_new_mod_project_receipt(binding["receipt_ref"], workspace_root=workspace,
        expected_scope_hash=binding["scope_hash"])
    if (receipt["project_id"] != run_id or Path(receipt["target_project"]).resolve()
            != Path(options["--target-project"]).resolve()):
        raise ValueError("DSH_NEW_PROJECT_TARGET_MISMATCH")
    return {"receipt": receipt, "request": request, "binding": binding}


def preview_new_project(*, execution_config: str, run_id: str) -> dict:
    selected = task_execution_config(execution_config=execution_config, run_id=run_id)
    _, options = _configuration_arguments(selected)
    bound = _verify_project_configuration(Path(selected), options, base_profile=execution_config)
    return {"entry": "dsh-new-project", "status": "initialized_awaiting_product_execution",
        "run_id": run_id, "request_id": options["--request-id"],
        "execution_config_path": selected, "run_root": str(Path(options["--workspace-root"]) / run_id),
        "target_project": options["--target-project"], "project_identity": bound["receipt"]["identity"],
        "initialization_scope_hash": bound["receipt"]["scope_hash"],
        "initialization_receipt_ref": bound["binding"]["receipt_ref"],
        "description": bound["request"]["description"],
        "configured_arguments": read_json_object(selected)["arguments"],
        "model_calls_performed": 0, "product_execution_started": False,
        "build_performed": False, "game_deployment_performed": False}


def task_execution_config(*, execution_config: str, run_id: str) -> str:
    """Recover an allocated task's immutable invocation input after host restart."""
    _, options = _configuration_arguments(execution_config)
    if options.get("--run-id") == run_id:
        if "new_project_binding" in read_json_object(execution_config):
            _verify_project_configuration(Path(execution_config), options)
        if "followup_binding" in read_json_object(execution_config):
            _verify_followup_configuration(Path(execution_config), options)
        return execution_config
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("DSH_RUN_INSPECTION_IDENTITY_INVALID")
    workspace = Path(options["--workspace-root"]).resolve()
    selected = _contained_record_path(workspace / ".dsh-inputs" / run_id / "execution.json", workspace)
    if not selected.is_file():
        raise ValueError("DSH_DELIVERY_CONFIGURED_RUN_MISMATCH")
    _, allocated = _configuration_arguments(str(selected))
    if (allocated.get("--run-id") != run_id or not allocated.get("--request-id")
            or Path(allocated["--workspace-root"]).resolve() != workspace):
        raise ValueError("DSH_NEW_TASK_CONFIGURATION_IDENTITY_MISMATCH")
    if "new_project_binding" in read_json_object(selected):
        _verify_project_configuration(selected, allocated, base_profile=execution_config)
    elif "followup_binding" in read_json_object(selected):
        _verify_followup_configuration(selected, allocated, base_profile=execution_config)
    elif Path(allocated["--target-project"]).resolve() != Path(options["--target-project"]).resolve():
        raise ValueError("DSH_NEW_TASK_CONFIGURATION_IDENTITY_MISMATCH")
    return str(selected)


def build_product_argv(*, execution_config: str, description: str) -> list[str]:
    """Validate administrative bounds, then bind one new public invocation."""
    arguments, parsed = validate_product_profile(execution_config=execution_config, description=description)
    _, seen = _configuration_arguments(execution_config)
    if "followup_binding" in read_json_object(execution_config):
        bound = _verify_followup_configuration(Path(execution_config), seen)
        if description != bound["request"]["description"]:
            raise ValueError("DSH_FOLLOWUP_REQUEST_CHANGED")
    if "new_project_binding" in read_json_object(execution_config):
        bound = _verify_project_configuration(Path(execution_config), seen)
        if description != bound["request"]["description"]:
            raise ValueError("DSH_NEW_PROJECT_REQUEST_CHANGED")
    if parsed.run_id is not None:
        if (Path(parsed.workspace_root) / parsed.run_id).exists():
            raise ValueError("DSH_RUN_CONFIGURATION_RUN_ALREADY_EXISTS")
        return ["run-request", *arguments, "--", description]
    identity = uuid4().hex
    return ["run-request", *arguments, "--run-id", "dsh_" + identity,
            "--request-id", "request_" + identity, "--", description]


def validate_product_profile(*, execution_config: str, description: str):
    """Read administrative options without allocating or requiring an unused run ID.

    为什么要有它：选路和恢复需要读取已执行配置，新调用的独占目录检查仍由build_product_argv负责。
    """
    if (not isinstance(description, str) or not description.strip() or "\0" in description
            or len(description.encode("utf-8")) > MAX_REQUEST_BYTES):
        raise ValueError("DSH_RUN_REQUEST_INVALID")
    arguments, seen = _configuration_arguments(execution_config)
    policy = seen.get("--model-call-budget-policy", "task_total_only")
    required = _REQUIRED
    external = "--external-patch-profile" in seen
    if external:
        required = (required - {"--local-ollama", "--ollama-model", "--ollama-context-window"}) | {"--allow-source-disclosure", "--api-key-env", "--base-url-env"}
    if policy in {"task_total_only", "meter_only"}:
        required = required - {"--stage-model-call-limits"}
    if policy == "meter_only":
        # Scope: numeric experiment quotas are optional in the user's count-only mode.
        # 为什么要有它：计量入口不能为了满足旧配置形状要求填写虚构上限；权限和模型实际能力检查仍保留。
        required = required - {"--task-model-call-limit"}
    if not required.issubset(seen):
        raise ValueError("DSH_RUN_CONFIGURATION_REQUIRED_OPTIONS_MISSING: "
                         + ", ".join(sorted(required - set(seen))))
    for option in _PATHS & set(seen):
        if not Path(seen[option]).is_absolute():
            raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE: " + option)
    if "--patch-worker-config" in seen and not Path(seen["--patch-worker-config"]).is_file():
        raise ValueError("DSH_RUN_CONFIGURATION_WORKER_FILE_MISSING")
    dependency_api_binding(seen)
    knowledge_evidence_binding(seen)
    repository_profile_binding(seen)
    if external:
        external_profile_binding(seen)

    # Use the real parser for option types/choices; '--' keeps all description
    # bytes literal even when they start with a CLI option or shell expression.
    try:
        parsed = cli.build_parser().parse_args(["run-request", *arguments, "--", description])
    except SystemExit as exc:
        raise ValueError("DSH_RUN_CONFIGURATION_PARSER_REJECTED") from exc
    if not external:
        validate_loopback_http_base_url(parsed.ollama_base_url)
    cli._task_agent_edit_paths(parsed.task_agent_edit_paths)
    if ((policy != "meter_only" and (parsed.task_model_call_limit is None or parsed.task_model_call_limit < 1))
            or not external and (parsed.ollama_context_window < 1
                or parsed.ollama_max_output_tokens < 1
                or parsed.ollama_max_output_tokens >= parsed.ollama_context_window
                or not parsed.ollama_model.strip())):
        raise ValueError("DSH_RUN_CONFIGURATION_MODEL_BUDGET_INVALID")
    if parsed.ollama_timeout_seconds is not None and (
        not math.isfinite(parsed.ollama_timeout_seconds) or parsed.ollama_timeout_seconds <= 0
    ):
        raise ValueError("DSH_RUN_CONFIGURATION_TIMEOUT_INVALID")
    if parsed.stage_model_call_limits is not None:
        limits = read_json_object(parsed.stage_model_call_limits)
        if set(limits) != _STAGES or any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("DSH_RUN_CONFIGURATION_STAGE_LIMITS_INVALID")
    if (parsed.run_id is None) != (parsed.request_id is None):
        raise ValueError("DSH_RUN_CONFIGURATION_IDENTITY_PAIR_REQUIRED")
    if parsed.run_id is not None:
        if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
               for value in (parsed.run_id, parsed.request_id)):
            raise ValueError("DSH_RUN_CONFIGURATION_IDENTITY_INVALID")
    return arguments, parsed


def _contained_record_path(path: Path, workspace: Path) -> Path:
    # Reject links/reparse points before opening. The canonical containment check
    # also covers a relocated intermediate directory on supported platforms.
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None and (stat.S_ISLNK(info.st_mode) or
                            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("DSH_RUN_INSPECTION_LINK_NOT_ALLOWED")
    resolved = path.resolve()
    if resolved == workspace or not resolved.is_relative_to(workspace):
        raise ValueError("DSH_RUN_INSPECTION_PATH_OUTSIDE_WORKSPACE")
    return resolved


def _read_run_record(path: Path, workspace: Path, run_id: str) -> dict | None:
    selected = _contained_record_path(path, workspace)
    try:
        info = selected.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RUN_RECORD_BYTES:
        raise ValueError("DSH_RUN_INSPECTION_RECORD_SIZE_OR_TYPE_INVALID")
    with selected.open("rb") as handle:
        raw = handle.read(MAX_RUN_RECORD_BYTES + 1)
    if len(raw) > MAX_RUN_RECORD_BYTES:
        raise ValueError("DSH_RUN_INSPECTION_RECORD_SIZE_OR_TYPE_INVALID")
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict) or result.get("run_id") != run_id:
        raise ValueError("DSH_RUN_INSPECTION_RECORD_IDENTITY_INVALID")
    return result


def inspect_run(*, execution_config: str, run_id: str) -> dict:
    """Return saved public-entry records; never infer success or start a run."""
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("DSH_RUN_INSPECTION_IDENTITY_INVALID")
    _, seen = _configuration_arguments(execution_config)
    configured_workspace = seen.get("--workspace-root")
    if configured_workspace is None or not Path(configured_workspace).is_absolute():
        raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE: --workspace-root")
    workspace = Path(configured_workspace).resolve()
    run_root = _contained_record_path(workspace / run_id, workspace)
    manifest = _read_run_record(run_root / "run_manifest.json", workspace, run_id)
    result = _read_run_record(run_root / "run_result.json", workspace, run_id)
    try:
        task_config = task_execution_config(execution_config=execution_config, run_id=run_id)
    except ValueError:
        task_config = None
    record = {
        "entry": "run-request-status", "read_only": True, "run_id": run_id,
        "run_root": str(run_root), "manifest": manifest, "result": result,
        "status": "recorded" if manifest is not None or result is not None else "not_found",
        "model_calls_performed": 0,
        "execution_config_path": task_config,
    }
    if any((run_root / "continuations").glob("initial_findings*.json")):
        from src.mod_agent.initial_findings_continuation import read_initial_findings_execution
        continuation = read_initial_findings_execution(run_root)
        if continuation is not None:
            record.update(original_result=result, continuation=continuation,
                          result={**(result or {}), **continuation})
    if (run_root / "continuations/pretransport_latest.json").exists():
        from src.mod_agent.interrupted_patch_continuation import read_interrupted_patch_execution
        continuation = read_interrupted_patch_execution(run_root, pretransport=True)
        record.update(original_result=result, continuation=continuation,
                      result={**(result or {}), **continuation})
    if (run_root / "continuations/interrupted_latest.json").exists():
        from src.mod_agent.interrupted_patch_continuation import read_interrupted_patch_execution
        continuation = read_interrupted_patch_execution(run_root)
        if continuation is not None:
            record.update(original_result=result, continuation=continuation,
                          result={**(result or {}), **continuation})
    return record


def evidence_locator(*, execution_config: str, run_id: str, evidence_context=(), evidence_roots=()) -> dict:
    """Describe a read-only invocation; do not create a report or execution config."""
    config = str(Path(execution_config).resolve())
    return {"execution_config_path": config, "run_id": run_id,
            "argv": ["python", "-m", "scripts.dsh_run_request", "--execution-config", config,
                     "--inspect-run", run_id, "--view", "evidence", "--format", "json",
                     *[arg for path in evidence_context for arg in ("--evidence-context", str(path))],
                     *[arg for path in evidence_roots for arg in ("--evidence-root", str(path))]]}


def inspect_evidence(*, execution_config: str, run_id: str, evidence_context=(), evidence_roots=()) -> dict:
    """Resolve the business identity using the fixed config, then query shared evidence."""
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("DSH_RUN_INSPECTION_IDENTITY_INVALID")
    _, options = _configuration_arguments(execution_config)
    configured = options.get("--run-id")
    if configured is not None and configured != run_id:
        raise ValueError("DSH_RUN_INSPECTION_CONFIG_IDENTITY_CONFLICT")
    workspace = options.get("--workspace-root")
    if not workspace or not Path(workspace).is_absolute():
        raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE: --workspace-root")
    from src.observability.dashboard.services.trace_service import TraceService

    extra = {}
    if evidence_context:
        extra["evidence_context"] = evidence_context
    if evidence_roots:
        extra["evidence_roots"] = evidence_roots
    result = TraceService().query_run_evidence(allowed_roots=[str(Path(workspace))], run_id=run_id, **extra)
    return {**result, "inspection": evidence_locator(execution_config=execution_config, run_id=run_id,
        evidence_context=evidence_context, evidence_roots=evidence_roots)}


def _with_inspection(result: dict, execution_config: str) -> dict:
    execution_config = result.get("execution_config_path") or execution_config
    if isinstance(result.get("product_result"), dict):
        result = {**result, "product_result": _with_inspection(result["product_result"], execution_config)}
    run_id = result.get("run_id")
    if isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        try:
            _, options = _configuration_arguments(execution_config)
            if options.get("--run-id") not in (None, run_id):
                execution_config = task_execution_config(execution_config=execution_config, run_id=run_id)
            return {**result, "inspection": evidence_locator(execution_config=execution_config, run_id=run_id)}
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            # 为什么要有它：查询配置缺失或冲突只能使入口不可用，不能伪造可用命令或改变已有产品结果。
            return {**result, "inspection": {"run_id": run_id, "status": "unavailable", "reason": str(error)}}
    return result


def _invoke_product(argv: list[str]) -> tuple[int, dict | None, str]:
    output = io.StringIO()
    with redirect_stdout(output):
        status = cli.main(argv)
    raw = output.getvalue()
    try:
        result = json.loads(raw)
    except ValueError:
        result = None
    return status, result if isinstance(result, dict) else None, raw


def _task_run_location(run_ref: dict[str, str]) -> tuple[Path, str, str, Path, Path]:
    selected = _verify_ref(run_ref)
    _, options = _configuration_arguments(str(selected))
    run_id, request_id = options.get("--run-id"), options.get("--request-id")
    workspace = Path(options.get("--workspace-root") or "")
    if (not isinstance(run_id, str) or not isinstance(request_id, str)
            or not workspace.is_absolute()):
        raise ValueError("DSH_TASK_RUN_IDENTITY_REQUIRED")
    run_root = _contained_record_path(workspace / run_id, workspace)
    return selected, run_id, request_id, run_root, run_root / "a"


def _suspended_task_run(run_ref: dict[str, str]) -> TaskRunSuspended:
    from src.mod_agent.model_qualification import canonical_hash

    _selected, run_id, request_id, run_root, store = _task_run_location(run_ref)
    manifest = read_json_object(run_root / "run_manifest.json")
    if manifest.get("run_id") != run_id or manifest.get("request_id") != request_id:
        raise ValueError("DSH_TASK_RUN_MANIFEST_IDENTITY_CHANGED")
    pointers = list((store / "v2/session-checkpoints").glob("*.json"))
    if len(pointers) != 1:
        raise ValueError("DSH_TASK_RUN_RESUME_POINT_AMBIGUOUS")
    pointer = read_json_object(pointers[0])
    checkpoint_path = _contained_record_path(store / pointer["checkpoint_ref"], store)
    checkpoint = read_json_object(checkpoint_path)
    pending = [call_id for call_id, row in checkpoint["loop"]["call_records"].items()
        if row.get("status") == "pending"]
    if checkpoint["loop"].get("phase") != "tool_pending" or len(pending) != 1:
        raise ValueError("DSH_TASK_RUN_EFFECT_RECEIPT_REQUIRED")
    output_root = Path(checkpoint["owner"]["output_root"])
    if not output_root.is_absolute() or not output_root.resolve().is_relative_to(store.resolve()):
        raise ValueError("DSH_TASK_RUN_RECEIPT_OUTSIDE_STORE")
    receipt_path = output_root / ("tool-receipt-" + canonical_hash(pending[0]) + ".json")
    receipt = read_json_object(receipt_path)
    if receipt.get("origin", {}).get("call_id") != pending[0]:
        raise ValueError("DSH_TASK_RUN_EFFECT_RECEIPT_CHANGED")
    # Scope: only an interrupted Effect with a durable Receipt is exposed as recoverable.
    # 为什么要有它：任意进程中断不能被补写成安全恢复点，未知效果必须先由原回执对账。
    return TaskRunSuspended(run_ref=deepcopy(run_ref), run_id=run_id, request_id=request_id,
        resume_ref=_file_ref(checkpoint_path), progress_receipt_ref=_file_ref(receipt_path))


def _completed_task_run(run_ref: dict[str, str], result: dict) -> TaskRunCompleted:
    _selected, run_id, request_id, run_root, _store = _task_run_location(run_ref)
    manifest_path = run_root / "run_manifest.json"
    manifest = read_json_object(manifest_path)
    if (manifest.get("run_id") != run_id or manifest.get("request_id") != request_id
            or result.get("run_id", run_id) != run_id or result.get("request_id", request_id) != request_id):
        raise ValueError("DSH_TASK_RUN_RESULT_IDENTITY_CHANGED")
    outcome_ref = result.get("outcome_summary_ref", manifest.get("outcome_summary_ref"))
    if not isinstance(outcome_ref, dict):
        raise ValueError("DSH_TASK_RUN_OUTCOME_MISSING: " + str({key: result.get(key)
            for key in ("status", "reason", "current_stage")}))
    _verify_ref(outcome_ref)
    outcome = read_json_object(outcome_ref["path"])
    status = result.get("status")
    if (outcome.get("run_id") != run_id or outcome.get("request_id") != request_id
            or not isinstance(status, str) or not status):
        raise ValueError("DSH_TASK_RUN_OUTCOME_INVALID")
    # Scope: the existing manifest is the Evidence Bundle root for this slice.
    # 为什么要有它：它已绑定终态、continuation 与Run Outcome，首切片无需复制原件或新增持久化bundle schema。
    return TaskRunCompleted(run_ref=deepcopy(run_ref), run_id=run_id, request_id=request_id,
        status=status, run_outcome_ref=deepcopy(outcome_ref), evidence_bundle_ref=_file_ref(manifest_path))


def execute_task_run(command: TaskRunCommand) -> TaskRunExecution:
    """Open or resume one repository Task Run behind the same interface.

    Scope: the registered repository Task Agent v2 path; legacy continuations remain outside this slice.
    为什么要有它：调用者只选择Open或Resume，生命周期、恢复点和唯一结果引用由同一deep module负责。
    """
    if isinstance(command, OpenTaskRun):
        base = _verify_ref(command.execution_config_ref)
        selected = prepare_new_task(execution_config=str(base), description=command.intent)
        run_ref = _file_ref(Path(selected))
        argv = build_product_argv(execution_config=selected, description=command.intent)
        try:
            _status, result, raw = _invoke_product(argv)
        except KeyboardInterrupt:
            try:
                return _suspended_task_run(run_ref)
            except (OSError, KeyError, TypeError, ValueError):
                raise KeyboardInterrupt() from None
        if result is None:
            raise ValueError("DSH_TASK_RUN_PRODUCT_RESULT_INVALID: " + raw[:256])
        return _completed_task_run(run_ref, result)
    if isinstance(command, ResumeTaskRun):
        _selected, _run_id, _request_id, _run_root, store = _task_run_location(command.run_ref)
        checkpoint = _verify_ref(command.resume_ref, root=store)
        from src.mod_agent.task_agent_v2_recovery import resume_v2_task_agent
        result = resume_v2_task_agent(artifact_store_root=store, checkpoint_ref=str(checkpoint))
        return _completed_task_run(command.run_ref, result)
    raise TypeError("DSH_TASK_RUN_COMMAND_REQUIRED")


def _dispatch_product(argv: list[str], *, execution_config: str) -> int:
    # 为什么要有它：新运行返回可用查询入口，但不能把产品失败或待批准状态改写为成功。
    status, result, raw = _invoke_product(argv)
    if isinstance(result, dict):
        print(json.dumps(_with_inspection(result, execution_config), ensure_ascii=True))
    else:
        print(raw, end="")
    return status


def list_runs(*, execution_config: str) -> dict:
    _, options = _configuration_arguments(execution_config)
    workspace = Path(options["--workspace-root"])
    if not workspace.is_absolute():
        raise ValueError("DSH_RUN_CONFIGURATION_PATH_MUST_BE_ABSOLUTE")
    records = []
    if workspace.is_dir():
        for root in sorted(workspace.iterdir()):
            if root.name.startswith(".") or not root.is_dir():
                continue
            record = inspect_run(execution_config=execution_config, run_id=root.name)
            if record["status"] == "recorded":
                result = record["result"] or record["manifest"]
                records.append({"run_id": root.name, "request_id": result.get("request_id"),
                                "status": result.get("status"), "run_root": record["run_root"]})
    return {"entry": "run-request-list", "read_only": True, "records": records,
            "model_calls_performed": 0}


def advance_task(*, execution_config: str, run_id: str, allocation_config: str | None = None) -> dict:
    """Start a prepared project or consume one supported, already authorized task revision."""
    selected = task_execution_config(execution_config=execution_config, run_id=run_id)
    record = inspect_run(execution_config=selected, run_id=run_id)
    if record["status"] == "not_found":
        if "followup_binding" in read_json_object(selected):
            # 为什么要有它：准备后中断的派生任务须恢复已绑定正文和身份，重新分配会丢失谱系并刷新预算。
            _, options = _configuration_arguments(selected)
            bound = _verify_followup_configuration(Path(selected), options)
            description = bound["request"]["description"]
        else:
            preview = preview_new_project(execution_config=execution_config, run_id=run_id)
            description = preview["description"]
        output = io.StringIO()
        with redirect_stdout(output):
            cli.main(build_product_argv(execution_config=selected, description=description))
        return json.loads(output.getvalue())
    # 为什么要有它：已到待审批且无需解释修订的任务先返回既有结果，不读取无关续接配置或重复派发。
    if record["result"] is not None and record["result"].get("status") == "awaiting_approval" and not record["result"].get("request_interpretation_revision"):
        return record["result"]
    if record["manifest"] is not None and allocation_config is not None:
        allocation = read_json_object(allocation_config)
        if allocation.get("pretransport_patch") is not None:
            if set(allocation) != {"pretransport_patch"}:
                raise ValueError("PRETRANSPORT_PATCH_ALLOCATION_INVALID")
            from src.mod_agent.interrupted_patch_continuation import resume_interrupted_patch
            return resume_interrupted_patch(execution_config=selected, run_root=record["run_root"],
                run_id=run_id, recovery_config=allocation["pretransport_patch"], pretransport=True)
        if allocation.get("interrupted_patch") is not None:
            from src.mod_agent.interrupted_patch_continuation import resume_interrupted_patch
            return resume_interrupted_patch(execution_config=selected, run_root=record["run_root"],
                run_id=run_id, recovery_config=allocation["interrupted_patch"])
    if record["manifest"] is None or record["result"] is None:
        return {**record, "status": "task_incomplete_inspect_before_retry",
                "reason": "DSH_TASK_PROGRESS_INCOMPLETE", "automatic_execution_started": False}
    if allocation_config is None:
        return record["result"]
    allocation = read_json_object(allocation_config)
    if allocation.get("candidate_reverification") is not None:
        from src.mod_agent.candidate_verification_continuation import resume_candidate_verification
        return resume_candidate_verification(execution_config=selected, run_root=record["run_root"],
            run_id=run_id, recovery_config=allocation)
    if allocation.get("request_interpretation_revision") is None:
        return record["result"]
    from src.mod_agent.request_interpretation_continuation import advance_authorized_revision
    return advance_authorized_revision(run_root=record["run_root"], run_id=run_id,
        execution_config=selected, allocation_config=allocation_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--inspect-run")
    parser.add_argument("--view", choices=["evidence"])
    parser.add_argument("--format", choices=["json", "markdown"])
    parser.add_argument("--evidence-context", action="append", default=[])
    parser.add_argument("--evidence-root", action="append", default=[])
    parser.add_argument("--new-task", action="store_true")
    parser.add_argument("--route-request", action="store_true")
    parser.add_argument("--route-preview", action="store_true")
    parser.add_argument("--conversation-id")
    parser.add_argument("--clarification-for")
    parser.add_argument("--routing-turn-id")
    parser.add_argument("--active-run-id")
    parser.add_argument("--followup-delivery-root")
    parser.add_argument("--parent-execution-config")
    parser.add_argument("--initialize-project-name")
    parser.add_argument("--preview-project")
    parser.add_argument("--start-project")
    parser.add_argument("--advance-task")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--prepare-findings-continuation")
    parser.add_argument("--execute-findings-continuation")
    parser.add_argument("--findings-allocation-config")
    parser.add_argument("--decision-stdin", action="store_true")
    args = parser.parse_args(argv)
    try:
        # 为什么要有它：只读查询必须在任何执行分支之前拒绝混用，不能由分支顺序决定副作用。
        if (args.inspect_run is not None or args.view is not None or args.format is not None
                or args.evidence_context or args.evidence_root):
            if args.inspect_run is None or ((args.format is not None or args.evidence_context or args.evidence_root) and args.view != "evidence"):
                raise ValueError("DSH_RUN_EVIDENCE_INSPECTION_REQUIRED")
            allowed = {"execution_config", "inspect_run", "view", "format", "evidence_context", "evidence_root"}
            if any(value not in (None, False) for name, value in vars(args).items() if name not in allowed):
                raise ValueError("DSH_RUN_INSPECTION_OPERATION_CONFLICT")
            if args.view == "evidence":
                result = inspect_evidence(execution_config=args.execution_config, run_id=args.inspect_run,
                    evidence_context=args.evidence_context, evidence_roots=args.evidence_root)
                if args.format == "markdown":
                    from src.observability.dashboard.services.run_evidence import (
                        render_evidence_markdown,
                    )
                    # 为什么要有它：Windows管道也需稳定UTF-8，不能让人读视图因本机代码页损坏中文。
                    reconfigure = getattr(sys.stdout, "reconfigure", None)
                    if callable(reconfigure):
                        reconfigure(encoding="utf-8")
                    print(render_evidence_markdown(result))
                else:
                    print(json.dumps(result, ensure_ascii=True))
            else:
                print(json.dumps(inspect_run(execution_config=args.execution_config,
                                            run_id=args.inspect_run), ensure_ascii=True))
            return 0
        if args.route_request or args.route_preview:
            if (args.route_request and args.route_preview or args.new_task or args.initialize_project_name
                    or args.preview_project or args.start_project or args.advance_task or args.list_runs
                    or args.inspect_run or args.prepare_findings_continuation or args.execute_findings_continuation
                    or args.decision_stdin):
                raise ValueError("DSH_TASK_ROUTING_OPERATION_CONFLICT")
            from scripts.dsh_request_routing import run_natural_request
            body = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
            if len(body) > MAX_REQUEST_BYTES:
                raise ValueError("DSH_RUN_REQUEST_INVALID")
            result = run_natural_request(execution_config=args.execution_config,
                description=body.decode("utf-8"), active_run_id=args.active_run_id,
                allocation_config=args.findings_allocation_config, delivery_root=args.followup_delivery_root,
                parent_execution_config=args.parent_execution_config, preview_only=args.route_preview,
                **({"conversation_id": args.conversation_id} if args.conversation_id is not None else {}),
                **({"clarification_for": args.clarification_for} if args.clarification_for is not None else {}),
                **({"turn_id": args.routing_turn_id} if args.routing_turn_id is not None else {}))
            print(json.dumps(_with_inspection(result, result.get("execution_config_path") or args.parent_execution_config or args.execution_config), ensure_ascii=True))
            # 为什么要有它：已交付任务的只读续接也是成功结果，非零退出码会让客户端把正确回执当成执行失败。
            return 0 if result["status"] in {"awaiting_approval", "route_selected", "needs_clarification", "built", "delivered"} else 2
        if (args.active_run_id or args.followup_delivery_root or args.parent_execution_config
                or args.conversation_id or args.clarification_for or args.routing_turn_id):
            raise ValueError("DSH_TASK_ROUTING_OPERATION_REQUIRED")
        if args.advance_task is not None:
            result = advance_task(execution_config=args.execution_config, run_id=args.advance_task,
                allocation_config=args.findings_allocation_config)
            print(json.dumps(_with_inspection(result, args.execution_config), ensure_ascii=True))
            return 0 if result.get("status") == "awaiting_approval" else 2
        if args.preview_project is not None:
            print(json.dumps(preview_new_project(execution_config=args.execution_config,
                run_id=args.preview_project), ensure_ascii=True))
            return 0
        if args.start_project is not None:
            preview = preview_new_project(execution_config=args.execution_config, run_id=args.start_project)
            product_argv = build_product_argv(execution_config=preview["execution_config_path"],
                description=preview["description"])
            return _dispatch_product(product_argv, execution_config=preview["execution_config_path"])
        if args.execute_findings_continuation is not None:
            if not args.findings_allocation_config or not args.decision_stdin:
                raise ValueError("INITIAL_FINDINGS_EXPLICIT_APPROVAL_CHANNEL_REQUIRED")
            run_id = args.execute_findings_continuation
            selected = task_execution_config(execution_config=args.execution_config, run_id=run_id)
            arguments, _ = _configuration_arguments(selected)
            record = inspect_run(execution_config=selected, run_id=run_id)
            return _dispatch_product(["run-request", *arguments, "--resume-initial-findings", selected,
                "--findings-stage-allocation", args.findings_allocation_config, "--findings-approval-stdin",
                "--", record["manifest"]["request"]], execution_config=selected)
        if args.prepare_findings_continuation is not None:
            from src.mod_agent.initial_findings_continuation import (
                prepare_initial_findings_continuation,
                preview_initial_findings_execution,
                read_initial_findings_execution,
            )
            run_id = args.prepare_findings_continuation
            record = inspect_run(execution_config=args.execution_config, run_id=run_id)
            if args.findings_allocation_config:
                prepared = preview_initial_findings_execution(run_root=record["run_root"], run_id=run_id,
                    execution_config=record["execution_config_path"], allocation_config=args.findings_allocation_config)
            else:
                current = read_initial_findings_execution(record["run_root"])
                if current is not None:
                    print(json.dumps(current, ensure_ascii=True))
                    return 0
                prepared = prepare_initial_findings_continuation(run_root=record["run_root"], run_id=run_id,
                    execution_config=record["execution_config_path"])
            # Captured source can contain BOM/non-console characters. JSON escapes
            # preserve the exact values on Windows consoles as well as DSH's UTF-8 pipe.
            print(json.dumps(prepared, ensure_ascii=True))
            return 0
        if args.list_runs:
            print(json.dumps(list_runs(execution_config=args.execution_config), ensure_ascii=False))
            return 0
        body = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("DSH_RUN_REQUEST_INVALID")
        description = body.decode("utf-8")
        if args.initialize_project_name is not None:
            selected = prepare_new_task(execution_config=args.execution_config, description=description,
                project_name=args.initialize_project_name)
            _, options = _configuration_arguments(selected)
            print(json.dumps(preview_new_project(execution_config=args.execution_config,
                run_id=options["--run-id"]), ensure_ascii=True))
            return 0
        selected = (prepare_new_task(execution_config=args.execution_config, description=description)
                    if args.new_task else args.execution_config)
        product_argv = build_product_argv(execution_config=selected, description=description)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "DSH_INPUT_ERROR", "reason": str(exc)}, ensure_ascii=False))
        return 2
    # The owner distinguishes pending approval, domain failure and platform
    # failure. Do not replace its result with a bridge-created success record.
    return _dispatch_product(product_argv, execution_config=selected)


if __name__ == "__main__":
    raise SystemExit(main())
