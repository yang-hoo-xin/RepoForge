import ast
import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import ChatResponse, LLMFactory, OllamaLLM, OllamaLLMError
from src.mod_agent.bindings import BindingResolver
from src.mod_agent.build import BuildAdapter, BuildCommandResult, BuildProfile
from src.mod_agent.execution import (
    ExactDiffApprovalRejectedError,
    ExactTextPatchApplier,
    PatchApplyError,
)
from src.mod_agent.proposal import LLMExactPatchGenerator, PatchGenerationError
from src.mod_agent.local_entry import (
    EXACT_DIFF_REJECTION_TOKEN,
    LOCAL_DIAGNOSTIC_CONTENT_FILENAME,
    LOCAL_DIAGNOSTIC_CONTENT_MAX_BYTES,
    CompiledArtifactInspectionProfile,
    LocalEntryError,
    NuGetIsolatedBuildExecution,
    RunLocalGameFolderBindingV1,
    _assert_no_absolute_local_paths,
    _assert_ollama_model_identity_stable,
    _atomic_write_frozen_patch_attempt,
    _bind_objective_failure_to_single_changed_file,
    _canonicalize_run_local_build_diagnostic_paths,
    _build_run_local_il_inspector,
    _build_local_patch_frozen_resume_identity,
    _claim_local_frozen_patch_resume,
    _create_nuget_isolated_build_profile,
    _create_run_local_member_access_resolver,
    _copy_validated_directory_tree,
    _directory_copy_path_argument,
    _external_evidence_items,
    _local_model_admission_record,
    _local_no_external_disclosure_record,
    _local_patch_generation_correction_allowed,
    _implementation_with_runtime_repair_bindings,
    _binary_tree_snapshot,
    _binary_tree_snapshot_sha256,
    _materialize_game_folder_build_dependency,
    _materialize_repair_span_change,
    _objective_verification_source_views,
    _overlay_generated_patch_changes_on_plan,
    _persist_graph_terminal_failure,
    _persist_graph_terminal_result,
    _persist_local_failure_diagnostics,
    _patch_generation_calls_remaining,
    _patch_model_input_disclosure_data_classes,
    _patch_generation_obligation_projection,
    _prepare_local_frozen_patch_attempt,
    _project_text_property,
    _repair_review_context,
    _read_exact_diff_approval,
    _read_frozen_patch_attempt,
    _read_ollama_server_version,
    _request_requirement_coverage_model,
    _read_utf8_source_snapshot,
    _repair_history_fact_constraints,
    _repair_provider_constraints,
    _repair_span_response_schema,
    _resume_local_frozen_patch_attempt,
    _remove_generated_build_directory,
    _resolve_project_path_property,
    _runtime_admission_record,
    _runtime_continuation_backup_report,
    _runtime_observation_retry_output_report,
    _runtime_profile_preflight_record,
    _runtime_repair_spans,
    _runtime_source_diagnostic_lines,
    _runtime_verification_chain,
    _run_nuget_isolated_build,
    _run_nuget_isolated_verification_command,
    _sanitized_endpoint_url,
    _select_ollama_model_identity,
    _source_disclosure_record,
    _unique_repair_diagnostics,
    _verify_bepin_plugin_identity,
    _verify_exact_snapshot_change,
    _verify_final_compiled_artifact_if_built,
    _validate_ollama_context_window,
    _validate_game_folder_build_binding,
    _windows_extended_length_path,
    _verify_patch_model_input_disclosure,
    _verify_source_disclosure,
    PATCH_PROVIDER_PAYLOAD_DISCLOSURE_BINDINGS,
    SOURCE_DISCLOSURE_DATA_CLASSES,
    run_local_request,
)
from tests.fixtures.local_request_profile import request_budget

_OLLAMA_DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"


def _feature_wide_patch_model_input() -> dict[str, object]:
    return {
        "request": "Change the behavior.",
        "evidence": [{"excerpt": "verified source"}],
        "binding_registry": {"edit_bindings": []},
        "required_change_units": [],
        "side_effect_boundaries": ["Preserve unrelated behavior."],
        "target_behavior": "Implement the Feature-wide behavior.",
        "acceptance_criteria": ["Observe the requested behavior."],
        "runtime_verification_plan": ["Restart and observe the behavior."],
        "repair_observation": None,
        "repair_span_inventory": [],
    }


def _patch_agent_context_fixture() -> dict[str, object]:
    members = {
        f"State{index}": {
            "member_name": f"State{index}",
            "member_kind": "field",
            "declared_type": "float",
            "declaring_type": "Example.SharedState",
            "is_static": True,
            "declaration_span": {"start_line": index + 10, "end_line": index + 10},
        }
        for index in range(16)
    }
    metadata = {
        "declared_members": members,
        "declared_member_types": {name: item["declared_type"] for name, item in members.items()},
    }
    return {
        "request": "Preserve First Shot Match while disabling the spread override.",
        "target_behavior": "Keep shared state and independent First Shot effects.",
        "acceptance_criteria": ["False preserves Match; true retains current behavior."],
        "runtime_verification_plan": ["Restart before each configuration value."],
        "constraints": ["No unapproved path or source changes."],
        "diagnostic": {"source": "human semantic review", "observed": False},
        "allowed_edit_binding_ids": [f"edit_binding_{index:03}" for index in range(5)],
        "edit_binding_contexts": {
            f"edit_binding_{index:03}": {
                **metadata,
                "binding_id": f"edit_binding_{index:03}",
                "source_excerpt": "private void Update() { FirstShot.Match(); }\r\n",
                "source_hash": "a" * 64,
                "canonical_target_path": "Example/Behavior.cs",
                "target_span": {"start_line": index + 30, "end_line": index + 30},
            }
            for index in range(5)
        },
        "supporting_evidence": [
            {
                **metadata,
                "binding_id": f"support_binding_{index:03}",
                "evidence_id": f"evidence_{index:03}",
                "source_path": "Example/Behavior.cs",
                "source_hash": "a" * 64,
                "source_type": "target_source",
                "excerpt": "证据\n" * 200,
                "target_span": {"start_line": index + 50, "end_line": index + 50},
                "claim_supported": "Shared state is read here; this is not edit authority.",
            }
            for index in range(24)
        ],
        "required_change_units": [{"edit_binding_id": "edit_binding_000"}],
        "required_output_schema": {"type": "object", "properties": {"changes": {"type": "array"}}},
    }


def test_patch_agent_context_losslessly_compacts_after_existing_excerpt_projection() -> None:
    from src.libs.llm import Message
    from src.mod_agent import local_entry

    original = _patch_agent_context_fixture()
    original_json = json.dumps(original, ensure_ascii=False)
    messages = (
        Message(role="system", content="Generate an exact Patch."),
        Message(role="user", content=original_json),
    )
    projected = local_entry._patch_agent_initial_messages(messages)
    encoded = json.loads(projected[1].content)
    assert "patch_agent_context_encoding" in encoded
    assert encoded["patch_agent_context_encoding"]["version"] == "patch_agent_context_tables_v1"
    restored = local_entry._restore_patch_agent_context(encoded)
    expected = json.loads(original_json)
    for item in expected["supporting_evidence"]:
        full_excerpt = item["excerpt"]
        item.update(
            excerpt="证据\n" * 146,
            excerpt_sha256=hashlib.sha256(full_excerpt.encode("utf-8")).hexdigest(),
            excerpt_utf8_bytes=len(full_excerpt.encode("utf-8")),
            excerpt_truncated=True,
        )
    expected["patch_agent_context_policy"] = {
        "editable_source": "exact_binding_excerpt",
        "supporting_evidence": "ref_first_bounded_excerpt_1024_bytes",
        "additional_source": "plan_binding_tool_observation",
    }
    assert json.dumps(restored, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert messages[1].content == original_json
    assert encoded["edit_binding_contexts"]["edit_binding_000"]["source_excerpt"] == (
        original["edit_binding_contexts"]["edit_binding_000"]["source_excerpt"]
    )
    assert len(restored["edit_binding_contexts"]) == 5
    assert len(restored["supporting_evidence"]) == 24


def test_patch_agent_context_retains_unusual_metadata_and_missing_fields() -> None:
    from src.mod_agent import local_entry

    original = _patch_agent_context_fixture()
    original = json.loads(json.dumps(original))
    unusual = original["edit_binding_contexts"]["edit_binding_000"]
    unusual["declared_members"]["State0"]["extra_type_fact"] = {"nullable": False}
    unusual["declared_member_types"]["State0"] = "System.Single"
    del original["supporting_evidence"][0]["source_hash"]
    original["supporting_evidence"][1]["source_hash"] = None
    encoded = local_entry._compact_patch_agent_context(original)
    restored = local_entry._restore_patch_agent_context(encoded)

    assert json.dumps(restored, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert "source_hash" not in restored["supporting_evidence"][0]
    assert restored["supporting_evidence"][1]["source_hash"] is None


def test_patch_agent_context_missing_local_ref_cannot_pass_restore() -> None:
    from src.mod_agent import local_entry

    encoded = local_entry._compact_patch_agent_context(_patch_agent_context_fixture())
    encoded["edit_binding_contexts"]["edit_binding_000"]["member_metadata_ref"] = "unknown"
    with pytest.raises(LocalEntryError, match="LOCAL_PATCH_AGENT_CONTEXT_PROJECTION_INVALID"):
        local_entry._restore_patch_agent_context(encoded)


def test_patch_provider_payload_fields_bind_to_exact_disclosure_classes() -> None:
    binding_by_provider_field = {
        provider_field: data_classes
        for _, provider_field, data_classes in (
            PATCH_PROVIDER_PAYLOAD_DISCLOSURE_BINDINGS
        )
    }

    assert binding_by_provider_field["target_behavior"] == (
        "feature_target_behavior",
    )
    assert binding_by_provider_field["acceptance_criteria"] == (
        "feature_acceptance_criteria",
    )
    assert binding_by_provider_field["runtime_verification_plan"] == (
        "feature_runtime_verification_plan",
    )
    assert binding_by_provider_field["constraints"] == (
        "side_effect_boundaries",
    )
    assert binding_by_provider_field["edit_binding_contexts"] == (
        "retrieved_local_source_excerpts",
        "run_local_api_member_metadata_facts",
        "binding_registry",
    )
    assert binding_by_provider_field["supporting_evidence"] == (
        "retrieved_local_source_excerpts",
        "run_local_api_member_metadata_facts",
        "binding_registry",
    )
    assert binding_by_provider_field["allowed_edit_binding_ids"] == (
        "binding_registry",
    )
    assert _patch_model_input_disclosure_data_classes(
        _feature_wide_patch_model_input()
    ) == SOURCE_DISCLOSURE_DATA_CLASSES


@pytest.mark.parametrize(
    ("model_input_field", "data_class"),
    [
        ("target_behavior", "feature_target_behavior"),
        ("acceptance_criteria", "feature_acceptance_criteria"),
        ("runtime_verification_plan", "feature_runtime_verification_plan"),
        ("side_effect_boundaries", "side_effect_boundaries"),
    ],
)
def test_patch_disclosure_classes_are_presence_bound_to_model_input(
    model_input_field: str,
    data_class: str,
) -> None:
    model_input = _feature_wide_patch_model_input()
    model_input.pop(model_input_field)

    assert data_class not in _patch_model_input_disclosure_data_classes(model_input)


@pytest.mark.parametrize(
    "missing_data_class",
    [
        "feature_target_behavior",
        "feature_acceptance_criteria",
        "feature_runtime_verification_plan",
    ],
)
def test_external_patch_missing_feature_disclosure_fails_before_provider_chat(
    missing_data_class: str,
) -> None:
    record = _source_disclosure_record(
        provider="openai",
        model="qwen3.6-plus",
        base_url="https://example.invalid/compatible-mode/v1",
        approved_at="2026-08-31T00:00:00+00:00",
    )
    record["approved_data_classes"].remove(missing_data_class)
    provider_calls: list[dict[str, object]] = []

    def invoke_patch_provider() -> None:
        _verify_patch_model_input_disclosure(
            record,
            model_input=_feature_wide_patch_model_input(),
        )
        provider_calls.append({"chat": "called"})

    with pytest.raises(LocalEntryError, match=missing_data_class):
        invoke_patch_provider()

    assert provider_calls == []


def test_patch_generation_obligation_projection_preserves_one_feature_contract() -> None:
    target_behavior = (
        "  When disabled, retain the game-supplied value without replacement.  "
    )
    disabled_acceptance = "Observe the disabled behavior after restart."
    enabled_acceptance = "Observe the enabled preservation behavior after restart."
    runtime_verification = "  Restart before observing both modes.  "
    non_goal = "  Do not change unrelated configuration behavior.  "

    projection = _patch_generation_obligation_projection(
        feature_spec={
            "target_behavior": target_behavior,
            "acceptance_criteria": [disabled_acceptance, enabled_acceptance],
            "runtime_verification_plan": [runtime_verification],
        },
        implementation={"side_effect_boundaries": [non_goal]},
        planned_changes=[
            {
                "change_id": "chg_disabled",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_disabled",
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "rationale": target_behavior,
                "expected_validation": [disabled_acceptance],
            },
            {
                "change_id": "chg_enabled",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_enabled",
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "rationale": target_behavior,
                "expected_validation": [enabled_acceptance],
            },
        ],
    )

    assert projection == {
        "target_behavior": target_behavior,
        "required_change_units": [
            {
                "change_id": "chg_disabled",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_disabled",
                "hook_target_binding_id": None,
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "expected_validation": [disabled_acceptance],
            },
            {
                "change_id": "chg_enabled",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_enabled",
                "hook_target_binding_id": None,
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "expected_validation": [enabled_acceptance],
            },
        ],
        "side_effect_boundaries": [non_goal],
        "acceptance_criteria": [disabled_acceptance, enabled_acceptance],
        "runtime_verification_plan": [runtime_verification],
    }


def test_plan_validation_overlay_requires_exact_change_identity_and_value() -> None:
    planned_changes = [
        {
            "change_id": "chg_001",
            "edit_binding_id": "edit_binding_001",
            "expected_validation": [
                "Validate this bounded edit against the Feature-wide test_plan."
            ],
        }
    ]
    generated_change = {
        "change_id": "chg_001",
        "edit_binding_id": "edit_binding_001",
        "rationale": "Model-authored explanation only.",
        "expected_validation": ["forged model validation"],
        "rollback_hint": "Restore the prior source.",
        "patch": {"kind": "exact_text_replace"},
    }

    with pytest.raises(LocalEntryError, match="expected_validation"):
        _overlay_generated_patch_changes_on_plan(
            planned_changes=planned_changes,
            generated_changes=(generated_change,),
        )

    generated_change["expected_validation"] = list(
        planned_changes[0]["expected_validation"]
    )
    overlaid = _overlay_generated_patch_changes_on_plan(
        planned_changes=planned_changes,
        generated_changes=(generated_change,),
    )

    assert overlaid[0]["expected_validation"] == planned_changes[0][
        "expected_validation"
    ]


@pytest.mark.parametrize(
    "expected_validation",
    [None, [], [""], ["same", "same"]],
)
def test_patch_generation_obligation_projection_rejects_invalid_plan_validation(
    expected_validation: object,
) -> None:
    with pytest.raises(LocalEntryError, match="expected_validation"):
        _patch_generation_obligation_projection(
            feature_spec={
                "target_behavior": "Implement the requested behavior.",
                "acceptance_criteria": ["Observe the behavior."],
                "runtime_verification_plan": ["Restart and observe the behavior."],
            },
            implementation={
                "side_effect_boundaries": ["Preserve unrelated behavior."]
            },
            planned_changes=[
                {
                    "change_id": "chg_001",
                    "change_type": "modify_behavior",
                    "edit_binding_id": "edit_binding_001",
                    "supporting_binding_ids": [],
                    "operation": "minimal_evidence_bound_change",
                    "expected_validation": expected_validation,
                }
            ],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_behavior", ""),
        ("target_behavior", 1),
        ("acceptance_criteria", []),
        ("acceptance_criteria", [""]),
        ("runtime_verification_plan", []),
        ("runtime_verification_plan", [""]),
    ],
)
def test_patch_generation_obligation_projection_rejects_invalid_feature_contract(
    field: str,
    value: object,
) -> None:
    feature_spec: dict[str, object] = {
        "target_behavior": "Implement the requested behavior.",
        "acceptance_criteria": ["Observe the behavior."],
        "runtime_verification_plan": ["Restart and observe the behavior."],
    }
    feature_spec[field] = value

    with pytest.raises(LocalEntryError, match=field):
        _patch_generation_obligation_projection(
            feature_spec=feature_spec,
            implementation={"side_effect_boundaries": ["Preserve unrelated behavior."]},
            planned_changes=[
                {
                    "change_id": "chg_001",
                    "change_type": "modify_behavior",
                    "edit_binding_id": "edit_binding_001",
                    "supporting_binding_ids": [],
                    "operation": "minimal_evidence_bound_change",
                }
            ],
        )


def test_requirement_coverage_model_request_uses_schema_and_admission_recorder() -> None:
    captured: dict[str, object] = {}

    class _FakeLLM:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            recorder = kwargs["provider_request_recorder"]
            recorder(
                {
                    "body_utf8": "must-not-persist",
                    "body_sha256": "a" * 64,
                    "body_size_bytes": 123,
                    "timeout_seconds": 600.0,
                    "context_admission": {
                        "mode": "provider_body_utf8_proxy_plus_output_budget_v1",
                        "admitted": True,
                    },
                }
            )
            return ChatResponse(content='{"requirements":[]}', model="fake-local")

    schema = {"type": "object", "additionalProperties": False}
    response, provider_request = _request_requirement_coverage_model(
        llm=_FakeLLM(),
        payload={"message_type": "product_requirement_projection"},
        schema=schema,
        max_output_tokens=4096,
        capture_provider_request=True,
    )

    assert response.model == "fake-local"
    assert captured["kwargs"]["response_format"] == schema
    assert captured["kwargs"]["max_tokens"] == 4096
    assert provider_request["context_admission"]["admitted"] is True
    assert provider_request["body_sha256"] == "a" * 64
    assert provider_request["timeout_seconds"] == 600.0
    assert "body_utf8" not in provider_request


def test_run_local_request_has_no_direct_graph_control_plane() -> None:
    function = ast.parse(inspect.getsource(run_local_request)).body[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    direct_names = {
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
    }

    assert "preflight_product_execution" in direct_names
    assert "execute_product_execution" in direct_names
    assert "build_graph" not in direct_names
    assert "create_initial_state" not in direct_names
    assert "invoke" not in attribute_calls


def test_run_local_request_threads_one_strict_git_admission_through_all_source_checks() -> None:
    signature = inspect.signature(run_local_request)
    source = inspect.getsource(run_local_request)

    assert "source_git_admission" in signature.parameters
    assert signature.parameters["planning_target_evidence_ids"].default == ()
    assert source.count("strict_git_admission=source_git_admission") == 3
    assert signature.parameters["game_folder_build_binding"].default is None


def _runtime_settings(
    tmp_path: Path,
    *,
    enabled: bool = True,
) -> SimpleNamespace:
    executable = tmp_path / "GTFO.exe"
    executable.write_bytes(b"test executable")
    config_path = tmp_path / "BepInEx" / "config" / "BepInEx.cfg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[Logging.Disk]\nAppendLog = false\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        runtime=SimpleNamespace(
            gtfo=SimpleNamespace(
                enabled=enabled,
                game_executable=str(executable),
                working_directory=str(tmp_path),
                launch_args=["-test"],
                log_paths=["BepInEx/LogOutput.log"],
                screenshot_directory=str(tmp_path / "screenshots"),
                screenshot_timeout_seconds=15,
                startup_markers=["Chainloader startup complete"],
                acceptance_markers={"behavior works": "BEHAVIOR_OK"},
                timeout_seconds=120,
                poll_interval_seconds=1.0,
                output_limit_chars=20_000,
                terminate_after_capture=True,
            )
        )
    )


def _runtime_destination(tmp_path: Path) -> Path:
    destination = tmp_path / "BepInEx" / "plugins" / "BetterHitFeedback.dll"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def test_runtime_admission_selects_one_same_run_chain(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)

    admission = _runtime_admission_record(
        settings,
        run_gtfo_runtime=True,
        allow_game_launch=True,
        runtime_plugin_destination=_runtime_destination(tmp_path),
        allow_game_directory_write=True,
    )
    assert admission["requested"] is True
    assert admission["evidence_source"] == "gtfo_real_runtime_automation"
    assert admission["capability_scope"] == "runtime_harness_preflight"
    assert admission["allow_game_launch"] is True
    assert admission["authorization_scope"] == "this_run_only"
    assert admission["capability_report"]["status"] == "ready"
    assert admission["plugin_deployment"] == {
        "requested": True,
        "destination_path": str(_runtime_destination(tmp_path).resolve()),
        "allow_game_directory_write": True,
        "restore_after_runtime": True,
        "performed_by_run_request": True,
        "verified_deployed_artifact_hash": False,
        "destination_existed_at_admission": False,
        "destination_sha256_at_admission": None,
    }


def test_compile_only_admission_keeps_runtime_out_of_graph(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)

    admission = _runtime_admission_record(
        settings,
        run_gtfo_runtime=False,
        allow_game_launch=False,
        runtime_plugin_destination=None,
        allow_game_directory_write=False,
    )
    assert admission["requested"] is False
    assert admission["allow_game_launch"] is False
    assert admission["capability_report"] is None
    assert admission["plugin_deployment"]["requested"] is False


def test_runtime_admission_rejects_unbound_launch_authorization(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalEntryError, match="requires --run-gtfo-runtime"):
        _runtime_admission_record(
            _runtime_settings(tmp_path),
            run_gtfo_runtime=False,
            allow_game_launch=True,
            runtime_plugin_destination=None,
            allow_game_directory_write=False,
        )


def test_runtime_admission_requires_enabled_settings(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalEntryError, match="runtime.gtfo.enabled=true"):
        _runtime_admission_record(
            _runtime_settings(tmp_path, enabled=False),
            run_gtfo_runtime=True,
            allow_game_launch=False,
            runtime_plugin_destination=_runtime_destination(tmp_path),
            allow_game_directory_write=True,
        )


def test_runtime_admission_requires_explicit_launch_approval(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        LocalEntryError,
        match="GTFO game launch requires explicit approval",
    ):
        _runtime_admission_record(
            _runtime_settings(tmp_path),
            run_gtfo_runtime=True,
            allow_game_launch=False,
            runtime_plugin_destination=_runtime_destination(tmp_path),
            allow_game_directory_write=True,
        )


def test_runtime_admission_requires_physical_deployment_authorization(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalEntryError, match="allow-game-directory-write"):
        _runtime_admission_record(
            _runtime_settings(tmp_path),
            run_gtfo_runtime=True,
            allow_game_launch=True,
            runtime_plugin_destination=_runtime_destination(tmp_path),
            allow_game_directory_write=False,
        )


def test_runtime_profile_preflight_binds_marker_without_claiming_visual_or_audio(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)
    marker_criterion = (
        "STAGGER_FEEDBACK_SHOWN records visual_spawn_succeeded=true and "
        "audio_event_accepted=true without proving visibility or audibility."
    )
    settings.runtime.gtfo.acceptance_markers = {
        marker_criterion: (
            "signal=STAGGER_FEEDBACK_SHOWN "
            "target=ES_Hitreact.ActivateState "
            "message=visual_spawn_succeeded_and_audio_event_accepted "
            "visual_spawn_succeeded=true audio_event_accepted=true"
        )
    }

    result = _runtime_profile_preflight_record(
        settings,
        runtime_plugin_destination=_runtime_destination(tmp_path),
        feature_spec={
            "acceptance_criteria": [
                "The yellow X is visibly observed.",
                "The feature cue is audibly observed.",
                marker_criterion,
            ],
            "success_marker_contract": {
                "signal": "STAGGER_FEEDBACK_SHOWN",
                "code_level_required_facts": [
                    "visual_spawn_succeeded=true",
                    "audio_event_accepted=true",
                ],
                "emit_only_after": [
                    "feature visual creation returned success",
                    "feature audio event submission returned a nonzero playing id",
                ],
                "runtime_claim_limit": (
                    "The marker does not by itself prove visibility or audibility."
                ),
            },
        },
    )

    boundary = result["acceptance_boundary"]
    assert boundary["feature_spec_marker_criterion"] == marker_criterion
    assert boundary["marker_does_not_prove_visual_or_audio"] is True
    assert boundary["remaining_acceptance_criteria_require_separate_observation"] == [
        "The yellow X is visibly observed.",
        "The feature cue is audibly observed.",
    ]
    assert result["plugin_deployment"]["allow_game_directory_write"] is False
    assert result["plugin_deployment"]["performed"] is False
    assert result["capability_report"]["would_launch"] is False
    assert result["capability_report"]["authorization_status"] == (
        "not_granted_preflight_only"
    )
    assert result["capability_report"]["checks"]["explicit_launch_approval"] == {
        "status": "not_granted_preflight_only",
        "required": True,
        "allow_game_launch": False,
    }
    assert result["capability_report"]["checks"]["acceptance_markers"] == {
        "status": "warning",
        "configured_markers": settings.runtime.gtfo.acceptance_markers,
        "acceptance_criteria": [
            "The yellow X is visibly observed.",
            "The feature cue is audibly observed.",
            marker_criterion,
        ],
        "missing_acceptance_criteria": [
            "The yellow X is visibly observed.",
            "The feature cue is audibly observed.",
        ],
    }
    assert result["capability_report"]["warnings"] == [
        "GTFO Runtime acceptance criteria without configured markers require "
        "separate observation and cannot be verified by plugin markers"
    ]


def test_runtime_profile_preflight_binds_two_process_stagger_plan_without_writing(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)
    marker_criterion = (
        "STAGGER_FEEDBACK_SHOWN records visual_spawn_succeeded=true and "
        "audio_event_accepted=true without proving visibility or audibility."
    )
    disabled_criterion = (
        "With General.Enabled=false, a qualifying reaction produces no "
        "feature output."
    )
    settings.runtime.gtfo.acceptance_markers = {
        marker_criterion: (
            "signal=STAGGER_FEEDBACK_SHOWN "
            "target=ES_Hitreact.ActivateState "
            "message=visual_spawn_succeeded_and_audio_event_accepted "
            "visual_spawn_succeeded=true audio_event_accepted=true"
        )
    }
    settings.runtime.gtfo.operator_completion_required = True
    settings.runtime.gtfo.operator_completion_timeout_seconds = 900
    config_destination = (
        tmp_path / "BepInEx" / "config" / "local.betterhitfeedback.cfg"
    )
    disabled_config = tmp_path / "reviewed" / "general_enabled_false.cfg"
    enabled_config = tmp_path / "reviewed" / "general_enabled_true.cfg"
    disabled_config.parent.mkdir()
    disabled_config.write_text(
        "[General]\nEnabled = false\n",
        encoding="utf-8",
    )
    enabled_config.write_text(
        "[General]\nEnabled = true\n",
        encoding="utf-8",
    )
    settings.runtime.gtfo.stagger_acceptance = SimpleNamespace(
        enabled=True,
        config_path=str(config_destination),
        disabled_config_path=str(disabled_config),
        disabled_config_sha256=hashlib.sha256(
            disabled_config.read_bytes()
        ).hexdigest(),
        enabled_config_path=str(enabled_config),
        enabled_config_sha256=hashlib.sha256(
            enabled_config.read_bytes()
        ).hexdigest(),
    )

    result = _runtime_profile_preflight_record(
        settings,
        runtime_plugin_destination=_runtime_destination(tmp_path),
        feature_spec={
            "acceptance_criteria": [
                disabled_criterion,
                marker_criterion,
            ],
            "success_marker_contract": {
                "signal": "STAGGER_FEEDBACK_SHOWN",
                "code_level_required_facts": [
                    "visual_spawn_succeeded=true",
                    "audio_event_accepted=true",
                ],
                "runtime_claim_limit": "Marker is code-level only.",
            },
        },
    )

    stagger = result["stagger_acceptance"]
    assert stagger["status"] == "ready"
    assert stagger["writes_performed"] is False
    assert stagger["process_launched"] is False
    assert config_destination.exists() is False
    assert [item["mode"] for item in stagger["attempts"]] == [
        "general_enabled_false",
        "general_enabled_true",
    ]
    assert result["profile"]["operator_completion_required"] is True
    assert result["profile"]["operator_completion_timeout_seconds"] == 900


def test_runtime_profile_preflight_rejects_product_marker_as_startup_marker(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)
    marker_criterion = (
        "STAGGER_FEEDBACK_SHOWN records visual_spawn_succeeded=true and "
        "audio_event_accepted=true."
    )
    settings.runtime.gtfo.startup_markers = ["STAGGER_FEEDBACK_SHOWN"]
    settings.runtime.gtfo.acceptance_markers = {
        marker_criterion: (
            "signal=STAGGER_FEEDBACK_SHOWN "
            "visual_spawn_succeeded=true audio_event_accepted=true"
        )
    }

    with pytest.raises(LocalEntryError, match="cannot be a startup marker"):
        _runtime_profile_preflight_record(
            settings,
            runtime_plugin_destination=_runtime_destination(tmp_path),
            feature_spec={
                "acceptance_criteria": [marker_criterion],
                "success_marker_contract": {
                    "signal": "STAGGER_FEEDBACK_SHOWN",
                    "code_level_required_facts": [
                        "visual_spawn_succeeded=true",
                        "audio_event_accepted=true",
                    ],
                    "runtime_claim_limit": "Marker is code-level only.",
                },
            },
        )


def test_runtime_profile_preflight_does_not_read_non_runtime_provider_settings(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)
    marker_criterion = (
        "STAGGER_FEEDBACK_SHOWN records visual_spawn_succeeded=true and "
        "audio_event_accepted=true."
    )
    settings.runtime.gtfo.acceptance_markers = {
        marker_criterion: (
            "signal=STAGGER_FEEDBACK_SHOWN "
            "visual_spawn_succeeded=true audio_event_accepted=true"
        )
    }

    class RuntimeOnlySettings:
        runtime = settings.runtime

        def __getattribute__(self, name: str) -> object:
            if name in {"llm", "embedding", "vector_store", "rerank"}:
                raise AssertionError(
                    f"Runtime preflight read unrelated settings: {name}"
                )
            return object.__getattribute__(self, name)

    guarded_settings = RuntimeOnlySettings()
    destination = _runtime_destination(tmp_path)
    result = _runtime_profile_preflight_record(
        guarded_settings,
        runtime_plugin_destination=destination,
        feature_spec={
            "acceptance_criteria": [marker_criterion],
            "success_marker_contract": {
                "signal": "STAGGER_FEEDBACK_SHOWN",
                "code_level_required_facts": [
                    "visual_spawn_succeeded=true",
                    "audio_event_accepted=true",
                ],
                "runtime_claim_limit": "Marker is code-level only.",
            },
        },
    )
    admission = _runtime_admission_record(
        guarded_settings,
        run_gtfo_runtime=True,
        allow_game_launch=True,
        runtime_plugin_destination=destination,
        allow_game_directory_write=True,
        acceptance_criteria=(marker_criterion,),
    )

    assert result["capability_report"]["status"] == "ready"
    assert admission["capability_report"]["status"] == "ready"


def test_runtime_backup_report_reuses_empty_directories_and_blocks_material_state(
    tmp_path: Path,
) -> None:
    backup_root = tmp_path / "runtime_continuation_deployment_backup"
    (backup_root / "general_enabled_false" / "config").mkdir(parents=True)
    (backup_root / "general_enabled_false" / "plugin").mkdir(parents=True)

    report = _runtime_continuation_backup_report(
        tmp_path,
        stagger_enabled=True,
    )

    assert report["status"] == "ready"
    assert report["existing_material_entries"] == []
    assert report["empty_directories_reusable"] is True
    assert all(
        Path(path).name == "original.bin"
        for path in report["planned_backup_paths"]
    )
    assert report["maximum_path_chars"] == max(
        len(path) for path in report["planned_backup_paths"]
    )
    assert {
        Path(path).name
        for path in report["planned_process_diagnostic_paths"]
    } == {"stdout.log", "stderr.log"}
    assert report["maximum_process_diagnostic_path_chars"] == max(
        len(path) for path in report["planned_process_diagnostic_paths"]
    )

    (backup_root / "general_enabled_false" / "config" / "original.bin").write_bytes(
        b"unresolved backup"
    )
    with pytest.raises(LocalEntryError, match="unresolved prior deployment backup"):
        _runtime_continuation_backup_report(
            tmp_path,
            stagger_enabled=True,
        )


def test_runtime_backup_report_compacts_process_diagnostics_below_windows_limit(
    tmp_path: Path,
) -> None:
    target_root_chars = 178
    padding_chars = target_root_chars - len(str(tmp_path)) - 1
    assert padding_chars > 0
    run_root = tmp_path / ("x" * padding_chars)

    report = _runtime_continuation_backup_report(
        run_root,
        stagger_enabled=True,
    )
    old_dynamic_path = (
        run_root
        / "runtime_process_diagnostics"
        / "general_enabled_false"
        / (
            "feature_4f4c63c3802d_1785298858384466900.stdout.log"
        )
    )

    assert len(str(old_dynamic_path.resolve())) >= 260
    assert report["maximum_path_chars"] < 260
    assert report["maximum_process_diagnostic_path_chars"] < 260
    assert all(
        Path(path).name in {"stdout.log", "stderr.log"}
        for path in report["planned_process_diagnostic_paths"]
    )


def test_runtime_observation_retry_output_requires_empty_separate_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = tmp_path / "retry"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(LocalEntryError, match="must be empty"):
        _runtime_observation_retry_output_report(
            source_run_root=source_root,
            runtime_output_root=output_root.resolve(),
            feature_id="feature_001",
            stagger_enabled=True,
        )
    with pytest.raises(LocalEntryError, match="outside the source Build run root"):
        _runtime_observation_retry_output_report(
            source_run_root=source_root,
            runtime_output_root=source_root.resolve(),
            feature_id="feature_001",
            stagger_enabled=True,
        )


def test_runtime_observation_retry_preflights_operator_exchange_without_writes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = tmp_path / "retry"
    output_root.mkdir()

    report = _runtime_observation_retry_output_report(
        source_run_root=source_root,
        runtime_output_root=output_root.resolve(),
        feature_id="feature_001",
        stagger_enabled=True,
    )

    exchange = report["operator_completion_exchange"]
    assert exchange == {
        "status": "ready",
        "directory": str(
            (output_root / "operator_completion").resolve()
        ),
        "request_identity_source": (
            "GTFOOperatorCompletionRequest.request_id"
        ),
        "response_path_source": "request_file.response_path_suffix",
        "confirmation_path_source": (
            "request_file.confirmation_path_suffix"
        ),
        "confirmation_identity": "confirmation_file.response_sha256",
        "maximum_path_chars": exchange["maximum_path_chars"],
        "writes_performed": False,
    }
    assert exchange["maximum_path_chars"] < 260
    assert report["writes_performed"] is False
    assert list(output_root.iterdir()) == []


def test_runtime_verification_chain_requires_all_bound_artifacts() -> None:
    complete = _runtime_verification_chain(
        {
            "current_stage": "done",
            "artifacts": {
                "build_result": "artifact://build",
                "runtime_evidence": "artifact://runtime",
                "eval_report": "artifact://eval",
                "workflow_summary": "artifact://summary",
            },
        },
        requested=True,
    )
    incomplete = _runtime_verification_chain(
        {
            "current_stage": "blocked",
            "artifacts": {
                "build_result": "artifact://build",
                "runtime_evidence": "artifact://runtime",
            },
        },
        requested=True,
    )

    assert complete["workflow_chain_completed"] is True
    assert incomplete["workflow_chain_completed"] is False
    assert incomplete["artifact_refs"]["eval_report"] is None
    assert incomplete["artifact_refs"]["workflow_summary"] is None


def test_exact_diff_rejection_is_distinct_from_hash_mismatch() -> None:
    with pytest.raises(ExactDiffApprovalRejectedError, match="User rejected"):
        _read_exact_diff_approval(
            lambda _: EXACT_DIFF_REJECTION_TOKEN,
            {"diff_hash": "sha256:displayed"},
        )

    assert _read_exact_diff_approval(
        lambda _: "sha256:other",
        {"diff_hash": "sha256:displayed"},
    ) == ("sha256:other", None)


def test_unproven_exact_diff_reader_requires_structured_human_decision() -> None:
    binding = {
        "schema_version": "human_semantic_decision_v1",
        "feature_id": "feature_001",
        "feature_spec_artifact_hash": "sha256:feature",
        "proposal_id": "proposal_001",
        "proposal_payload_sha256": "sha256:proposal",
        "proposal_review_payload_sha256": "sha256:review",
        "semantic_assurance_payload_sha256": "sha256:assurance",
        "diff_hash": "sha256:displayed",
    }
    dry_run = {
        "diff_hash": "sha256:displayed",
        "human_semantic_review_required": True,
        "human_semantic_decision_binding": binding,
    }

    with pytest.raises(LocalEntryError, match="structured human semantic decision"):
        _read_exact_diff_approval(
            lambda _: "sha256:displayed",
            dry_run,
        )
    with pytest.raises(LocalEntryError, match="structured reject"):
        _read_exact_diff_approval(
            lambda _: EXACT_DIFF_REJECTION_TOKEN,
            dry_run,
        )

    decision = {
        **binding,
        "decision": "approve",
        "reviewer_identity": "human:test-reviewer",
    }
    assert _read_exact_diff_approval(
        lambda _: json.dumps(decision),
        dry_run,
    ) == ("sha256:displayed", decision)


def test_unproven_exact_diff_reader_forwards_structured_reject_for_binding_check() -> None:
    decision = {
        "schema_version": "human_semantic_decision_v1",
        "feature_id": "feature_001",
        "feature_spec_artifact_hash": "sha256:feature",
        "proposal_id": "proposal_001",
        "proposal_payload_sha256": "sha256:proposal",
        "proposal_review_payload_sha256": "sha256:review",
        "semantic_assurance_payload_sha256": "sha256:assurance",
        "diff_hash": "sha256:displayed",
        "decision": "reject",
        "reviewer_identity": "human:test-reviewer",
    }
    dry_run = {
        "diff_hash": "sha256:displayed",
        "human_semantic_review_required": True,
        "human_semantic_decision_binding": {
            key: value
            for key, value in decision.items()
            if key not in {"decision", "reviewer_identity"}
        },
    }

    assert _read_exact_diff_approval(
        lambda _: decision,
        dry_run,
    ) == ("sha256:displayed", decision)


def test_final_artifact_verification_requires_real_final_build(tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []

    def verify(path: Path, stage: str) -> dict[str, object]:
        calls.append((path, stage))
        return {"stage": stage}

    blocked = _verify_final_compiled_artifact_if_built(
        graph_result={"build_summary": {}},
        target_copy=tmp_path,
        expected_artifact=Path("bin/Debug/Example.dll"),
        verifier=verify,
    )

    assert blocked is None
    assert calls == []

    final_artifact = tmp_path / "bin" / "Debug" / "Example.dll"
    final_artifact.parent.mkdir(parents=True)
    final_artifact.write_bytes(b"final")
    verified = _verify_final_compiled_artifact_if_built(
        graph_result={"build_summary": {"claims_real_build_success": True}},
        target_copy=tmp_path,
        expected_artifact=Path("bin/Debug/Example.dll"),
        verifier=verify,
    )

    assert verified == {"stage": "final"}
    assert calls == [(final_artifact, "final")]


def test_graph_terminal_result_updates_same_run_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "request_id": "request_001",
                "status": "baseline_verified",
                "graph_invoked": False,
                "model_call_count": 0,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "run_id": "run_001",
        "current_stage": "blocked",
        "blockers": [{"code": "EXACT_DIFF_REJECTED"}],
        "graph_invoked": True,
        "model_call_count": 1,
        "patch_payload_parser_entered": True,
        "actual_write_performed": False,
    }

    _persist_graph_terminal_result(
        run_root=tmp_path,
        run_id="run_001",
        request_id="request_001",
        summary=summary,
        model_call_count=1,
        patch_payload_parser_entered=True,
        actual_write_performed=False,
    )

    result_path = tmp_path / "run_result.json"
    result_bytes = result_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_summary = {**summary, "status": "blocked"}
    expected_result_bytes = (
        json.dumps(expected_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert result_bytes == expected_result_bytes
    assert json.loads(result_bytes) == expected_summary
    assert manifest["status"] == "blocked"
    assert manifest["terminal_stage"] == "blocked"
    assert manifest["graph_invoked"] is True
    assert manifest["model_call_count"] == 1
    assert manifest["patch_payload_parser_entered"] is True
    assert manifest["actual_write_performed"] is False
    assert manifest["terminal_record"] == {
        "path": str(result_path),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }


def test_graph_patch_failure_updates_same_run_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "run_002", "request_id": "request_002"}),
        encoding="utf-8",
    )
    error = LocalEntryError("bound void method rejected")
    error.diagnostics = {
        "failure_boundary": "deterministic_patch_validation",
        "failure_classification": "csharp_return_value_in_void_method",
    }

    failure = _persist_graph_terminal_failure(
        run_root=tmp_path,
        run_id="run_002",
        request_id="request_002",
        error=error,
        model_call_count=1,
    )

    failure_path = tmp_path / "run_failure.json"
    failure_bytes = failure_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "deterministic_patch_validation"
    assert failure["graph_invoked"] is True
    assert failure["model_call_count"] == 1
    assert failure["patch_payload_parser_entered"] is True
    # A diagnostic phase name does not prove that the whole prior Graph run
    # performed no writes; an executor receipt is needed for that assertion.
    assert failure["actual_write_performed"] is None
    assert manifest["status"] == "failed"
    assert manifest["terminal_failure_stage"] == (
        "deterministic_patch_validation"
    )
    assert manifest["terminal_record"] == {
        "path": str(failure_path),
        "sha256": hashlib.sha256(failure_bytes).hexdigest(),
    }


def _nuget_isolated_profile(tmp_path: Path):
    run_root = tmp_path / "run"
    project = run_root / "generated_mods" / "BetterHitFeedback"
    bepinex = run_root / "build_dependencies" / "BepInEx"
    project.mkdir(parents=True)
    bepinex.mkdir(parents=True)
    config = project / "nuget.config"
    config.write_text("<configuration />", encoding="utf-8")
    return _create_nuget_isolated_build_profile(
        run_root=run_root,
        working_directory=project,
        config_file=config,
        bepinex_dir=bepinex,
        timeout_seconds=120,
    )


def _ollama_identity(*, digest: str = _OLLAMA_DIGEST) -> dict[str, object]:
    return {
        "name": "qwen3:4b-instruct",
        "model": "qwen3:4b-instruct",
        "digest": digest,
        "modified_at": "2026-07-16T14:06:42.9800065+08:00",
        "size": 2497293803,
        "details": {
            "context_length": 262144,
            "parameter_size": "4.0B",
            "quantization_level": "Q4_K_M",
        },
        "capabilities": ["completion", "tools"],
        "base_url": "http://127.0.0.1:11434",
    }


def test_nuget_isolated_profile_binds_explicit_run_local_inputs(
    tmp_path: Path,
) -> None:
    profile = _nuget_isolated_profile(tmp_path)

    assert profile.profile_id == "v2_codegen_real_local_dotnet_nuget_isolated"
    assert profile.restore_command == (
        "dotnet",
        "restore",
        "--configfile",
        str(profile.config_file),
        "--packages",
        str(profile.packages_path),
        f"-p:BepInExDir={profile.bepinex_dir}",
    )
    assert profile.build_command == (
        "dotnet",
        "build",
        "--no-restore",
        f"-p:BepInExDir={profile.bepinex_dir}",
    )
    assert profile.environment == {
        "NUGET_PACKAGES": str(profile.packages_path),
        "NUGET_HTTP_CACHE_PATH": str(profile.http_cache_path),
        "NUGET_SCRATCH": str(profile.scratch_path),
        "NUGET_PLUGINS_CACHE_PATH": str(profile.plugins_cache_path),
        "APPDATA": str(profile.appdata_path),
    }
    assert profile.timeout_seconds == 120
    assert profile.to_dict()["fully_hermetic_build_claimed"] is False
    for path in (
        profile.config_file,
        profile.packages_path,
        profile.http_cache_path,
        profile.scratch_path,
        profile.plugins_cache_path,
        profile.appdata_path,
        profile.user_config_file,
        profile.additional_user_config_path,
        profile.bepinex_dir,
    ):
        path.relative_to(profile.run_root)
    serialized = profile.to_dict()
    assert serialized["user_config_file"] == str(profile.user_config_file)
    assert serialized["user_config_source_sha256"] == hashlib.sha256(
        profile.config_file.read_bytes()
    ).hexdigest()


def test_nuget_isolated_profile_supports_standard_dotnet_project(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    project = run_root / "generated_mods" / "Inspector"
    project.mkdir(parents=True)
    config = project / "nuget.config"
    config.write_text("<configuration />", encoding="utf-8")

    profile = _create_nuget_isolated_build_profile(
        run_root=run_root,
        working_directory=project,
        config_file=config,
        bepinex_dir=None,
        timeout_seconds=120,
    )

    assert profile.bepinex_dir is None
    assert profile.restore_command == (
        "dotnet",
        "restore",
        "--configfile",
        str(profile.config_file),
        "--packages",
        str(profile.packages_path),
    )
    assert profile.build_command == (
        "dotnet",
        "build",
        "--no-restore",
    )
    assert profile.to_dict()["bepinex_dir"] is None


def test_run_local_build_diagnostics_project_only_bound_target_paths(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    target_root = run_root / "generated_mods" / "Inspector"
    target_file = target_root / "Program.cs"
    outside_file = tmp_path / "outside" / "Other.cs"
    target_file.parent.mkdir(parents=True)
    outside_file.parent.mkdir(parents=True)
    target_file.write_text("class Program {}\n", encoding="utf-8")
    outside_file.write_text("class Other {}\n", encoding="utf-8")
    profile = BuildProfile(
        profile_id="standard_dotnet",
        build_system="dotnet",
        working_directory=str(target_root),
        command=("dotnet", "build"),
        timeout_seconds=120,
    )
    result = BuildAdapter().from_supplied_output(
        feature_id="feature_001",
        profile=profile,
        exit_code=1,
        stderr=(
            f"{target_file}(21,3): error CS0103: target failure\n"
            f"{outside_file}(4,2): error CS0103: outside failure"
        ),
    )

    projected = _canonicalize_run_local_build_diagnostic_paths(
        result,
        run_root=run_root,
        isolated_target_root=target_root,
    )

    assert projected.diagnostics[0].file == (
        "generated_mods/Inspector/Program.cs"
    )
    assert projected.diagnostics[1].file == str(outside_file)
    resolver = BindingResolver(
        binding_registry={
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "canonical_target_path": (
                        "generated_mods/Inspector/Program.cs"
                    ),
                }
            ]
        }
    )
    assert resolver.resolve_edit_binding_for_path(
        str(projected.diagnostics[0].file),
        allowed_binding_ids={"edit_binding_001"},
    ) == ("edit_binding_001", None)
    assert resolver.resolve_edit_binding_for_path(
        str(projected.diagnostics[1].file),
        allowed_binding_ids={"edit_binding_001"},
    ) == (None, "unresolved")


def test_nuget_isolated_runner_restores_then_builds_with_same_profile(
    tmp_path: Path,
) -> None:
    profile = _nuget_isolated_profile(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="restore ok" if command == profile.restore_command else "build ok",
            stderr="",
        )

    first = _run_nuget_isolated_build(profile, command_runner=run)
    second = _run_nuget_isolated_build(profile, command_runner=run)

    assert [item[0] for item in calls] == [
        profile.restore_command,
        profile.build_command,
        profile.restore_command,
        profile.build_command,
    ]
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.restore_exit_code == 0
    assert first.build_exit_code == 0
    for _, kwargs in calls:
        assert kwargs["cwd"] == profile.working_directory
        assert kwargs["timeout"] == 120
        assert kwargs["check"] is False
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        for key, value in profile.environment.items():
            assert environment[key] == value
    for path in profile.environment.values():
        assert Path(path).is_dir()
    assert profile.user_config_file.read_bytes() == profile.config_file.read_bytes()
    assert profile.additional_user_config_path.is_dir()


def test_nuget_isolated_verification_command_uses_bound_workspace_and_environment(
    tmp_path: Path,
) -> None:
    profile = _nuget_isolated_profile(tmp_path)
    command = (
        "dotnet",
        "run",
        "--project",
        "RetryLedgerApp.csproj",
        "--no-restore",
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(selected, **kwargs):
        calls.append((selected, kwargs))
        return subprocess.CompletedProcess(
            selected,
            3,
            stdout="",
            stderr="RetryLedger.cs(1,1): error IA0001: objective failed",
        )

    result = _run_nuget_isolated_verification_command(
        profile,
        command,
        command_runner=run,
    )

    assert result.exit_code == 3
    assert result.stderr == (
        "RetryLedger.cs(1,1): error IA0001: objective failed"
    )
    assert len(calls) == 1
    selected, kwargs = calls[0]
    assert selected == command
    assert kwargs["cwd"] == profile.working_directory
    assert kwargs["timeout"] == profile.timeout_seconds
    assert kwargs["check"] is False
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    for key, value in profile.environment.items():
        assert environment[key] == value


def test_objective_failure_binds_to_only_dry_run_changed_file() -> None:
    result = _bind_objective_failure_to_single_changed_file(
        BuildCommandResult(
            exit_code=1,
            stderr="OBJECTIVE_FAILED total=19 operations=op-1,op-2",
            duration_ms=8,
        ),
        ("generated_mods/source/RetryLedger.cs",),
    )

    assert result.exit_code == 1
    assert result.stderr == (
        "generated_mods/source/RetryLedger.cs(1,1): error IA0001: "
        "Objective verification failed with exit code 1. Observation: "
        "OBJECTIVE_FAILED total=19 operations=op-1,op-2"
    )
    assert result.duration_ms == 8


def test_objective_failure_does_not_infer_ambiguous_changed_file() -> None:
    original = BuildCommandResult(exit_code=1, stderr="OBJECTIVE_FAILED")

    assert _bind_objective_failure_to_single_changed_file(
        original,
        ("one.cs", "two.cs"),
    ) is original


def test_repair_history_facts_exclude_historical_editable_text() -> None:
    old_text = "historical-before-text"
    old_replacement = "historical-after-text"
    constraints = _repair_history_fact_constraints(
        patch_proposal={
            "diff_intent": [
                {
                    "change_id": "chg_001",
                    "edit_binding_id": "edit_binding_001",
                    "operation": "minimal_evidence_bound_change",
                    "rationale": "historical rationale",
                    "patch": {
                        "before_text": old_text,
                        "after_text": old_replacement,
                    },
                }
            ]
        },
        patch_proposal_hash="patch-hash",
        build_result={
            "exit_code": 1,
            "diagnostics": [
                {
                    "code": "IA0001",
                    "message": "objective failed",
                    "error_class": "unknown",
                }
            ],
        },
        build_result_hash="build-hash",
        run_id="run-001",
        request_id="request-001",
    )

    rendered = "\n".join(constraints)
    assert old_text not in rendered
    assert old_replacement not in rendered
    assert "historical rationale" not in rendered
    prefix = "Non-editable prior execution facts: "
    facts = json.loads(constraints[2][len(prefix) :])
    assert facts == {
        "committed_artifacts": {
            "build_result": "build-hash",
            "patch_proposal": "patch-hash",
        },
        "objective_transition": {
            "after_prior_patch": {
                "exit_code": 1,
            },
            "before_prior_patch": "unavailable_in_current_run",
        },
        "prior_change_scope": [
            {
                "change_id": "chg_001",
                "edit_binding_id": "edit_binding_001",
                "operation": "minimal_evidence_bound_change",
            }
        ],
        "run_identity": {
            "request_id": "request-001",
            "run_id": "run-001",
        },
    }


def test_repair_provider_constraints_deduplicate_diagnostic_messages() -> None:
    constraints = _repair_provider_constraints(
        base_constraints=["base constraint"],
        diagnostics=[
            {"message": "same diagnostic", "line": 10},
            {"message": "same diagnostic", "line": 20},
            {"message": "different diagnostic", "line": 30},
            {"message": None, "line": 40},
        ],
        history_constraints=("history fact",),
    )

    assert constraints == [
        "base constraint",
        (
            "Use the supplied objective verification diagnostics to produce a "
            "different repair; do not repeat the prior failed behavior."
        ),
        "same diagnostic",
        "different diagnostic",
        "history fact",
    ]


def _game_folder_binding(root: Path) -> RunLocalGameFolderBindingV1:
    snapshot = _binary_tree_snapshot(root)
    return RunLocalGameFolderBindingV1(
        source_root=str(root.resolve()),
        snapshot_sha256=_binary_tree_snapshot_sha256(snapshot),
        file_count=len(snapshot),
    )


def _game_folder_dependency(tmp_path: Path) -> Path:
    root = tmp_path / "dependency_source" / "GameFolder"
    nested = root / "BepInEx" / "interop"
    nested.mkdir(parents=True)
    (root / "doorstop_config.ini").write_bytes(b"enabled=true\r\n")
    (nested / "GameAssembly.dll").write_bytes(b"\x00\x01\xffbinary\x00")
    return root


@pytest.mark.parametrize(
    ("path_text", "expected"),
    [
        (r"C:\repo\dependency", r"\\?\C:\repo\dependency"),
        (
            r"\\server\share\dependency",
            r"\\?\UNC\server\share\dependency",
        ),
        (r"\\?\C:\already\extended", r"\\?\C:\already\extended"),
    ],
)
def test_windows_extended_length_path_preserves_drive_and_unc_semantics(
    path_text: str,
    expected: str,
) -> None:
    assert _windows_extended_length_path(path_text) == expected


def test_directory_copy_path_argument_is_unchanged_off_windows() -> None:
    selected = Path("unchanged")
    assert _directory_copy_path_argument(selected, windows=False) is selected


def test_windows_extended_length_path_rejects_relative_and_device_paths() -> None:
    for selected in (r"relative\dependency", r"\\.\PhysicalDrive0"):
        with pytest.raises(LocalEntryError, match="absolute filesystem paths"):
            _windows_extended_length_path(selected)


def test_binary_tree_snapshot_sorts_normalized_paths_by_utf8_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    for name in ("é.bin", "z.bin", "a.bin"):
        (root / name).write_bytes(name.encode("utf-8"))

    snapshot = _binary_tree_snapshot(root)

    assert [item["path"] for item in snapshot] == sorted(
        ("é.bin", "z.bin", "a.bin"),
        key=lambda value: value.encode("utf-8"),
    )
    assert all("\\\\?\\" not in str(item["path"]) for item in snapshot)


def test_binary_tree_snapshot_rejects_reparse_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    selected = root / "reparse.bin"
    selected.write_bytes(b"not trusted as a regular dependency")
    original_lstat = os.lstat

    def reparse_lstat(path):
        observed = original_lstat(path)
        if str(path).endswith("reparse.bin"):
            return SimpleNamespace(
                st_mode=stat.S_IFREG,
                st_file_attributes=0x400,
                st_size=observed.st_size,
            )
        return observed

    monkeypatch.setattr("src.mod_agent.local_entry.os.lstat", reparse_lstat)

    with pytest.raises(LocalEntryError, match="symlink or reparse"):
        _binary_tree_snapshot(root)


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows long paths")
def test_validated_directory_copy_reads_real_source_path_over_260_chars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_parent"
    source.mkdir()
    padding = 250 - len(str(source)) - 1
    if padding <= 0 or padding > 240:
        pytest.skip("temporary root cannot form a bounded long-path fixture")
    nested = source / ("s" * padding)
    Path(_windows_extended_length_path(str(nested))).mkdir()
    long_file = nested / "payload_a.bin"
    second_long_file = nested / "payload_b.bin"
    assert len(str(long_file)) > 260
    assert len(str(second_long_file)) > 260
    payload = b"\x00long-source-a\xff\x00"
    second_payload = b"\x00long-source-b\xfe\x00"
    extended_file = Path(_windows_extended_length_path(str(long_file)))
    extended_second = Path(
        _windows_extended_length_path(str(second_long_file))
    )
    extended_file.write_bytes(payload)
    extended_second.write_bytes(second_payload)
    (source / "short.bin").write_bytes(b"short")
    assert extended_file.is_file()
    assert extended_second.is_file()

    initial_snapshot = _binary_tree_snapshot(source)
    initial_hash = _binary_tree_snapshot_sha256(initial_snapshot)
    assert len(initial_snapshot) == 3
    assert {item["path"] for item in initial_snapshot} == {
        "short.bin",
        f"{nested.name}/payload_a.bin",
        f"{nested.name}/payload_b.bin",
    }
    extended_second.write_bytes(b"changed-long-source-b")
    changed_snapshot = _binary_tree_snapshot(source)
    assert len(changed_snapshot) == 3
    assert _binary_tree_snapshot_sha256(changed_snapshot) != initial_hash

    legacy_incomplete_binding = RunLocalGameFolderBindingV1(
        source_root=str(source.resolve()),
        snapshot_sha256=(
            "c3d4c14ae937b72aa251ccadc239ebb73852d0f7a22a3b9758e969f04ea490b9"
        ),
        file_count=137,
    )
    canonical_source = tmp_path / "canonical" / "Project"
    canonical_source.mkdir(parents=True)
    with pytest.raises(LocalEntryError, match="file count drifted"):
        _validate_game_folder_build_binding(
            legacy_incomplete_binding,
            canonical_source_root=canonical_source,
        )

    destination = tmp_path / "normal_destination"
    _copy_validated_directory_tree(
        source=source.resolve(strict=True),
        destination=destination,
    )

    assert not str(destination).startswith("\\\\?\\")
    copied = destination / long_file.relative_to(source)
    copied_bytes = Path(_windows_extended_length_path(str(copied))).read_bytes()
    assert copied_bytes == payload
    assert hashlib.sha256(copied_bytes).hexdigest() == hashlib.sha256(
        payload
    ).hexdigest()
    assert extended_file.read_bytes() == payload
    assert _binary_tree_snapshot(destination) == changed_snapshot
    assert _binary_tree_snapshot(source) == changed_snapshot


def test_game_folder_source_containment_rejects_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy_calls: list[object] = []
    monkeypatch.setattr(
        "src.mod_agent.build.dependencies._copy_validated_directory_tree",
        lambda **kwargs: copy_calls.append(kwargs),
    )
    canonical = tmp_path / "canonical" / "Project"
    dependency_inside = canonical / "dependency"
    dependency_inside.mkdir(parents=True)
    (dependency_inside / "inside.bin").write_bytes(b"inside")

    with pytest.raises(LocalEntryError, match="cannot be inside canonical"):
        _validate_game_folder_build_binding(
            _game_folder_binding(dependency_inside),
            canonical_source_root=canonical,
        )

    container = tmp_path / "container"
    canonical_inside = container / "Project"
    canonical_inside.mkdir(parents=True)
    (container / "dependency.bin").write_bytes(b"container")
    with pytest.raises(LocalEntryError, match="cannot contain canonical"):
        _validate_game_folder_build_binding(
            _game_folder_binding(container),
            canonical_source_root=canonical_inside,
        )

    assert copy_calls == []


def test_game_folder_binding_snapshots_and_copies_nested_binary_tree(
    tmp_path: Path,
) -> None:
    canonical_source = tmp_path / "canonical" / "Project"
    canonical_source.mkdir(parents=True)
    dependency = _game_folder_dependency(tmp_path)
    binding = _game_folder_binding(dependency)
    before_source = _binary_tree_snapshot(dependency)
    run_root = tmp_path / "run"
    run_root.mkdir()

    identity = _validate_game_folder_build_binding(
        binding,
        canonical_source_root=canonical_source,
    )
    record = _materialize_game_folder_build_dependency(
        identity,
        run_root=run_root,
    )

    isolated = run_root / "build_dependencies" / "GameFolder"
    assert record["property_name"] == "GameFolder"
    assert record["source_file_count"] == 2
    assert record["copy_file_count"] == 2
    assert record["source_snapshot_sha256"] == record["copy_snapshot_sha256"]
    assert _binary_tree_snapshot(isolated) == before_source
    assert _binary_tree_snapshot(dependency) == before_source


def test_game_folder_binding_rejects_arbitrary_msbuild_property() -> None:
    with pytest.raises(ValueError, match="binding is invalid"):
        RunLocalGameFolderBindingV1(
            source_root="C:/dependency",
            snapshot_sha256="a" * 64,
            file_count=1,
            property_name="ArbitraryProperty",
        )


@pytest.mark.parametrize("mutation", ["count", "hash"])
def test_game_folder_binding_drift_rejects_before_any_dotnet_call(
    tmp_path: Path,
    mutation: str,
) -> None:
    canonical_source = tmp_path / "canonical" / "Project"
    canonical_source.mkdir(parents=True)
    dependency = _game_folder_dependency(tmp_path)
    binding = _game_folder_binding(dependency)
    dotnet_calls: list[tuple[str, ...]] = []
    if mutation == "count":
        (dependency / "new.bin").write_bytes(b"new")
    else:
        (dependency / "doorstop_config.ini").write_bytes(b"changed")

    with pytest.raises(LocalEntryError, match="file count|snapshot SHA256"):
        _validate_game_folder_build_binding(
            binding,
            canonical_source_root=canonical_source,
        )

    assert dotnet_calls == []


def test_game_folder_destination_reuse_and_escape_are_rejected(
    tmp_path: Path,
) -> None:
    canonical_source = tmp_path / "canonical" / "Project"
    canonical_source.mkdir(parents=True)
    dependency = _game_folder_dependency(tmp_path)
    identity = _validate_game_folder_build_binding(
        _game_folder_binding(dependency),
        canonical_source_root=canonical_source,
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    _materialize_game_folder_build_dependency(identity, run_root=run_root)

    with pytest.raises(LocalEntryError, match="already exists"):
        _materialize_game_folder_build_dependency(identity, run_root=run_root)

    project = run_root / "project"
    project.mkdir()
    config = project / "nuget.config"
    config.write_text("<configuration />", encoding="utf-8")
    outside = tmp_path / "outside_game"
    outside.mkdir()
    with pytest.raises(LocalEntryError, match="game_folder_dir must remain"):
        _create_nuget_isolated_build_profile(
            run_root=run_root,
            working_directory=project,
            config_file=config,
            bepinex_dir=None,
            timeout_seconds=120,
            game_folder_dir=outside,
        )


def test_game_folder_profile_binds_unique_property_to_restore_and_build(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    target = run_root / "generated_mods" / "Neutral"
    baseline = run_root / "baseline_source" / "Neutral"
    game_folder = run_root / "build_dependencies" / "GameFolder"
    for directory in (target, baseline, game_folder):
        directory.mkdir(parents=True)
    for project in (target, baseline):
        (project / "nuget.config").write_text("<configuration />", encoding="utf-8")

    profiles = tuple(
        _create_nuget_isolated_build_profile(
            run_root=run_root,
            working_directory=project,
            config_file=project / "nuget.config",
            bepinex_dir=None,
            timeout_seconds=120,
            game_folder_dir=game_folder,
        )
        for project in (target, baseline)
    )

    expected = f"-p:GameFolder={game_folder.resolve()}"
    assert profiles[0].game_folder_dir == profiles[1].game_folder_dir
    for profile in profiles:
        assert profile.restore_command.count(expected) == 1
        assert profile.build_command.count(expected) == 1
        assert profile.to_dict()["game_folder_dir"] == str(game_folder.resolve())


def test_actual2_relative_game_folder_fixture_passes_fake_restore_and_build(
    tmp_path: Path,
) -> None:
    canonical_source = tmp_path / "canonical" / "Neutral"
    canonical_source.mkdir(parents=True)
    project_bytes = b"<Project><Import Project='GameFolder.props' /></Project>\n"
    props_bytes = (
        b"<Project><PropertyGroup><GameFolder>../../build_experiment/fake_game"
        b"</GameFolder></PropertyGroup></Project>\n"
    )
    (canonical_source / "Neutral.csproj").write_bytes(project_bytes)
    (canonical_source / "GameFolder.props").write_bytes(props_bytes)
    dependency = _game_folder_dependency(tmp_path)
    dependency_before = _binary_tree_snapshot(dependency)
    canonical_before = _binary_tree_snapshot(canonical_source)
    run_root = tmp_path / "run"
    target = run_root / "generated_mods" / "Neutral"
    baseline = run_root / "baseline_source" / "Neutral"
    shutil.copytree(canonical_source, target)
    shutil.copytree(canonical_source, baseline)
    identity = _validate_game_folder_build_binding(
        _game_folder_binding(dependency),
        canonical_source_root=canonical_source,
    )
    record = _materialize_game_folder_build_dependency(identity, run_root=run_root)
    game_folder = Path(str(record["isolated_copy"]))
    calls: list[tuple[str, ...]] = []

    def command_runner(command, **_kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    for project in (target, baseline):
        config = project / "nuget.config"
        config.write_text("<configuration />", encoding="utf-8")
        profile = _create_nuget_isolated_build_profile(
            run_root=run_root,
            working_directory=project,
            config_file=config,
            bepinex_dir=None,
            timeout_seconds=120,
            game_folder_dir=game_folder,
        )
        result = _run_nuget_isolated_build(profile, command_runner=command_runner)
        assert result.restore_exit_code == 0
        assert result.build_exit_code == 0

    exact_argument = f"-p:GameFolder={game_folder.resolve()}"
    assert len(calls) == 4
    assert all(command.count(exact_argument) == 1 for command in calls)
    assert (target / "GameFolder.props").read_bytes() == props_bytes
    assert (baseline / "GameFolder.props").read_bytes() == props_bytes
    assert _binary_tree_snapshot(canonical_source) == canonical_before
    assert _binary_tree_snapshot(dependency) == dependency_before

def test_unique_repair_diagnostics_removes_only_exact_duplicates() -> None:
    first = {
        "file": "generated_mods/source/Program.cs",
        "line": 20,
        "column": 22,
        "code": "CS0103",
        "severity": "error",
        "message": "当前上下文中不存在名称 assemblyPath",
        "error_class": "unknown",
    }
    different_line = {**first, "line": 22, "column": 77}
    different_message = {**first, "message": "当前上下文中不存在名称 peReader"}
    diagnostics = [first, dict(first), different_line, different_message, dict(first)]

    projected = _unique_repair_diagnostics(diagnostics)

    assert projected == [first, different_line, different_message]
    assert projected[0] is not first
    assert diagnostics == [
        first,
        dict(first),
        different_line,
        different_message,
        dict(first),
    ]


def test_runtime_repair_spans_are_unique_current_source_handles() -> None:
    live_text = (
        "public void Record()\n"
        "{\n"
        "    if (retry)\n"
        "    {\n"
        "        Total += amount;\n"
        "        return;\n"
        "    }\n"
        "    Total += amount;\n"
        "}\n"
    )

    spans = _runtime_repair_spans(live_text, source_hash="live-hash")

    assert len(spans) == 1
    assert spans[0].selection_id == "s01"
    assert spans[0].source_hash == "live-hash"
    assert spans[0].method_name == "Record"
    assert spans[0].start_line == 1
    assert spans[0].end_line == 9
    assert spans[0].before_text == live_text
    assert spans[0].before_text.count("Total += amount;") == 2
    assert "{" in spans[0].before_text and "}" in spans[0].before_text
    schema = _repair_span_response_schema(spans)
    assert set(schema["properties"]) == {
        "selection_id",
        "replacement_text",
        "rationale",
    }
    assert "before_text" not in schema["properties"]


def test_runtime_source_diagnostic_lines_uses_exact_source_path_match() -> None:
    working_root = Path("work/root")
    source_path = working_root / "Program.cs"
    diagnostics = (
        {"file": "src/Program.cs", "line": "8"},
        {"file": "other/Program.cs", "line": "13"},
        {"file": "WORK/ROOT/Program.cs", "line": "21"},
        {"file": "Program.cs", "line": "34"},
    )

    observed = _runtime_source_diagnostic_lines(
        diagnostics,
        source_path=source_path,
        working_root=working_root,
    )

    assert observed == (34,)


def test_runtime_repair_spans_include_missing_bound_diagnostics() -> None:
    live_text = (
        "using System;\n"
        "using System.Text;\n"
        "using System.Linq;\n"
        "\n"
        "public class Worker\n"
        "{\n"
        "    public int Compute()\n"
        "    {\n"
        "        return 1;\n"
        "    }\n"
        "}\n"
    )
    line_count = len(live_text.splitlines())
    spans = _runtime_repair_spans(
        live_text,
        source_hash="live-hash",
        diagnostic_lines=(1, 2, 3, 4),
    )
    diagnostic_spans = [span for span in spans if span.method_name == "diagnostic_bound_region"]
    assert diagnostic_spans
    assert len(diagnostic_spans) == 1
    assert diagnostic_spans[0].start_line == 1
    assert diagnostic_spans[0].end_line == 4
    assert diagnostic_spans[0].before_text == "".join(
        live_text.splitlines(keepends=True)[:4]
    )
    method_span = next(span for span in spans if span.method_name == "Compute")
    assert method_span.start_line > diagnostic_spans[0].end_line
    assert diagnostic_spans[0].selection_id == "s01"
    assert diagnostic_spans[0].end_line <= line_count


def test_repair_span_materialization_restores_current_before_text() -> None:
    live_text = (
        "public void Record()\n"
        "{\n"
        "    Total += amount;\n"
        "    return;\n"
        "}\n"
    )
    spans = _runtime_repair_spans(live_text, source_hash="live-hash")
    selected = next(span for span in spans if "Total += amount;" in span.before_text)

    change = _materialize_repair_span_change(
        json.dumps(
            {
                "selection_id": selected.selection_id,
                "replacement_text": (
                    "public void Record()\n"
                    "{\n"
                    "    return;\n"
                    "}\n"
                ),
                "rationale": "Use the objective failure to change retry behavior.",
            }
        ),
        spans=spans,
        required_change={
            "change_id": "repair_build_0",
            "edit_binding_id": "edit_binding_001",
        },
        expected_validation=("behavior evidence observed",),
    )

    assert change["patch"]["before_text"] == selected.before_text
    assert change["patch"]["after_text"] == (
        "public void Record()\n"
        "{\n"
        "    return;\n"
        "}\n"
    )
    assert change["edit_binding_id"] == "edit_binding_001"
    assert change["expected_validation"] == ["behavior evidence observed"]

    with pytest.raises(PatchGenerationError, match="must change"):
        _materialize_repair_span_change(
            json.dumps(
                {
                    "selection_id": selected.selection_id,
                    "replacement_text": selected.before_text,
                    "rationale": "No change.",
                }
            ),
            spans=spans,
            required_change={
                "change_id": "repair_build_0",
                "edit_binding_id": "edit_binding_001",
            },
            expected_validation=(),
        )


def test_read_utf8_source_snapshot_preserves_exact_mixed_line_endings(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Mixed.cs"
    source_bytes = (
        "public void Record()\r\n"
        "{\n"
        "    Console.WriteLine(\"修复\");\r\n"
        "}\n"
    ).encode("utf-8")
    source_path.write_bytes(source_bytes)

    live_text, live_hash = _read_utf8_source_snapshot(source_path)

    assert live_text.encode("utf-8") == source_bytes
    assert live_text.count("\r\n") == 2
    assert live_text.count("\n") == 4
    assert "修复" in live_text
    assert live_hash == hashlib.sha256(source_bytes).hexdigest()


def test_read_utf8_source_snapshot_rejects_invalid_utf8_without_extra_writes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Invalid.cs"
    source_path.write_bytes(b"valid prefix\n\xff")

    with pytest.raises(UnicodeDecodeError):
        _read_utf8_source_snapshot(source_path)

    assert list(tmp_path.iterdir()) == [source_path]


def test_repair_review_context_binds_current_evidence_without_changing_path() -> None:
    live_text = "public void Record()\r\n{\n    // 修复\r\n    Total += amount;\n}\r\n"
    live_hash = hashlib.sha256(live_text.encode("utf-8")).hexdigest()
    implementation = {
        "feature_id": "feature-001",
        "binding_registry": {
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "evidence_id": "old-evidence",
                    "canonical_target_path": "generated_mods/RetryLedger.cs",
                    "source_hash": "old-hash",
                    "target_span": {"start_line": 1, "end_line": 4},
                }
            ]
        },
    }
    evidence_pack = {
        "feature_id": "feature-001",
        "items": [{"evidence_id": "old-evidence", "excerpt": "old source"}],
    }
    runtime_binding = {
        "binding_id": "edit_binding_001",
        "evidence_id": "repair-live",
        "canonical_target_path": "generated_mods/RetryLedger.cs",
        "source_hash": live_hash,
        "target_span": {"start_line": 1, "end_line": 4},
    }
    runtime_evidence = {
        "evidence_id": "repair-live",
        "source_path": "generated_mods/RetryLedger.cs",
        "source_hash": live_hash,
        "file_hash": live_hash,
        "doc_hash": live_hash,
        "excerpt": live_text,
    }

    projected_implementation, projected_evidence = _repair_review_context(
        implementation,
        evidence_pack,
        runtime_bindings=(runtime_binding,),
        runtime_evidence=(runtime_evidence,),
    )

    projected_binding = projected_implementation["binding_registry"][
        "edit_bindings"
    ][0]
    assert projected_binding == runtime_binding
    assert projected_evidence["items"][-1] == runtime_evidence
    assert projected_evidence["items"][0]["evidence_id"] == "old-evidence"
    assert implementation["binding_registry"]["edit_bindings"][0][
        "evidence_id"
    ] == "old-evidence"

    with pytest.raises(LocalEntryError, match="excerpt does not match its source hash"):
        _repair_review_context(
            implementation,
            evidence_pack,
            runtime_bindings=(runtime_binding,),
            runtime_evidence=(
                {**runtime_evidence, "excerpt": live_text + "// drift\n"},
            ),
        )

    with pytest.raises(LocalEntryError, match="current source binding"):
        _repair_review_context(
            implementation,
            evidence_pack,
            runtime_bindings=(
                {**runtime_binding, "source_hash": "0" * 64},
            ),
            runtime_evidence=(runtime_evidence,),
        )

    with pytest.raises(LocalEntryError, match="cannot change an edit binding path"):
        _repair_review_context(
            implementation,
            evidence_pack,
            runtime_bindings=(
                {**runtime_binding, "canonical_target_path": "generated_mods/Other.cs"},
            ),
            runtime_evidence=(runtime_evidence,),
        )


def test_objective_verification_source_views_are_bounded_current_sources(
    tmp_path: Path,
) -> None:
    (tmp_path / "Program.cs").write_text(
        "var actual = ledger.Total;\nif (actual != 12) return 1;\n",
        encoding="utf-8",
    )
    (tmp_path / "RetryLedger.cs").write_text(
        "public sealed class RetryLedger {}\n",
        encoding="utf-8",
    )
    generated = tmp_path / "obj" / "Generated.cs"
    generated.parent.mkdir()
    generated.write_text("generated", encoding="utf-8")

    views = _objective_verification_source_views(tmp_path)

    assert [view["source_path"] for view in views] == [
        "Program.cs",
        "RetryLedger.cs",
    ]
    assert "actual != 12" in views[0]["source_text"]
    assert all(len(view["source_sha256"]) == 64 for view in views)


def test_runtime_repair_binding_revision_authorizes_current_live_span(
    tmp_path: Path,
) -> None:
    target = tmp_path / "RetryLedger.cs"
    target.write_text("initial line\ncurrent repair line\n", encoding="utf-8")
    original_plan = {
        "binding_registry": {
            "schema_version": "binding_registry_v1",
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "canonical_target_path": "RetryLedger.cs",
                    "source_hash": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "target_span": {"start_line": 1, "end_line": 1},
                }
            ],
            "supporting_bindings": [],
        }
    }
    proposal = {
        "proposal_id": "repair-proposal",
        "diff_intent": [
            {
                "change_id": "repair_build_0",
                "edit_binding_id": "edit_binding_001",
                "patch": {
                    "kind": "exact_text_replace",
                    "before_text": "current repair line",
                    "after_text": "repaired current line",
                    "expected_occurrences": 1,
                },
            }
        ],
    }
    review = {"allowed_next_stage": "dry_run_apply", "decision": "approve_dry_run"}
    applier = ExactTextPatchApplier()

    stale_plan = deepcopy(original_plan)
    stale_plan["binding_registry"]["edit_bindings"][0]["source_hash"] = "0" * 64
    with pytest.raises(PatchApplyError, match="source_hash does not match"):
        applier.run(
            workspace_root=tmp_path,
            proposal=proposal,
            implementation_plan=stale_plan,
            review=review,
            allow_write=False,
        )

    with pytest.raises(PatchApplyError, match="outside edit binding target_span"):
        applier.run(
            workspace_root=tmp_path,
            proposal=proposal,
            implementation_plan=original_plan,
            review=review,
            allow_write=False,
        )

    revised_plan = _implementation_with_runtime_repair_bindings(
        original_plan,
        runtime_bindings=(
            {
                "binding_id": "edit_binding_001",
                "canonical_target_path": "RetryLedger.cs",
                "source_hash": hashlib.sha256(target.read_bytes()).hexdigest(),
                "target_span": {"start_line": 1, "end_line": 2},
            },
        ),
    )
    result = applier.run(
        workspace_root=tmp_path,
        proposal=proposal,
        implementation_plan=revised_plan,
        review=review,
        allow_write=False,
    )

    assert result["execution_mode"] == "dry_run"
    span_check = next(
        item
        for item in result["static_checks"]
        if item["check_id"] == "binding_target_span"
    )
    assert span_check["bound_end_line"] == 2
    assert span_check["observed_start_line"] == 2

    with pytest.raises(LocalEntryError, match="cannot change an edit binding path"):
        _implementation_with_runtime_repair_bindings(
            original_plan,
            runtime_bindings=(
                {
                    "binding_id": "edit_binding_001",
                    "canonical_target_path": "Other.cs",
                    "source_hash": "current-hash",
                    "target_span": {"start_line": 1, "end_line": 2},
                },
            ),
        )


def test_nuget_isolated_runner_rejects_run_local_user_config_drift(
    tmp_path: Path,
) -> None:
    profile = _nuget_isolated_profile(tmp_path)
    profile.user_config_file.parent.mkdir(parents=True)
    profile.user_config_file.write_text(
        "<configuration><drift /></configuration>",
        encoding="utf-8",
    )

    with pytest.raises(LocalEntryError, match="user config does not match"):
        _run_nuget_isolated_build(profile, command_runner=MagicMock())


def test_nuget_isolated_runner_stops_when_restore_fails(tmp_path: Path) -> None:
    profile = _nuget_isolated_profile(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fail_restore(command, **_):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="restore failed",
        )

    result = _run_nuget_isolated_build(profile, command_runner=fail_restore)

    assert calls == [profile.restore_command]
    assert result.exit_code == 1
    assert result.restore_exit_code == 1
    assert result.build_exit_code is None
    assert result.to_dict()["build_executed"] is False


def test_nuget_isolated_profile_rejects_paths_outside_run_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    project = tmp_path / "outside-project"
    config = project / "nuget.config"
    bepinex = run_root / "build_dependencies" / "BepInEx"
    project.mkdir()
    config.write_text("<configuration />", encoding="utf-8")
    bepinex.mkdir(parents=True)

    with pytest.raises(LocalEntryError, match="working_directory must remain"):
        _create_nuget_isolated_build_profile(
            run_root=run_root,
            working_directory=project,
            config_file=config,
            bepinex_dir=bepinex,
            timeout_seconds=120,
        )


def test_nuget_isolated_runner_propagates_timeout_fail_closed(
    tmp_path: Path,
) -> None:
    profile = _nuget_isolated_profile(tmp_path)

    def time_out(command, **_):
        raise subprocess.TimeoutExpired(command, timeout=120)

    with pytest.raises(subprocess.TimeoutExpired):
        _run_nuget_isolated_build(profile, command_runner=time_out)


def test_generated_build_cleanup_tolerates_only_vanished_entries(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "obj"
    generated.mkdir()
    (generated / "remaining.tmp").write_bytes(b"generated")
    vanished = generated / "already-gone.tmp"

    def remove(path: Path, *, onerror):
        error = FileNotFoundError(str(vanished))
        onerror(Path.unlink, str(vanished), (FileNotFoundError, error, None))
        shutil.rmtree(path)

    result = _remove_generated_build_directory(
        generated,
        tree_remover=remove,
    )

    assert result == {
        "path": str(generated.resolve()),
        "removed": True,
        "ignored_missing_paths": [str(vanished)],
        "cleanup_attempt_count": 1,
        "directory_not_empty_retry_count": 0,
    }
    assert not generated.exists()


def test_generated_build_cleanup_retries_winerror_145_once(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "obj"
    generated.mkdir()
    (generated / "late.tmp").write_bytes(b"generated")
    calls = 0

    def remove(path: Path, *, onerror):
        nonlocal calls
        calls += 1
        if calls == 1:
            error = OSError(41, "directory not empty", str(path))
            error.winerror = 145
            raise error
        shutil.rmtree(path, onerror=onerror)

    result = _remove_generated_build_directory(
        generated,
        tree_remover=remove,
    )

    assert calls == 2
    assert result["removed"] is True
    assert result["cleanup_attempt_count"] == 2
    assert result["directory_not_empty_retry_count"] == 1
    assert not generated.exists()


def test_generated_build_cleanup_stops_after_second_winerror_145(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "obj"
    generated.mkdir()
    calls = 0

    def remain_nonempty(path: Path, *, onerror):
        nonlocal calls
        calls += 1
        error = OSError(41, "directory not empty", str(path))
        error.winerror = 145
        raise error

    with pytest.raises(OSError) as captured:
        _remove_generated_build_directory(
            generated,
            tree_remover=remain_nonempty,
        )

    assert calls == 2
    assert captured.value.winerror == 145
    assert generated.exists()


def test_generated_build_cleanup_propagates_other_io_errors(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "obj"
    generated.mkdir()
    calls = 0

    def deny(_path: Path, *, onerror):
        nonlocal calls
        calls += 1
        error = PermissionError("denied")
        onerror(Path.unlink, str(generated / "locked.tmp"), (PermissionError, error, None))

    with pytest.raises(PermissionError, match="denied"):
        _remove_generated_build_directory(
            generated,
            tree_remover=deny,
        )

    assert calls == 1
    assert generated.exists()


def test_run_local_inspector_build_is_source_and_output_hash_bound(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "tracked-inspector"
    project_dir.mkdir()
    project = project_dir / "r8_il_inspector.csproj"
    project.write_text("<Project Sdk=\"Microsoft.NET.Sdk\" />", encoding="utf-8")
    (project_dir / "Program.cs").write_text(
        "internal static class Program { static void Main() {} }",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "build":
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "r8_il_inspector.dll").write_bytes(b"compiled inspector")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = _build_run_local_il_inspector(
        CompiledArtifactInspectionProfile(
            project_path=project,
            method_symbol="Example.Plugin.Load",
            expected_guid="example.mod",
            expected_name="Example",
            expected_version="1.0.0",
        ),
        run_root=run_root,
        command_runner=run,
    )

    assert len(calls) == 2
    assert calls[0][0][0:2] == ("dotnet", "restore")
    assert calls[1][0][0:2] == ("dotnet", "build")
    assert "--no-restore" in calls[1][0]
    assert result["source_project_sha256"] == hashlib.sha256(
        project.read_bytes()
    ).hexdigest()
    assert result["inspector_sha256"] == hashlib.sha256(
        b"compiled inspector"
    ).hexdigest()
    assert result["fully_hermetic_build_claimed"] is False
    for _, kwargs in calls:
        environment = kwargs["env"]
        for name in (
            "NUGET_PACKAGES",
            "NUGET_HTTP_CACHE_PATH",
            "NUGET_SCRATCH",
            "NUGET_PLUGINS_CACHE_PATH",
            "APPDATA",
        ):
            Path(environment[name]).resolve().relative_to(run_root.resolve())
    record = json.loads(
        (
            run_root
            / "verification_tools"
            / "r8_il_inspector"
            / "verification_tool_build.json"
        ).read_text(encoding="utf-8")
    )
    assert record == result


def test_run_local_inspector_build_records_restore_failure_without_build(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "tracked-inspector"
    project_dir.mkdir()
    project = project_dir / "r8_il_inspector.csproj"
    project.write_text("<Project Sdk=\"Microsoft.NET.Sdk\" />", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[tuple[str, ...]] = []

    def fail_restore(command, **_):
        calls.append(command)
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="failed")

    with pytest.raises(LocalEntryError, match="restore failed with exit code 7"):
        _build_run_local_il_inspector(
            CompiledArtifactInspectionProfile(
                project_path=project,
                method_symbol="Example.Plugin.Load",
                expected_guid="example.mod",
                expected_name="Example",
                expected_version="1.0.0",
            ),
            run_root=run_root,
            command_runner=fail_restore,
        )

    assert len(calls) == 1
    record = json.loads(
        (
            run_root
            / "verification_tools"
            / "r8_il_inspector"
            / "verification_tool_build.json"
        ).read_text(encoding="utf-8")
    )
    assert record["restore_exit_code"] == 7
    assert record["build_exit_code"] is None
    assert record["inspector_exists"] is False


def test_member_access_resolver_prefers_standalone_run_local_bepinex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    standalone = run_root / "build_dependencies" / "BepInEx"
    game_folder = run_root / "build_dependencies" / "GameFolder" / "BepInEx"
    standalone.mkdir(parents=True)
    game_folder.mkdir(parents=True)
    inspector = run_root / "verification_tools" / "inspector.dll"
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(b"inspector")
    received: list[Path] = []

    class RecordingResolver:
        def __init__(self, _inspector_path: Path, *, dependency_root: Path) -> None:
            received.append(dependency_root)

    monkeypatch.setattr(
        "src.mod_agent.local_entry.RunLocalMemberAccessResolver",
        RecordingResolver,
    )

    resolver = _create_run_local_member_access_resolver(
        inspector_path=inspector,
        run_root=run_root,
    )

    assert isinstance(resolver, RecordingResolver)
    assert received == [standalone.resolve()]
    assert received[0] is not None


def test_member_access_resolver_falls_back_to_game_folder_bepinex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    game_folder = run_root / "build_dependencies" / "GameFolder" / "BepInEx"
    game_folder.mkdir(parents=True)
    inspector = run_root / "verification_tools" / "inspector.dll"
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(b"inspector")
    received: list[Path] = []

    class RecordingResolver:
        def __init__(self, _inspector_path: Path, *, dependency_root: Path) -> None:
            received.append(dependency_root)

    monkeypatch.setattr(
        "src.mod_agent.local_entry.RunLocalMemberAccessResolver",
        RecordingResolver,
    )

    _create_run_local_member_access_resolver(
        inspector_path=inspector,
        run_root=run_root,
    )

    assert received == [game_folder.resolve()]
    assert received[0].is_relative_to(run_root.resolve())


def test_member_access_resolver_rejects_missing_run_local_dependency_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    host_bepinex = tmp_path / "host" / "BepInEx"
    host_bepinex.mkdir(parents=True)
    inspector = run_root / "verification_tools" / "inspector.dll"
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(b"inspector")
    received: list[Path] = []

    class RecordingResolver:
        def __init__(self, _inspector_path: Path, *, dependency_root: Path) -> None:
            received.append(dependency_root)

    monkeypatch.setattr(
        "src.mod_agent.local_entry.RunLocalMemberAccessResolver",
        RecordingResolver,
    )

    with pytest.raises(
        LocalEntryError,
        match="compiled artifact member access dependency root is missing",
    ):
        _create_run_local_member_access_resolver(
            inspector_path=inspector,
            run_root=run_root,
        )
    assert received == []
    assert not host_bepinex.resolve().is_relative_to(run_root.resolve())


def test_member_access_resolver_rejects_non_directory_dependency_paths(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    standalone = run_root / "build_dependencies" / "BepInEx"
    game_folder = run_root / "build_dependencies" / "GameFolder" / "BepInEx"
    standalone.parent.mkdir(parents=True)
    standalone.write_bytes(b"not a directory")
    game_folder.parent.mkdir(parents=True)
    game_folder.write_bytes(b"not a directory")
    inspector = run_root / "verification_tools" / "inspector.dll"
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(b"inspector")

    with pytest.raises(LocalEntryError):
        _create_run_local_member_access_resolver(
            inspector_path=inspector,
            run_root=run_root,
        )


def test_resolve_project_path_property_uses_project_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "generated_mods" / "BetterHitFeedback"
    project_dir.mkdir(parents=True)
    expected = tmp_path / "local_refs" / "gtfo_r8" / "BepInEx"
    expected.mkdir(parents=True)
    project_file = project_dir / "BetterHitFeedback.csproj"
    project_file.write_text(
        "<Project><PropertyGroup>"
        "<BepInExDir>..\\..\\local_refs\\gtfo_r8\\BepInEx</BepInExDir>"
        "</PropertyGroup></Project>",
        encoding="utf-8",
    )

    resolved = _resolve_project_path_property(
        project_file,
        property_name="BepInExDir",
    )

    assert resolved == expected.resolve()


def test_resolve_project_path_property_rejects_missing_property(tmp_path: Path) -> None:
    project_file = tmp_path / "Target.csproj"
    project_file.write_text("<Project />", encoding="utf-8")

    with pytest.raises(LocalEntryError, match="does not define BepInExDir"):
        _resolve_project_path_property(project_file, property_name="BepInExDir")


def _local_run_test_target(tmp_path: Path) -> Path:
    source = tmp_path / "canonical" / "BetterHitFeedback"
    source.mkdir(parents=True)
    bepinex = tmp_path / "references" / "BepInEx"
    bepinex.mkdir(parents=True)
    (source / "BetterHitFeedback.csproj").write_text(
        "<Project><PropertyGroup>"
        "<AssemblyName>BetterHitFeedback</AssemblyName>"
        f"<BepInExDir>{bepinex}</BepInExDir>"
        "</PropertyGroup></Project>",
        encoding="utf-8",
    )
    (source / "Plugin.cs").write_text(
        "namespace BetterHitFeedback; public sealed class Plugin {}\n",
        encoding="utf-8",
    )
    (source / "nuget.config").write_text(
        "<configuration />",
        encoding="utf-8",
    )
    return source


def _install_local_ollama_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(settings, **kwargs):
        return OllamaLLM(settings, **kwargs)

    monkeypatch.setattr(LLMFactory, "create", staticmethod(create))


def test_pre_graph_verifier_failure_persists_terminal_lineage(
    short_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_run_test_target(short_workspace)
    workspace = short_workspace / "runs"
    _install_local_ollama_factory(monkeypatch)
    inspector_project = short_workspace / "tracked-inspector" / "r8_il_inspector.csproj"
    inspector_project.parent.mkdir()
    inspector_project.write_text(
        "<Project Sdk=\"Microsoft.NET.Sdk\" />",
        encoding="utf-8",
    )

    def build(profile):
        artifact = profile.working_directory / "bin" / "Debug" / "BetterHitFeedback.dll"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"baseline")
        return NuGetIsolatedBuildExecution(
            restore_exit_code=0,
            build_exit_code=0,
            restore_stdout="restore ok",
            restore_stderr="",
            build_stdout="build ok",
            build_stderr="",
            duration_ms=12,
        )

    monkeypatch.setattr("src.mod_agent.local_entry._run_nuget_isolated_build", build)

    def build_inspector(_profile, *, run_root, **_):
        inspector = run_root / "verification_tools" / "r8_il_inspector.dll"
        inspector.parent.mkdir(parents=True)
        inspector.write_bytes(b"inspector")
        return {
            "inspector_path": str(inspector),
            "inspector_sha256": hashlib.sha256(b"inspector").hexdigest(),
        }

    monkeypatch.setattr(
        "src.mod_agent.local_entry._build_run_local_il_inspector",
        build_inspector,
    )

    def fail_verification(_artifact: Path, *, inspector_path: Path, **_) -> dict[str, object]:
        assert inspector_path.read_bytes() == b"inspector"
        raise LocalEntryError("compiled identity mismatch")

    monkeypatch.setattr(
        "src.mod_agent.local_entry._verify_bepin_plugin_identity",
        fail_verification,
    )

    with pytest.raises(LocalEntryError, match="compiled identity mismatch"):
        run_local_request(
            **request_budget(),
            description="make General.Enabled control startup patch registration",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            compiled_artifact_inspection_profile=CompiledArtifactInspectionProfile(
                project_path=inspector_project,
                method_symbol="BetterHitFeedback.Plugin.Load",
                expected_guid="local.betterhitfeedback",
                expected_name="BetterHitFeedback",
                expected_version="0.1.0",
            ),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    failure_path = run_root / "run_failure.json"
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes)
    assert manifest["run_id"] == failure["run_id"]
    assert manifest["request_id"] == failure["request_id"]
    assert manifest["status"] == "failed"
    assert manifest["terminal_failure_stage"] == "baseline_artifact_verification"
    assert manifest["ollama_patch_generation_profile"][
        "context_window"
    ] == 32768
    assert manifest["ollama_patch_generation_profile"][
        "model_context_limit"
    ] == 262144
    assert manifest["ollama_patch_generation_profile"][
        "pre_call_context_admission"
    ] == "provider_body_utf8_proxy_plus_output_budget_v1"
    assert manifest["ollama_patch_generation_profile"][
        "exact_prompt_tokens_available_pre_call"
    ] is False
    assert manifest["terminal_record"] == {
        "path": str(failure_path),
        "sha256": hashlib.sha256(failure_bytes).hexdigest(),
    }
    assert failure["failure_stage"] == "baseline_artifact_verification"
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert failure["patch_payload_parser_entered"] is False
    assert failure["actual_write_performed"] is False
    assert failure["baseline_build"]["artifact_exists"] is True
    assert failure["verification_tool_build"]["inspector_sha256"] == hashlib.sha256(
        b"inspector"
    ).hexdigest()
    assert not (run_root / "a").exists()
    assert (source / "Plugin.cs").read_text(encoding="utf-8").endswith("{}\n")


def test_member_access_dependency_failure_is_persisted_before_graph(
    short_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = short_workspace / "canonical" / "BetterHitFeedback"
    source.mkdir(parents=True)
    (source / "BetterHitFeedback.csproj").write_text(
        "<Project><PropertyGroup>"
        "<AssemblyName>BetterHitFeedback</AssemblyName>"
        "</PropertyGroup></Project>",
        encoding="utf-8",
    )
    (source / "Plugin.cs").write_text(
        "namespace BetterHitFeedback; public sealed class Plugin {}\n",
        encoding="utf-8",
    )
    workspace = short_workspace / "runs"
    inspector_project = short_workspace / "tracked-inspector" / "r8_il_inspector.csproj"
    inspector_project.parent.mkdir()
    inspector_project.write_text(
        '<Project Sdk="Microsoft.NET.Sdk" />',
        encoding="utf-8",
    )
    _install_local_ollama_factory(monkeypatch)

    def build(profile):
        artifact = profile.working_directory / "bin" / "Debug" / "BetterHitFeedback.dll"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"baseline")
        return NuGetIsolatedBuildExecution(
            restore_exit_code=0,
            build_exit_code=0,
            restore_stdout="restore ok",
            restore_stderr="",
            build_stdout="build ok",
            build_stderr="",
            duration_ms=12,
        )

    def build_inspector(_profile, *, run_root, **_):
        inspector = run_root / "verification_tools" / "r8_il_inspector.dll"
        inspector.parent.mkdir(parents=True)
        inspector.write_bytes(b"inspector")
        return {
            "inspector_path": str(inspector),
            "inspector_sha256": hashlib.sha256(b"inspector").hexdigest(),
        }

    monkeypatch.setattr("src.mod_agent.local_entry._run_nuget_isolated_build", build)
    monkeypatch.setattr(
        "src.mod_agent.local_entry._build_run_local_il_inspector",
        build_inspector,
    )
    monkeypatch.setattr(
        "src.mod_agent.local_entry._verify_bepin_plugin_identity",
        lambda *_args, **_kwargs: {"identity_verified": True},
    )

    with pytest.raises(
        LocalEntryError,
        match="compiled artifact member access dependency root is missing",
    ):
        run_local_request(
            **request_budget(),
            description="make General.Enabled control startup patch registration",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            compiled_artifact_inspection_profile=CompiledArtifactInspectionProfile(
                project_path=inspector_project,
                method_symbol="BetterHitFeedback.Plugin.Load",
                expected_guid="local.betterhitfeedback",
                expected_name="BetterHitFeedback",
                expected_version="0.1.0",
            ),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "run_failure.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["terminal_failure_stage"] == (
        "compiled_artifact_member_access_dependency"
    )
    assert manifest["graph_invoked"] is False
    assert failure["failure_stage"] == "compiled_artifact_member_access_dependency"
    assert failure["error_type"] == "LocalEntryError"
    assert failure["error"] == (
        "Run-local compiled artifact member access dependency root is missing."
    )
    assert failure["graph_invoked"] is False
    assert failure["model_call_count"] == 0
    assert not (run_root / "a").exists()


def test_pre_graph_restore_failure_records_exact_stage_and_stops(
    short_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_run_test_target(short_workspace)
    workspace = short_workspace / "runs"
    _install_local_ollama_factory(monkeypatch)

    def fail_restore(_profile):
        return NuGetIsolatedBuildExecution(
            restore_exit_code=9,
            build_exit_code=None,
            restore_stdout="",
            restore_stderr="restore failed",
            build_stdout="",
            build_stderr="",
            duration_ms=4,
        )

    monkeypatch.setattr(
        "src.mod_agent.local_entry._run_nuget_isolated_build",
        fail_restore,
    )

    with pytest.raises(LocalEntryError, match="baseline build failed"):
        run_local_request(
            **request_budget(),
            description="make General.Enabled control startup patch registration",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    failure = json.loads((run_root / "run_failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "baseline_restore"
    assert failure["baseline_build"]["restore_exit_code"] == 9
    assert failure["baseline_build"]["build_executed"] is False
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert not (run_root / "a").exists()


@pytest.mark.parametrize("configure_dependency_api", [False, True])
def test_run_local_request_records_game_folder_binding_before_fake_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configure_dependency_api: bool,
) -> None:
    source = _local_run_test_target(tmp_path)
    dependency = _game_folder_dependency(tmp_path)
    dependency_before = _binary_tree_snapshot(dependency)
    binding = _game_folder_binding(dependency)
    workspace = tmp_path / "runs"
    _install_local_ollama_factory(monkeypatch)

    api_config = None
    if configure_dependency_api:
        inspector = tmp_path / "inspector.dll"
        inspector.write_bytes(b"read-only-inspector-test")
        api_config = tmp_path / "dependency-api.json"
        api_config.write_text(json.dumps({"schema_version": "run_dependency_api_config_v1",
            "inspector_ref": {"path": str(inspector), "sha256": hashlib.sha256(inspector.read_bytes()).hexdigest()},
            "max_method_inspections": 2}), encoding="utf-8")

    def fail_restore(profile):
        exact = f"-p:GameFolder={profile.game_folder_dir}"
        assert profile.restore_command.count(exact) == 1
        assert profile.build_command.count(exact) == 1
        return NuGetIsolatedBuildExecution(
            restore_exit_code=9,
            build_exit_code=None,
            restore_stdout="",
            restore_stderr="injected restore failure",
            build_stdout="",
            build_stderr="",
            duration_ms=4,
        )

    monkeypatch.setattr(
        "src.mod_agent.local_entry._run_nuget_isolated_build",
        fail_restore,
    )

    with pytest.raises(LocalEntryError, match="baseline build failed"):
        run_local_request(
            description="make General.Enabled control startup patch registration",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            task_model_call_limit=1,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            game_folder_build_binding=binding,
            dependency_api_config=api_config,
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "run_failure.json").read_text(encoding="utf-8"))
    game_folder = manifest["game_folder_build_dependency"]
    assert manifest["game_folder_build_binding"] == binding.to_dict()
    assert manifest["small_project_context_policy"] == {
        "schema_version": "small_project_source_context_v1", "enabled": True,
        "max_files": 4, "max_total_bytes": 8192, "max_entries": 256}
    if configure_dependency_api:
        api_binding = manifest["dependency_api"]
        assert api_binding["status"] == "configured"
        assert api_binding["configuration_ref"]["path"] == str(api_config)
        assert api_binding["source_snapshot_sha256"] == binding.snapshot_sha256
        assert api_binding["target_code_execution_allowed"] is False
        assert api_binding["model_calls_performed"] == 0
        assert api_binding["max_method_inspections"] == 2
    else:
        assert manifest["dependency_api"]["status"] == "not_configured"
    assert game_folder["source_snapshot_sha256"] == binding.snapshot_sha256
    assert game_folder["copy_snapshot_sha256"] == binding.snapshot_sha256
    assert Path(game_folder["isolated_copy"]).is_relative_to(run_root)
    assert failure["failure_stage"] == "baseline_restore"
    assert failure["game_folder_build_dependency"] == game_folder
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert _binary_tree_snapshot(dependency) == dependency_before


def test_run_local_request_game_folder_snapshot_drift_stops_before_dotnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_run_test_target(tmp_path)
    dependency = _game_folder_dependency(tmp_path)
    binding = _game_folder_binding(dependency)
    (dependency / "doorstop_config.ini").write_bytes(b"drifted")
    workspace = tmp_path / "runs"
    dotnet_calls: list[object] = []
    _install_local_ollama_factory(monkeypatch)
    monkeypatch.setattr(
        "src.mod_agent.local_entry._run_nuget_isolated_build",
        lambda profile: dotnet_calls.append(profile),
    )

    with pytest.raises(LocalEntryError, match="snapshot SHA256 drifted"):
        run_local_request(
            description="change local source",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            task_model_call_limit=1,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            game_folder_build_binding=binding,
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    failure = json.loads((run_root / "run_failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "build_dependency_snapshot"
    assert failure["game_folder_build_binding"] == binding.to_dict()
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert dotnet_calls == []


def test_run_local_request_game_folder_copy_failure_is_bound_before_dotnet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_run_test_target(tmp_path)
    dependency = _game_folder_dependency(tmp_path)
    binding = _game_folder_binding(dependency)
    source_before = _binary_tree_snapshot(source)
    dependency_before = _binary_tree_snapshot(dependency)
    workspace = tmp_path / "runs"
    dotnet_calls: list[object] = []
    _install_local_ollama_factory(monkeypatch)

    def fail_copy(*, source: Path, destination: Path) -> None:
        assert source == dependency.resolve()
        assert destination.name == "GameFolder"
        raise OSError("injected long-path copy failure")

    monkeypatch.setattr(
        "src.mod_agent.build.dependencies._copy_validated_directory_tree",
        fail_copy,
    )
    monkeypatch.setattr(
        "src.mod_agent.local_entry._run_nuget_isolated_build",
        lambda profile: dotnet_calls.append(profile),
    )

    with pytest.raises(OSError, match="injected long-path copy failure"):
        run_local_request(
            description="change local source",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            task_model_call_limit=1,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            game_folder_build_binding=binding,
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    failure = json.loads(
        (run_root / "run_failure.json").read_text(encoding="utf-8")
    )
    assert failure["failure_stage"] == "build_dependency_copy"
    assert failure["game_folder_build_binding"] == binding.to_dict()
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert dotnet_calls == []
    assert _binary_tree_snapshot(source) == source_before
    assert _binary_tree_snapshot(dependency) == dependency_before


def test_pre_graph_cleanup_failure_is_terminally_bound(
    short_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_run_test_target(short_workspace)
    workspace = short_workspace / "runs"
    _install_local_ollama_factory(monkeypatch)

    built_working_directories: list[Path] = []

    def build(profile):
        built_working_directories.append(profile.working_directory)
        artifact = profile.working_directory / "bin" / "Debug" / "BetterHitFeedback.dll"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"baseline")
        (profile.working_directory / "obj" / "Debug").mkdir(parents=True)
        return NuGetIsolatedBuildExecution(
            restore_exit_code=0,
            build_exit_code=0,
            restore_stdout="restore ok",
            restore_stderr="",
            build_stdout="build ok",
            build_stderr="",
            duration_ms=8,
        )

    monkeypatch.setattr("src.mod_agent.local_entry._run_nuget_isolated_build", build)

    def deny_cleanup(_directory: Path) -> dict[str, object]:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(
        "src.mod_agent.local_entry._remove_generated_build_directory",
        deny_cleanup,
    )

    with pytest.raises(PermissionError, match="cleanup denied"):
        run_local_request(
            **request_budget(),
            description="make General.Enabled control startup patch registration",
            target_project=source,
            workspace_base=workspace,
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    run_root = next(path for path in workspace.iterdir() if path.is_dir())
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    failure_path = run_root / "run_failure.json"
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes)
    assert manifest["status"] == "failed"
    assert manifest["terminal_failure_stage"] == (
        "pre_graph_generated_directory_cleanup"
    )
    assert manifest["terminal_record"]["sha256"] == hashlib.sha256(
        failure_bytes
    ).hexdigest()
    assert failure["failure_stage"] == "pre_graph_generated_directory_cleanup"
    assert failure["error_type"] == "PermissionError"
    assert failure["model_call_count"] == 0
    assert failure["graph_invoked"] is False
    assert failure["patch_payload_parser_entered"] is False
    assert failure["actual_write_performed"] is False
    assert failure["baseline_build"]["artifact_exists"] is True
    assert len(built_working_directories) == 1
    assert built_working_directories[0] == run_root / "baseline_source" / source.name
    assert Path(failure["baseline_build"]["artifact_path"]).is_relative_to(
        run_root / "baseline_source"
    )
    assert not (run_root / "generated_mods" / source.name / "bin").exists()
    assert not (run_root / "generated_mods" / source.name / "obj").exists()
    assert not (run_root / "a").exists()


def test_run_request_blocks_source_access_without_explicit_disclosure_consent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "runs"

    with pytest.raises(LocalEntryError, match="--allow-source-disclosure"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=workspace,
            settings_path=tmp_path / "missing-settings.yaml",
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
        )

    assert not workspace.exists()


def test_local_run_selects_ollama_without_api_key_or_external_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def create(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        llm = OllamaLLM(settings, **kwargs)
        captured["llm"] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "create", staticmethod(create))
    monkeypatch.delenv("LOCAL_OLLAMA_UNUSED_API_KEY", raising=False)

    with pytest.raises(LocalEntryError, match="Target project"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            api_key_env="LOCAL_OLLAMA_UNUSED_API_KEY",
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:4b-instruct",
            ollama_max_output_tokens=3072,
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    settings = captured["settings"]
    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "qwen3:4b-instruct"
    assert settings.llm.max_tokens == 3072
    assert isinstance(captured["llm"], OllamaLLM)
    assert captured["kwargs"] == {
        "base_url": "http://127.0.0.1:11434",
        "context_window": 32768,
    }
    assert not (tmp_path / "runs").exists()


def test_local_run_forwards_main_timeout_without_reusing_it_for_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    def create(settings, **kwargs):
        llm = OllamaLLM(settings, **kwargs)
        created.append(
            {
                "model": settings.llm.model,
                "kwargs": kwargs,
                "timeout": llm.timeout,
            }
        )
        return llm

    def identity(_base_url: str, model: str) -> dict[str, object]:
        return {
            **_ollama_identity(),
            "name": model,
            "model": model,
        }

    monkeypatch.setattr(LLMFactory, "create", staticmethod(create))

    with pytest.raises(LocalEntryError, match="Target project"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:4b-instruct",
            ollama_timeout_seconds=600.0,
            repair_ollama_model="qwen3.6:27b",
            ollama_context_window=32768,
            ollama_model_identity_reader=identity,
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    assert created == [
        {
            "model": "qwen3:4b-instruct",
            "kwargs": {
                "base_url": "http://127.0.0.1:11434",
                "context_window": 32768,
                "timeout": 600.0,
            },
            "timeout": 600.0,
        },
        {
            "model": "qwen3.6:27b",
            "kwargs": {
                "base_url": "http://127.0.0.1:11434",
                "context_window": 32768,
            },
            "timeout": 120.0,
        },
    ]
    assert not (tmp_path / "runs").exists()


def test_local_run_applies_timeout_only_to_repair_ollama_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    def create(settings, **kwargs):
        llm = OllamaLLM(settings, **kwargs)
        created.append(
            {
                "model": settings.llm.model,
                "kwargs": kwargs,
                "timeout": llm.timeout,
            }
        )
        return llm

    def identity(_base_url: str, model: str) -> dict[str, object]:
        return {
            **_ollama_identity(),
            "name": model,
            "model": model,
        }

    monkeypatch.setattr(LLMFactory, "create", staticmethod(create))

    with pytest.raises(LocalEntryError, match="Target project"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3:4b-instruct",
            repair_ollama_model="qwen3.6:27b",
            repair_ollama_timeout_seconds=600.0,
            ollama_context_window=32768,
            ollama_model_identity_reader=identity,
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    assert created == [
        {
            "model": "qwen3:4b-instruct",
            "kwargs": {
                "base_url": "http://127.0.0.1:11434",
                "context_window": 32768,
            },
            "timeout": 120.0,
        },
        {
            "model": "qwen3.6:27b",
            "kwargs": {
                "base_url": "http://127.0.0.1:11434",
                "context_window": 32768,
                "timeout": 600.0,
            },
            "timeout": 600.0,
        },
    ]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_local_run_rejects_invalid_repair_ollama_timeout(timeout: float) -> None:
    with pytest.raises(
        LocalEntryError,
        match="Repair Ollama timeout seconds must be a positive finite number",
    ):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=Path("missing-target"),
            workspace_base=Path("missing-runs"),
            settings_path=Path("missing-settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            repair_ollama_model="qwen3.6:27b",
            repair_ollama_timeout_seconds=timeout,
            ollama_context_window=32768,
        )


@pytest.mark.parametrize(
    "timeout",
    [True, "600", 0, 0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_local_run_rejects_invalid_main_ollama_timeout(timeout: object) -> None:
    with pytest.raises(
        LocalEntryError,
        match="Main Ollama timeout seconds must be a positive finite number",
    ):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=Path("missing-target"),
            workspace_base=Path("missing-runs"),
            settings_path=Path("missing-settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_timeout_seconds=timeout,
            ollama_context_window=32768,
        )


def test_non_local_run_rejects_main_ollama_timeout() -> None:
    with pytest.raises(
        LocalEntryError,
        match="Main Ollama timeout can only be configured for local Ollama execution",
    ):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=Path("missing-target"),
            workspace_base=Path("missing-runs"),
            settings_path=Path("missing-settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=True,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=False,
            ollama_timeout_seconds=600.0,
        )


def test_patch_generation_role_budgets_preserve_both_repair_attempts(
    tmp_path: Path,
) -> None:
    assert _patch_generation_calls_remaining(
        "repair",
        initial_patch_call_count=1,
        repair_call_count=0,
    ) == 2
    assert _patch_generation_calls_remaining(
        "repair",
        initial_patch_call_count=2,
        repair_call_count=0,
    ) == 2
    assert _patch_generation_calls_remaining(
        "repair",
        initial_patch_call_count=2,
        repair_call_count=1,
    ) == 1
    assert _local_patch_generation_correction_allowed(
        "initial_patch",
        initial_patch_call_count=1,
    ) is True
    assert _local_patch_generation_correction_allowed(
        "initial_patch",
        initial_patch_call_count=2,
    ) is False
    assert _local_patch_generation_correction_allowed(
        "repair",
        initial_patch_call_count=1,
    ) is False
    assert list(tmp_path.iterdir()) == []


def test_patch_generation_role_budgets_reject_extra_provider_calls() -> None:
    with pytest.raises(
        LocalEntryError,
        match="LOCAL_PATCH_GENERATION_CALL_BUDGET_EXHAUSTED",
    ):
        _patch_generation_calls_remaining(
            "initial_patch",
            initial_patch_call_count=2,
            repair_call_count=0,
        )

    with pytest.raises(
        LocalEntryError,
        match="LOCAL_PATCH_GENERATION_CALL_BUDGET_EXHAUSTED",
    ):
        _patch_generation_calls_remaining(
            "repair",
            initial_patch_call_count=2,
            repair_call_count=2,
        )

    with pytest.raises(
        LocalEntryError,
        match="LOCAL_PATCH_GENERATION_CALL_COUNT_INVALID",
    ):
        _patch_generation_calls_remaining(
            "repair",
            initial_patch_call_count=0,
            repair_call_count=-1,
        )


def _local_frozen_resume_identity() -> dict[str, object]:
    return _build_local_patch_frozen_resume_identity(
        run_id="run_001",
        request_id="request_001",
        feature_id="feature_001",
        model_role="initial_patch",
        model_call_count=2,
        initial_patch_call_count=0,
        repair_call_count=0,
        model="qwen3:4b-instruct",
        output_budget=4096,
        context_window=32768,
        model_context_limit=40960,
        ollama_server_version="0.32.1",
        model_identity=_ollama_identity(),
        source_revision_binding={"revision": "source-revision-001"},
        source_snapshot_sha256="a" * 64,
        request_sha256="b" * 64,
        model_input_payload_sha256="c" * 64,
        feature_spec_ref="a/feature_spec.json",
        feature_spec_hash="d" * 64,
        evidence_pack_ref="a/evidence_pack.json",
        evidence_pack_hash="e" * 64,
        implementation_plan_ref="a/implementation_plan.json",
        implementation_plan_hash="f" * 64,
        repair_plan_ref=None,
        repair_plan_hash=None,
        build_result_ref=None,
        build_result_hash=None,
        binding_registry_sha256="1" * 64,
        evidence_sha256="2" * 64,
        editable_paths=["generated_mods/Example/Plugin.cs"],
        change_scope=["Preserve unrelated behavior."],
    )


def test_local_frozen_resume_identity_binds_role_ordinal_budget_and_owner_refs() -> None:
    identity = _local_frozen_resume_identity()

    assert identity["model_role"] == "initial_patch"
    assert identity["role_ordinal"] == 1
    assert identity["model_call_ordinal"] == 3
    assert identity["remaining_call_budget"] == 2
    assert identity["frozen_attempt_call_budget"] == 1
    assert identity["artifact_bindings"] == {
        "feature_spec_ref": "a/feature_spec.json",
        "feature_spec_hash": "d" * 64,
        "evidence_pack_ref": "a/evidence_pack.json",
        "evidence_pack_hash": "e" * 64,
        "implementation_plan_ref": "a/implementation_plan.json",
        "implementation_plan_hash": "f" * 64,
        "repair_plan_ref": None,
        "repair_plan_hash": None,
        "build_result_ref": None,
        "build_result_hash": None,
    }
    assert identity["binding_registry_sha256"] == "1" * 64
    assert identity["evidence_sha256"] == "2" * 64


def test_frozen_patch_attempt_atomic_roundtrip_has_complete_readback(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "frozen_patch_attempt.json"
    attempt = {
        "schema_version": "frozen_native_patch_attempt_v1",
        "status": "paused_frozen",
        "consumed": False,
        "resume_identity": _local_frozen_resume_identity(),
        "provider_request": {
            "body_utf8": '{"messages":[{"content":"内容"}]}'
        },
        "materialization_context": {"binding_contexts": {}},
        "terminal": None,
    }

    _atomic_write_frozen_patch_attempt(destination, attempt)

    assert _read_frozen_patch_attempt(destination) == attempt
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_frozen_patch_attempt_atomic_failure_leaves_no_resumable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "frozen_patch_attempt.json"

    def fail_replace(*_):
        raise OSError("replace failed")

    monkeypatch.setattr("src.mod_agent.local_entry.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        _atomic_write_frozen_patch_attempt(
            destination,
            {
                "status": "paused_frozen",
                "resume_identity": _local_frozen_resume_identity(),
            },
        )

    assert destination.exists() is False
    assert list(tmp_path.iterdir()) == []


def _prepare_local_resume_order_fixture(
    tmp_path: Path,
) -> tuple[LLMExactPatchGenerator, Path, dict[str, object], MagicMock]:
    source_hash = "3" * 64
    evidence = {
        "evidence_id": "evidence_001",
        "source_path": "generated_mods/Example/Plugin.cs",
        "source_hash": source_hash,
        "file_hash": source_hash,
        "doc_hash": source_hash,
        "source_type": "target_source",
        "symbol": "Example.Plugin.Apply",
        "span": {"start_line": 1, "end_line": 1},
        "excerpt": "ApplyOld();",
    }
    generation_inputs = {
        "user_request": "Change the bound behavior.",
        "evidence_items": [evidence],
        "binding_registry": {
            "schema_version": "binding_registry_v1",
            "owner_artifact_type": "implementation_plan",
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "evidence_id": "evidence_001",
                    "canonical_target_path": evidence["source_path"],
                    "source_hash": source_hash,
                    "target_symbol": evidence["symbol"],
                    "target_span": evidence["span"],
                }
            ],
            "supporting_bindings": [],
        },
        "constraints": ["Preserve unrelated behavior."],
        "required_change_units": [
            {
                "change_id": "change_001",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_001",
                "hook_target_binding_id": None,
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "rationale": "Apply the bounded change.",
                "expected_validation": ["compile or static validation"],
            }
        ],
        "required_validation_by_binding": {
            "edit_binding_001": ["compile or static validation"]
        },
    }
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            model="qwen3:4b-instruct",
            temperature=0.0,
            max_tokens=4096,
        )
    )
    generator = LLMExactPatchGenerator(
        OllamaLLM(
            settings,
            base_url="http://127.0.0.1:11434",
            context_window=32768,
            timeout=600.0,
        )
    )
    attempt_path = tmp_path / "frozen_patch_attempt.json"
    identity = _local_frozen_resume_identity()
    _prepare_local_frozen_patch_attempt(
        generator=generator,
        destination=attempt_path,
        resume_identity=identity,
        generation_inputs=generation_inputs,
    )
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "qwen3:4b-instruct",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "changes": [
                        {
                            "edit_binding_id": "edit_binding_001",
                            "rationale": "Apply the bounded change.",
                            "rollback_hint": "Restore the prior source.",
                            "after_text": "ApplyNew();",
                        }
                    ]
                }
            ),
        },
        "done": True,
        "done_reason": "stop",
    }
    return generator, attempt_path, identity, response


def _rehash_frozen_attempt(attempt: dict[str, object]) -> None:
    immutable = {
        "resume_identity": attempt["resume_identity"],
        "provider_request": attempt["provider_request"],
        "materialization_context": attempt["materialization_context"],
    }
    attempt["frozen_identity_sha256"] = hashlib.sha256(
        json.dumps(
            immutable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_wrong_resume_identity_does_not_claim_and_correct_identity_then_succeeds(
    tmp_path: Path,
) -> None:
    generator, attempt_path, identity, response = _prepare_local_resume_order_fixture(
        tmp_path
    )
    wrong_identity = json.loads(json.dumps(identity))
    wrong_identity["model"] = "wrong-model"
    claim_path = attempt_path.with_name(f"{attempt_path.name}.resume.claim")

    with patch("httpx.Client") as client:
        post = client.return_value.__enter__.return_value.post
        post.return_value = response
        with pytest.raises(PatchGenerationError):
            _resume_local_frozen_patch_attempt(
                generator=generator,
                source=attempt_path,
                current_resume_identity=wrong_identity,
            )
        assert post.call_count == 0
        assert claim_path.exists() is False
        paused = _read_frozen_patch_attempt(attempt_path)
        assert paused["status"] == "paused_frozen"
        assert paused["consumed"] is False

        generated = _resume_local_frozen_patch_attempt(
            generator=generator,
            source=attempt_path,
            current_resume_identity=identity,
        )

    assert len(generated.changes) == 1
    assert post.call_count == 1


def test_corrupted_frozen_provider_body_does_not_create_claim_or_http(
    tmp_path: Path,
) -> None:
    generator, attempt_path, identity, response = _prepare_local_resume_order_fixture(
        tmp_path
    )
    attempt = _read_frozen_patch_attempt(attempt_path)
    provider_request = attempt["provider_request"]
    body = json.loads(provider_request["body_utf8"])
    body["tools"] = []
    body_utf8 = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    body_bytes = body_utf8.encode("utf-8")
    provider_request["body_utf8"] = body_utf8
    provider_request["body_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    provider_request["body_size_bytes"] = len(body_bytes)
    admission = provider_request["context_admission"]
    admission["provider_body_utf8_bytes"] = len(body_bytes)
    admission["conservative_required_context"] = len(body_bytes) + 4096
    admission["admitted"] = True
    _rehash_frozen_attempt(attempt)
    _atomic_write_frozen_patch_attempt(attempt_path, attempt)
    claim_path = attempt_path.with_name(f"{attempt_path.name}.resume.claim")

    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        with pytest.raises(OllamaLLMError):
            _resume_local_frozen_patch_attempt(
                generator=generator,
                source=attempt_path,
                current_resume_identity=identity,
            )

    client.assert_not_called()
    assert claim_path.exists() is False


def test_frozen_attempt_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _, attempt_path, _, _ = _prepare_local_resume_order_fixture(tmp_path)
    raw = attempt_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"status": "paused_frozen"',
        '"status": "paused_frozen",\n  "status": "paused_frozen"',
        1,
    )
    attempt_path.write_text(raw, encoding="utf-8")

    with pytest.raises(LocalEntryError, match="duplicate"):
        _read_frozen_patch_attempt(attempt_path)


def test_post_claim_second_validation_drift_blocks_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, attempt_path, identity, response = _prepare_local_resume_order_fixture(
        tmp_path
    )
    claim_path = attempt_path.with_name(f"{attempt_path.name}.resume.claim")

    def claim_then_drift(**kwargs):
        result = _claim_local_frozen_patch_resume(**kwargs)
        attempt = _read_frozen_patch_attempt(attempt_path)
        attempt["resume_identity"]["run_id"] = "drift-after-claim"
        _rehash_frozen_attempt(attempt)
        _atomic_write_frozen_patch_attempt(attempt_path, attempt)
        return result

    monkeypatch.setattr(
        "src.mod_agent.local_entry._claim_local_frozen_patch_resume",
        claim_then_drift,
    )
    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        with pytest.raises(PatchGenerationError):
            _resume_local_frozen_patch_attempt(
                generator=generator,
                source=attempt_path,
                current_resume_identity=identity,
            )

    client.assert_not_called()
    assert claim_path.is_file() is True


def test_concurrent_local_frozen_resume_has_one_filesystem_consumer_and_one_post(
    tmp_path: Path,
) -> None:
    source_hash = "3" * 64
    evidence = {
        "evidence_id": "evidence_001",
        "source_path": "generated_mods/Example/Plugin.cs",
        "source_hash": source_hash,
        "file_hash": source_hash,
        "doc_hash": source_hash,
        "source_type": "target_source",
        "symbol": "Example.Plugin.Apply",
        "span": {"start_line": 1, "end_line": 1},
        "excerpt": "ApplyOld();",
    }
    generation_inputs = {
        "user_request": "Change the bound behavior.",
        "evidence_items": [evidence],
        "binding_registry": {
            "schema_version": "binding_registry_v1",
            "owner_artifact_type": "implementation_plan",
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "evidence_id": "evidence_001",
                    "canonical_target_path": evidence["source_path"],
                    "source_hash": source_hash,
                    "target_symbol": evidence["symbol"],
                    "target_span": evidence["span"],
                }
            ],
            "supporting_bindings": [],
        },
        "constraints": ["Preserve unrelated behavior."],
        "required_change_units": [
            {
                "change_id": "change_001",
                "change_type": "modify_behavior",
                "edit_binding_id": "edit_binding_001",
                "hook_target_binding_id": None,
                "supporting_binding_ids": [],
                "operation": "minimal_evidence_bound_change",
                "rationale": "Apply the bounded change.",
                "expected_validation": ["compile or static validation"],
            }
        ],
        "required_validation_by_binding": {
            "edit_binding_001": ["compile or static validation"]
        },
    }
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            model="qwen3:4b-instruct",
            temperature=0.0,
            max_tokens=4096,
        )
    )
    generator = LLMExactPatchGenerator(
        OllamaLLM(
            settings,
            base_url="http://127.0.0.1:11434",
            context_window=32768,
            timeout=600.0,
        )
    )
    attempt_path = tmp_path / "frozen_patch_attempt.json"
    identity = _local_frozen_resume_identity()
    _prepare_local_frozen_patch_attempt(
        generator=generator,
        destination=attempt_path,
        resume_identity=identity,
        generation_inputs=generation_inputs,
    )
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "qwen3:4b-instruct",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "changes": [
                        {
                            "edit_binding_id": "edit_binding_001",
                            "rationale": "Apply the bounded change.",
                            "rollback_hint": "Restore the prior source.",
                            "after_text": "ApplyNew();",
                        }
                    ]
                }
            ),
        },
        "done": True,
        "done_reason": "stop",
    }
    start = threading.Barrier(3)
    post_entered = threading.Event()
    release_post = threading.Event()
    results: list[object] = []
    errors: list[Exception] = []

    def run_resume() -> None:
        start.wait()
        try:
            results.append(
                _resume_local_frozen_patch_attempt(
                    generator=generator,
                    source=attempt_path,
                    current_resume_identity=identity,
                )
            )
        except Exception as exc:
            errors.append(exc)

    def post_once(*_, **__) -> MagicMock:
        post_entered.set()
        assert release_post.wait(timeout=5.0)
        return response

    first = threading.Thread(target=run_resume)
    second = threading.Thread(target=run_resume)
    with patch("httpx.Client") as client:
        post = client.return_value.__enter__.return_value.post
        post.side_effect = post_once
        first.start()
        second.start()
        start.wait()
        assert post_entered.wait(timeout=5.0)
        second.join(timeout=0.2)
        first.join(timeout=0.2)
        release_post.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert post.call_count == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], LocalEntryError)
    assert str(errors[0]) == "LOCAL_PATCH_FROZEN_ATTEMPT_CONSUMED_OR_IN_PROGRESS"
    final_attempt = _read_frozen_patch_attempt(attempt_path)
    assert final_attempt["status"] == "terminal_succeeded"
    assert final_attempt["consumed"] is True
    assert final_attempt["terminal"]["outcome"] == "succeeded"
    assert (
        attempt_path.with_name(f"{attempt_path.name}.resume.claim").is_file()
        is True
    )


def test_local_run_rejects_external_disclosure_boundary_mix(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalEntryError, match="exactly one model boundary"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=tmp_path / "missing-settings.yaml",
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=True,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
        )

    assert not (tmp_path / "runs").exists()


def test_local_run_requires_exact_model_without_inference(tmp_path: Path) -> None:
    with pytest.raises(LocalEntryError, match="exact --ollama-model"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model=None,
        )


def test_local_run_requires_explicit_context_window_before_catalog_read(
    tmp_path: Path,
) -> None:
    catalog_read = False

    def read_identity(*_):
        nonlocal catalog_read
        catalog_read = True
        return _ollama_identity()

    with pytest.raises(LocalEntryError, match="--ollama-context-window"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_model_identity_reader=read_identity,
        )

    assert catalog_read is False


def test_local_run_rejects_context_window_not_above_output_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalEntryError, match="must exceed"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_max_output_tokens=4096,
            ollama_context_window=4096,
        )


def test_local_run_rejects_non_literal_loopback_before_catalog_read(
    tmp_path: Path,
) -> None:
    catalog_read = False

    def read_identity(*_):
        nonlocal catalog_read
        catalog_read = True
        return _ollama_identity()

    with pytest.raises(
        LocalEntryError,
        match="LOCAL_AGENT_ENDPOINT_MUST_BE_LITERAL_LOOPBACK_HTTP_ORIGIN",
    ):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_base_url="http://192.168.1.9:11434",
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=read_identity,
        )

    assert catalog_read is False


def test_local_provider_initialization_failure_has_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail(settings, **kwargs):
        calls.append(settings.llm.provider)
        raise RuntimeError("local provider unavailable")

    monkeypatch.setattr(LLMFactory, "create", staticmethod(fail))

    with pytest.raises(LocalEntryError, match="Local Ollama initialization failed"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
            local_ollama=True,
            ollama_model="qwen3:4b-instruct",
            ollama_context_window=32768,
            ollama_model_identity_reader=lambda *_: _ollama_identity(),
            ollama_server_version_reader=lambda *_: "0.32.1",
        )

    assert calls == ["ollama"]


def test_read_ollama_server_version_uses_literal_local_transport_profile() -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"version": "0.32.1"}

    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.get.return_value = response
        version = _read_ollama_server_version("http://127.0.0.1:11434")

    assert version == "0.32.1"
    client.assert_called_once_with(
        timeout=10.0,
        trust_env=False,
        follow_redirects=False,
    )
    client.return_value.__enter__.return_value.get.assert_called_once_with(
        "http://127.0.0.1:11434/api/version"
    )


def test_local_failure_content_is_preserved_exactly_within_policy_limit(
    tmp_path: Path,
) -> None:
    content = "Patch failure 内容\r\n"
    error = LocalEntryError("patch failed")
    error.diagnostic_content = content
    error.diagnostics = {
        "failure_classification": "malformed_structured_output",
        "message_content": content,
    }

    diagnostics = _persist_local_failure_diagnostics(
        run_root=tmp_path,
        error=error,
        enabled=True,
    )

    assert diagnostics is not None
    retention = diagnostics["diagnostic_content"]
    assert retention["max_utf8_bytes"] == 262_144
    assert retention["utf8_bytes"] == len(content.encode("utf-8"))
    assert retention["retained"] is True
    assert "message_content" not in diagnostics
    assert (tmp_path / LOCAL_DIAGNOSTIC_CONTENT_FILENAME).read_bytes() == content.encode(
        "utf-8"
    )


def test_local_failure_content_over_policy_limit_is_not_written_or_truncated(
    tmp_path: Path,
) -> None:
    content = "界" * ((LOCAL_DIAGNOSTIC_CONTENT_MAX_BYTES // 3) + 1)
    error = LocalEntryError("patch failed")
    error.diagnostic_content = content

    diagnostics = _persist_local_failure_diagnostics(
        run_root=tmp_path,
        error=error,
        enabled=True,
    )

    assert diagnostics is not None
    retention = diagnostics["diagnostic_content"]
    assert retention["utf8_bytes"] > LOCAL_DIAGNOSTIC_CONTENT_MAX_BYTES
    assert retention["retained"] is False
    assert retention["path"] is None
    assert retention["not_retained_reason"] == "policy_limit_exceeded"
    assert diagnostics["diagnostic_sufficiency"] == "reduced"
    assert not (tmp_path / LOCAL_DIAGNOSTIC_CONTENT_FILENAME).exists()


def test_external_run_still_requires_api_key_after_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from src.core.settings import load_settings
    from tests.unit.test_external_patch import _profile

    profile = replace(_profile(), credential_env="MISSING_EXTERNAL_KEY")
    settings = load_settings("config/settings.yaml")
    settings = replace(settings, llm=replace(settings.llm, model=profile.model, base_url=profile.base_url))
    monkeypatch.setattr("src.mod_agent.local_entry.load_settings", lambda _path: settings)
    monkeypatch.delenv("MISSING_EXTERNAL_KEY", raising=False)
    monkeypatch.setenv("EXTERNAL_KEY_TEST_BASE_URL", profile.base_url)

    with pytest.raises(LocalEntryError, match="MISSING_EXTERNAL_KEY"):
        run_local_request(
            **request_budget(),
            description="change a local source file",
            target_project=tmp_path / "missing-target",
            workspace_base=tmp_path / "runs",
            settings_path=Path("config/settings.yaml"),
            api_key_env="MISSING_EXTERNAL_KEY",
            base_url_env="EXTERNAL_KEY_TEST_BASE_URL",
            external_patch_profile=profile,
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=True,
            diff_approval_reader=lambda dry_run: str(dry_run["diff_hash"]),
        )


def test_source_disclosure_record_is_explicit_and_does_not_contain_api_key() -> None:
    record = _source_disclosure_record(
        provider="openai",
        model="qwen3.6-plus",
        base_url="https://example.invalid/compatible-mode/v1",
        approved_at="2026-07-12T00:00:00+00:00",
    )

    assert record == {
        "approved": True,
        "approved_at": "2026-07-12T00:00:00+00:00",
        "approval_id": None,
        "approval_actor_type": "local_cli_invoker",
        "consent_mechanism": "--allow-source-disclosure",
        "provider_adapter": None,
        "destination": {
            "provider": "openai",
            "model": "qwen3.6-plus",
            "base_url": "https://example.invalid/compatible-mode/v1",
            "request_url": None,
        },
        "approved_data_classes": [
            "natural_language_request",
            "retrieved_local_source_excerpts",
            "run_local_api_member_metadata_facts",
            "binding_registry",
            "side_effect_boundaries",
            "feature_target_behavior",
            "feature_acceptance_criteria",
            "feature_runtime_verification_plan",
        ],
        "requested_data_classes": [
            "natural_language_request",
            "retrieved_local_source_excerpts",
            "run_local_api_member_metadata_facts",
            "binding_registry",
            "side_effect_boundaries",
            "feature_target_behavior",
            "feature_acceptance_criteria",
            "feature_runtime_verification_plan",
        ],
        "scope": "this_run_only",
    }
    assert "api_key" not in str(record).lower()


def test_local_admission_records_local_execution_without_external_disclosure() -> None:
    no_disclosure = _local_no_external_disclosure_record()
    admission = _local_model_admission_record(
        model_identity=_ollama_identity(),
        provider_adapter="src.libs.llm.ollama_llm.OllamaLLM",
    )

    assert no_disclosure["occurred"] is False
    assert no_disclosure["approved"] is False
    assert no_disclosure["approved_data_classes"] == []
    assert no_disclosure["requested_data_classes"] == []
    assert admission == {
        "provider": "ollama",
        "model": "qwen3:4b-instruct",
        "model_digest": _OLLAMA_DIGEST,
        "provider_adapter": "src.libs.llm.ollama_llm.OllamaLLM",
        "execution_locality": "local",
        "base_url": "http://127.0.0.1:11434",
        "destination": "http://127.0.0.1:11434/api/chat",
        "external_disclosure_occurred": False,
        "api_key_required": False,
        "api_key_read": False,
        "api_key_persisted": False,
        "automatic_provider_fallback": False,
        "automatic_external_retry": False,
        "data_processing_location": "machine_local",
    }


def test_ollama_catalog_binds_one_exact_model_identity() -> None:
    identity = _select_ollama_model_identity(
        {
            "models": [
                {
                    "name": "qwen3:4b-instruct",
                    "model": "qwen3:4b-instruct",
                    "digest": _OLLAMA_DIGEST,
                    "modified_at": "2026-07-16T14:06:42.9800065+08:00",
                    "size": 2497293803,
                    "details": {
                        "context_length": 262144,
                        "parameter_size": "4.0B",
                        "quantization_level": "Q4_K_M",
                    },
                    "capabilities": ["completion", "tools"],
                }
            ]
        },
        base_url="http://127.0.0.1:11434",
        expected_model="qwen3:4b-instruct",
    )

    assert identity == _ollama_identity()


def test_ollama_context_window_binds_declared_model_limit() -> None:
    assert (
        _validate_ollama_context_window(
            _ollama_identity(),
            requested_context_window=32768,
        )
        == 262144
    )


def test_ollama_context_window_rejects_value_above_declared_model_limit() -> None:
    with pytest.raises(
        LocalEntryError,
        match="OLLAMA_CONTEXT_WINDOW_EXCEEDS_MODEL_LIMIT",
    ):
        _validate_ollama_context_window(
            _ollama_identity(),
            requested_context_window=262145,
        )


def test_ollama_context_window_requires_declared_model_limit() -> None:
    identity = _ollama_identity()
    identity["details"] = {
        "parameter_size": "4.0B",
        "quantization_level": "Q4_K_M",
    }

    with pytest.raises(
        LocalEntryError,
        match="OLLAMA_MODEL_CONTEXT_LENGTH_INVALID",
    ):
        _validate_ollama_context_window(
            identity,
            requested_context_window=32768,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"models": []}, "OLLAMA_MODEL_LIST_EMPTY"),
        (
            {
                "models": [
                    {
                        "name": "another-model",
                        "model": "another-model",
                        "digest": "1" * 64,
                        "details": {},
                        "capabilities": ["completion"],
                    }
                ]
            },
            "OLLAMA_MODEL_NOT_EXACTLY_PRESENT",
        ),
    ),
)
def test_ollama_catalog_fails_closed_without_exact_model(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(LocalEntryError, match=message):
        _select_ollama_model_identity(
            payload,
            base_url="http://127.0.0.1:11434",
            expected_model="qwen3:4b-instruct",
        )


def test_ollama_model_digest_drift_fails_closed() -> None:
    with pytest.raises(LocalEntryError, match="OLLAMA_MODEL_DIGEST_DRIFT"):
        _assert_ollama_model_identity_stable(
            expected=_ollama_identity(),
            observed=_ollama_identity(digest="1" * 64),
        )


def test_external_evidence_replaces_absolute_source_ref() -> None:
    sanitized = _external_evidence_items(
        [
            {
                "evidence_id": "evidence_001",
                "source_type": "target_source",
                "source_path": "generated_mods/BetterHitFeedback/Plugin.cs",
                "source_ref": r"C:\private\Plugin.cs",
                "excerpt": 'public const string VERSION = "0.1.0";',
            }
        ]
    )

    assert sanitized[0]["source_ref"] == (
        "source://target_source/generated_mods/BetterHitFeedback/Plugin.cs"
    )
    _assert_no_absolute_local_paths(sanitized)


def test_external_payload_rejects_absolute_local_path() -> None:
    with pytest.raises(LocalEntryError, match="absolute local path"):
        _assert_no_absolute_local_paths({"source_ref": r"C:\private\Plugin.cs"})


def test_exact_snapshot_change_requires_only_bound_file() -> None:
    before = [
        {"path": "Plugin.cs", "sha256": "before", "size": 10},
        {"path": "Other.cs", "sha256": "same", "size": 20},
    ]
    after = [
        {"path": "Plugin.cs", "sha256": "after", "size": 10},
        {"path": "Other.cs", "sha256": "same", "size": 20},
    ]

    result = _verify_exact_snapshot_change(
        before,
        after,
        relative_path="Plugin.cs",
        before_sha256="before",
        after_sha256="after",
    )

    assert result["status"] == "passed"
    assert result["changed_paths"] == ["Plugin.cs"]


def test_bepin_plugin_identity_reads_compiled_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "BetterHitFeedback.dll"
    artifact.write_bytes(b"compiled")
    inspector = tmp_path / "r8_il_inspector.dll"
    inspector.write_bytes(b"inspector")

    class Completed:
        returncode = 0
        stderr = ""
        stdout = (
            '{"schema_version":"r8_il_inspection_v1","methods":['
            '{"symbol":"BetterHitFeedback.Plugin.Load",'
            '"declaring_type_custom_attributes":['
            '{"constructor":"BepInEx.BepInPlugin..ctor",'
            '"printable_ascii_runs":["local.betterhitfeedback",'
            '"BetterHitFeedback","0.1.0"]}]}],'
            '"target_code_executed":false}'
        )

    monkeypatch.setattr("src.mod_agent.local_entry.subprocess.run", lambda *_, **__: Completed())

    result = _verify_bepin_plugin_identity(
        artifact,
        inspector_path=inspector,
        method_symbol="BetterHitFeedback.Plugin.Load",
        expected_guid="local.betterhitfeedback",
        expected_name="BetterHitFeedback",
        expected_version="0.1.0",
    )

    assert result["status"] == "passed"
    assert result["bepin_plugin"]["version"] == "0.1.0"
    assert result["target_code_executed"] is False


def test_source_disclosure_blocks_unapproved_new_data_class() -> None:
    record = _source_disclosure_record(
        provider="openai",
        model="qwen3.6-plus",
        base_url="https://example.invalid/compatible-mode/v1",
        approved_at="2026-07-12T00:00:00+00:00",
    )

    with pytest.raises(LocalEntryError, match="runtime_logs"):
        _verify_source_disclosure(
            record,
            requested_data_classes=("retrieved_local_source_excerpts", "runtime_logs"),
        )


def test_endpoint_metadata_strips_userinfo_query_and_fragment() -> None:
    sanitized = _sanitized_endpoint_url(
        "https://local-key@example.invalid/compatible/v1?token=secret#fragment"
    )

    assert sanitized == "https://example.invalid/compatible/v1"
    assert "local-key" not in sanitized
    assert "secret" not in sanitized


def test_project_text_property_reads_non_better_hit_feedback_assembly(
    tmp_path: Path,
) -> None:
    project_file = tmp_path / "AnotherMod.csproj"
    project_file.write_text(
        "<Project><PropertyGroup><AssemblyName>AnotherMod</AssemblyName>"
        "</PropertyGroup></Project>",
        encoding="utf-8",
    )

    assert (
        _project_text_property(project_file, property_name="AssemblyName")
        == "AnotherMod"
    )
