from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.mod_agent.actual_execution import (
    ACTUAL_CONSUMPTION_FILENAME,
    ACTUAL_JOURNAL_FILENAME,
    ACTUAL_PACKET_FILENAME,
    ACTUAL_SEAL_FILENAME,
    ActualExecutionDependenciesV1,
    ActualProductExecutionSelectionV1,
    ActualRunError,
    ActualRunPreparationRequestV1,
    ActualRunSpecV1,
    _execution_arguments,
    _verify_current_facts,
    execute_sealed_actual_run,
    prepare_sealed_actual_run,
)
from src.mod_agent.cli import build_parser, main as cli_main
from src.mod_agent.entry_admission import (
    EntryAdmissionDependenciesV1,
    admit_sealed_entry,
    entry_request_from_preflight_packet,
    verify_entry_current_facts,
)
from src.mod_agent.local_entry import (
    CompiledArtifactInspectionProfile,
    LocalEntryError,
    RunLocalGameFolderBindingV1,
    run_local_request,
)
from src.mod_agent.preflight_preparation import (
    _load_actual_record,
    prepare_sealed_entry_preflight,
)
from tests.unit.test_preflight_preparation import (
    _file_sha256,
    _fixture as preflight_fixture,
    _git,
)


_DIRTY_PLATFORM_PATHS = (
    "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/ENGINEERING_RAG_EVIDENCE_INTELLIGENCE_TARGET_DRAFT.md",
    "docs/SPEC_COVERAGE_MATRIX.md",
    "docs/LOCAL_MODEL_COMPARATIVE_EVALUATION_EXPERIMENT.md",
)


def _admit_preflight(
    request: Any,
    preflight_dependencies: Any,
) -> tuple[Path, Path]:
    root = Path(request.packet_root)
    packet = json.loads((root / "packet.json").read_text(encoding="utf-8"))
    entry = entry_request_from_preflight_packet(
        packet,
        payload_manifest_sha256=json.loads(
            (root / "PREFLIGHT.json").read_text(encoding="utf-8")
        )["payload_manifest_sha256"],
    )
    run = admit_sealed_entry(
        entry,
        dependencies=EntryAdmissionDependenciesV1(
            git_admission=entry.strict_git_admission(),
            model_identity_reader=preflight_dependencies.model_identity_reader,
            server_version_reader=preflight_dependencies.server_version_reader,
            context_validator=preflight_dependencies.context_validator,
        ),
    )
    assert run.result.status == "ready"
    return Path(str(run.receipt_path)), Path(str(run.receipt_manifest_path))


def _actual_request_from_preflight(
    preflight_request: Any,
    preflight_dependencies: Any,
    tmp_path: Path,
):
    preflight = prepare_sealed_entry_preflight(
        preflight_request,
        dependencies=preflight_dependencies,
    )
    receipt, receipt_manifest = _admit_preflight(
        preflight_request,
        preflight_dependencies,
    )
    request = ActualRunPreparationRequestV1(
        contract_version="1",
        accepted_preflight_packet_path=str(
            Path(preflight_request.packet_root) / "packet.json"
        ),
        accepted_preflight_packet_sha256=preflight.packet_sha256,
        accepted_preflight_manifest_sha256=preflight.preflight_sha256,
        accepted_payload_manifest_sha256=preflight.payload_manifest_sha256,
        accepted_receipt_path=str(receipt),
        accepted_receipt_sha256=_file_sha256(receipt),
        accepted_receipt_manifest_path=str(receipt_manifest),
        accepted_receipt_manifest_sha256=_file_sha256(receipt_manifest),
        product_execution=ActualProductExecutionSelectionV1(
            contract_version="1",
            workspace_base=str((tmp_path / "actual_workspaces").resolve()),
            settings_path=str((tmp_path / "settings.yaml").resolve()),
            requested_completion_scope="compile_only",
            completion_classification=(
                ("changed_product_build", True),
                ("runtime_required", False),
            ),
            objective_verification_command=("dotnet", "test"),
            game_folder_build_binding=None,
            compiled_artifact_inspection_profile=None,
        ),
    )
    dependencies = ActualExecutionDependenciesV1(
        model_identity_reader=preflight_dependencies.model_identity_reader,
        server_version_reader=preflight_dependencies.server_version_reader,
        context_validator=preflight_dependencies.context_validator,
    )
    return request, dependencies, preflight


def _actual_request(tmp_path: Path):
    preflight_request, preflight_dependencies, _ = preflight_fixture(tmp_path)
    return _actual_request_from_preflight(
        preflight_request,
        preflight_dependencies,
        tmp_path,
    )


def _actual_request_with_dirty_platform(tmp_path: Path):
    preflight_request, preflight_dependencies, _ = preflight_fixture(tmp_path)
    platform = Path(preflight_request.platform_root)
    for path_text in _DIRTY_PLATFORM_PATHS[:3]:
        path = platform / Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline:{path_text}\n", encoding="utf-8")
    _git(platform, "add", ".")
    _git(platform, "commit", "-m", "tracked documentation baseline")
    for path_text in _DIRTY_PLATFORM_PATHS:
        path = platform / Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"wip:{path_text}\n", encoding="utf-8")
    head = _git(platform, "rev-parse", "HEAD")
    preflight_request = replace(
        preflight_request,
        allowed_dirty_paths=_DIRTY_PLATFORM_PATHS,
        selected_configuration=replace(
            preflight_request.selected_configuration,
            projection_contract_checkpoint=head,
        ),
    )
    return _actual_request_from_preflight(
        preflight_request,
        preflight_dependencies,
        tmp_path,
    )


def _prepared_actual(tmp_path: Path):
    request, dependencies, preflight = _actual_request(tmp_path)
    result = prepare_sealed_actual_run(request, dependencies=dependencies)
    return request, dependencies, result, preflight


def test_prepare_derives_future_identities_and_preserves_owner_inputs(
    tmp_path: Path,
) -> None:
    request, _, result, preflight = _prepared_actual(tmp_path)

    assert result.outer_run_id == preflight.identities.outer_run_id
    assert result.actual_attempt_id == preflight.identities.future_actual_id
    assert result.actual_authorized is False
    assert result.model_generation_calls == 0
    assert result.ProductExecution_runs == 0
    assert result.Builds == 0
    assert result.Apply_runs == 0
    root = Path(request.packet_root)
    preflight_root = Path(request.accepted_preflight_packet_path).parent
    assert root.parent == preflight_root.parent
    assert root != preflight_root
    assert root.name == (
        f"actual_{preflight.identities.outer_run_id}_"
        f"{preflight.identities.future_actual_id}"
    )
    assert {path.name for path in root.iterdir()} == {
        ACTUAL_PACKET_FILENAME,
        ACTUAL_SEAL_FILENAME,
        "RUN.md",
    }
    packet = json.loads((root / ACTUAL_PACKET_FILENAME).read_text(encoding="utf-8"))
    spec = ActualRunSpecV1.from_dict(packet["actual_run_spec"])
    preflight_packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )
    assert spec.outer_run_id == preflight.identities.outer_run_id
    assert spec.actual_attempt_id == preflight.identities.future_actual_id
    assert spec.request == "Enable a deterministic AccurateCrosshair spread toggle."
    assert spec.platform["HEAD"] == preflight_packet["platform"]["current_checkpoint"]
    assert spec.model["digest"] == "b" * 64
    assert spec.model["quantization_level"] == "Q4_K_M"
    assert spec.model["configured_context"] == 32768
    assert spec.model["output_budget"] == 4096
    assert spec.selected_configuration["main_ollama_timeout_seconds"] == 600.0
    assert spec.selected_configuration["route"] == (
        "qwen3.6:27b × current full_correction"
    )
    assert spec.product_execution == request.product_execution
    assert spec.accepted_preflight["receipt_sha256"] == _file_sha256(
        Path(request.accepted_receipt_path)
    )
    assert "feature_spec_sha256" not in json.dumps(spec.to_dict())
    run_text = (root / "RUN.md").read_text(encoding="utf-8")
    assert result.packet_sha256 in result.run_command_requiring_separate_authorization
    assert result.seal_sha256 in result.run_command_requiring_separate_authorization
    assert result.run_command_requiring_separate_authorization in run_text
    assert "receipt remains non-authorizing" in run_text


def test_public_prepare_request_has_no_independent_outer_or_actual_identity() -> None:
    fields = set(inspect.signature(ActualRunPreparationRequestV1).parameters)
    assert fields.isdisjoint(
        {
            "outer_run_id",
            "actual_attempt_id",
            "future_actual_id",
            "future_canary_number",
            "packet_root",
            "actual_lineage_packet_paths",
        }
    )


def test_run_local_request_requires_explicit_identities_as_one_pair() -> None:
    signature = inspect.signature(run_local_request)
    assert signature.parameters["run_id"].default is None
    assert signature.parameters["request_id"].default is None

    with pytest.raises(LocalEntryError, match="must be supplied together"):
        run_local_request(
            description="request",
            target_project="missing",
            workspace_base="missing",
            settings_path="missing",
            confirm_isolated_write=True,
            allow_build=True,
            allow_source_disclosure=False,
            local_ollama=True,
            model_call_budget_policy="meter_only",
            diff_approval_reader=lambda _diff: "literal user input",
            run_id="outer_001",
        )


@pytest.mark.parametrize(
    "failure",
    (
        "receipt_status",
        "receipt_hash",
        "manifest_hash",
        "payload_manifest",
        "future_identity",
        "platform_head",
        "source",
        "model",
    ),
)
def test_prepare_fails_closed_before_actual_packet_write(
    tmp_path: Path,
    failure: str,
) -> None:
    preflight_request, preflight_dependencies, predecessor = preflight_fixture(tmp_path)
    prepared_preflight = prepare_sealed_entry_preflight(
        preflight_request,
        dependencies=preflight_dependencies,
    )
    receipt, receipt_manifest = _admit_preflight(
        preflight_request,
        preflight_dependencies,
    )
    accepted_receipt_sha256 = _file_sha256(receipt)
    accepted_receipt_manifest_sha256 = _file_sha256(receipt_manifest)
    expected_actual_root = Path(preflight_request.packet_root).parent / (
        f"actual_{prepared_preflight.identities.outer_run_id}_"
        f"{prepared_preflight.identities.future_actual_id}"
    )
    if failure == "receipt_status":
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["result"]["status"] = "blocked"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "receipt_hash":
        receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif failure == "manifest_hash":
        receipt_manifest.write_text(
            receipt_manifest.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    elif failure == "payload_manifest":
        payload = json.loads(receipt_manifest.read_text(encoding="utf-8"))
        payload["payload_manifest_sha256"] = "e" * 64
        receipt_manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "future_identity":
        packet_path = Path(preflight_request.packet_root) / "packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["future_execution_binding"]["future_attempt_id"] = "actual_999"
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
    elif failure == "platform_head":
        platform = Path(preflight_request.platform_root)
        (platform / "new.txt").write_text("new\n", encoding="utf-8")
        from tests.unit.test_preflight_preparation import _git

        _git(platform, "add", ".")
        _git(platform, "commit", "-m", "drift")
    elif failure == "source":
        Path(preflight_request.target_project, "Loader.cs").write_text(
            "public sealed class Changed {}\n", encoding="utf-8"
        )
    else:
        preflight_dependencies = replace(
            preflight_dependencies,
            model_identity_reader=lambda base_url, model: {
                "name": model,
                "model": model,
                "digest": "d" * 64,
                "base_url": base_url,
                "details": {
                    "context_length": 65536,
                    "quantization_level": "Q4_K_M",
                },
                "capabilities": ["completion"],
            },
        )
    request = ActualRunPreparationRequestV1(
        contract_version="1",
        accepted_preflight_packet_path=str(
            Path(preflight_request.packet_root) / "packet.json"
        ),
        accepted_preflight_packet_sha256=prepared_preflight.packet_sha256,
        accepted_preflight_manifest_sha256=prepared_preflight.preflight_sha256,
        accepted_payload_manifest_sha256=(
            prepared_preflight.payload_manifest_sha256
        ),
        accepted_receipt_path=str(receipt),
        accepted_receipt_sha256=accepted_receipt_sha256,
        accepted_receipt_manifest_path=str(receipt_manifest),
        accepted_receipt_manifest_sha256=accepted_receipt_manifest_sha256,
        product_execution=ActualProductExecutionSelectionV1(
            contract_version="1",
            workspace_base=str((tmp_path / "workspaces").resolve()),
            settings_path=str((tmp_path / "settings.yaml").resolve()),
            requested_completion_scope="compile_only",
        ),
    )
    dependencies = ActualExecutionDependenciesV1(
        model_identity_reader=preflight_dependencies.model_identity_reader,
        server_version_reader=preflight_dependencies.server_version_reader,
        context_validator=preflight_dependencies.context_validator,
    )

    with pytest.raises(ActualRunError):
        prepare_sealed_actual_run(request, dependencies=dependencies)

    assert not expected_actual_root.exists()
    assert Path(predecessor["packet"]).is_file()


def test_execute_requires_runtime_authorization_and_revalidates_before_owner(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    packet_root = prepared.packet_root
    calls: list[dict[str, Any]] = []
    dependencies = replace(
        dependencies,
        execution_owner=lambda **kwargs: calls.append(kwargs) or {},
    )

    with pytest.raises(ActualRunError, match="ACTUAL_EXECUTION_AUTHORIZATION_REQUIRED"):
        execute_sealed_actual_run(
            packet_root,
            expected_packet_sha256=prepared.packet_sha256,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=False,
            diff_approval_reader=lambda _diff: "user-input",
            dependencies=dependencies,
        )

    assert calls == []
    assert not (Path(packet_root) / ACTUAL_CONSUMPTION_FILENAME).exists()

    with pytest.raises(ActualRunError, match="ACTUAL_REVIEW_HASH_DRIFT"):
        execute_sealed_actual_run(
            request.packet_root,
            expected_packet_sha256="f" * 64,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=True,
            diff_approval_reader=lambda _diff: "user-input",
            dependencies=dependencies,
        )
    assert calls == []
    assert not (Path(request.packet_root) / ACTUAL_CONSUMPTION_FILENAME).exists()


def test_repeat_root_does_not_overwrite_and_concurrent_seal_has_one_winner(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    marker = Path(request.packet_root) / ACTUAL_PACKET_FILENAME
    packet_sha256 = _file_sha256(marker)

    with pytest.raises(ActualRunError, match="ACTUAL_PACKET_ROOT_ALREADY_EXISTS"):
        prepare_sealed_actual_run(request, dependencies=dependencies)
    assert _file_sha256(marker) == packet_sha256

    race, race_dependencies, _ = _actual_request(tmp_path / "race")

    def prepare() -> str:
        try:
            return prepare_sealed_actual_run(
                race,
                dependencies=race_dependencies,
            ).status
        except ActualRunError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: prepare(), range(2)))

    assert outcomes.count("sealed") == 1
    assert outcomes.count("ACTUAL_PACKET_ROOT_ALREADY_EXISTS") == 1
    assert {path.name for path in Path(race.packet_root).iterdir()} == set(
        ACTUAL_SEALED_FILE_SET_FOR_TEST
    )


ACTUAL_SEALED_FILE_SET_FOR_TEST = (
    ACTUAL_PACKET_FILENAME,
    ACTUAL_SEAL_FILENAME,
    "RUN.md",
)


@pytest.mark.parametrize(
    "drift_file",
    (ACTUAL_PACKET_FILENAME, ACTUAL_SEAL_FILENAME),
)
def test_execute_rejects_actual_seal_drift_before_owner(
    tmp_path: Path,
    drift_file: str,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    selected = Path(request.packet_root) / drift_file
    selected.write_text(selected.read_text(encoding="utf-8") + " ", encoding="utf-8")
    calls = 0

    def owner(**_kwargs):
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(ActualRunError):
        execute_sealed_actual_run(
            request.packet_root,
            expected_packet_sha256=prepared.packet_sha256,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=True,
            diff_approval_reader=lambda _diff: "user-input",
            dependencies=replace(dependencies, execution_owner=owner),
        )

    assert calls == 0


@pytest.mark.parametrize(
    "drift",
    ("receipt", "preflight_payload", "platform_head", "source", "model"),
)
def test_execute_rejects_bound_fact_drift_before_owner(
    tmp_path: Path,
    drift: str,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    packet_root = prepared.packet_root
    if drift == "receipt":
        path = Path(request.accepted_receipt_path)
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif drift == "preflight_payload":
        path = Path(request.accepted_preflight_packet_path).parent / "sealed_entry_runner.py"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif drift == "platform_head":
        packet = json.loads(
            Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
        )
        platform = Path(packet["platform"]["repository_root"])
        (platform / "drift.txt").write_text("drift\n", encoding="utf-8")
        from tests.unit.test_preflight_preparation import _git

        _git(platform, "add", ".")
        _git(platform, "commit", "-m", "drift")
    elif drift == "source":
        packet = json.loads(
            Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
        )
        Path(packet["source_identity"]["target_project"], "Loader.cs").write_text(
            "public sealed class Drift {}\n", encoding="utf-8"
        )
    else:
        dependencies = replace(
            dependencies,
            model_identity_reader=lambda base_url, model: {
                "name": model,
                "model": model,
                "digest": "c" * 64,
                "base_url": base_url,
                "details": {
                    "context_length": 65536,
                    "quantization_level": "Q4_K_M",
                },
                "capabilities": ["completion"],
            },
        )
    calls = 0

    def owner(**_kwargs):
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(ActualRunError):
        execute_sealed_actual_run(
            packet_root,
            expected_packet_sha256=prepared.packet_sha256,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=True,
            diff_approval_reader=lambda _diff: "literal user input",
            dependencies=replace(dependencies, execution_owner=owner),
        )

    assert calls == 0
    assert not (Path(packet_root) / ACTUAL_CONSUMPTION_FILENAME).exists()


def test_execute_consumes_before_exactly_one_owner_call_and_projects_safe_journal(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    calls: list[dict[str, Any]] = []
    diff_payload = {
        "diff_hash": "a" * 64,
        "unified_diff": "FULL DIFF\nraw source sentinel",
        "unified_diff_sha256": "b" * 64,
    }
    approvals: list[dict[str, Any]] = []

    def read_approval(value: dict[str, Any]) -> str:
        approvals.append(value)
        return "literal user input"

    def owner(**kwargs):
        assert (Path(request.packet_root) / ACTUAL_CONSUMPTION_FILENAME).is_file()
        calls.append(kwargs)
        assert kwargs["diff_approval_reader"](diff_payload) == "literal user input"
        return {
            "run_id": kwargs["run_id"],
            "current_stage": "blocked",
            "run_root": str(Path(kwargs["workspace_base"]) / kwargs["run_id"]),
            "model_call_count": 2,
            "local_model_input_audit": {"model_role": "requirement_findings"},
            "product_execution_result": {
                "platform_status": "completed",
                "execution_disposition": "domain_blocked",
                "failure_stage": None,
                "terminal_node": "blocked",
            },
            "raw_prompt": "raw prompt sentinel",
            "reasoning": "raw reasoning sentinel",
            "source": "raw source sentinel",
        }

    result = execute_sealed_actual_run(
        request.packet_root,
        expected_packet_sha256=prepared.packet_sha256,
        expected_seal_sha256=prepared.seal_sha256,
        actual_execution_authorized=True,
        diff_approval_reader=read_approval,
        dependencies=replace(dependencies, execution_owner=owner),
    )

    assert result.status == "terminal"
    assert len(calls) == 1
    assert approvals == [diff_payload]
    kwargs = calls[0]
    assert kwargs["run_id"] == prepared.outer_run_id
    assert kwargs["request_id"] == prepared.actual_attempt_id
    assert kwargs["description"] == "Enable a deterministic AccurateCrosshair spread toggle."
    assert kwargs["ollama_timeout_seconds"] == 600.0
    assert kwargs["ollama_context_window"] == 32768
    assert kwargs["ollama_max_output_tokens"] == 4096
    assert kwargs["local_ollama"] is True
    journal_text = (Path(request.packet_root) / ACTUAL_JOURNAL_FILENAME).read_text(
        encoding="utf-8"
    )
    journal = json.loads(journal_text)
    assert journal["model_calls"] == 2
    assert journal["model_call_accounting_complete"] is True
    assert journal["last_observed_model_role"] == "requirement_findings"
    assert journal["stage"] == "blocked"
    assert journal["terminal"] == {
        "execution_disposition": "domain_blocked",
        "failure_stage": None,
        "platform_status": "completed",
        "terminal_node": "blocked",
    }
    assert all(
        sentinel not in journal_text
        for sentinel in ("raw prompt sentinel", "raw reasoning sentinel", "raw source sentinel")
    )

    with pytest.raises(ActualRunError, match="ACTUAL_IDENTITY_ALREADY_CONSUMED"):
        execute_sealed_actual_run(
            request.packet_root,
            expected_packet_sha256=prepared.packet_sha256,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=True,
            diff_approval_reader=read_approval,
            dependencies=replace(dependencies, execution_owner=owner),
        )
    assert len(calls) == 1


def test_owner_exception_is_one_terminal_without_retry_or_automatic_replay(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    calls = 0
    approvals = 0

    def owner(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("raw source and reasoning must not persist")

    def approval(_diff):
        nonlocal approvals
        approvals += 1
        return "not used"

    result = execute_sealed_actual_run(
        request.packet_root,
        expected_packet_sha256=prepared.packet_sha256,
        expected_seal_sha256=prepared.seal_sha256,
        actual_execution_authorized=True,
        diff_approval_reader=approval,
        dependencies=replace(dependencies, execution_owner=owner),
    )

    assert result.status == "failed"
    assert calls == 1
    assert approvals == 0
    journal_text = (Path(request.packet_root) / ACTUAL_JOURNAL_FILENAME).read_text(
        encoding="utf-8"
    )
    journal = json.loads(journal_text)
    assert journal["terminal"] == {
        "exception_type": "RuntimeError",
        "status": "owner_exception",
    }
    assert "raw source and reasoning" not in journal_text


def test_historical_v6_fixture_compatibility_claim_is_identity_and_receipt_only(
    tmp_path: Path,
) -> None:
    request, _, result, _ = _prepared_actual(tmp_path)
    packet = json.loads(
        (Path(request.packet_root) / ACTUAL_PACKET_FILENAME).read_text(encoding="utf-8")
    )
    spec = packet["actual_run_spec"]
    historical = json.loads(
        Path(request.actual_lineage_packet_paths[0]).read_text(encoding="utf-8")
    )

    assert historical["packet_version"] == "p2_receipt_bound_actual_packet_v6"
    assert set(historical) == {
        "packet_version",
        "outer_run_id",
        "actual_attempt_id",
        "accepted_entry_receipt",
        "seal_status",
    }
    assert spec["accepted_preflight"]["receipt_status"] == "ready"
    assert spec["accepted_preflight"]["receipt_actual_authorized"] is False
    assert result.semantic_compatibility_boundary == (
        "repository_v6_fixture_identity_and_receipt_lineage_only"
    )


def test_stable_v7_packet_is_readable_by_existing_actual_lineage_owner(
    tmp_path: Path,
) -> None:
    request, _, _, preflight = _prepared_actual(tmp_path)

    record = _load_actual_record(
        str(Path(request.packet_root) / ACTUAL_PACKET_FILENAME)
    )

    assert record.outer_run_id == preflight.identities.outer_run_id
    assert record.actual_id == preflight.identities.future_actual_id
    assert record.accepted_receipt_sha256 == request.accepted_receipt_sha256


def test_historical_semantic_contract_uses_real_spec_and_owner_projection(
    tmp_path: Path,
) -> None:
    request, _, prepared, _ = _prepared_actual(tmp_path)
    packet = json.loads(
        (Path(prepared.packet_root) / ACTUAL_PACKET_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    spec = ActualRunSpecV1.from_dict(packet["actual_run_spec"])
    owner_inputs = _execution_arguments(
        spec,
        lambda _diff: "literal user input",
    )
    preflight_packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )

    assert spec.request == preflight_packet["request"]
    assert spec.source["target_project"] == preflight_packet["source_identity"][
        "target_project"
    ]
    assert spec.source["source_revision_binding"] == preflight_packet[
        "source_identity"
    ]["source_revision_binding"]
    assert spec.model == {
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "local-neutral-model",
        "digest": "b" * 64,
        "quantization_level": "Q4_K_M",
        "server_version": "1.2.3",
        "context_limit": 65536,
        "configured_context": 32768,
        "output_budget": 4096,
        "local_only": True,
    }
    assert spec.platform["HEAD"] == preflight_packet["platform"][
        "current_checkpoint"
    ]
    assert spec.accepted_preflight["receipt_sha256"] == (
        request.accepted_receipt_sha256
    )
    assert spec.selected_configuration["route"] == (
        "qwen3.6:27b × current full_correction"
    )
    assert spec.selected_configuration["model_projection_fields"] == [
        "requirements",
        "input_dispositions",
    ]
    assert spec.selected_configuration["provider_fallback"] is False
    assert spec.selected_configuration["automatic_retry"] is False
    assert owner_inputs["description"] == spec.request
    assert owner_inputs["target_project"] == spec.source["target_project"]
    assert owner_inputs["source_revision_binding"] == spec.source[
        "source_revision_binding"
    ]
    assert owner_inputs["ollama_model"] == spec.model["model"]
    assert owner_inputs["ollama_context_window"] == spec.model[
        "configured_context"
    ]
    assert owner_inputs["ollama_max_output_tokens"] == spec.model[
        "output_budget"
    ]
    assert owner_inputs["ollama_timeout_seconds"] == 600.0
    assert owner_inputs["completion_classification"] == dict(
        request.product_execution.completion_classification
    )
    assert owner_inputs["run_gtfo_runtime"] is False


def test_cli_separates_prepare_from_later_actual_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
            "prepare-sealed-actual",
            "--accepted-preflight-packet",
            "C:/preflight/packet.json",
            "--accepted-preflight-packet-sha256",
            "a" * 64,
            "--accepted-preflight-manifest-sha256",
            "b" * 64,
            "--accepted-payload-manifest-sha256",
            "c" * 64,
            "--accepted-receipt",
            "C:/preflight/receipt.json",
            "--accepted-receipt-sha256",
            "d" * 64,
            "--accepted-receipt-manifest",
            "C:/preflight/manifest.json",
            "--accepted-receipt-manifest-sha256",
            "e" * 64,
            "--workspace-root",
            "C:/workspaces",
            "--settings",
            "C:/settings.yaml",
    ]
    prepare = build_parser().parse_args(arguments)
    assert prepare.command == "prepare-sealed-actual"
    assert set(vars(prepare)).isdisjoint({"outer_run_id", "actual_attempt_id"})

    captured: dict[str, Any] = {}

    class Result:
        def to_dict(self) -> dict[str, Any]:
            return {"status": "sealed", "actual_authorized": False}

    def prepare_owner(request: ActualRunPreparationRequestV1) -> Result:
        captured["request"] = request
        return Result()

    monkeypatch.setattr(
        "src.mod_agent.cli.prepare_sealed_actual_run",
        prepare_owner,
    )
    assert cli_main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "sealed",
        "actual_authorized": False,
    }
    prepared_request = captured["request"]
    assert isinstance(prepared_request, ActualRunPreparationRequestV1)
    assert prepared_request.accepted_preflight_packet_sha256 == "a" * 64
    assert prepared_request.product_execution.requested_completion_scope == (
        "compile_only"
    )

    exit_code = cli_main(
        [
            "execute-sealed-actual",
            "--packet-root",
            "C:/missing",
            "--expected-packet-sha256",
            "a" * 64,
            "--expected-seal-sha256",
            "b" * 64,
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["code"] == "ACTUAL_EXECUTION_AUTHORIZATION_REQUIRED"


def test_same_ready_receipt_cannot_prepare_and_execute_from_two_roots(
    tmp_path: Path,
) -> None:
    first_request, dependencies, first, _ = _prepared_actual(tmp_path)
    public_fields = set(inspect.signature(ActualRunPreparationRequestV1).parameters)
    assert "packet_root" not in public_fields
    assert "actual_lineage_packet_paths" not in public_fields
    with pytest.raises(ActualRunError, match="ACTUAL_PACKET_ROOT_ALREADY_EXISTS"):
        prepare_sealed_actual_run(first_request, dependencies=dependencies)
    owner_calls = 0

    def owner(**_kwargs):
        nonlocal owner_calls
        owner_calls += 1
        return {"model_call_count": 0, "current_stage": "blocked"}

    selected = replace(dependencies, execution_owner=owner)
    execute_sealed_actual_run(
        first_request.packet_root,
        expected_packet_sha256=first.packet_sha256,
        expected_seal_sha256=first.seal_sha256,
        actual_execution_authorized=True,
        diff_approval_reader=lambda _diff: "literal user input",
        dependencies=selected,
    )

    assert owner_calls == 1


def test_same_receipt_concurrent_prepare_has_one_physical_winner_even_without_lineage(
    tmp_path: Path,
) -> None:
    request, dependencies, _ = _actual_request(tmp_path)

    def prepare(selected: ActualRunPreparationRequestV1) -> str:
        try:
            return prepare_sealed_actual_run(
                selected,
                dependencies=dependencies,
            ).status
        except ActualRunError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(prepare, (request, request)))

    assert outcomes.count("sealed") == 1
    assert outcomes.count("ACTUAL_PACKET_ROOT_ALREADY_EXISTS") == 1


def test_preexisting_journal_blocks_before_consumption_and_owner(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    journal_path = Path(request.packet_root) / ACTUAL_JOURNAL_FILENAME
    journal_path.write_text("{}\n", encoding="utf-8")
    owner_calls = 0

    def owner(**_kwargs):
        nonlocal owner_calls
        owner_calls += 1
        return {}

    with pytest.raises(ActualRunError, match="ACTUAL_JOURNAL_ALREADY_EXISTS"):
        execute_sealed_actual_run(
            request.packet_root,
            expected_packet_sha256=prepared.packet_sha256,
            expected_seal_sha256=prepared.seal_sha256,
            actual_execution_authorized=True,
            diff_approval_reader=lambda _diff: "literal user input",
            dependencies=replace(dependencies, execution_owner=owner),
        )

    assert owner_calls == 0
    assert not (Path(request.packet_root) / ACTUAL_CONSUMPTION_FILENAME).exists()


def test_concurrent_execute_has_one_journal_claim_and_one_owner_call(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)
    owner_calls = 0

    def owner(**_kwargs):
        nonlocal owner_calls
        owner_calls += 1
        return {"model_call_count": 0, "current_stage": "blocked"}

    selected = replace(dependencies, execution_owner=owner)

    def execute() -> str:
        try:
            return execute_sealed_actual_run(
                prepared.packet_root,
                expected_packet_sha256=prepared.packet_sha256,
                expected_seal_sha256=prepared.seal_sha256,
                actual_execution_authorized=True,
                diff_approval_reader=lambda _diff: "literal user input",
                dependencies=selected,
            ).status
        except ActualRunError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: execute(), range(2)))

    assert outcomes.count("terminal") == 1
    assert len(outcomes) == 2
    assert owner_calls == 1


def test_packet_contains_no_generated_per_run_python_launcher(tmp_path: Path) -> None:
    request, _, _, _ = _prepared_actual(tmp_path)

    assert not (Path(request.packet_root) / "actual_runner.py").exists()
    assert "actual_runner.py" not in json.loads(
        (Path(request.packet_root) / ACTUAL_SEAL_FILENAME).read_text(encoding="utf-8")
    )["files"]
    run_text = (Path(request.packet_root) / "RUN.md").read_text(encoding="utf-8")
    assert "-m src.main execute-sealed-actual" in run_text
    assert "actual_runner.py" not in run_text


def test_owner_exception_journal_does_not_invent_zero_model_calls(
    tmp_path: Path,
) -> None:
    request, dependencies, prepared, _ = _prepared_actual(tmp_path)

    def owner(**_kwargs):
        raise RuntimeError("unknown model-call progress")

    execute_sealed_actual_run(
        request.packet_root,
        expected_packet_sha256=prepared.packet_sha256,
        expected_seal_sha256=prepared.seal_sha256,
        actual_execution_authorized=True,
        diff_approval_reader=lambda _diff: "literal user input",
        dependencies=replace(dependencies, execution_owner=owner),
    )
    journal = json.loads(
        (Path(request.packet_root) / ACTUAL_JOURNAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert journal["model_calls"] is None
    assert journal["model_call_accounting_complete"] is False


def test_cli_accepts_typed_game_folder_and_inspection_profile_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    game_folder = tmp_path / "game_folder_binding.json"
    inspection = tmp_path / "inspection_profile.json"
    game_payload = {
        "binding_version": "run_local_game_folder_binding_v1",
        "source_root": str((tmp_path / "GameFolder").resolve()),
        "snapshot_sha256": "f" * 64,
        "file_count": 1,
        "property_name": "GameFolder",
    }
    inspection_payload = {
        "project_path": str((tmp_path / "Inspector.csproj").resolve()),
        "method_symbol": "Inspector.Program.Main",
        "expected_guid": "plugin.guid",
        "expected_name": "Plugin",
        "expected_version": "1.0.0",
        "timeout_seconds": 120,
    }
    game_folder.write_text(json.dumps(game_payload), encoding="utf-8")
    inspection.write_text(json.dumps(inspection_payload), encoding="utf-8")

    arguments = [
            "prepare-sealed-actual",
            "--accepted-preflight-packet",
            "C:/preflight/packet.json",
            "--accepted-preflight-packet-sha256",
            "a" * 64,
            "--accepted-preflight-manifest-sha256",
            "b" * 64,
            "--accepted-payload-manifest-sha256",
            "c" * 64,
            "--accepted-receipt",
            "C:/preflight/receipt.json",
            "--accepted-receipt-sha256",
            "d" * 64,
            "--accepted-receipt-manifest",
            "C:/preflight/manifest.json",
            "--accepted-receipt-manifest-sha256",
            "e" * 64,
            "--workspace-root",
            "C:/workspaces",
            "--settings",
            "C:/settings.yaml",
            "--game-folder-build-binding-file",
            str(game_folder),
            "--compiled-artifact-inspection-profile-file",
            str(inspection),
    ]
    args = build_parser().parse_args(arguments)

    assert args.game_folder_build_binding_file == str(game_folder)
    assert args.compiled_artifact_inspection_profile_file == str(inspection)

    captured: dict[str, Any] = {}

    class Result:
        def to_dict(self) -> dict[str, Any]:
            return {"status": "sealed", "actual_authorized": False}

    def prepare_owner(request: ActualRunPreparationRequestV1) -> Result:
        captured["request"] = request
        return Result()

    monkeypatch.setattr(
        "src.mod_agent.cli.prepare_sealed_actual_run",
        prepare_owner,
    )
    assert cli_main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "sealed"
    selection = captured["request"].product_execution
    assert dict(selection.game_folder_build_binding) == game_payload
    assert dict(selection.compiled_artifact_inspection_profile) == inspection_payload


def test_typed_profiles_roundtrip_from_spec_into_stable_owner_inputs(
    tmp_path: Path,
) -> None:
    request, dependencies, _ = _actual_request(tmp_path)
    game_payload = {
        "binding_version": "run_local_game_folder_binding_v1",
        "source_root": str((tmp_path / "GameFolder").resolve()),
        "snapshot_sha256": "f" * 64,
        "file_count": 3,
        "property_name": "GameFolder",
    }
    inspection_payload = {
        "project_path": str((tmp_path / "Inspector.csproj").resolve()),
        "method_symbol": "Inspector.Program.Main",
        "expected_guid": "plugin.guid",
        "expected_name": "Plugin",
        "expected_version": "1.0.0",
        "timeout_seconds": 120,
    }
    request = replace(
        request,
        product_execution=replace(
            request.product_execution,
            game_folder_build_binding=game_payload,
            compiled_artifact_inspection_profile=inspection_payload,
        ),
    )
    prepared = prepare_sealed_actual_run(request, dependencies=dependencies)
    packet = json.loads(
        (Path(prepared.packet_root) / ACTUAL_PACKET_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    spec = ActualRunSpecV1.from_dict(packet["actual_run_spec"])
    owner_inputs = _execution_arguments(
        spec,
        lambda _diff: "literal user input",
    )

    game_binding = owner_inputs["game_folder_build_binding"]
    inspection_profile = owner_inputs["compiled_artifact_inspection_profile"]
    assert isinstance(game_binding, RunLocalGameFolderBindingV1)
    assert game_binding.to_dict() == game_payload
    assert isinstance(inspection_profile, CompiledArtifactInspectionProfile)
    assert inspection_profile.to_dict() == inspection_payload


def test_real_platform_gate_path_status_shape_allows_dirty_bytes_at_seal(
    tmp_path: Path,
) -> None:
    request, dependencies, _ = _actual_request_with_dirty_platform(tmp_path)
    packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )
    entry = entry_request_from_preflight_packet(
        packet,
        payload_manifest_sha256=request.accepted_payload_manifest_sha256,
    )
    gates = verify_entry_current_facts(
        entry,
        dependencies=EntryAdmissionDependenciesV1(
            git_admission=entry.strict_git_admission(),
            model_identity_reader=dependencies.model_identity_reader,
            server_version_reader=dependencies.server_version_reader,
            context_validator=dependencies.context_validator,
        ),
    )
    platform_gate = next(item for item in gates if item["gate"] == "platform")
    dirty_records = platform_gate["facts"]["dirty_records"]

    assert {frozenset(record) for record in dirty_records} == {
        frozenset(("path", "status"))
    }
    assert {record["path"] for record in dirty_records} == set(
        _DIRTY_PLATFORM_PATHS
    )
    result = prepare_sealed_actual_run(request, dependencies=dependencies)
    assert Path(result.packet_root).is_dir()


def test_dirty_platform_byte_drift_blocks_before_packet_write(tmp_path: Path) -> None:
    request, dependencies, _ = _actual_request_with_dirty_platform(tmp_path)
    root = Path(request.packet_root)
    packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )
    selected = Path(packet["platform"]["repository_root"]) / Path(
        _DIRTY_PLATFORM_PATHS[0]
    )
    selected.write_text(
        selected.read_text(encoding="utf-8") + "drift\n",
        encoding="utf-8",
    )

    with pytest.raises(ActualRunError, match="ACTUAL_PLATFORM_DIRTY_BYTES_DRIFT"):
        prepare_sealed_actual_run(request, dependencies=dependencies)

    assert not root.exists()


def test_dirty_platform_path_set_drift_blocks_before_packet_write(tmp_path: Path) -> None:
    request, dependencies, _ = _actual_request_with_dirty_platform(tmp_path)
    root = Path(request.packet_root)
    packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )
    platform_root = Path(packet["platform"]["repository_root"])
    (platform_root / Path(_DIRTY_PLATFORM_PATHS[-1])).unlink()

    with pytest.raises(ActualRunError, match="ACTUAL_PLATFORM_DIRTY_BYTES_DRIFT"):
        prepare_sealed_actual_run(request, dependencies=dependencies)

    assert not root.exists()


@pytest.mark.parametrize("unsafe_kind", ("escape", "non_file"))
def test_dirty_platform_unsafe_path_or_non_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    request, dependencies, _ = _actual_request_with_dirty_platform(tmp_path)
    root = Path(request.packet_root)
    packet = json.loads(
        Path(request.accepted_preflight_packet_path).read_text(encoding="utf-8")
    )
    entry = entry_request_from_preflight_packet(
        packet,
        payload_manifest_sha256=request.accepted_payload_manifest_sha256,
    )
    platform_root = Path(packet["platform"]["repository_root"])
    if unsafe_kind == "escape":
        dirty_path = "../escape.txt"
    else:
        dirty_path = "docs/not_a_file"
        (platform_root / Path(dirty_path)).mkdir(parents=True)
    packet["platform"]["allowed_dirty_path_sha256_at_seal"] = {
        dirty_path: "f" * 64
    }
    monkeypatch.setattr(
        "src.mod_agent.actual_execution.verify_entry_current_facts",
        lambda *_args, **_kwargs: (
            {
                "gate": "platform",
                "decision": "passed",
                "facts": {
                    "dirty_records": [
                        {"path": dirty_path, "status": " M"}
                    ]
                },
            },
        ),
    )

    with pytest.raises(ActualRunError, match="ACTUAL_PLATFORM_DIRTY_BYTES_DRIFT"):
        _verify_current_facts(entry, packet, dependencies)

    assert not root.exists()
