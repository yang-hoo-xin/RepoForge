"""Public Repair CLI acceptance with only the HTTP transport replaced offline.

The selected evaluator is a real hashed fixture file. Its static marker check
and all model responses are synthetic test data, not a Mod implementation or a
claim of live-model quality. Adapter, worker, Graph and serializers stay real.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.cli import main
from src.mod_agent.evidence import (
    RgLiveEvidenceProvider,
    build_requirement_coverage_pack,
    materialize_requirement_coverage_projection,
)
from src.mod_agent.fresh_planning_continuation import (
    PLANNING_CONTINUATION_CONTRACT_VERSION,
    PlanningContinuationInputV1,
    _canonical_sha256,
    run_fresh_patch_continuation,
)
from src.mod_agent.local_entry import (
    _restore_patch_agent_context,
    make_exact_no_write_dry_run_factory,
)
from src.mod_agent.proposal.patch_agent_loop import (
    PatchAgentLoopPolicy,
    PlanBoundSourceInspector,
    expand_patch_source_observations,
)
from tests.unit.test_fresh_planning_continuation import (
    _evidence_by_requirement,
    _feature_payload,
    _fresh_execution_receipt,
    _Inputs,
    _local_patch_adapter,
    _patch_cli_args,
    _patch_request,
    _positive_findings,
    _projection,
    _source_hashes,
)
from tests.unit.test_patch_repair_continuation import _repair_case

_FACTORY_SOURCE = '''import hashlib
import json
import sys
from pathlib import Path

def digest(value):
    return hashlib.sha256(value).hexdigest()

class Evaluator:
    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.calls = 0

    def verification_identity(self):
        return {"validator_sha256": digest(Path(__file__).read_bytes()),
                "inputs_sha256": digest(b"require RecordRepairMarker();"),
                "environment_sha256": digest(sys.version.encode()),
                "deterministic_reuse_allowed": True}

    def __call__(self, context):
        self.calls += 1
        passed = all("RecordRepairMarker();" in change["patch"]["after_text"]
                     for change in context["generated_changes"])
        report = {"status": "passed" if passed else "behavior_failed",
                  "failure_owner": None if passed else "patch_implementation",
                  "owner_binding_ids": [] if passed else [context["generated_changes"][0]["edit_binding_id"]],
                  "observations": [] if passed else [{"check_id": "cli_synthetic_repair_marker",
                      "input": {"enabled": True}, "expected": {"repair_marker_present": True},
                      "observed": {"repair_marker_present": False}}],
                  "generated_changes_sha256": context["generated_changes_sha256"],
                  "source_snapshot_sha256": context["source_snapshot_sha256"],
                  "materialized_code_identity": context["materialized_code_identity"],
                  "verification_identity": self.verification_identity(),
                  "execution_origin": "executed", "verification_scope": "synthetic static marker check"}
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root / ("report-%03d.json" % self.calls)
        raw = json.dumps(report, ensure_ascii=False, sort_keys=True).encode()
        path.write_bytes(raw)
        report["report_ref"] = {"path": str(path), "sha256": digest(raw)}
        return report

def build_evaluator(*, output_root, construction_marker):
    marker = Path(construction_marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("a", encoding="utf-8") as stream:
        stream.write("constructed\\n")
    return Evaluator(output_root)
'''


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _public_case(tmp_path, *, preparation=None, selected_policy=None):
    inputs, request, parent = preparation or _repair_case(tmp_path / "case")
    factory = tmp_path / "selected_evaluator.py"
    factory.write_text(_FACTORY_SOURCE, encoding="utf-8")
    marker = tmp_path / "factory-constructions.txt"
    reports = tmp_path / "verifier-results"
    policy = PatchAgentLoopPolicy(
        max_total_model_turns=4, max_tool_rounds=3,
        finalize_after_first_observation=False, require_source_inspection=False,
        enable_supporting_reads=True, allow_upstream_request=True, allow_batched_reads=True,
    )
    config = {
        "schema_version": "patch_worker_configuration_v1", "policy": selected_policy or asdict(policy),
        "enable_recovery_loop": True,
        "behavior_evaluator": {
            "factory_ref": {"path": str(factory), "sha256": _sha(factory.read_bytes())},
            "factory_name": "build_evaluator",
            "kwargs": {"output_root": str(reports), "construction_marker": str(marker)},
        },
    }
    config_path = _write_json(tmp_path / "worker.json", config)
    repair = {key: asdict(getattr(request, key)) if key in (
        "repair_parent_proposal_ref", "repair_parent_review_ref") else dict(getattr(request, key))
        for key in ("repair_parent_proposal_ref", "repair_parent_review_ref", "repair_failure_report_ref",
                    "repair_parent_response_ref", "repair_parent_attempt_ref")} if request.repair_parent_proposal_ref is not None else {}
    repair_path = _write_json(tmp_path / "repair-inputs.json", repair)
    argv = _patch_cli_args(request)
    argv[argv.index("--ollama-context-window") + 1] = "49152"
    argv.extend(["--patch-worker-config", str(config_path), "--repair-inputs", str(repair_path),
                 "--attempt-id", "public-cli-synthetic-recovery"])
    return SimpleNamespace(inputs=inputs, request=request, parent=parent, argv=argv, config=config,
                           config_path=config_path, repair=repair, repair_path=repair_path,
                           factory=factory, marker=marker, reports=reports)


def _five_binding_repair_case(tmp_path, *, source_files=None):
    """Build genuine synthetic Evidence/Plan revisions before retaining a failed parent."""
    workspace = tmp_path / "workspace"
    source = workspace / "generated_mods" / "SyntheticMod"
    source.mkdir(parents=True)
    symbols = ["ChangeMarkerA", "ChangeMarkerB", "ChangeMarkerC", "ChangeMarkerD", "ChangeMarkerE"]
    # The long shared comment makes five duplicate full-file windows exceed 32 KiB.
    # It is fixture source, and is never a patch answer or a real Mod implementation.
    shared_context = "// SHARED_SYNTHETIC_CONTEXT " + "state-observation " * 425 + "\n"
    behavior = source / "Behavior.cs"
    if source_files is None:
        source_files = {
            "Behavior.cs": "public sealed class Behavior\n{\n" + shared_context + "".join(
                f"    public void {symbol}()\n    {{\n        KeepExistingFeedback();\n    }}\n"
                for symbol in symbols) + "}\n",
            "Configuration.cs": 'public sealed class Configuration\n{\n    public const string ConfigKey = "Marker.Enabled";\n}\n',
            "Preservation.cs": "public sealed class Preservation\n{\n    public void PreserveFeedback() { }\n}\n",
        }
        duplicate_full_windows = True
    else:
        duplicate_full_windows = False
    for name, text in source_files.items():
        (source / name).write_text(text, encoding="utf-8")
    feature = _feature_payload()
    feature["target_behavior"] = "Update the five marker methods: " + ", ".join(symbols) + "."
    projection = _projection()
    projection["requirements"][0].update(text=feature["target_behavior"], query_leads=["ChangeMarker"])
    materialized = materialize_requirement_coverage_projection(
        feature_data=feature, request=feature["user_request"], projection=projection,
    )
    retrieved = RgLiveEvidenceProvider(workspace).collect_query_set(
        queries=tuple(dict(value) for value in materialized["queries"]),
        search_root="generated_mods/SyntheticMod", max_items=8,
    )
    findings = _positive_findings(materialized, retrieved)
    change_requirement = materialized["requirements"][0]["requirement_id"]
    edit_ids = list(dict.fromkeys(_evidence_by_requirement(materialized, retrieved)[change_requirement]))
    assert len(edit_ids) == 5
    findings["findings"][0].update(evidence_ids=edit_ids, edit_target_evidence_ids=edit_ids)
    pack = build_requirement_coverage_pack(
        query=feature["user_request"], materialized_projection=materialized,
        retrieved_pack=retrieved, findings=findings,
    )
    pack = replace(pack, budget_summary={**pack.budget_summary,
                                        "fresh_coverage_model_execution": _fresh_execution_receipt()})
    store = ArtifactStore(tmp_path / "artifact-store")
    feature_ref = store.write(feature_id=feature["feature_id"], artifact_type="feature_spec", payload=feature)
    evidence_ref = store.write(feature_id=feature["feature_id"], artifact_type="evidence_pack",
                               payload=pack.to_artifact_payload(), upstream_refs=[feature_ref])
    planning = PlanningContinuationInputV1(
        contract_version=PLANNING_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=str(store.root), workspace_root=str(workspace),
        feature_spec_ref=feature_ref, evidence_pack_ref=evidence_ref,
        planning_target_evidence_ids=tuple(edit_ids),
    )
    inputs = _Inputs(request=planning, store=store, workspace=workspace, behavior=behavior,
                     feature_ref=feature_ref, evidence_ref=evidence_ref, feature_payload=feature)
    request = _patch_request(inputs)
    plan = store.read(request.implementation_plan_ref)["data"]
    inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan,
                                          evidence_pack=store.read(evidence_ref)["data"],
                                          policy=PatchAgentLoopPolicy(enable_supporting_reads=True))
    assert len(inspector.available_binding_ids) == 5
    assert len(inspector.available_read_binding_ids) >= 6
    windows = [inspector.inspect(binding_id=binding, observation_id=f"fixture-check-{number}").excerpt
               for number, binding in enumerate(inspector.available_binding_ids)]
    if duplicate_full_windows:
        assert all(window == windows[0] for window in windows)
        assert sum(len(window.encode()) for window in windows) > 32768
    parent_adapter, _ = _local_patch_adapter(inputs)
    parent = run_fresh_patch_continuation(
        request, patch_change_factory=parent_adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(workspace),
    )
    assert parent.status == "terminal", parent.to_dict()
    proposal = store.read(ArtifactRef(**parent.patch_proposal_artifact))["data"]
    receipt_path = Path(parent.attempt_receipt["path"])
    parent_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    response = parent_receipt["factory_receipt"]["model_response"]
    native = [{
        "change_id": change["change_id"], "change_type": change["change_type"],
        "edit_binding_id": change["edit_binding_id"], "supporting_binding_ids": change["supporting_binding_ids"],
        "operation": "exact_text_replace", "rationale": change["rationale"], "risk_tags": ["medium"],
        "expected_validation": change["expected_validation"], "rollback_hint": change["rollback_hint"],
        "patch": change["patch"],
    } for change in proposal["diff_intent"]]
    observed = all("RecordRepairMarker();" in change["patch"]["after_text"] for change in native)
    assert observed is False
    report_path = _write_json(tmp_path / "five-binding-parent-failure.json", {
        "status": "behavior_failed", "failure_owner": "patch_implementation",
        "owner_binding_ids": [change["edit_binding_id"] for change in native],
        "observations": [{"check_id": "synthetic_repair_marker", "input": {"enabled": True},
                          "expected": {"recorded": True}, "observed": {"recorded": observed}}],
        "generated_changes_sha256": _canonical_sha256(native),
        "source_snapshot_sha256": inspector.source_snapshot_sha256,
    })
    recovery = replace(request,
        repair_parent_proposal_ref=ArtifactRef(**parent.patch_proposal_artifact),
        repair_parent_review_ref=ArtifactRef(**parent.proposal_review_artifact),
        repair_failure_report_ref={"path": str(report_path), "sha256": _sha(report_path.read_bytes())},
        repair_parent_response_ref={"path": response["path"], "sha256": response["output_sha256"]},
        repair_parent_attempt_ref={"path": str(receipt_path), "sha256": _sha(receipt_path.read_bytes())},
        attempt_id="synthetic-five-binding-recovery",
    )
    return inputs, recovery, parent


def _message_jsons(payload):
    values = []
    for message in payload["messages"]:
        try:
            decoded = json.loads(message.get("content") or "null")
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            values.append((message["role"], decoded))
    return values


class _OfflineOllamaTransport:
    def __init__(self, root, *, scenario="recover"):
        self.root = root
        self.scenario = scenario
        self.requests = []
        self.chat_requests = []
        self.read_bindings = []
        self.generated_texts = []

    def __call__(self, request):
        self.requests.append((request.method, str(request.url)))
        assert request.url.scheme == "http" and request.url.host == "127.0.0.1" and request.url.port == 11434
        model = "qwen3:4b-instruct"
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{
                "name": model, "model": model, "digest": "a" * 64,
                "modified_at": "2026-09-04T00:00:00Z", "size": 1,
                "details": {"context_length": 262144}, "capabilities": ["completion", "tools"],
            }]}, request=request)
        assert request.method == "POST" and request.url.path == "/api/chat", self.requests
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["stream"] is False and payload["think"] is False
        self.chat_requests.append((request.content, payload))
        number = len(self.chat_requests)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"request-{number:03d}.json").write_bytes(request.content)
        decoded = _message_jsons(payload)
        feedback = [value for _, value in decoded if value.get("message_type") == "patch_behavior_correction"]
        tool_results = [value for role, value in decoded if role == "tool"]
        if self.scenario == "direct_pass" and number == 1:
            prompt = next(value for _, value in decoded if "required_change_units" in value)
            prompt = _restore_patch_agent_context(prompt)
            missing = [
                unit["edit_binding_id"]
                for unit in prompt["required_change_units"]
                if "source_excerpt" not in prompt["edit_binding_contexts"][unit["edit_binding_id"]]
            ]
            if missing:
                source_tool = next(
                    tool["function"]
                    for tool in payload["tools"]
                    if tool["function"]["name"] == "inspect_plan_binding_source"
                )
                self.read_bindings.extend(missing)
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": source_tool["name"], "arguments": {"binding_id": binding}},
                } for binding in missing]}
                return httpx.Response(200, json={"model": model, "message": message, "done": True,
                                                "done_reason": "stop", "prompt_eval_count": 200, "eval_count": 40},
                                      request=request)
        if self.scenario != "direct_pass" and number in (1, 3):
            if number == 1:
                assert any(value.get("message_type") == "patch_repair_context" for _, value in decoded)
            else:
                assert feedback and tool_results
                assert feedback[-1]["validator_feedback"]["observations"][0]["check_id"] == "cli_synthetic_repair_marker"
                assert any(message["role"] == "assistant" and message.get("content") == self.generated_texts[0]
                           for message in payload["messages"])
            source_tool = next(tool["function"] for tool in payload["tools"]
                               if tool["function"]["name"] == "inspect_plan_binding_source")
            available = source_tool["parameters"]["properties"]["binding_id"]["enum"]
            if self.scenario == "batch_dedup" and number == 1:
                state = next(value for _, value in reversed(decoded)
                             if value.get("message_type") == "patch_agent_loop_state")
                selected = [value for value in available if value in state["allowed_binding_ids"]]
                assert len(selected) == 5
            else:
                selected = [next(value for value in available if value not in self.read_bindings)]
            self.read_bindings.extend(selected)
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": source_tool["name"], "arguments": {"binding_id": binding}},
            } for binding in selected]}
        else:
            if self.scenario != "direct_pass":
                assert number in (2, 4)
                assert {value["binding_id"] for value in tool_results} == set(self.read_bindings)
                if number == 4:
                    assert feedback and len(tool_results) == (6 if self.scenario == "batch_dedup" else 2)
            else:
                assert number in (1, 2)
            prompt = next(value for _, value in decoded if "required_change_units" in value)
            prompt = _restore_patch_agent_context(prompt)
            contexts = prompt["edit_binding_contexts"]
            missing = [
                unit["edit_binding_id"]
                for unit in prompt["required_change_units"]
                if "source_excerpt" not in contexts[unit["edit_binding_id"]]
            ]
            if missing:
                restored = {
                    item["binding_id"]: item["excerpt"]
                    for item in expand_patch_source_observations(tool_results)
                }
                for binding_id in missing:
                    contexts[binding_id]["source_excerpt"] = restored[binding_id]
            repaired = self.scenario == "direct_pass" or (
                self.scenario in ("recover", "batch_dedup") and number == 4)
            marker = "RecordRepairMarker" if repaired else (
                "RecordTemporaryMarker" if number == 2 else "RecordAnotherMissingMarker")
            changes = [{
                "edit_binding_id": unit["edit_binding_id"], "rationale": "Synthetic CLI transport fixture.",
                "rollback_hint": "Restore the original exact span.",
                "after_text": prompt["edit_binding_contexts"][unit["edit_binding_id"]]["source_excerpt"].replace(
                    "KeepExistingFeedback();", "KeepExistingFeedback();\n        RecordRuntimeMarker();\n        " + marker + "();"),
            } for unit in prompt["required_change_units"]]
            content = json.dumps({"changes": changes})
            self.generated_texts.append(content)
            message = {"role": "assistant", "content": content}
        return httpx.Response(200, json={"model": model, "message": message, "done": True,
                                        "done_reason": "stop", "prompt_eval_count": 200, "eval_count": 40},
                              request=request)


def _install_transport(monkeypatch, root, *, scenario="recover"):
    handler = _OfflineOllamaTransport(root, scenario=scenario)
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: transport.handle_request(request))
    def forbid_socket(*_args, **_kwargs):
        raise AssertionError("No real network connection is allowed in this CLI acceptance.")
    monkeypatch.setattr("socket.socket.connect", forbid_socket)
    monkeypatch.setattr("socket.socket.connect_ex", forbid_socket)
    return handler


def _run(case, capsys):
    code = main(case.argv)
    captured = capsys.readouterr()
    return code, json.loads(captured.out)


def test_public_repair_cli_runs_read_generate_fail_read_revise_and_exact_review(tmp_path, monkeypatch, capsys):
    case = _public_case(tmp_path)
    before = _source_hashes(case.inputs.workspace)
    handler = _install_transport(monkeypatch, tmp_path / "wire")
    code, public = _run(case, capsys)
    assert code == 0, public
    assert public["status"] == "terminal"
    assert public["execution_counts"]["model"] == 4
    assert public["execution_counts"]["review"] == public["execution_counts"]["dry_run"] == 1
    assert public["execution_counts"]["apply"] == public["execution_counts"]["build"] == public["execution_counts"]["runtime"] == 0
    assert public["patch_proposal_artifact"]["artifact_hash"] != case.parent.patch_proposal_artifact["artifact_hash"]
    assert _source_hashes(case.inputs.workspace) == before
    journal = json.loads(Path(public["attempt_receipt"]["path"]).read_text(encoding="utf-8"))
    factory = journal["factory_receipt"]
    assert factory["adapter"] == "LocalOllamaInitialPatchAdapter"
    assert factory["behavior_evaluator_configuration"]["configuration_ref"] == {
        "path": str(case.config_path), "sha256": _sha(case.config_path.read_bytes()),
    }
    assert factory["behavior_evaluator_configuration"]["factory_ref"] == case.config["behavior_evaluator"]["factory_ref"]
    assert factory["behavior_evaluator_configuration"]["kwargs_sha256"] == _canonical_sha256(
        case.config["behavior_evaluator"]["kwargs"])
    assert journal["worker_configuration_ref"] == factory["behavior_evaluator_configuration"]["configuration_ref"]
    assert journal["worker_profile_sha256"] == _canonical_sha256(factory["profile"])
    assert len(handler.chat_requests) == 4 and len(handler.read_bindings) == 2
    assert len(list(case.reports.glob("report-*.json"))) == 2
    assert case.marker.read_text(encoding="utf-8").splitlines() == ["constructed"]
    loop = factory["patch_agent_loop"]
    assert loop["tool_rounds_used"] == loop["behavior_correction"]["execution"]["tool_rounds_used"] == 1
    assert loop["behavior_correction"]["model_call_count"] == 2
    assert [row["status"] for row in loop["behavior_evaluations"]] == ["behavior_failed", "passed"]
    assert all(row["execution_origin"] == "executed" for row in loop["behavior_evaluations"])
    capacity = loop["recovery_capacity_at_failure"]
    assert capacity["can_read_consume_and_revise"] is True
    assert capacity["remaining_model_turns"] == 2 and capacity["remaining_binding_reads"] >= 1
    first_messages = handler.chat_requests[0][1]["messages"]
    final_messages = handler.chat_requests[-1][1]["messages"]
    assert final_messages[:2] == first_messages[:2]
    assert "PreserveFeedback must remain unchanged." in json.dumps(final_messages)
    for turn, (wire, _) in zip(factory["worker_turns"], handler.chat_requests, strict=True):
        assert turn["provider_diagnostic"]["request_identity"]["provider_body_sha256"] == _sha(wire)
        assert turn["response_origin"] == "live_provider_response"


def test_public_repair_cli_projected_input_reads_b0_once_and_cached_attempt_sends_nothing(tmp_path, monkeypatch, capsys):
    case = _public_case(tmp_path)
    handler = _install_transport(monkeypatch, tmp_path / "wire", scenario="direct_pass")
    code, first = _run(case, capsys)
    assert code == 0, first
    assert len(handler.chat_requests) == 2 and len(handler.read_bindings) == 1
    assert len(list(case.reports.glob("report-*.json"))) == 1
    count = len(handler.requests)
    marker = case.marker.read_bytes()
    code, resumed = _run(case, capsys)
    assert code == 0, resumed
    assert len(handler.requests) == count
    assert case.marker.read_bytes() == marker
    assert resumed["patch_proposal_artifact"] == first["patch_proposal_artifact"]


def test_public_repair_cli_persistent_failure_stops_without_apply(tmp_path, monkeypatch, capsys):
    case = _public_case(tmp_path)
    before = _source_hashes(case.inputs.workspace)
    handler = _install_transport(monkeypatch, tmp_path / "wire", scenario="persistent_failure")
    code, public = _run(case, capsys)
    assert code != 0
    assert public["reason"] == "PATCH_AGENT_CORRECTION_BUDGET_EXHAUSTED"
    assert len(handler.chat_requests) == 4
    assert public["patch_proposal_artifact"] is None
    assert public["execution_counts"]["apply"] == 0
    assert public["product_source_writes_performed"] is False
    assert _source_hashes(case.inputs.workspace) == before


@pytest.mark.parametrize("invalid", ["worker_schema", "factory_hash", "repair_field", "repair_hash"])
def test_public_repair_cli_invalid_config_or_refs_stop_before_transport(tmp_path, monkeypatch, capsys, invalid):
    case = _public_case(tmp_path)
    if invalid == "worker_schema":
        del case.config["policy"]
        _write_json(case.config_path, case.config)
    elif invalid == "factory_hash":
        case.config["behavior_evaluator"]["factory_ref"]["sha256"] = "0" * 64
        _write_json(case.config_path, case.config)
    elif invalid == "repair_field":
        del case.repair["repair_parent_attempt_ref"]
        _write_json(case.repair_path, case.repair)
    else:
        case.repair["repair_failure_report_ref"]["sha256"] = "0" * 64
        _write_json(case.repair_path, case.repair)
    handler = _install_transport(monkeypatch, tmp_path / "wire")
    code, public = _run(case, capsys)
    assert code != 0, public
    assert handler.requests == []
    assert not case.marker.exists()
    assert public["execution_counts"]["model"] == 0


def test_public_repair_cli_deduplicates_five_reads_and_preserves_a_sixth_after_failure(tmp_path, monkeypatch, capsys):
    from src.mod_agent.proposal.patch_agent_loop import expand_patch_source_observations

    preparation = _five_binding_repair_case(tmp_path / "case")
    policy = asdict(PatchAgentLoopPolicy(
        max_tool_rounds=6, max_total_model_turns=5, max_observation_bytes=262144,
        context_radius_lines=64, finalize_after_first_observation=False,
        require_source_inspection=False, enable_supporting_reads=True,
        allow_upstream_request=True, allow_batched_reads=True,
        deduplicate_source_excerpts=True,
    ))
    case = _public_case(tmp_path, preparation=preparation, selected_policy=policy)
    context_index = case.argv.index("--ollama-context-window")
    case.argv[context_index + 1] = "262144"
    before = _source_hashes(case.inputs.workspace)
    handler = _install_transport(monkeypatch, tmp_path / "wire", scenario="batch_dedup")
    code, public = _run(case, capsys)
    assert code == 0, public
    assert public["status"] == "terminal"
    assert len(handler.chat_requests) == 4
    assert len(handler.read_bindings) == len(set(handler.read_bindings)) == 6
    shared_source = case.inputs.behavior.read_bytes().decode("utf-8")

    def all_strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from all_strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from all_strings(item)

    for request_index, expected_reads in ((1, 5), (3, 6)):
        decoded = _message_jsons(handler.chat_requests[request_index][1])
        tools = [value for role, value in decoded if role == "tool"]
        assert len(tools) == expected_reads
        assert sum(text == shared_source for _, value in decoded for text in all_strings(value)) == 1
        assert sum("excerpt" in value for value in tools[:5]) == 1
        assert sum("excerpt_ref" in value for value in tools[:5]) == 4
        expanded = expand_patch_source_observations(tools)
        assert all(value["excerpt"] == shared_source for value in expanded[:5])
        assert len({value["observation_sha256"] for value in expanded[:5]}) == 5
        assert len({json.dumps(value["target_span"], sort_keys=True) for value in expanded[:5]}) == 5
        for value in tools[1:5]:
            assert value["excerpt_ref"]["observation_id"] == tools[0]["observation_id"]
            assert value["excerpt_ref"]["excerpt_sha256"] == _sha(shared_source.encode())
    journal = json.loads(Path(public["attempt_receipt"]["path"]).read_text(encoding="utf-8"))
    factory = journal["factory_receipt"]
    loop = factory["patch_agent_loop"]
    correction = loop["behavior_correction"]["execution"]
    assert loop["tool_rounds_used"] == 5
    assert correction["tool_rounds_used"] == 1
    assert len(loop["observation_trace"]) == 5 and len(correction["observation_trace"]) == 1
    assert all(row["admitted_to_model"] is True for row in loop["observation_trace"])
    assert loop["total_observation_bytes"] == sum(row["encoded_payload_bytes"] for row in loop["observation_trace"])
    assert sum(row["expanded_payload_bytes"] for row in loop["observation_trace"]) > 32768
    assert loop["total_observation_bytes"] < 32768
    capacity = loop["recovery_capacity_at_failure"]
    assert capacity["remaining_binding_reads"] == 1
    assert capacity["remaining_model_turns"] >= 2
    assert capacity["remaining_observation_bytes_after_feedback"] > 0
    assert capacity["can_read_consume_and_revise"] is True
    assert [row["status"] for row in loop["behavior_evaluations"]] == ["behavior_failed", "passed"]
    assert all(row["execution_origin"] == "executed" for row in loop["behavior_evaluations"])
    assert public["execution_counts"]["model"] == 4
    assert public["execution_counts"]["review"] == public["execution_counts"]["dry_run"] == 1
    assert public["execution_counts"]["apply"] == 0 and public["product_source_writes_performed"] is False
    assert _source_hashes(case.inputs.workspace) == before
    for turn, (wire, _) in zip(factory["worker_turns"], handler.chat_requests, strict=True):
        assert turn["provider_diagnostic"]["request_identity"]["provider_body_sha256"] == _sha(wire)


def _automatic_binding_case(tmp_path):
    from tests.unit.test_patch_repair_binding import _binding_case

    retained = _binding_case(tmp_path / "retained", different=True)
    case = _public_case(tmp_path, preparation=(retained["inputs"], retained["request"], retained["parent"]))
    selection = {"source_attempt_ref": retained["source_attempt_ref"],
                 "verification_result_ref": retained["verification_result_ref"]}
    selection_path = _write_json(tmp_path / "verification-selection.json", selection)
    index = case.argv.index("--repair-inputs")
    case.argv[index:index + 2] = ["--repair-from-verification", str(selection_path)]
    case.retained = retained
    case.selection = selection
    case.selection_path = selection_path
    return case


def test_public_cli_automatically_binds_a_different_unadmitted_revision_and_preserves_state(tmp_path, monkeypatch, capsys):
    case = _automatic_binding_case(tmp_path)
    before = _source_hashes(case.inputs.workspace)
    handler = _install_transport(monkeypatch, tmp_path / "wire")
    current_ref = case.retained["failed_response_ref"]
    current_raw = Path(current_ref["path"]).read_bytes()
    parent_receipt = json.loads(Path(case.parent.attempt_receipt["path"]).read_text(encoding="utf-8"))
    parent_raw = Path(parent_receipt["factory_receipt"]["model_response"]["path"]).read_bytes()
    assert current_raw != parent_raw
    code, public = _run(case, capsys)
    assert code == 0, public
    assert public["status"] == "terminal"
    assert public["execution_counts"]["model"] == 4
    assert public["execution_counts"]["review"] == public["execution_counts"]["dry_run"] == 1
    assert _source_hashes(case.inputs.workspace) == before
    journal = json.loads(Path(public["attempt_receipt"]["path"]).read_text(encoding="utf-8"))
    repair = journal["repair_context"]
    assert repair["parent_proposal"] == case.parent.patch_proposal_artifact
    assert repair["source_attempt_ref"] == case.selection["source_attempt_ref"]
    assert repair["verification_result_ref"] == case.selection["verification_result_ref"]
    assert repair["failed_response_ref"] == current_ref
    derived_ref = repair["failure_report_ref"]
    derived_raw = Path(derived_ref["path"]).read_bytes()
    assert _sha(derived_raw) == derived_ref["sha256"]
    derived = json.loads(derived_raw)
    first_messages = handler.chat_requests[0][1]["messages"]
    projected_history = json.loads(first_messages[2]["content"])
    assert projected_history["input_only"] is True
    assert all("after_text" not in item and "after_text_ref" in item
               for item in projected_history["changes"])
    assert journal["factory_receipt"]["repair_input_projection"]["raw_response_sha256"] == _sha(current_raw)
    feedback = json.loads(first_messages[3]["content"])["validator_feedback"]
    assert feedback["observations"]
    assert all("observed_state" in item for item in feedback["observations"])
    for original, delivered in zip(case.retained["observations"], feedback["observations"], strict=True):
        assert all(original[key] == delivered[key]
                   for key in ("check_id", "input", "expected", "observed", "observed_state"))
    assert feedback["code_revision"] == derived["code_revision"]
    assert feedback["code_revision"]["same_materialized_code"] is False
    assert feedback["code_revision"]["different_code_is_not_proof_of_repair"] is True
    assert derived["execution_origin"] != "executed"
    prior_http = len(handler.requests)
    prior_factory = case.marker.read_bytes()
    code, cached = _run(case, capsys)
    assert code == 0, cached
    assert cached["patch_proposal_artifact"] == public["patch_proposal_artifact"]
    assert len(handler.requests) == prior_http
    assert case.marker.read_bytes() == prior_factory
    assert Path(derived_ref["path"]).read_bytes() == derived_raw


@pytest.mark.parametrize("failure", ["verification_hash", "wrong_source"])
def test_public_cli_automatic_binding_rejects_changed_or_wrong_sources_before_network(tmp_path, monkeypatch, capsys, failure):
    case = _automatic_binding_case(tmp_path)
    if failure == "verification_hash":
        case.selection["verification_result_ref"] = {**case.selection["verification_result_ref"], "sha256": "0" * 64}
    else:
        parent_path = Path(case.parent.attempt_receipt["path"])
        case.selection["source_attempt_ref"] = {"path": str(parent_path), "sha256": _sha(parent_path.read_bytes())}
    _write_json(case.selection_path, case.selection)
    handler = _install_transport(monkeypatch, tmp_path / "wire")
    code, public = _run(case, capsys)
    assert code != 0, public
    assert public["execution_counts"]["model"] == 0
    assert handler.requests == []
    assert not case.marker.exists()
