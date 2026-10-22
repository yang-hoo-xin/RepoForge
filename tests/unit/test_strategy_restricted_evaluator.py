"""Production checker and adapter over controlled Docker transport originals."""
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.evaluation import restricted_csharp as runtime
from src.mod_agent.evaluation import restricted_owned_judge as judge
from src.mod_agent.evaluation.strategy_adoption import decide_strategy_adoption
from src.mod_agent.evaluation.strategy_restricted_evaluator import (
    bind_restricted_strategy_evidence,
    generate_restricted_strategy_verdict,
    qualify_restricted_strategy_evaluator,
    validate_restricted_strategy_verdict,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
from tests.unit.test_restricted_csharp import _inspect_document
from tests.unit.test_restricted_owned_judge import _fixture


def controlled_docker(tmp_path, monkeypatch):
    docker = tmp_path / "controlled-docker.exe"
    docker.write_bytes(b"CONTROLLED TRANSPORT; NOT AN EXECUTABLE")
    monkeypatch.setattr(runtime, "_docker", lambda: [str(docker)])
    containers = {}
    def command(root, name, argv):
        if name == "create":
            container = (uuid4().hex * 2)
            entrypoint = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--entrypoint="))
            token = next(arg.split("=", 2)[2] for arg in argv if arg.startswith("--label="))
            mounts = {}
            for index, arg in enumerate(argv):
                if arg == "--mount":
                    entries = argv[index + 1].split(",")
                    values = dict(item.split("=", 1) for item in entries if "=" in item)
                    mounts[values["target"]] = (Path(values["source"]), "readonly" in entries)
            doc, _ = _inspect_document(tmp_path)
            doc["Config"].update(Entrypoint=[entrypoint], Cmd=argv[argv.index(runtime.IMAGE) + 1:])
            doc["Config"]["Labels"] = {"codex.restricted-csharp": token}
            if entrypoint == "/bin/sh":
                doc["HostConfig"]["Tmpfs"].pop("/operation-output")
            doc["Mounts"] = [{"Type": "bind", "Destination": dest, "Source": str(path), "RW": not readonly}
                for dest, (path, readonly) in mounts.items()]
            doc.update(State={"Running": False, "ExitCode": 0}, Image="sha256:" + "a" * 64)
            containers[container] = {"doc": doc, "mounts": mounts}
            stdout = container
        elif name in {"inspect-before", "inspect-after"}:
            stdout = json.dumps([containers[argv[-1]]["doc"]])
        else:
            stdout = "removed"
        _save(root / (name + ".json"), {"argv": argv, "exit_code": 0, "stdout": stdout, "stderr": "", "elapsed_seconds": 0})
        return subprocess.CompletedProcess(argv, 0, stdout.encode(), b"")
    def attach(argv, *, stdin, timeout, stop):
        item = containers[argv[-1]]
        if "/inputs" in item["mounts"]:
            source = (item["mounts"]["/inputs"][0] / "generated_mods/Lease/Subject.cs").read_text()
            actual = 1 if "return 1;" in source or "return 1 + 0;" in source else 2
            (item["mounts"]["/output"][0] / "Subject.dll").write_bytes(str(actual).encode())
            stdout = "controlled compiler completed"
        else:
            actual = int((item["mounts"]["/assembly"][0] / "Subject.dll").read_text())
            stdout = json.dumps({"value": actual})
        return {"argv": argv, "exit_code": 0, "timed_out": False, "output_limit_exceeded": False,
            "local_process_reaped": True, "local_cleanup_errors": [], "elapsed_seconds": 0,
            "stdout": stdout, "stderr": "", "stdout_sha256": runtime._hash(stdout.encode()),
            "stderr_sha256": runtime._hash(b""), "output_hash_scope": "complete stream"}
    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime, "_attach", attach)


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    controlled_docker(tmp_path, monkeypatch)
    cases = {"schema_version": judge.CASE_BUNDLE_SCHEMA, "task_id": "unseen-1", "eligibility": "independent_strategy_acceptance",
        "cases": [{"case_id": "controlled-case", "operation": {}, "assertions": [{"path": ["value"], "equals": 1}]}]}
    kwargs, base, source = _fixture(tmp_path, cases=cases)
    kwargs["case_bundle_ref"] = kwargs.pop("cases_ref")
    driver = tmp_path / "driver.cs"
    driver.write_text("using System; using System.Text.Json; Console.WriteLine(JsonSerializer.Serialize(new { value = new ExportPipeline().Apply() }));")
    kwargs.update(public_worker_source_ref=runtime._ref(driver), task_id="unseen-1")
    evaluator = judge.RestrictedCSharpEvaluator(**kwargs)
    controls, contexts = {}, {}
    for role, expression in (("correct", "1"), ("equivalent", "1 + 0"), ("wrong", "2")):
        context = deepcopy(base)
        context["generated_changes"] = deepcopy(base["generated_changes"])
        context["generated_changes"][0]["patch"]["after_text"] = context["generated_changes"][0]["patch"]["after_text"].replace("return 1;", "return " + expression + ";")
        response = _save(tmp_path / (role + "-response.json"), {"changes": [{"edit_binding_id": context["generated_changes"][0]["edit_binding_id"],
            "after_text": context["generated_changes"][0]["patch"]["after_text"]}]})
        context["model_response"] = {"path": response["path"], "output_sha256": response["sha256"],
            "output_utf8_bytes": Path(response["path"]).stat().st_size}
        context["generated_changes_sha256"] = canonical_hash(context["generated_changes"])
        from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
        materialized = materialize_exact_text_patch(workspace_root=kwargs["workspace_root"],
            implementation_plan=context["implementation_plan"], diff_intent=context["generated_changes"])
        context["materialized_code_identity"] = {"baseline_sha256": materialized.baseline_sha256,
            "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}
        reference = _save(tmp_path / (role + "-context.json"), context)
        report = evaluator(context)
        token = Path(report["report_ref"]["path"]).parent.name
        private = runtime._ref(Path(kwargs["private_output_root"]) / token / "judge-report.json")
        controls[role] = bind_restricted_strategy_evidence(report_ref=report["report_ref"], context_ref=reference,
            private_report_ref=private, output_path=tmp_path / (role + "-bundle.json"))
        contexts[role] = (context, evaluator._materialize(context))
    evaluator_ref = qualify_restricted_strategy_evaluator(kwargs=kwargs, controls=controls,
        source_path_map={"Subject.cs": "generated_mods/Lease/Subject.cs"}, output_root=tmp_path / "qualified")
    store = tmp_path / "v2"
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": ["Subject.cs"], "read_roots": {"target": str(source.parent)}})
    state = TaskAgentV2State.create(store, "unseen-1", source.parent, allowed_edit_paths=["Subject.cs"], authority_ref=authority)
    candidates = {}
    for arm, role in (("B", "wrong"), ("B+", "correct")):
        view = state.read_source(state.current_revision_ref, "Subject.cs")
        binding = state.add_binding(state.registry_ref, view["source_view_ref"], "Subject.cs")
        selected = state.submit(state.current_revision_ref, [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"],
            "before_text": view["text"], "after_text": contexts[role][1]["generated_mods/Lease/Subject.cs"].decode()}])
        candidates[arm] = {"candidate_source_id": selected["candidate_source_id"], "candidate_state": {
            "artifact_store_root": str(store), "task_contract_ref": state.contract_ref, "candidate_ref": selected["candidate_ref"]},
            "bundle_ref": controls[role]}
    return {"evaluator_ref": evaluator_ref, "candidates": candidates, "controls": controls}


def test_qualified_production_checker_originals_allow_policy_transition(tmp_path, qualified):
    from tests.unit.test_strategy_adoption import pair
    from tests.unit.test_strategy_experience import setup
    fixture = setup.__wrapped__(tmp_path / "memory")
    memory, strategy, _, args = pair(tmp_path / "pair", fixture, minimum_pairs=1, qualified_evaluator=qualified)
    decision_ref = decide_strategy_adoption(memory, strategy, **args)
    decision = memory.read(decision_ref)
    assert decision["target_state"] == "active"
    assert decision["claim"] == "verified_restricted_csharp_paired_evidence"
    assert all(row["status"] == "verified_restricted_csharp" for row in decision["results"][0]["evaluator_evidence"].values())
    assert memory.read(memory.transition(strategy, decision_ref))["state"] == "active"


def test_wrong_candidate_version_and_unexecuted_original_reject(tmp_path, qualified):
    row = qualified["candidates"]["B+"]
    execution = _save(tmp_path / "execution.json", {"task_id": "unseen-1", **row})
    verdict_ref = generate_restricted_strategy_verdict(evaluator_ref=qualified["evaluator_ref"], execution_ref=execution,
        evidence_bundle_ref=row["bundle_ref"], output_path=tmp_path / "verdict.json")
    verdict = read_bound_json(verdict_ref)
    assert "assertions" not in json.dumps(verdict) and "observations" not in json.dumps(verdict)
    with pytest.raises(ValueError, match="CANDIDATE_VERSION_CHANGED"):
        validate_restricted_strategy_verdict({**verdict, "candidate_source_id": "different"})
    with pytest.raises(ValueError, match="VERDICT_CANDIDATE_CHANGED"):
        generate_restricted_strategy_verdict(evaluator_ref=qualified["evaluator_ref"], execution_ref=execution,
            evidence_bundle_ref=qualified["controls"]["wrong"], output_path=tmp_path / "wrong-code-verdict.json")
    bundle = read_bound_json(row["bundle_ref"])
    compilation = read_bound_json(bundle["compilation_receipt_ref"])
    process = read_bound_json(compilation["execution_ref"])
    execution_ref = _save(Path(compilation["execution_ref"]["path"]).with_name("unexecuted.json"), {**process, "completed": False})
    altered_compilation = _save(Path(bundle["compilation_receipt_ref"]["path"]).with_name("unexecuted-compilation.json"),
        {**compilation, "execution_ref": execution_ref})
    records = {**bundle["process_records"], execution_ref["sha256"]: bundle["process_records"][compilation["execution_ref"]["sha256"]]}
    altered_bundle = _save(tmp_path / "unexecuted-bundle.json", {**bundle,
        "compilation_receipt_ref": altered_compilation, "process_records": records})
    with pytest.raises(ValueError, match="PROCESS_NOT_COMPLETED"):
        validate_restricted_strategy_verdict({**verdict, "evaluator_result_ref": altered_bundle})


def test_missing_original_for_supported_evaluator_stays_trial(tmp_path, qualified):
    from tests.unit.test_strategy_adoption import pair
    from tests.unit.test_strategy_experience import setup
    memory, strategy, _, args = pair(tmp_path / "pair", setup.__wrapped__(tmp_path / "memory"),
        minimum_pairs=1, qualified_evaluator=qualified)
    result = read_bound_json(args["result_refs"][0])
    verdict = read_bound_json(result["arms"]["B+"]["evaluation_ref"])
    verdict.pop("evaluator_result_ref")
    result["arms"]["B+"]["evaluation_ref"] = _save(tmp_path / "missing-verdict.json", verdict)
    args["result_refs"] = [_save(tmp_path / "missing-result.json", result)]
    decision = memory.read(decide_strategy_adoption(memory, strategy, **args))
    assert decision["target_state"] == "trial"
    assert decision["results"][0]["evaluator_evidence"]["B+"]["status"] == "missing_original_evaluator_evidence"
