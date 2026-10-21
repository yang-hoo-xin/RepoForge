"""Native delivery uses its native provenance gate, not legacy worker fields."""
import json

import pytest

from src.mod_agent.evaluation import isolated_delivery_adapter as adapter


@pytest.mark.parametrize("wrong_factory", [False, True])
def test_native_dispatch_precedes_legacy_response_fields_but_keeps_factory_gate(tmp_path, monkeypatch, wrong_factory):
    worker_ref = {"path": str(tmp_path / "worker.json"), "sha256": "a" * 64}
    factory_ref = {"path": str(tmp_path / "factory.py"), "sha256": "b" * 64}
    snapshot_ref = {"path": str(tmp_path / "snapshot.json"), "sha256": "c" * 64}
    check_ref = {"path": str(tmp_path / "check.json"), "sha256": "d" * 64}
    worker = {"behavior_evaluator": {"factory_ref": factory_ref, "kwargs": {}}}
    snapshot = {"business_argv": ["run-request", "--patch-worker-config", worker_ref["path"]],
        "external_refs": [{"role": "bound_input", **worker_ref}]}
    check = {"status": "verified", "run_id": "original-run", "execution_version_ref": snapshot_ref,
        "external_bindings": [{"role": "extra_python_load", "execution_observed": not wrong_factory, **factory_ref}]}
    monkeypatch.setattr(adapter, "_read_reference", lambda selected, **_: json.dumps(
        snapshot if selected == snapshot_ref else check).encode())
    monkeypatch.setattr(adapter, "_bound_input", lambda *_: worker)
    invoked = []
    monkeypatch.setattr(adapter, "_task_agent_verifier_binding", lambda **kwargs: invoked.append(kwargs) or {"native": True})
    context = {"trial": {"snapshot_root": str(tmp_path), "bound_inputs": [worker_ref], "extra_load_refs": [factory_ref]}}
    initial = {"run_id": "original-run", "task_agent_execution_ref": {"path": "native", "sha256": "e" * 64}}
    manifest = {"execution_version_ref": snapshot_ref, "execution_version_check_ref": check_ref}
    # Native candidates do not claim the old patch-worker generation envelope.
    # 为什么要有它：来源由真实native调用链验证，不能为通过旧字段检查伪造worker模型输出。
    arguments = (context, tmp_path, initial, manifest, {}, {}, {"diff_intent": [{"generation": {}}]})
    if wrong_factory:
        with pytest.raises(ValueError, match="CALLER_BOUND_EXECUTED_FACTORY"):
            adapter._bound_verifier(*arguments)
        assert not invoked
    else:
        assert adapter._bound_verifier(*arguments) == {"native": True}
        assert invoked[0]["worker_ref"] == worker_ref
