"""Only the bounded argument-feedback helper's worker-version coverage."""
from src.mod_agent import model_qualification as qualification


def test_only_argument_feedback_helper_change_invalidates_worker_identity(tmp_path, monkeypatch):
    helper = "proposal/tool_argument_feedback.py"
    # Scope: isolated byte fixtures, no actual worker/provider execution.
    # 为什么要有它：本测试只证明新helper的单文件漂移会使旧执行身份失效，不审计其他模块覆盖范围。
    for name in set(qualification._WORKER_FILES) | {helper}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# unchanged identity fixture\n")
    monkeypatch.setattr(qualification, "__file__", str(tmp_path / "model_qualification.py"))
    before = qualification.worker_implementation_identity()
    (tmp_path / helper).write_bytes(b"# changed argument feedback fixture\n")
    after = qualification.worker_implementation_identity()
    assert before["implementation_sha256"] != after["implementation_sha256"]
    assert {name for name in before["files"] if before["files"][name] != after["files"][name]} == {helper}
