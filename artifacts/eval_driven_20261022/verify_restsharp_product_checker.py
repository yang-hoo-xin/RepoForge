"""Confirm current public project feedback through the actual shared backend.

为什么要有它：直接dotnet通过不能替代模型将要调用的产品检查器真实可用。
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryBackend
from src.mod_agent.domain.full_project_compilation import _ref, _save


def main():
    spec = importlib.util.spec_from_file_location("restsharp_run", Path(__file__).with_name("run_restsharp_v2.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    root = Path("C:/tmp/rfext-restsharp-2241-product-checker-v2")
    root.mkdir()
    source = root / "input/RestSharp"
    subprocess.run(["git", "-C", str(run.SOURCE), "worktree", "add", "--detach", str(source), run.BASE],
        check=True, capture_output=True)
    for name in ("restsharp-2241-public-reproduction-v2.patch", "restsharp-2241-public-environment.patch"):
        subprocess.run(["git", "apply", str(Path(__file__).with_name(name))], cwd=source, check=True, capture_output=True)
    allowed = sorted(p.relative_to(run.SOURCE).as_posix() for p in (run.SOURCE / "src/RestSharp").rglob("*.cs"))
    profile = _save(root / "repository-profile.json", run.repository_profile(allowed, _ref(run.DOTNET)))
    backend = CSharpRepositoryBackend(profile_ref=profile, source_root=source, workspace_root=root,
        output_root=root / "checks")
    result = backend.run_build_test(source, "restsharp-public-qualification", "isolated")
    result_ref = _save(root / "result.json", {**result, "build_result": result["build_result"].to_artifact_payload()})
    print(json.dumps({"result_ref": result_ref, "status": result["status"],
        "verification_summary": result.get("verification_summary"), "observations": result.get("observations")}))


if __name__ == "__main__":
    main()
