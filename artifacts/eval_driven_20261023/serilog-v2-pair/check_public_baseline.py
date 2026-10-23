"""Execute the public baseline once using only the retained local package feed.

为什么要有它：真实模型前确认公开反例能执行并失败，环境准备不消耗求解次数或提供修法。
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.csharp_repository_backend import (  # noqa: E402
    _IGNORED,
    CSharpRepositoryBackend,
    _snapshot,
)
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402

ROOT = Path("C:/tmp/rf-serilog-v2-pair-main")
CACHE = Path("C:/Users/10160/AppData/Local/Temp/rfext1020/serilog-base-v1/nuget-packages")


def main():
    preparation = read_bound_json(_ref(ROOT / "public-preparation.json"))
    if not CACHE.is_dir():
        raise ValueError("RETAINED_PACKAGE_FEED_REQUIRED")
    os.environ.update(RestoreSources=str(CACHE), RestoreFallbackFolders=str(CACHE), NuGetAudit="false",
        MinVerSkip="true", DOTNET_CLI_UI_LANGUAGE="en-US")
    source = ROOT / "baseline-source"
    if source.exists():
        if (ROOT / "baseline-check").exists() or _snapshot(source) != preparation["source_inputs"]:
            raise ValueError("PRIOR_BASELINE_REQUIRES_INSPECTION")
    else:
        shutil.copytree(preparation["source_root"], source, ignore=shutil.ignore_patterns(*_IGNORED))
    (ROOT / "baseline-work").mkdir(exist_ok=True)
    backend = CSharpRepositoryBackend(profile_ref=preparation["profile_ref"], source_root=source,
        workspace_root=ROOT, output_root=ROOT / "baseline-check",
        expected_fixed_inputs=preparation["fixed_inputs"])
    outcome = backend.run_build_test(source, "public-baseline", "development_preflight")
    report = read_bound_json(outcome["report_ref"])
    sources = set()
    for file in source.rglob("project.assets.json"):
        value = json.loads(file.read_bytes())
        sources.update(value.get("project", {}).get("restore", {}).get("sources", {}))
    local_only = bool(sources) and all(Path(path).resolve() == CACHE.resolve() for path in sources)
    result = _save(ROOT / "baseline-result.json", {"report_ref": outcome["report_ref"],
        "status": report["status"], "summary": report["verification_summary"],
        "observed_restore_sources": sorted(sources), "local_feed_only": local_only,
        "model_calls": 0, "scope": "public_baseline_not_model_execution",
        "environment": {key: os.environ[key] for key in ("RestoreSources", "RestoreFallbackFolders", "NuGetAudit", "MinVerSkip")}})
    print(json.dumps({"result_ref": result, "status": report["status"], "summary": report["verification_summary"],
        "local_feed_only": local_only}))
    if not local_only or report["status"] != "behavior_failed":
        raise ValueError("PUBLIC_BASELINE_NOT_READY")


if __name__ == "__main__":
    main()
