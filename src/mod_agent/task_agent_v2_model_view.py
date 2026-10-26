"""Coding navigation summaries, separate from the authoritative task contract."""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from posixpath import commonpath


def repository_navigation_view(state):
    """Keep exact membership available without sending every path on every turn.

    为什么要有它：目录概览用于定位，不能扩张注册文件授权；完整清单继续保存并由现有history_read按需读取。
    """
    contract = state.contract
    paths = sorted(state.load(contract["baseline_manifest_ref"])["files"])
    allowed = list(contract["allowed_edit_paths"])
    catalog = state.save("context-archive", {
        "schema": "task-agent-context-archive.v2", "task_contract_ref": state.contract_ref,
        "exchanges": [{"origin": "execution_host", "kind": "repository_navigation_catalog",
            "source_paths": paths, "allowed_edit_paths": allowed,
            "scope": "Exact registered membership; reading this catalog grants no additional authority."}],
    })
    top_level = Counter(PurePosixPath(path).parts[0] for path in paths)
    parents = [str(PurePosixPath(path).parent) for path in allowed]
    scope = {"operation": "modify_existing", "registered_file_count": len(allowed),
        "common_directory_hint": commonpath(parents) if parents else None,
        "registered_extensions": sorted({PurePosixPath(path).suffix for path in allowed}),
        "catalog_ref": catalog,
        "meaning": "Directory and extension summaries are navigation hints, not wildcard write permission. "
            "The host checks exact registered paths and source preimages. Use history_read on catalog_ref if exact membership is needed."}
    # 为什么要有它：短的精确清单直接可读；大清单转为同一可追溯目录，不把固定字节上限强加给整个历史。
    if len(allowed) <= 16:
        scope["registered_paths"] = allowed
    return {"edit_scope": scope, "source_inventory": {
        "file_count": len(paths), "top_level_entries": [
            {"name": name, "file_count": count} for name, count in sorted(top_level.items())],
        "catalog_ref": catalog, "scope": "Navigation inventory; exact original paths are available through history_read."}}
