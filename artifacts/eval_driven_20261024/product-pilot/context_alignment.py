"""One bounded native local-provider configuration check, not a solver adapter.

为什么要有它：只用现有API核对同权重与服务端上下文，失败不自动扩建或反复重试。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.run_benchmark_arm import codex_argv, codex_events, execute  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402

ROOT = Path("C:/tmp/rf-codex-local-provider-probe2")
INTENT = Path(__file__).with_name("context-alignment-intent.json")
EXECUTABLE = Path("C:/Users/10160/AppData/Local/OpenAI/Codex/bin/37f54d68e5533a51/codex.exe")
EXECUTABLE_SHA = "c81ed464e5e7cafd09a5fa26af390c480bed8e9fab2a87a54b285fbd56e7122a"


def api(name, body=None):
    request = Request("http://127.0.0.1:11434/api/" + name,
        data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def manifest(tag):
    name, version = tag.split(":")
    candidates = [Path.home() / ".ollama/models" / directory / host / "library" / name / version
        for directory, host in (("manifests", "registry.ollama.ai"), ("manifests-v2", "ollama.com"))]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("MODEL_MANIFEST_MISSING_OR_AMBIGUOUS")
    reference = _ref(found[0])
    return reference, read_bound_json(reference)


def main():
    continuation = sys.argv[1:] == ["--continue-created-alias"]
    if ROOT.exists() and not continuation:
        raise ValueError("ALREADY_PREPARED_INSPECT_EXISTING_PROBE")
    intent_ref = _ref(INTENT)
    intent = read_bound_json(intent_ref)
    change = intent["preferred_change"]
    original, alias = change["from"], change["new_tag"]
    tags, loaded = api("tags"), api("ps")
    if loaded.get("models"):
        raise ValueError("SHARED_SERVER_MUST_BE_IDLE_BEFORE_CONFIGURATION_PROBE")
    known = {row["name"]: row for row in tags["models"]}
    if (not continuation and alias in known) or known[original]["digest"] != change["expected_original_digest"]:
        raise ValueError("ALIAS_EXISTS_OR_ORIGINAL_IDENTITY_CHANGED")
    if _ref(EXECUTABLE)["sha256"] != EXECUTABLE_SHA:
        raise ValueError("INSTALLED_CODEX_CHANGED")
    before_ref, before = manifest(original)
    if before_ref["sha256"] != change["expected_original_digest"]:
        raise ValueError("SERVER_MANIFEST_IDENTITY_CHANGED")
    body = {"model": alias, "from": original, "parameters": change["parameters"], "stream": False}
    if continuation:
        previous = read_bound_json(_ref(ROOT / "before.json"))
        created = read_bound_json(_ref(ROOT / "create-response.json"))
        if (previous["intent_ref"] != intent_ref or previous["original_manifest_ref"] != before_ref
                or created["response"].get("status") != "success"
                or read_bound_json(created["request_ref"]) != body or (ROOT / "plan.json").exists()
                or (ROOT / "execution").exists() or (ROOT / "continuation.json").exists()):
            raise ValueError("EXACT_UNDISPATCHED_CONFIGURATION_CONTINUATION_REQUIRED")
        retained = _ref(Path(__file__).with_name("context_alignment-v1.py"))
        if retained["sha256"] != previous["controller_ref"]["sha256"]:
            raise ValueError("FIRST_CONTROLLER_BYTES_MISSING")
        _save(ROOT / "continuation.json", {"prior_controller_bytes_ref": retained,
            "current_controller_ref": _ref(__file__), "prior_model_generations": 0,
            "correction": "Ollama0.40 writes new tags to manifests-v2/ollama.com; legacy original remains in manifests/registry.ollama.ai. Reuse successful creation, never repeat it.",
            "why": "为什么要有它：配置读取错误不能抹去已发生的创建，也不能冒充模型失败或重开一次生成。"})
    else:
        ROOT.mkdir()
        _save(ROOT / "before.json", {"intent_ref": intent_ref, "controller_ref": _ref(__file__),
            "tags": tags, "loaded": loaded, "original_manifest_ref": before_ref,
            "original_show": api("show", {"model": original}), "model_generations": 0})
        request_ref = _save(ROOT / "create-request.json", body)
        response = api("create", body)
        _save(ROOT / "create-response.json", {"request_ref": request_ref, "response": response})
    after_ref, after = manifest(original)
    alias_ref, alias_manifest = manifest(alias)
    def content_layers(document):
        return [{key: row[key] for key in ("mediaType", "digest", "size")} for row in document["layers"]
            if row["mediaType"] in {"application/vnd.ollama.image.model", "application/vnd.ollama.image.license"}]
    if after_ref != before_ref or after != before or content_layers(alias_manifest) != content_layers(before):
        raise ValueError("INHERITED_CONTENT_OR_ORIGINAL_CHANGED")
    after_tags = api("tags")
    alias_identity = next(row for row in after_tags["models"] if row["name"] == alias)
    show = api("show", {"model": alias})
    original_show = read_bound_json(_ref(ROOT / "before.json"))["original_show"]
    if any(show.get(key) != original_show.get(key) for key in ("template", "model_info", "capabilities")):
        raise ValueError("INHERITED_MODEL_BEHAVIOR_METADATA_CHANGED")
    _save(ROOT / "alias-verification.json", {"original_manifest_ref": after_ref,
        "alias_manifest_ref": alias_ref, "alias_identity": alias_identity, "show": show,
        "weight_and_license_layers_identical": True, "template_model_info_capabilities_identical": True,
        "container_config_byte_identical": alias_manifest["config"] == before["config"],
        "original_unchanged": True, "new_weight_downloads": 0})
    workspace = ROOT / "workspace"
    workspace.mkdir()
    marker = "context-alignment-" + uuid4().hex
    marker_path = workspace / "marker.txt"
    with marker_path.open("x", encoding="utf-8") as stream:
        stream.write(marker + "\n")
    prompt_path = ROOT / "prompt.txt"
    with prompt_path.open("x", encoding="utf-8") as stream:
        stream.write("Read marker.txt in this workspace using one actual file-read command. Return only its contents. "
            "Do not modify files, inspect any other project or use network. This checks local configuration only, not a development task. "
            "Why: confirm actual tool execution without exposing business answers.")
    command = codex_argv(EXECUTABLE, workspace, ROOT / "last-message.txt",
        [REPO, Path.home() / ".codex", Path.home() / ".ssh", Path.home() / ".aws", Path("C:/tmp/rfext-spectre-1531-task"),
         Path("C:/tmp/rfext-spectre-1531-evaluation")])
    command[command.index("--model") + 1] = alias
    index = command.index('model_reasoning_effort="medium"')
    del command[index - 1:index + 1]
    index = next(i for i, value in enumerate(command) if value.startswith("permissions.benchmark.filesystem="))
    command[index] = command[index].replace('":workspace_roots"="write"', '":workspace_roots"="read"')
    command[-1:-1] = ["--oss", "--local-provider", "ollama", "--disable", "unbounded_connection_retries",
                      "-c", "model_context_window=65536"]
    plan_ref = _save(ROOT / "plan.json", {"intent_ref": intent_ref, "argv": command,
        "marker_ref": _ref(marker_path), "prompt_ref": _ref(prompt_path), "executable_ref": _ref(EXECUTABLE),
        "alias_verification_ref": _ref(ROOT / "alias-verification.json"), "wall_seconds": 180})
    process = execute(command, cwd=workspace, stdin_path=prompt_path, output=ROOT / "execution",
        deadline=time.monotonic() + 180, environment={**os.environ, "PYTHONUTF8": "1"})
    loaded_after = api("ps")
    events = codex_events(process["stdout_ref"]["path"])
    final = (ROOT / "last-message.txt").read_text(encoding="utf-8") if (ROOT / "last-message.txt").is_file() else None
    matching = [row for row in loaded_after.get("models", []) if row["name"] == alias]
    result = {"plan_ref": plan_ref, "process_ref": _ref(ROOT / "execution/process.json"),
        "events_ref": _save(ROOT / "native-events.json", events), "loaded_after": loaded_after,
        "marker_unchanged": _ref(marker_path) == read_bound_json(plan_ref)["marker_ref"],
        "marker_final_text_matches_ignoring_terminal_newline": final is not None and final.rstrip("\r\n") == marker,
        "observed_context65536": len(matching) == 1 and matching[0].get("context_length") == 65536,
        "native_usage": events["usage_totals"], "task_success": None,
        "limits": "not business task or product superiority; default sampling parameters are not actual-wire override evidence"}
    reference = _save(ROOT / "result.json", result)
    print(json.dumps({"result_ref": reference, "process_status": process["status"],
        "context65536": result["observed_context65536"],
        "marker_matches": result["marker_final_text_matches_ignoring_terminal_newline"]}))


if __name__ == "__main__":
    main()
