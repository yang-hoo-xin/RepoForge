"""Navigation projection preserves exact scope without making hints authority."""
import json

import pytest

from src.mod_agent.task_agent_v2_context import history_text
from src.mod_agent.task_agent_v2_model_view import repository_navigation_view
from src.mod_agent.task_agent_v2_state import (
    TaskAgentV2State,
    TaskAgentV2StateError,
    save_v2_artifact,
)
from src.mod_agent.task_agent_v2_tools import prepare_v2_contract
from tests.unit.test_task_agent_v2_state import state as state


def test_small_scope_is_exact_and_catalog_does_not_grant_an_unlisted_file(state):
    before = state.current_revision_ref
    view = repository_navigation_view(state)
    assert view["edit_scope"]["registered_paths"] == ["A.cs", "B.cs"]
    catalog = json.loads(history_text(state, view["edit_scope"]["catalog_ref"]))[0]
    assert catalog["allowed_edit_paths"] == ["A.cs", "B.cs"]
    assert "private.cs" in catalog["source_paths"]
    observed = state.read_source(before, "private.cs")
    with pytest.raises(TaskAgentV2StateError, match="edit_path_denied"):
        state.add_binding(state.registry_ref, observed["source_view_ref"], "private.cs")
    assert state.current_revision_ref == before


def test_large_scope_is_not_repeated_in_initial_request_and_remains_recoverable(tmp_path):
    source = tmp_path / "source"
    directory = source / "src/Library"
    directory.mkdir(parents=True)
    allowed = [f"src/Library/RegisteredFile{number:03d}.cs" for number in range(100)]
    for path in allowed:
        (source / path).write_text("public class Example {}\n", encoding="utf-8")
    protected = "src/Library/ProtectedSibling.cs"
    (source / protected).write_text("public class Protected {}\n", encoding="utf-8")
    store = tmp_path / "artifacts"
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": allowed, "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "scope-projection", source, {}, allowed, authority)
    contract_before = state.load(state.contract_ref)
    message = prepare_v2_contract(state).messages[1].content
    assert "allowed_edit_paths" not in message and "source_paths" not in message
    scope = message["edit_scope"]
    assert scope["registered_file_count"] == 100 and scope["common_directory_hint"] == "src/Library"
    assert "registered_paths" not in scope
    assert "RegisteredFile050.cs" not in json.dumps(message)
    restored = json.loads(history_text(state, scope["catalog_ref"]))[0]
    assert restored["allowed_edit_paths"] == allowed
    assert set(restored["source_paths"]) == set(allowed) | {protected}
    assert state.load(state.contract_ref) == contract_before
    # 为什么要有它：同目录、同扩展名的未授权文件不能借导航摘要获得编辑资格。
    observed = state.read_source(state.current_revision_ref, protected)
    with pytest.raises(TaskAgentV2StateError, match="edit_path_denied"):
        state.add_binding(state.registry_ref, observed["source_view_ref"], protected)
