"""Current-version defaults and focused work view through the local normal CLI."""
from copy import deepcopy

from tests.unit.test_continuous_repair_product_entry import (
    test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs as _normal_repair,
)
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def test_local_normal_work_view_repairs_without_repeating_host_identifiers(repository_workspace, monkeypatch, capsys):
    observed = []

    def omit_current_identifiers(*, name, arguments, current, request, **_kwargs):
        original = deepcopy(arguments)
        changed = {key: value for key, value in arguments.items()
            if key not in {"parent_revision_ref", "candidate_ref", "check_ref", "expected_registry_ref", "working_state"}}
        observed.append((name, original, deepcopy(changed)))
        assert current["current_revision_ref"]
        assert all("working_state" not in row["function"]["parameters"].get("required", []) for row in request["tools"])
        assert current["working_state"]["model_statement_projection"]["semantic_summary_available"] is False
        return changed

    _normal_repair(repository_workspace, monkeypatch, capsys, semantic_observer=omit_current_identifiers)
    assert len(observed) == 10 and observed[-1][0] == "finish" and observed[-1][2] == {}
    assert all(original != changed for _name, original, changed in observed)
