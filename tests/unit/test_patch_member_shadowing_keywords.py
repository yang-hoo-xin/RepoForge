"""Statement keywords do not turn valid bound-member uses into local declarations."""
import pytest

from src.mod_agent.proposal.code_generator import PatchGenerationError, _validate_bound_member_shadowing


def _validate(after):
    return _validate_bound_member_shadowing(after=after, declared_member_types={"Pending": "Queue<string>"},
        binding_id="edit_binding_001", context={"method_name": "Apply"}, index=1)


@pytest.mark.parametrize("after", [
    "return Pending;", "return Pending.ToArray();", "if (selected) removed++; else Pending.Enqueue(item);",
    "throw Pending;", "Use(out Pending);", "Use(ref Pending);", "new Pending();",
    "#if FEATURE\nUse(Pending);\n#endif\nPending.Enqueue(item);",
])
def test_non_type_statement_keywords_do_not_shadow(after):
    _validate(after)


@pytest.mark.parametrize("after", [
    "var Pending = new Queue<string>();", "Queue<string> Pending = new();",
    "dynamic Pending = source;", "record Pending = source;", "yield Pending = source;", "@return Pending = source;",
    "foreach(var Pending in values) { }", "Read(out var Pending);",
])
def test_real_declarations_including_contextual_type_names_remain_rejected(after):
    with pytest.raises(PatchGenerationError) as raised:
        _validate(after)
    assert raised.value.diagnostics["failure_classification"] == "csharp_bound_member_shadowed"
