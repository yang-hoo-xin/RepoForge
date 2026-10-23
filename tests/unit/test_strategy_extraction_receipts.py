"""Rejected extraction responses remain attributable without another provider call."""
from dataclasses import replace

import pytest

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from tests.unit.test_strategy_experience import Provider, extraction_inputs
from tests.unit.test_strategy_experience import setup as experience_setup


@pytest.mark.parametrize("failure", ["json", "fields", "length", "accounting"])
def test_completed_response_failure_is_not_left_as_dispatch_started(tmp_path, failure):
    setup = experience_setup.__wrapped__(tmp_path)
    memory, _, strategy, _ = setup
    episode, admission = extraction_inputs(tmp_path, setup)
    provider = Provider(strategy)
    complete = provider.complete_turn

    def respond(request):
        response = complete(request)
        if failure == "json":
            return replace(response, content="not JSON")
        if failure == "fields":
            return replace(response, content="{}")
        if failure == "length":
            return replace(response, stop_reason="length")
        return response

    provider.complete_turn = respond
    receipts = []

    def account(receipt):
        receipts.append(receipt)
        if failure == "accounting":
            raise OSError("controlled ledger unavailable")

    with pytest.raises((ValueError, OSError)):
        memory.propose_from_episode(provider, episode, admission_ref=admission,
            before_call=lambda ref: True, after_call=account)
    repeated = memory.propose_from_episode(provider, episode, admission_ref=admission,
        before_call=lambda ref: pytest.fail("must not reserve again"),
        after_call=lambda receipt: pytest.fail("must not account again"))
    assert repeated["status"] == ("accounting_failed" if failure == "accounting" else "response_rejected")
    assert repeated["response_received"] is True
    assert repeated["model_calls"] == 1
    assert repeated["usage"] == receipts[0]["usage"]
    assert repeated["strategy_ref"] is None
    assert len(provider.requests) == len(receipts) == 1
    response = memory.read(repeated["response_ref"])
    assert response["request_ref"] == repeated["request_ref"]
    if failure in {"json", "fields"}:
        assert response["response"]["content"] == ("not JSON" if failure == "json" else "{}")
    elif failure == "length":
        assert response["response"]["stop_reason"] == "length"
    else:
        assert repeated["accounting_status"] == "unconfirmed"
    assert memory.current_strategy("episode:" + canonical_hash(read_bound_json(episode))) is None
    with pytest.raises(ValueError, match="MODEL_PROVENANCE_MISMATCH"):
        memory.record_strategy("must-not-promote-rejected-extraction", strategy, evidence_refs=setup[1],
            origin={"kind": "model", "extraction_ref": repeated["extraction_ref"]})
