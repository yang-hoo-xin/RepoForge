import json

import pytest

from src.observability.patch_langfuse import PatchLangfuseError
from src.observability.patch_langfuse_query import CloudExecutionReader, summarize_execution


TRACE = "a" * 32


def observation(identity, kind="SPAN", name="patch.execution", **changes):
    return {"id": identity, "traceId": TRACE, "type": kind, "name": name,
            "parentObservationId": None, "isRootObservation": False,
            "startTime": "2026-09-09T01:00:00Z", "endTime": "2026-09-09T01:00:02Z",
            "level": "DEFAULT", "metadata": {}, **changes}


def generation(identity="g", **changes):
    return observation(identity, "GENERATION", "patch.generate", **changes)


def test_queries_join_actual_consumption_and_output_without_counting_parent_twice():
    rows = [observation("root", isRootObservation=True, metadata={"workflow_status": "partial"}),
        observation("request", name="model.request", parentObservationId="root"),
        observation("read", "TOOL", "source.inspect", parentObservationId="root",
                    metadata={"call_id_sha256": "c" * 64, "encoded_payload_sha256": "d" * 64}),
        generation(parentObservationId="request", providedModelName="model-1",
            metadata={"output_sha256": "e" * 64, "usage_available": True,
                "usage_mapping_status": "inclusive_provider_counts_to_exclusive_buckets",
                "consumed_tool_results": [{"call_id_sha256": "c" * 64, "observation_sha256": "d" * 64}]},
            usageDetails={"input": 8, "input_cached_tokens": 2, "output": 3, "output_reasoning_tokens": 1, "total": 14}),
        observation("verify", name="patch.verify", parentObservationId="root",
            metadata={"verification_status": "behavior_failed", "output_sha256": "e" * 64})]
    result = summarize_execution(TRACE, rows)
    assert result["generation_count"] == result["model_request_observations"] == 1
    assert result["generations"][0]["duration_ms"] == 2000
    assert result["generations"][0]["ttft_ms"] is None
    assert result["generations"][0]["cost"] is None
    assert result["usage"]["measured_totals"] == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    assert result["tool_consumption_links"][0]["source_observation_ids"] == ["read"]
    assert result["verification"][0]["matching_generation_ids"] == ["g"]
    assert result["verification"][0]["level"] == "DEFAULT"
    assert result["verification"][0]["facts"]["verification_status"] == "behavior_failed"
    assert result["scores"] == [] and result["missing_parent_observation_ids"] == []


def test_mismatched_tool_bytes_are_not_consumed_and_missing_usage_is_not_zero():
    result = summarize_execution(TRACE, [
        observation("read", "TOOL", "source.inspect", metadata={"call_id_sha256": "c", "encoded_payload_sha256": "old"}),
        generation(metadata={"consumed_tool_results": [{"call_id_sha256": "c", "observation_sha256": "new"}]}),
        observation("verify", name="patch.verify", metadata={"output_sha256": "different"})])
    assert result["tool_consumption_links"][0]["binding"] == "unresolved"
    assert result["verification"][0]["binding"] == "unresolved"
    assert result["usage"] == {"measured_generations": 0, "missing_generations": 1, "measured_totals": None}


@pytest.mark.parametrize("change", [{"endTime": None}, {"endTime": "2026-09-08T01:00:00Z"},
                                  {"startTime": "2026-09-09T01:00:00"}])
def test_missing_or_invalid_timing_is_unknown(change):
    assert summarize_execution(TRACE, [generation(**change)])["generations"][0]["duration_ms"] is None


def test_duplicate_identity_is_deduped_but_conflicts_and_other_traces_are_rejected():
    row = generation()
    assert summarize_execution(TRACE, [row, row])["generation_count"] == 1
    with pytest.raises(PatchLangfuseError, match="CONFLICTING_OBSERVATION"):
        summarize_execution(TRACE, [row, {**row, "level": "ERROR"}])
    with pytest.raises(PatchLangfuseError, match="IDENTITY_MISMATCH"):
        summarize_execution(TRACE, [{**row, "traceId": "b" * 32}])


def test_score_requires_its_actual_trace_binding():
    score = {"id": "s", "name": "test_fraction", "dataType": "NUMERIC", "value": 0.5,
             "subject": {"kind": "observation", "id": "v", "traceId": TRACE}}
    assert summarize_execution(TRACE, [], [score])["scores"] == [score]
    with pytest.raises(PatchLangfuseError, match="SCORE_IDENTITY_MISMATCH"):
        summarize_execution(TRACE, [], [{**score, "subject": {"kind": "trace", "id": "other"}}])


class Transport:
    def __init__(self):
        self.requests = []
        self.repeat_cursor = False
        self.required_new_field = False

    def _request(self, method, path, *, payload):
        self.requests.append((method, path, payload))
        assert method == "POST" and path == "/api/public/mcp"
        if payload["method"] == "tools/list":
            tools = [{"name": name, "inputSchema": {"type": "object", "required":
                     ["future_required_field"] if self.required_new_field and name == "listObservations" else []}}
                     for name in ("getObservationFieldSchema", "listObservations", "listScores")]
            return {"result": {"tools": tools}}
        name, args = payload["params"]["name"], payload["params"]["arguments"]
        if name == "getObservationFieldSchema":
            fields = ("id traceId parentObservationId isRootObservation type name startTime endTime level statusMessage "
                      "providedModelName usageDetails totalCost completionStartTime promptName metadata sessionId").split()
            result = {"fields": {f: {} for f in fields}}
        elif name == "listObservations":
            assert args["traceId"] == TRACE and args["limit"] == 100
            assert not {"input", "output", "*"}.intersection(args["fields"])
            assert set(args["expandMetadataKeys"]) == {"artifact_refs", "consumed_tool_results", "tool_call_id_sha256s",
                "consumed_materials", "produced_materials", "selected_evidence_id_sha256s", "relationship_counts",
                "decision_summary", "local_detail_refs", "terminal_record_ref", "symbol_declaration_checks",
                "requirement_findings_attempt_audit", "supplemental_findings_replay", "terminal_model_call_budget"}
            result = {"data": [generation()], "meta": {"cursor": "next"} if not args.get("cursor") or self.repeat_cursor else {}}
        elif name == "listScores":
            assert args["traceId"] == [TRACE]
            result = {"data": [], "meta": {}}
        else:
            raise AssertionError("No mutation or model tool is permitted")
        return {"result": {"content": [{"type": "text", "text": json.dumps(result)}]}}


def test_cloud_read_discovers_schemas_pages_expands_metadata_and_excludes_bodies():
    transport = Transport()
    result = CloudExecutionReader(transport).read(TRACE)
    assert result["generation_count"] == 1  # Overlapping pages do not duplicate a generation.
    assert sum(r[2]["method"] == "tools/list" for r in transport.requests) == 1
    assert len(transport.requests) == 5


def test_changed_server_schema_stops_before_observation_query():
    transport = Transport()
    transport.required_new_field = True
    with pytest.raises(PatchLangfuseError, match="SCHEMA_CHANGED"):
        CloudExecutionReader(transport).read(TRACE)
    assert len(transport.requests) == 2


def test_pagination_loop_is_bounded_and_never_reports_partial_as_complete():
    transport = Transport()
    transport.repeat_cursor = True
    with pytest.raises(PatchLangfuseError, match="CURSOR_INVALID"):
        CloudExecutionReader(transport).read(TRACE)
    assert len(transport.requests) == 4


def test_existing_http_transport_declares_mcp_response_types(monkeypatch):
    import httpx
    from src.observability.patch_langfuse import LangfuseHttpTransport

    original = httpx.Client
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200 if request.headers.get("accept") == "application/json, text/event-stream" else 406,
                              json={"result": {"tools": []}})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    transport = LangfuseHttpTransport("https://example.com", "test-public", "test-secret")
    assert transport._request("POST", "/api/public/mcp", payload={"method": "tools/list"}) == {"result": {"tools": []}}
    assert len(requests) == 1


def test_explicit_observation_time_window_survives_every_page():
    class CurrentTransport(Transport):
        def _request(self, method, path, *, payload):
            result = super()._request(method, path, payload=payload)
            if payload["method"] == "tools/list":
                for tool in result["result"]["tools"]:
                    if tool["name"] == "listObservations":
                        tool["inputSchema"]["required"] = ["fromStartTime", "toStartTime"]
            return result

    transport = CurrentTransport()
    window = {"fromStartTime": "2026-09-09T00:59:59Z", "toStartTime": "2026-09-09T01:00:03Z"}
    result = CloudExecutionReader(transport).read(TRACE,
        from_start_time=window["fromStartTime"], to_start_time=window["toStartTime"])
    queries = [payload["params"]["arguments"] for _, _, payload in transport.requests
               if payload.get("params", {}).get("name") == "listObservations"]
    assert len(queries) == 2
    assert all(all(query[key] == value for key, value in window.items()) for query in queries)
    assert result["query_window"] == window


@pytest.mark.parametrize("start,end", [
    ("2026-09-09T00:00:00Z", None),
    ("2026-09-09T00:00:00", "2026-09-10T00:00:00Z"),
    ("2026-09-10T00:00:00Z", "2026-09-09T00:00:00Z"),
    ("2026-09-09T00:00:00Z", "2026-09-24T00:00:00Z"),
])
def test_invalid_time_window_stops_before_network(start, end):
    transport = Transport()
    with pytest.raises(PatchLangfuseError, match="TIME_RANGE_INVALID"):
        CloudExecutionReader(transport).read(TRACE, from_start_time=start, to_start_time=end)
    assert transport.requests == []


@pytest.mark.parametrize("window", [[], ["--from-start-time", "2026-09-09T00:59:59Z"]])
def test_public_cli_requires_time_bounds_before_loading_credentials(monkeypatch, capsys, window):
    from src.observability import patch_langfuse_query as query

    monkeypatch.setattr(query.LangfuseHttpTransport, "from_environment",
                        lambda: pytest.fail("Missing arguments must not initialize a cloud connection"))
    with pytest.raises(SystemExit) as error:
        query.main(["--trace-id", TRACE, *window])
    assert error.value.code == 2
    assert "--to-start-time" in capsys.readouterr().err


def test_public_cli_forwards_time_window_to_shared_reader(monkeypatch, capsys):
    from src.observability import patch_langfuse_query as query

    transport = Transport()
    monkeypatch.setattr(query.LangfuseHttpTransport, "from_environment", lambda: transport)
    assert query.main(["--trace-id", TRACE, "--from-start-time", "2026-09-09T00:59:59Z",
                       "--to-start-time", "2026-09-09T01:00:03Z"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trace_id"] == TRACE and result["generation_count"] == 1
    assert result["query_window"]["fromStartTime"] == "2026-09-09T00:59:59Z"
    queries = [payload["params"]["arguments"] for _, _, payload in transport.requests
               if payload.get("params", {}).get("name") == "listObservations"]
    assert len(queries) == 2
    assert all(query_args["toStartTime"] == "2026-09-09T01:00:03Z" for query_args in queries)
