"""Optional local wire-body references for the existing Patch turn recorder.

The HTTP sender owns whether bytes were actually prepared for transport. Replay
and admission failure never call this observer; it cannot create send evidence.
为什么要有它：应用层请求和重放记录不是本次真实HTTP正文，不能靠再次序列化补写。
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy

from src.mod_agent.upstream_model_trace import _local_body


class PatchTransportBodyRecorder:
    """Add optional local refs without changing Patch persistence or exceptions."""

    def __init__(self, *, record, observation, body_root, provider=None):
        self.record = record
        self.observation = observation
        self.body_root = body_root
        self.capture_boundary = "https_httpx_request_event" if provider == "openai_responses" else "loopback_httpx_request_event"
        self.reference_scope = "actual_https_httpx_request_content" if provider == "openai_responses" else "actual_loopback_httpx_request_content"
        record["provider_request_ref_scope"] = "unknown_not_observed_this_turn"
        record["provider_request_local_body_status"] = "not_observed"
        self.publish_refs()

    @property
    def callback(self):
        return self.capture if self.observation is not None else None

    def publish_refs(self, **facts):
        if self.observation is None:
            return
        try:
            refs = {role: deepcopy(self.record[field]) for role, field in (
                ("request", "request_ref"), ("response", "response_ref"),
                ("provider_request", "provider_request_ref"))
                if isinstance(self.record.get(field), Mapping)}
            self.observation.update(local_detail_refs=refs, **facts)
        except Exception:
            # A trace ref cannot replace a business result or persistence failure.
            pass

    def capture(self, raw):
        try:
            digest = hashlib.sha256(raw).hexdigest()
            facts = {"provider_request_sha256": digest, "provider_request_bytes": len(raw),
                     "provider_request_capture_boundary": self.capture_boundary}
            self.record.update(facts)
            self.record.pop("provider_request_ref", None)
            try:
                reference = _local_body(self.body_root, self.observation.observation_id,
                                        "provider_request.json", raw)
                if reference is not None:
                    self.record["provider_request_ref"] = reference
                    self.record["provider_request_ref_scope"] = self.reference_scope
                    status = "saved"
                else:
                    self.record["provider_request_ref_scope"] = "unknown_body_destination_unavailable"
                    status = "not_configured"
            except Exception:
                self.record["provider_request_ref_scope"] = "unknown_local_body_persistence_failed"
                status = "write_failed"
            self.record["provider_request_local_body_status"] = status
            self.publish_refs(**facts, provider_request_local_body_status=status)
        except Exception:
            # Capture is observational even if hashing or metadata assembly fails.
            pass
