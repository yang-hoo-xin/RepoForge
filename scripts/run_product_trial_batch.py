"""Prepare or serially execute frozen Product trials with one caller-owned delivery adapter.

The JSON spec supplies judge_ref, trials and (for --execute) delivery_adapter_ref.
Each trial supplies task_id, arm, repeat, launcher, snapshot_root, allowed_root,
run_root, run_id, original business_argv, process_deadline_seconds, and optional
bound_inputs/extra_load_refs. Missing deadline/budget evidence disables resource comparison.
The selected adapter module exports deliver(context), invoking the existing public
continuation APIs and common judge. It returns status (delivered/failed/
infrastructure_failure/unknown), source_origin=product_model_proposal,
delivery_complete, acceptance_passed, verification_report_ref, evidence_refs and execution_refs. The latter
are exact references to public continuation result records, not developer closure.

For v2, also return v2_delivery_receipt_ref and a verification_report_ref pointing
to a qualified independent_strategy_verdict_v1 bound to the final native execution.
If recovery changed the execution, its verified checkpoint/rebase lineage must
reach the original execution. Existing outcome boundaries index immutable recovery
trace receipts; the batch remeasures those after delivery, including adapter failure.
为什么要有它：v2交付复用原生回执与独立判分，不伪造旧Plan或仅以某次真实模型调用替代最终补丁来源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mod_agent.evaluation.product_trial_batch import (  # noqa: E402
    _read_reference,
    run_product_trial_batch,
    summarize_trials,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run the selected commands; otherwise only prepare them.")
    args = parser.parse_args(argv)
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    output = args.output.resolve()
    for row in spec["trials"]:
        if any(output.is_relative_to(Path(row[key]).resolve()) for key in ("run_root", "snapshot_root")):
            raise ValueError("Batch output must stay outside run and snapshot history.")
    if output.exists():
        raise ValueError("Select a new batch output; earlier records are never overwritten.")
    adapter = None
    adapter_ref = spec.get("delivery_adapter_ref")
    owned_hash = None

    def save(report):
        nonlocal owned_hash
        report.update(spec_ref={"path": str(args.spec.resolve()), "sha256": hashlib.sha256(spec_bytes).hexdigest()},
            delivery_adapter_ref=adapter_ref)
        raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if owned_hash is None:
            with output.open("xb") as stream:
                stream.write(raw)
        else:
            if hashlib.sha256(output.read_bytes()).hexdigest() != owned_hash:
                raise ValueError("The active batch report changed outside this writer.")
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("xb") as stream:
                stream.write(raw)
            os.replace(temporary, output)
        owned_hash = hashlib.sha256(raw).hexdigest()

    report = run_product_trial_batch(spec)
    save(report)
    if args.execute:
        try:
            raw = _read_reference(adapter_ref)
            name = "_selected_product_delivery_" + hashlib.sha256(raw).hexdigest()
            module = types.ModuleType(name)
            module.__file__ = adapter_ref["path"]
            sys.modules[name] = module
            exec(compile(raw, adapter_ref["path"], "exec"), module.__dict__)
            adapter = getattr(module, "deliver")
            if not callable(adapter):
                raise ValueError("Selected delivery export is not callable.")
        except Exception as error:
            for row in report["trials"]:
                row.update(status="infrastructure_failure", success=False,
                    failure={"stage": "adapter_load", "type": type(error).__name__, "message": str(error)})
            report["summary"] = summarize_trials(report["trials"])
            save(report)
            print(json.dumps({"output": str(output), "executed": False, "error": str(error)}, ensure_ascii=False))
            return 2
        report = run_product_trial_batch(spec, execute=True, delivery_adapter=adapter, checkpoint=save)
    print(json.dumps({"output": str(output), "executed": args.execute, "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
