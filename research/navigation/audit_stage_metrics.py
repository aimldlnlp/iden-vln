#!/usr/bin/env python3
"""Read canonical traces; compute stage metrics without altering Table 4."""

import argparse
import json
from collections import Counter
from pathlib import Path


NAMES = (
    "random", "frame_bc_balanced", "target_bc_mild", "target_bc_strong",
    "target_bc_calibrated", "sequence_bc_mild", "sequence_bc_strong",
    "sequence_bc_calibrated", "sequence_hybrid", "dagger_lite",
)


def row_metrics(records):
    assert len(records) == 100 and len({r["index"] for r in records}) == 100
    assert all(r["status"] == "completed" and r["target_count"] == 2 and len(r["successes"]) == 2 for r in records)
    patterns = Counter(tuple(bool(v) for v in r["successes"]) for r in records)
    stage_hits = [sum(bool(r["successes"][i]) for r in records) for i in (0, 1)]
    total = sum(stage_hits)
    # Adjacent-stage reading of LH-VLN Eq. 3, with preceding success=1 at stage 1.
    # This local adaptation remains provisional until the evaluation contract is frozen.
    csr_numerator = sum(
        int(a) * 2 + int(b) * (1 + int(a)) for a, b in patterns.elements()
    )
    prefix = sum(r["completed_prefix"] for r in records)
    assert prefix == stage_hits[0] + patterns[(True, True)]
    return {
        "episodes": len(records), "stages_per_episode": 2,
        "stage1_success": stage_hits[0], "stage2_success": stage_hits[1],
        "stage_successes": total, "ISR_local": total / 200,
        "CSR_adjacent_provisional": csr_numerator / 400,
        "OCS_from_trace": prefix / 200,
        "patterns_first_second": {"00": patterns[(False, False)], "10": patterns[(True, False)],
                                  "01": patterns[(False, True)], "11": patterns[(True, True)]},
        "full_chain_successes": sum(bool(r["full_chain_success"]) for r in records),
        "reference_total_present": sum(r.get("reference_path_length") is not None for r in records),
        "reference_stage_lengths_present": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = {}
    episode_bank = None
    for name in NAMES:
        path = args.trace_dir / f"{name}.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        indices = {r["index"] for r in records}
        if episode_bank is None:
            episode_bank = indices
        assert indices == episode_bank, name
        rows[name] = row_metrics(records)
    result = {
        "status": "sidecar_audit_not_table_4",
        "source": str(args.trace_dir),
        "ISR_local": "sum of successful stages / 200; all 100 tasks have two stages",
        "CSR_adjacent_provisional": "mean over tasks of [2*s1 + s2*(1+s1)]/4; adjacent-stage interpretation of LH-VLN Eq. 3, initial preceding success=1",
        "CGT": "not computed: trace has only full-chain reference length, not validated per-stage ground-truth trajectory lengths",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: value[k] for k in ("stage_successes", "ISR_local", "CSR_adjacent_provisional", "OCS_from_trace")}
                      for name, value in rows.items()}, indent=2))


if __name__ == "__main__":
    assert row_metrics([{"index": i, "status": "completed", "target_count": 2,
                         "successes": [i == 0, False], "completed_prefix": int(i == 0),
                         "full_chain_success": False, "reference_path_length": 1}
                        for i in range(100)])["ISR_local"] == 0.005
    main()
