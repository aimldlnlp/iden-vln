#!/usr/bin/env python3
"""Reconstruct canonical Table 4 reference segments and audit local CGT."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import habitat_sim

from audit_stage_metrics import NAMES


def load_ordered(lh_root):
    source = lh_root / "runs/idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py"
    spec = importlib.util.spec_from_file_location("ordered_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_navmesh(lh_root, scene):
    split = "train" if int(scene[2]) < 8 else "val"
    return lh_root / "data/hm3d" / split / scene / (scene.split("-")[-1] + ".basis.navmesh")


def cgt_episode(lengths, successes):
    first, second = map(bool, successes)
    p1, p2 = lengths
    total = p1 + p2
    assert total > 0 and all(math.isfinite(p) and p >= 0 for p in lengths)
    # LH-VLN Eq. 4, N=2; local convention: predecessor of first stage succeeds.
    return (p1 / total) * first + (p2 / total) * second * (1 + first) / 2


def audit(trace_dir, lh_root, limit):
    traces = {}
    for name in NAMES:
        path = trace_dir / f"{name}.jsonl"
        records = {r["index"]: r for r in (json.loads(s) for s in path.read_text(encoding="utf-8").splitlines())}
        assert len(records) == 100, (name, len(records))
        traces[name] = records
    bank = sorted(traces[NAMES[0]])[:limit]
    assert all(sorted(records)[:limit] == bank for records in traces.values())
    ordered = load_ordered(lh_root)
    lengths_by_index = {}
    maximum_error = 0.0
    for ordinal, index in enumerate(bank, 1):
        baseline = traces[NAMES[0]][index]
        for name in NAMES[1:]:
            record = traces[name][index]
            for key in ("scene", "start_state", "reference_target_points", "reference_path_length"):
                assert record[key] == baseline[key], (index, name, key)
        points = baseline["reference_target_points"]
        assert len(points) == 2 and all(point is not None for point in points), index
        pathfinder = habitat_sim.PathFinder()
        navmesh = scene_navmesh(lh_root, baseline["scene"])
        assert pathfinder.load_nav_mesh(str(navmesh)), navmesh
        current = baseline["start_state"]["position"]
        lengths = []
        for point in points:
            distance, selected = ordered.geodesic_between(pathfinder, current, [point])
            assert math.isfinite(distance) and selected is not None, index
            lengths.append(distance)
            current = point
        expected = baseline["reference_path_length"]
        assert expected is not None and expected > 0, index
        error = abs(sum(lengths) - expected)
        assert error <= 1e-3, (index, lengths, expected, error)
        maximum_error = max(maximum_error, error)
        lengths_by_index[index] = {
            "scene": baseline["scene"], "stage_lengths_m": lengths,
            "reference_total_m": expected, "sum_error_m": error,
        }
        if ordinal % 10 == 0 or ordinal == len(bank):
            print(f"validated {ordinal}/{len(bank)}; max sum error {maximum_error:.8f} m", flush=True)
    rows = {}
    for name, records in traces.items():
        scores = [cgt_episode(lengths_by_index[i]["stage_lengths_m"], records[i]["successes"]) for i in bank]
        rows[name] = {"episodes": len(bank), "CGT_local": sum(scores) / len(scores),
                      "successful_stages": sum(sum(map(bool, records[i]["successes"])) for i in bank)}
    return {
        "status": "sidecar_audit_not_table_4",
        "source_traces": str(trace_dir),
        "reference_method": "Habitat geodesic_between on original navmesh, start to saved selected goal 1, then saved goal 1 to saved goal 2",
        "formula": "mean[(P1/P)*s1 + (P2/P)*s2*(1+s1)/2], P=P1+P2, initial predecessor success=1",
        "reference_episodes_validated": len(bank),
        "max_reference_sum_error_m": maximum_error,
        "stage_lengths": lengths_by_index,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lh-root", type=Path, default=Path("/home/aimldl/LH-VLN"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    assert 1 <= args.limit <= 100
    result = audit(args.trace_dir, args.lh_root, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    assert cgt_episode([2, 8], [True, False]) == 0.2
    assert cgt_episode([2, 8], [True, True]) == 1.0
    assert cgt_episode([2, 8], [False, True]) == 0.4
    main()
