#!/usr/bin/env python3
"""Geodesic nDTW on fixed canonical frozen traces (diagnostic only)."""

import argparse
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np

ROOT = Path('/home/aimldl/LH-VLN')
SOURCE = ROOT / 'runs/idenvln_policy_sweep_20260805_014026/results/traces'


def distinct(points):
    result = []
    for point in points:
        point = tuple(float(x) for x in point)
        if not result or point != result[-1]:
            result.append(point)
    return result


def dtw(reference, predicted, distance):
    previous = [math.inf] * (len(predicted) + 1)
    previous[0] = 0.0
    for target in reference:
        current = [math.inf] * (len(predicted) + 1)
        for j, position in enumerate(predicted, 1):
            current[j] = distance(target, position) + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--policies", default="random,frame_bc_balanced,target_bc_mild,target_bc_strong,target_bc_calibrated,sequence_bc_mild,sequence_bc_strong,sequence_bc_calibrated,sequence_hybrid,dagger_lite")
    args = parser.parse_args()
    indices = list(range(100)) if args.indices == "all" else [int(x) for x in args.indices.split(",")]
    policies = args.policies.split(",")
    refs = {row["index"]: row for row in json.loads(args.references.read_text())}
    traces = {name: {row["index"]: row for row in
                     (json.loads(line) for line in (SOURCE / f"{name}.jsonl").open())}
              for name in policies}
    rows = []
    for index in indices:
        ref = refs[index]
        scene = ref["scene"]
        split = "train" if int(scene[2]) < 8 else "val"
        navmesh = ROOT / "data/hm3d" / split / scene / (scene.split("-")[-1] + ".basis.navmesh")
        pathfinder = habitat_sim.PathFinder()
        assert pathfinder.load_nav_mesh(str(navmesh))
        reference = distinct(ref["reference_paths"][0] + ref["reference_paths"][1])
        assert reference[0] == tuple(ref["start"])
        assert reference[-1] == tuple(ref["stage_endpoints"][-1])
        cache = {}

        def distance(start, end):
            key = (start, end)
            if key not in cache:
                path = habitat_sim.nav.ShortestPath()
                path.requested_start = np.asarray(start, dtype=np.float32)
                path.requested_end = np.asarray(end, dtype=np.float32)
                cache[key] = float(path.geodesic_distance) if pathfinder.find_path(path) else math.inf
            return cache[key]

        for name in policies:
            trace = traces[name][index]
            assert trace["scene"] == scene and trace["target_count"] == 2
            assert np.allclose(trace["start_state"]["position"], ref["start"], atol=1e-5)
            predicted = distinct([ref["start"]] + [step["position_after"] for step in trace["steps"]])
            cost = dtw(reference, predicted, distance)
            final_error = distance(predicted[-1], reference[-1])
            final3 = final_error < 3.0
            stage_stop3 = [False, False]
            for step in trace["steps"]:
                if step["action"] == "stop":
                    stage = step["active_target"]
                    stage_stop3[stage] |= distance(tuple(step["position_after"]),
                                                    tuple(ref["stage_endpoints"][stage])) < 3.0
            ndtw = math.exp(-cost / (len(reference) * 3.0))
            row = {"index": index, "policy": name, "reference_points": len(reference),
                   "agent_points": len(predicted), "final_error_m": final_error,
                   "final3": final3, "ndtw": ndtw, "sdtw": ndtw if final3 else 0.0,
                   "stage0_stop3": stage_stop3[0], "stage1_stop3": stage_stop3[1],
                   "ordered_stage_stop3": all(stage_stop3) and trace["termination_reason"] == "completed_chain"}
            rows.append(row)
            print(index, name, "nDTW", round(ndtw, 4), "final3", final3, flush=True)
    output = {"scope": "fixed canonical100 subset; frozen two-stage traces; Habitat geodesic nDTW",
              "threshold_m": 3.0, "indices": indices, "policies": policies, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    assert dtw([(0,), (1,)], [(0,), (1,)], lambda a, b: abs(a[0] - b[0])) == 0
    assert dtw([(0,), (1,)], [(0,)], lambda a, b: abs(a[0] - b[0])) == 1
    main()
