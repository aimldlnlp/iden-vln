#!/usr/bin/env python3
"""Freeze a single-endpoint bank and run an honest one-STOP Habitat pilot."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import habitat_sim
import numpy as np

sys.path.insert(0, "/home/aimldl/LH-VLN")
from habitat_base.simulation import SceneSimulator
from run_ordered_metrics_eval import choose_action, install_index_aware_target_patch, json_default, load_legacy_module
from audit_release_simulator_assets import HM3D, NEW, OLD, rows
from pilot_level_balanced_route import BANK, geodesic, source_path
from pilot_route_fidelity import distinct, dtw
from pilot_simulator_route_stop import reset


ROOT = Path("/home/aimldl/LH-VLN")
MANIFEST = ROOT / "runs/single_goal_route_bank_100_20260929.jsonl"
OUTPUT = ROOT / "runs/single_goal_route_pilot_20260929.jsonl"
POLICIES = ("stationary", "random", "tiered_bc", "target_conditioned_bc")


def interpolate(points):
    output = [list(points[0])]
    for start, end in zip(points, points[1:]):
        segments = max(1, math.ceil((math.dist(start, end) - 1e-5) / 0.25))
        for k in range(1, segments + 1):
            output.append([float(a + (b - a) * k / segments) for a, b in zip(start, end)])
    return output


def build_manifest():
    bank, old, new = rows(BANK), rows(OLD), rows(NEW)
    assert len(bank) == 100 and len({r["task_id"] for r in bank}) == 100
    with MANIFEST.open("w") as handle:
        for index, record in enumerate(bank):
            task_path = source_path(record, old, new)
            task = json.loads(task_path.read_text())
            level = record["level"]
            assert task["Scene"] == record["scene"] and len(task["trial"]) == level
            assert task["trial"]["trial_0"]["yaw"][0] == 180
            paths = [task["trial"][f"trial_{i}"]["pos"] for i in range(level)]
            assert all(path and len(path) >= 2 for path in paths)
            scene = record["scene"]
            navmesh_name = scene.split("-")[-1] + ".basis.navmesh"
            navmesh = next(p for split in ("train", "val") if (p := HM3D / split / scene / navmesh_name).exists())
            pathfinder = habitat_sim.PathFinder()
            assert pathfinder.load_nav_mesh(str(navmesh))
            reference = list(paths[0])
            for path in paths[1:]:
                if math.dist(reference[-1], path[0]) > 1e-5:
                    bridge = habitat_sim.nav.ShortestPath()
                    bridge.requested_start = np.asarray(reference[-1], dtype=np.float32)
                    bridge.requested_end = np.asarray(path[0], dtype=np.float32)
                    assert pathfinder.find_path(bridge) and math.isfinite(bridge.geodesic_distance)
                    reference.extend(interpolate(bridge.points)[1:])
                reference.extend(path)
            reference = [list(p) for p in distinct(interpolate(reference))]
            assert math.dist(reference[0], paths[0][0]) < 1e-5
            assert math.dist(reference[-1], paths[-1][-1]) < 1e-5
            row = {"index": index, "task_id": record["task_id"], "level": level,
                   "scene": scene, "source_task_json": str(task_path), "navmesh": str(navmesh),
                   "training_scene_overlap": record["tiered_scaled_train_scene_overlap"],
                   "start": paths[0][0], "final": paths[-1][-1],
                   "stage_endpoints": [path[-1] for path in paths], "reference": reference}
            handle.write(json.dumps(row) + "\n")
    print("manifest", MANIFEST, "sha256", hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), flush=True)


def evaluate(trace, row, pathfinder):
    positions = distinct([row["start"]] + [step["position"] for step in trace])
    reference = [tuple(p) for p in row["reference"]]
    cache = {}

    def distance(start, end):
        key = (start, end)
        if key not in cache:
            cache[key] = geodesic(pathfinder, start, end)
        return cache[key]

    final_error = distance(positions[-1], tuple(row["final"]))
    success = final_error < 3.0
    score = math.exp(-dtw(reference, positions, distance) / (3.0 * len(reference)))
    prefix = 0
    for position in positions:
        while prefix < row["level"] and distance(position, tuple(row["stage_endpoints"][prefix])) < 3.0:
            prefix += 1
    return {"final_error_m": final_error, "final3": success,
            "stop3": success and bool(trace) and trace[-1]["action"] == "stop",
            "ndtw": score, "sdtw": score if success else 0.0,
            "ordered_stage_visit_prefix": prefix}


def run(per_level, policy_names, output):
    assert MANIFEST.exists(), "build manifest first"
    manifest = rows(MANIFEST)
    assert len(manifest) == 100 and {r["level"] for r in manifest} == {2, 3, 4, 5}
    selected = [r for level in range(2, 6) for r in [x for x in manifest if x["level"] == level][:per_level]]
    install_index_aware_target_patch()
    legacy = load_legacy_module(ROOT)
    policy_states = {name: legacy.load_policy(name) for name in policy_names if name not in ("stationary", "random")}
    assert all(value["status"] == "loaded" for value in policy_states.values()), policy_states
    args = legacy.make_args()
    with output.open("w") as handle:
        for name in policy_names:
            for row in selected:
                task = json.loads(Path(row["source_task_json"]).read_text())
                context = {"instruction": str(task.get("Task instruction") or ""),
                           "objects": legacy.object_labels(task.get("Object")),
                           "regions": legacy.normalize_regions(task.get("Region Name")),
                           "robot": str(task.get("Robot") or "").lower(),
                           "level": row["level"], "index": row["index"]}
                assert set(context) == {"instruction", "objects", "regions", "robot", "level", "index"}
                args.max_step = 150 * row["level"]
                with contextlib.redirect_stdout(io.StringIO()):
                    sim = SceneSimulator(args=args, config=task)
                    try:
                        obs = reset(sim, row["start"])
                        previous = legacy.ACTION_TO_IDX["stop"]
                        recent, buffer, positions, trace = [], [], [list(sim.agent.get_state().position)], []
                        reason = "horizon"
                        for step in range(args.max_step):
                            if name == "stationary":
                                action_index = legacy.ACTION_TO_IDX["stop"]
                            else:
                                state = sim.agent.get_state()
                                action_index, buffer = choose_action(
                                    legacy, name, policy_states.get(name, {}), obs,
                                    list(state.position), float(state.rotation.angle()),
                                    previous, buffer, context, "raw", recent, positions[-16:], step)
                            action = legacy.IDX_TO_ACTION[action_index]
                            if action == "stop":
                                reason = "stop"
                                sim.episode_over = True  # One STOP ends this single-goal episode.
                            else:
                                obs, _, _ = sim.actor(action)
                            current = list(sim.agent.get_state().position)
                            trace.append({"action": action, "position": current})
                            positions.append(current)
                            previous = action_index
                            recent = (recent + [action_index])[-16:]
                            if sim.episode_over:
                                break
                        metrics = evaluate(trace, row, sim.pathfinder)
                    finally:
                        sim.close()
                result = {"index": row["index"], "task_id": row["task_id"], "level": row["level"],
                          "policy": name, "training_scene_overlap": row["training_scene_overlap"],
                          "actions": len(trace), "reason": reason, "checkpoint": policy_states.get(name, {}).get("ckpt"),
                          "trace": trace, **metrics}
                handle.write(json.dumps(result, default=json_default) + "\n")
                handle.flush()
                print(row["index"], name, "final3", metrics["final3"],
                      "nDTW", round(metrics["ndtw"], 3), flush=True)


if __name__ == "__main__":
    assert interpolate([[0, 0, 0], [0.5, 0, 0]]) == [[0, 0, 0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--per-level", type=int, default=1)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.build_manifest:
        build_manifest()
    else:
        assert 1 <= args.per_level <= 25
        names = args.policies.split(",")
        assert names and len(names) == len(set(names)) and set(names) <= set(POLICIES)
        run(args.per_level, names, args.output)
