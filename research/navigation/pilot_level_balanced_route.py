#!/usr/bin/env python3
"""Pilot endpoint geodesics and ordered STOP semantics on one bank task per level."""

import json
import math
import sys
from pathlib import Path

import habitat_sim
import numpy as np

from audit_release_simulator_assets import HM3D, NEW, OLD, RELEASE, rows


ROOT = Path("/home/aimldl/LH-VLN")
BANK = ROOT / "runs/level-balanced-route-bank-100-20260929.jsonl"
THRESHOLD_M = 3.0


def stop_transition(stage, stage_count, distance):
    if not math.isfinite(distance) or distance > THRESHOLD_M:
        return stage, "failed_stop"
    next_stage = stage + 1
    return next_stage, "completed" if next_stage == stage_count else "next_stage"


def geodesic(pathfinder, start, end):
    path = habitat_sim.nav.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    return float(path.geodesic_distance) if pathfinder.find_path(path) else math.inf


def source_path(record, old, new):
    if record["source_run"] == "old":
        item = old[record["source_index"]]
        source = Path(item["source"]) / "success/trial_1/task.json"
        return source if source.exists() else Path(item["link"]) / "success/trial_1/task.json"
    return Path(new[record["source_index"]]["source_task_json"])


def main():
    bank, old, new = rows(BANK), rows(OLD), rows(NEW)
    if "--test-all" in sys.argv:
        selected = [x for x in rows(RELEASE) if x["scene_heldout_split"] == "test"]
        for record in selected:
            record["level"] = int(record["level"])
    else:
        selected = bank if "--all" in sys.argv else [next(x for x in bank if x["level"] == level) for level in range(2, 6)]
    for record in selected:
        task = json.loads(source_path(record, old, new).read_text())
        scene = record["scene"]
        navmesh_name = scene.split("-")[-1] + ".basis.navmesh"
        navmesh = next(p for split in ("train", "val") if (p := HM3D / split / scene / navmesh_name).exists())
        pathfinder = habitat_sim.PathFinder()
        assert pathfinder.load_nav_mesh(str(navmesh)), navmesh
        stage_count = record["level"]
        paths = [task["trial"][f"trial_{i}"]["pos"] for i in range(stage_count)]
        stationary_distances = [geodesic(pathfinder, paths[0][0], positions[-1]) for positions in paths]
        stationary_complete = all(distance <= THRESHOLD_M for distance in stationary_distances)
        stage = 0
        stage_rows = []
        for i, positions in enumerate(paths):
            endpoint = positions[-1]
            start_distance = geodesic(pathfinder, positions[0], endpoint)
            endpoint_distance = geodesic(pathfinder, endpoint, endpoint)
            assert math.isfinite(endpoint_distance) and endpoint_distance <= THRESHOLD_M
            stage, outcome = stop_transition(stage, stage_count, endpoint_distance)
            assert stage == i + 1
            assert outcome == ("completed" if i == stage_count - 1 else "next_stage")
            stage_rows.append({"stage": i, "start_to_endpoint_m": round(start_distance, 4) if math.isfinite(start_distance) else None,
                               "endpoint_self_m": round(endpoint_distance, 4), "stop_outcome": outcome})
        early_distance = geodesic(pathfinder, paths[0][0], paths[0][-1])
        early_stage, early_outcome = stop_transition(0, stage_count, early_distance)
        if early_distance > THRESHOLD_M:
            assert (early_stage, early_outcome) == (0, "failed_stop")
        assert stop_transition(0, stage_count, math.inf) == (0, "failed_stop")
        print(json.dumps({"task_id": record["task_id"], "level": stage_count,
                          "navmesh": str(navmesh), "stages": stage_rows,
                          "early_stop_distance_m": round(early_distance, 4) if math.isfinite(early_distance) else None,
                          "early_stop_outcome": early_outcome,
                          "stationary_all_stop_complete": stationary_complete,
                          "stationary_endpoint_distances_m": [round(d, 4) if math.isfinite(d) else None for d in stationary_distances],
                          "unreachable_stop_outcome": "failed_stop"}, sort_keys=True))


if __name__ == "__main__":
    main()
