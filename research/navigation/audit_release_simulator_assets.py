#!/usr/bin/env python3
"""Read-only audit of the 2,500 release tasks against source trajectories."""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import median


RUNS = Path("/home/aimldl/LH-VLN/nav_gen/runs")
RELEASE = RUNS / "ieee_access_release_export_20260708/iden_vln_tasks.jsonl"
OLD = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_split_step_success1206/logs/success_pool_manifest.jsonl"
NEW = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704/logs/downstream_manifest.jsonl"
HM3D = Path("/home/aimldl/LH-VLN/data/hm3d")


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main():
    released, old, new = rows(RELEASE), rows(OLD), rows(NEW)
    counts = Counter()
    example = {}
    test_details = {level: {"actions": [], "gaps": []} for level in range(2, 6)}
    for record in released:
        level = int(record["level"])
        counts[level, "release"] += 1
        if record["scene_heldout_split"] == "test":
            counts[level, "test"] += 1
        if record["source_run"] == "old":
            item = old[record["source_index"]]
            parent = Path(item["source"])
            source = parent / "success/trial_1/task.json"
            if not source.exists():
                source = Path(item["link"]) / "success/trial_1/task.json"
        else:
            item = new[record["source_index"]]
            source = Path(item["source_task_json"])
        if not source.exists():
            counts[level, "missing_source"] += 1
            continue
        task = json.loads(source.read_text())
        trial = task.get("trial")
        if not isinstance(trial, dict) or len(trial) != level:
            counts[level, "wrong_stage_count"] += 1
            example.setdefault(level, {"source": str(source), "trial_type": type(trial).__name__,
                                       "trial_keys": list(trial) if isinstance(trial, dict) else None})
            continue
        if not all(isinstance(trial.get(f"trial_{i}"), dict) and
                   isinstance(trial[f"trial_{i}"].get("pos"), list) and
                   len(trial[f"trial_{i}"]["pos"]) >= 2
                   for i in range(level)):
            counts[level, "missing_position_path"] += 1
            continue
        counts[level, "paths"] += 1
        scene = record["scene"]
        navmesh = scene.split("-")[-1] + ".basis.navmesh"
        if not any((HM3D / split / scene / navmesh).exists() for split in ("train", "val")):
            counts[level, "missing_navmesh"] += 1
            continue
        counts[level, "navmesh"] += 1
        if record["scene_heldout_split"] == "test":
            counts[level, "test_eligible"] += 1
            actions = [trial[f"trial_{i}"].get("action", []) for i in range(level)]
            if all(isinstance(part, list) and part and part[-1] == "stop" for part in actions):
                counts[level, "test_all_stages_end_stop"] += 1
            if all(len(part) == len(trial[f"trial_{i}"]["pos"]) for i, part in enumerate(actions)):
                counts[level, "test_action_position_lengths_match"] += 1
            total_actions = sum(len(part) for part in actions)
            test_details[level]["actions"].append(total_actions)
            gaps = [math.dist(trial[f"trial_{i}"]["pos"][-1], trial[f"trial_{i+1}"]["pos"][0])
                    for i in range(level - 1)]
            test_details[level]["gaps"].append(max(gaps, default=0))
        for i in range(level - 1):
            gap = math.dist(trial[f"trial_{i}"]["pos"][-1], trial[f"trial_{i+1}"]["pos"][0])
            if gap <= 0.1:
                continue
            counts[level, "boundary_offset_gt_0p1m"] += 1
            if record["scene_heldout_split"] == "test":
                counts[level, "test_boundary_offset_gt_0p1m"] += 1
            if trial[f"trial_{i+1}"]["action"][0] != "move_forward" or gap > 0.35:
                counts[level, "unexplained_boundary_offset"] += 1
                if record["scene_heldout_split"] == "test":
                    counts[level, "test_unexplained_boundary_offset"] += 1
    print("manifest_counts", len(old), len(new), "release", len(released))
    for level in range(2, 6):
        print(level, {name: counts[level, name] for name in
                      ("release", "test", "paths", "navmesh", "test_eligible", "missing_source",
                       "wrong_stage_count", "missing_position_path", "missing_navmesh",
                       "boundary_offset_gt_0p1m", "test_boundary_offset_gt_0p1m",
                       "unexplained_boundary_offset", "test_unexplained_boundary_offset",
                       "test_all_stages_end_stop", "test_action_position_lengths_match")})
        detail = test_details[level]
        print("test_detail", level, {"expert_actions_median": median(detail["actions"]),
                                     "expert_actions_min": min(detail["actions"]),
                                     "expert_actions_max": max(detail["actions"]),
                                     "expert_actions_le150": sum(x <= 150 for x in detail["actions"]),
                                     "gap_max_median": round(median(detail["gaps"]), 3),
                                     "gap_max_max": round(max(detail["gaps"]), 3)})
    print("examples", example)


if __name__ == "__main__":
    main()
