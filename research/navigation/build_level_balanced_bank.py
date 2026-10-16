#!/usr/bin/env python3
"""Emit a reproducible mixed-scene 25-per-level route diagnostic bank."""

import hashlib
import json
import random
from pathlib import Path


ROOT = Path("/home/aimldl/LH-VLN")
RELEASE = ROOT / "nav_gen/runs/ieee_access_release_export_20260708/iden_vln_tasks.jsonl"
OLD = ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706/data_expert_scaled"
SEED = 20260929


def main():
    trials = sorted(OLD.glob("task/batch_generated_level23/*/*/success/trial_1"))
    assert len(trials) == 300
    ids = list(range(len(trials)))
    random.Random(49000).shuffle(ids)
    train_scenes = {json.loads((trials[i] / "task.json").read_text())["Scene"] for i in ids[:210]}
    tasks = [json.loads(line) for line in RELEASE.open(encoding="utf-8") if line.strip()]
    chosen = []
    for level in range(2, 6):
        candidates = [x for x in tasks if int(x["level"]) == level and x["scene_heldout_split"] == "test"]
        candidates.sort(key=lambda x: hashlib.sha256(f'{SEED}:{x["task_id"]}'.encode()).hexdigest())
        scenes, selected = set(), []
        for task in candidates:
            if task["scene"] not in scenes:
                selected.append(task)
                scenes.add(task["scene"])
        selected.extend(x for x in candidates if x not in selected)
        chosen.extend(selected[:25])
    assert len(chosen) == 100 and len({x["task_id"] for x in chosen}) == 100
    for task in chosen:
        print(json.dumps({
            "task_id": task["task_id"],
            "level": int(task["level"]),
            "scene": task["scene"],
            "source_run": task["source_run"],
            "source_index": task["source_index"],
            "tiered_scaled_train_scene_overlap": task["scene"] in train_scenes,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
