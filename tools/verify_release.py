"""Verify the compact IDEN-VLN release without optional ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(root):
    checks = json.loads((root / "release_checksums.json").read_text(encoding="utf-8"))
    required = {"iden_vln_tasks.jsonl", "iden_vln_step_tasks.jsonl", "splits_random.json", "splits_scene_heldout.json"}
    require(required <= checks.keys(), "Checksum inventory is missing required files")
    for name, expected in checks.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Checksum path escapes dataset directory")
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected["sha256"], "Checksum mismatch: " + name)
        require(len(content) == expected["size_bytes"], "Size mismatch: " + name)
    tasks = read_rows(root / "iden_vln_tasks.jsonl")
    steps = read_rows(root / "iden_vln_step_tasks.jsonl")
    ids = [row["task_id"] for row in tasks]
    by_id = {row["task_id"]: row for row in tasks}
    require(len(tasks) == len(set(ids)) == 2500, "Expected 2,500 unique tasks")
    require(len(steps) == len({row["step_task_id"] for row in steps}) == 3291, "Expected 3,291 unique segments")
    require(all(row["parent_task_id"] in by_id for row in steps), "Unknown segment parent")
    require(len({row["scene"] for row in tasks}) == 155, "Expected 155 scenes")
    require(all(2 <= int(row["level"]) <= 5 for row in tasks), "Unexpected task level")
    for name in ["random", "scene_heldout"]:
        split = json.loads((root / ("splits_" + name + ".json")).read_text(encoding="utf-8"))
        all_ids = []
        scenes = {}
        for part in ["train", "val", "test"]:
            selected = split[part]["task_ids"]
            require(len(selected) == split[part]["count"], "Split count mismatch")
            require(all(task_id in by_id for task_id in selected), "Unknown task in split")
            require(all(by_id[task_id][name + "_split"] == part for task_id in selected), "Record/split disagreement")
            all_ids.extend(selected)
            scenes[part] = {by_id[task_id]["scene"] for task_id in selected}
        require(len(all_ids) == len(set(all_ids)) == len(ids), "Split overlap or missing tasks")
        require(set(all_ids) == set(ids), "Incomplete split coverage")
        if name == "scene_heldout":
            require(not any(scenes[a] & scenes[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]), "Scene-held-out overlap")
    return {"tasks": len(tasks), "segments": len(steps), "scenes": 155, "checksums": "passed", "splits": "passed", "parent_mapping": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    print(json.dumps(verify(parser.parse_args().data_root), indent=2))
