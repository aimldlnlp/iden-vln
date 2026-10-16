#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NAV_GEN = Path("/home/aimldl/LH-VLN/nav_gen")
RUNS = NAV_GEN / "runs"
EXPORT_ROOT = RUNS / "ieee_access_release_export_20260708"

OLD_MANIFEST = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_split_step_success1206/logs/success_pool_manifest.jsonl"
DOWNSTREAM_MANIFEST = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704/logs/downstream_manifest.jsonl"
STEP_TASK_DIR = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704/step_task"
TRAIL_LIST = RUNS / "audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704/split/trail_list.txt"
BENCHMARK_RESULTS = RUNS / "ieee_access_experiments_2500_20260704/results"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def norm(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def object_entries(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                name = norm(item[0]) if item else ""
                region_text = norm(item[1]) if len(item) > 1 else ""
                match = re.match(r"Region\s+([^:]+):\s*(.*)", region_text)
                out.append(
                    {
                        "name": name,
                        "region_id": match.group(1).strip() if match else "",
                        "region_name": match.group(2).strip() if match else region_text,
                    }
                )
            elif isinstance(item, dict):
                out.append(
                    {
                        "name": norm(item.get("name") or item.get("object") or item.get("label")),
                        "region_id": norm(item.get("region_id") or item.get("region")),
                        "region_name": norm(item.get("region_name") or item.get("room")),
                    }
                )
            elif item not in (None, ""):
                out.append({"name": norm(item), "region_id": "", "region_name": ""})
    return out


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [norm(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [norm(item) for item in value.values() if item not in (None, "")]
    return [norm(value)]


def task_hash(source: str, synthetic_path: str, data: dict[str, Any]) -> str:
    payload = synthetic_path + norm(data.get("Scene")) + norm(data.get("Task instruction"))
    return f"{source}_{hashlib.sha1(payload.encode()).hexdigest()[:16]}"


def portable_task_record(
    *,
    source_run: str,
    source_index: int,
    source_id: str,
    synthetic_path: str,
    data: dict[str, Any],
    source_slug: str,
) -> dict[str, Any]:
    level = norm(data.get("Generated level")) or source_slug.split("/", 1)[0]
    objects = object_entries(data.get("Object"))
    region_names = as_list(data.get("Region Name"))
    region_ids = as_list(data.get("Region"))
    task_id = task_hash(source_run, synthetic_path, data)
    return {
        "task_id": task_id,
        "source_run": source_run,
        "source_index": source_index,
        "source_id": source_id,
        "source_slug": source_slug,
        "level": str(level),
        "robot": norm(data.get("Robot")).lower(),
        "scene": norm(data.get("Scene")),
        "task_instruction": norm(data.get("Task instruction")),
        "subtask_list": as_list(data.get("Subtask list")),
        "objects": objects,
        "object_names": [item["name"] for item in objects if item.get("name")],
        "region_ids": region_ids,
        "region_names": region_names or [item["region_name"] for item in objects if item.get("region_name")],
        "audit": data.get("Audit") if isinstance(data.get("Audit"), dict) else {},
        "generator": norm(data.get("Generator")),
        "validation_status": "trajectory_success",
    }


def discover_old_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(read_jsonl(OLD_MANIFEST)):
        source_dir = Path(item["source"])
        link_dir = Path(item["link"])
        config_path = source_dir / "config.json"
        if not config_path.exists():
            config_path = link_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing old config for {item}")
        data = read_json(config_path)
        synthetic_path = str(link_dir / "config.json")
        source_slug = f"{item.get('level', '')}/{item.get('task', '')}"
        rows.append(
            portable_task_record(
                source_run="old",
                source_index=idx,
                source_id=f"old_{idx:05d}",
                synthetic_path=synthetic_path,
                data=data,
                source_slug=source_slug,
            )
        )
    return rows


def discover_new_tasks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    slug_to_task_id: dict[str, str] = {}
    for idx, item in enumerate(read_jsonl(DOWNSTREAM_MANIFEST)):
        task_json = Path(item["source_task_json"])
        if not task_json.exists():
            raise FileNotFoundError(f"Missing downstream task json: {task_json}")
        data = read_json(task_json)
        source_run = norm(item.get("source_run"))
        if source_run not in {"v2", "topup"}:
            source_run = "v2" if "v2" in str(task_json) else "topup"
        source_id = norm(item.get("id")) or f"{source_run}_{idx:05d}"
        source_slug = f"{item.get('level', '')}/{Path(item.get('source_path', '')).name}"
        row = portable_task_record(
            source_run=source_run,
            source_index=idx,
            source_id=source_id,
            synthetic_path=str(task_json),
            data=data,
            source_slug=source_slug,
        )
        rows.append(row)
        slug_to_task_id[source_id] = row["task_id"]
        slug_to_task_id[Path(norm(item.get("source_path"))).name] = row["task_id"]
    return rows, slug_to_task_id


def split_maps(tasks: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    sorted_tasks = sorted(tasks, key=lambda row: stable_hash(row["task_id"]))
    random_map: dict[str, str] = {}
    for index, row in enumerate(sorted_tasks):
        frac = index / len(sorted_tasks)
        random_map[row["task_id"]] = "train" if frac < 0.8 else ("val" if frac < 0.9 else "test")

    scene_count = Counter(row["scene"] for row in tasks)
    scenes = sorted(scene_count, key=stable_hash)
    scene_split: dict[str, str] = {}
    running = 0
    total = len(tasks)
    for scene in scenes:
        split = "train" if running < 0.8 * total else ("val" if running < 0.9 * total else "test")
        scene_split[scene] = split
        running += scene_count[scene]
    scene_map = {row["task_id"]: scene_split[row["scene"]] for row in tasks}
    return {"random": random_map, "scene_heldout": scene_map}


def split_summary(tasks: list[dict[str, Any]], split_map: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in tasks if split_map[row["task_id"]] == split]
        out[split] = {
            "count": len(rows),
            "scene_count": len({row["scene"] for row in rows}),
            "level": dict(sorted(Counter(row["level"] for row in rows).items())),
            "robot": dict(sorted(Counter(row["robot"] for row in rows).items())),
            "task_ids": [row["task_id"] for row in rows],
        }
    split_scenes = {split: {row["scene"] for row in tasks if split_map[row["task_id"]] == split} for split in ("train", "val", "test")}
    out["scene_overlap"] = {
        "train_val": len(split_scenes["train"] & split_scenes["val"]),
        "train_test": len(split_scenes["train"] & split_scenes["test"]),
        "val_test": len(split_scenes["val"] & split_scenes["test"]),
    }
    return out


def step_parent_slug(trajectory_path: str) -> str:
    parts = Path(trajectory_path).parts
    if "task" in parts:
        idx = parts.index("task")
        if len(parts) > idx + 2:
            return parts[idx + 2]
    return ""


def export_step_tasks(slug_to_task_id: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_parent_count: dict[str, int] = defaultdict(int)
    for path in sorted(STEP_TASK_DIR.glob("step_*.json")):
        data = read_json(path)
        match = re.match(r"step_(\d+)_", path.name)
        global_index = int(match.group(1)) if match else len(rows)
        trajectory_ref = norm(data.get("trajectory path"))
        slug = step_parent_slug(trajectory_ref)
        parent_task_id = slug_to_task_id.get(slug, "")
        local_index = per_parent_count[parent_task_id or slug]
        per_parent_count[parent_task_id or slug] += 1
        rows.append(
            {
                "step_task_id": f"step_{global_index:05d}",
                "parent_task_id": parent_task_id,
                "parent_source_slug": slug,
                "step_index_within_parent": local_index,
                "global_step_index": global_index,
                "trajectory_ref": trajectory_ref,
                "start": data.get("start"),
                "end": data.get("end"),
                "robot": norm(data.get("Robot")).lower(),
                "scene": norm(data.get("Scene")),
                "target": as_list(data.get("target")),
                "region": as_list(data.get("Region")),
                "start_pos": data.get("start_pos"),
                "start_yaw": data.get("start_yaw"),
                "mapping": data.get("mapping") if isinstance(data.get("mapping"), dict) else {},
                "step_instruction": norm(data.get("Task instruction")),
            }
        )
    return rows


def copy_benchmark_results() -> None:
    dst = EXPORT_ROOT / "benchmark_results"
    dst.mkdir(parents=True, exist_ok=True)
    names = [
        "baseline_rule_extractor_full2500.json",
        "baseline_rule_extractor_full2500.jsonl",
        "baseline_llm_instruction_parse_full2500.json",
        "baseline_llm_instruction_parse_full2500.jsonl",
        "baseline_llm_instruction_parse_llama3.1_8b_full2500.json",
        "baseline_llm_instruction_parse_llama3.1_8b_full2500.jsonl",
        "baseline_llm_instruction_parse_mistral_7b_full2500.json",
        "baseline_llm_instruction_parse_mistral_7b_full2500.jsonl",
        "classic_ml_baselines.json",
        "scene_heldout_benchmark.json",
        "code_mixing_token_metrics.json",
        "code_mixing_token_metrics.jsonl",
        "failure_mode_breakdown.json",
        "split_overlap_audit.json",
        "sequence_metrics.json",
        "publishable_benchmark_summary.json",
    ]
    for name in names:
        src = BENCHMARK_RESULTS / name
        if src.exists():
            shutil.copy2(src, dst / name)


def file_inventory(root: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": size,
                    "sha256": sha256_file(path),
                }
            )
    return {"total_size_bytes": total, "files": files}


def main() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = discover_old_tasks()
    new_tasks, slug_to_task_id = discover_new_tasks()
    tasks.extend(new_tasks)
    if len(tasks) != 2500:
        raise RuntimeError(f"Expected 2500 high-level tasks, got {len(tasks)}")
    if len({row["task_id"] for row in tasks}) != len(tasks):
        raise RuntimeError("Duplicate task_id detected")

    maps = split_maps(tasks)
    for row in tasks:
        row["random_split"] = maps["random"][row["task_id"]]
        row["scene_heldout_split"] = maps["scene_heldout"][row["task_id"]]

    step_tasks = export_step_tasks(slug_to_task_id)
    if len(step_tasks) != 3291:
        raise RuntimeError(f"Expected 3291 step tasks, got {len(step_tasks)}")
    missing_parent = sum(1 for row in step_tasks if not row["parent_task_id"])

    write_jsonl(EXPORT_ROOT / "iden_vln_tasks.jsonl", sorted(tasks, key=lambda row: row["task_id"]))
    write_jsonl(EXPORT_ROOT / "iden_vln_step_tasks.jsonl", sorted(step_tasks, key=lambda row: row["global_step_index"]))
    write_json(EXPORT_ROOT / "splits_random.json", split_summary(tasks, maps["random"]))
    write_json(EXPORT_ROOT / "splits_scene_heldout.json", split_summary(tasks, maps["scene_heldout"]))
    copy_benchmark_results()

    manifest = {
        "generated_at": now(),
        "export_root": str(EXPORT_ROOT),
        "counts": {
            "high_level_tasks": len(tasks),
            "step_task_segments": len(step_tasks),
            "step_tasks_missing_parent_mapping": missing_parent,
        },
        "source_paths": {
            "old_manifest": str(OLD_MANIFEST),
            "downstream_manifest": str(DOWNSTREAM_MANIFEST),
            "step_task_dir": str(STEP_TASK_DIR),
            "trail_list": str(TRAIL_LIST),
            "benchmark_results": str(BENCHMARK_RESULTS),
        },
        "split_summary": {
            "random": {k: {kk: vv for kk, vv in v.items() if kk != "task_ids"} if isinstance(v, dict) else v for k, v in split_summary(tasks, maps["random"]).items()},
            "scene_heldout": {k: {kk: vv for kk, vv in v.items() if kk != "task_ids"} if isinstance(v, dict) else v for k, v in split_summary(tasks, maps["scene_heldout"]).items()},
        },
        "known_limitations": [
            "License is TBD.",
            "Code-mixing labels are automatic heuristics, not human linguistic annotation.",
            "Scene-held-out split is not zero-shot object, room, or language generalization.",
            "Agent results are controlled downstream diagnostics, not competitive navigation performance.",
            "Portable release excludes raw trajectory assets and simulator scene files.",
        ],
    }
    write_json(EXPORT_ROOT / "release_manifest.json", manifest)
    inventory = file_inventory(EXPORT_ROOT)
    write_json(EXPORT_ROOT / "release_checksums.json", inventory)
    print(json.dumps({"status": "ok", **manifest["counts"], "size_bytes": inventory["total_size_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
