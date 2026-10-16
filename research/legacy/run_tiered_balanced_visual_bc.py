#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import ImageStat


LH_ROOT = Path("/home/aimldl/LH-VLN")
SOURCE_RUN = LH_ROOT / "nav_gen/runs/audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704"
SOURCE_EXPERT_ROOT = LH_ROOT / "runs/idenvln_expert_traj_bc_level23_20260706/data_idenvln_generated_level23"
RUN_ROOT = LH_ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706"
RESULTS = RUN_ROOT / "results"
BRIDGE_100_ROOT = RUN_ROOT / "data_expert_100"
BRIDGE_SCALED_ROOT = RUN_ROOT / "data_expert_scaled"
BATCH = "/batch_generated_level23"

SEED = 49000
MIN_GPU_FREE_MIB = 12_000
GPU_WAIT_TIMEOUT_SEC = 24 * 60 * 60
GPU_POLL_SEC = 60
TARGET_100 = 100
TARGET_SCALED = 300
MAX_EXPERT_STEPS = 150
MAX_EVAL_STEPS = 80
EVAL_EPISODES = 30
TRAIN_STEPS_100 = 12_000
TRAIN_STEPS_SCALED = 20_000
BATCH_SIZE = 64
ACTION_TO_IDX = {"stop": 0, "turn_left": 1, "move_forward": 2, "turn_right": 3}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(name: str, data: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip()[:180]


def safe_target(value: str) -> str:
    return str(value).replace("/", "").replace("\\", "").replace("_", "-").replace(" ", "-").strip()[:80] or "target"


def first_word(value: Any) -> str:
    if isinstance(value, list) and value:
        item = value[0]
        if isinstance(item, list) and item:
            return str(item[0]).strip().lower()
        return str(item).strip().lower()
    return str(value or "").strip().lower()


def object_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, list) and item:
                out.append(str(item[0]).strip().lower())
            elif isinstance(item, dict):
                out.append(str(item.get("name") or item.get("object") or item.get("label") or "").strip().lower())
            elif item not in ("", None):
                out.append(str(item).strip().lower())
        return [x for x in out if x]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(object_labels(item))
        return out
    return [str(value).strip().lower()] if str(value).strip() else []


def object_region_pairs(task: dict[str, Any]) -> list[list[str]]:
    objects = object_labels(task.get("Object"))
    regions = task.get("Region Name") or task.get("Region") or []
    if isinstance(regions, dict):
        regions = list(regions.values())
    if not isinstance(regions, list):
        regions = [regions]
    pairs = []
    for i, obj in enumerate(objects):
        room = regions[i] if i < len(regions) else (regions[0] if regions else "Unknown")
        pairs.append([str(obj), f"Room: {room}"])
    return pairs


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def gpu_snapshot() -> dict[str, Any]:
    code, out, err = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return {"available": False, "error": err.strip() or out.strip(), "gpus": []}
    gpus = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        used, total, util = int(parts[2]), int(parts[3]), int(parts[4])
        gpus.append({"index": int(parts[0]), "name": parts[1], "memory_used_mib": used, "memory_total_mib": total, "memory_free_mib": total - used, "utilization_gpu_percent": util})
    return {"available": bool(gpus), "gpus": gpus}


def wait_for_gpu() -> dict[str, Any]:
    started = time.time()
    attempt = 0
    history = []
    while True:
        attempt += 1
        snap = gpu_snapshot()
        best_free = max((gpu["memory_free_mib"] for gpu in snap.get("gpus", [])), default=0)
        rec = {"stage": "gpu_wait", "generated_at": now(), "attempt": attempt, "min_gpu_free_mib_required": MIN_GPU_FREE_MIB, "best_gpu_free_mib": best_free, "elapsed_sec": round(time.time() - started, 1), "snapshot": snap}
        history.append(rec)
        history = history[-20:]
        write_json("tiered_bc_preflight.json", {"stage": "tiered_bc_preflight", "status": "waiting_for_gpu", "gpu_wait": rec, "recent_gpu_wait_history": history})
        print(f"[{now()}] GPU wait attempt={attempt} best_free={best_free} MiB required={MIN_GPU_FREE_MIB}", flush=True)
        if best_free >= MIN_GPU_FREE_MIB:
            rec["status"] = "passed"
            rec["waited_sec"] = round(time.time() - started, 1)
            return rec
        if time.time() - started > GPU_WAIT_TIMEOUT_SEC:
            rec["status"] = "timeout"
            rec["waited_sec"] = round(time.time() - started, 1)
            return rec
        time.sleep(GPU_POLL_SEC)


def make_args(root: Path, max_step: int = MAX_EXPERT_STEPS) -> SimpleNamespace:
    return SimpleNamespace(
        task_data=str(root / "task"),
        step_task_data=str(root / "step_task"),
        episode_data=str(root / "episode_task"),
        train_batch=[BATCH],
        val_batch=[BATCH],
        test_batch=[BATCH],
        scene=str(LH_ROOT / "data/hm3d") + "/",
        scene_dataset=str(LH_ROOT / "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"),
        max_step=max_step,
        success_dis=1,
        gradient_accumulation_step=1,
        rank=0,
    )


def install_habitat_sim_api_patch() -> dict[str, Any]:
    from habitat_base.simulation import SceneSimulator

    original = SceneSimulator.get_coord

    def patched_get_coord(self, obj_target):
        scene = self.sim.semantic_scene
        coord_list = []
        index = self.target.index(obj_target)
        region_id = self.region[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    center = obj.aabb.center
                    coord_list.append(center() if callable(center) else center)
        if coord_list == []:
            print("wrong target")
            return 0
        return coord_list

    SceneSimulator.get_coord = patched_get_coord
    return {"patch": "habitat_sim_aabb_center_callable", "installed": True, "original_method": getattr(original, "__name__", "get_coord")}


def prepare_100_copy() -> dict[str, Any]:
    if not SOURCE_EXPERT_ROOT.exists():
        return {"status": "failed", "reason": f"missing source expert root: {SOURCE_EXPERT_ROOT}"}
    if BRIDGE_100_ROOT.exists():
        shutil.rmtree(BRIDGE_100_ROOT)
    shutil.copytree(SOURCE_EXPERT_ROOT, BRIDGE_100_ROOT)
    task_count = len(list((BRIDGE_100_ROOT / "task" / BATCH.strip("/")).glob("*/*/config.json")))
    step_count = len(list((BRIDGE_100_ROOT / "step_task" / BATCH.strip("/")).glob("*.json")))
    return {"status": "completed", "bridge_root": str(BRIDGE_100_ROOT), "high_level_task_count": task_count, "step_task_count": step_count}


def build_step_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((SOURCE_RUN / "step_task").glob("*.json")):
        try:
            data = read_json(path)
            parts = str(data.get("trajectory path", "")).split("/")
            if len(parts) >= 4:
                index[parts[2]].append(path)
        except Exception:
            continue
    return index


def select_manifest_rows(n: int) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_RUN / "logs/downstream_manifest.jsonl")
    step_index = build_step_index()
    candidates = []
    for rec in rows:
        if str(rec.get("level")) not in {"2", "3"} or not step_index.get(str(rec.get("id"))):
            continue
        p = Path(str(rec.get("source_task_json", "")))
        if not p.exists():
            continue
        try:
            task = read_json(p)
        except Exception:
            continue
        robot = str(task.get("Robot", "")).lower()
        if robot in {"stretch", "spot"}:
            candidates.append((rec, task))
    selected, seen = [], set()
    quota = n // 4
    for level in ("2", "3"):
        for robot in ("stretch", "spot"):
            bucket = [(r, t) for r, t in candidates if str(r.get("level")) == level and str(t.get("Robot", "")).lower() == robot]
            for rec, _task in sorted(bucket, key=lambda item: str(item[0].get("id")))[:quota]:
                selected.append(rec)
                seen.add(str(rec["id"]))
    for rec, _task in sorted(candidates, key=lambda item: str(item[0].get("id"))):
        if len(selected) >= n:
            break
        if str(rec["id"]) not in seen:
            selected.append(rec)
            seen.add(str(rec["id"]))
    return selected[:n]


def build_scaled_bridge(n: int = TARGET_SCALED) -> dict[str, Any]:
    step_index = build_step_index()
    selected = select_manifest_rows(n)
    if BRIDGE_SCALED_ROOT.exists():
        shutil.rmtree(BRIDGE_SCALED_ROOT)
    task_root = BRIDGE_SCALED_ROOT / "task" / BATCH.strip("/")
    step_root = BRIDGE_SCALED_ROOT / "step_task" / BATCH.strip("/")
    episode_root = BRIDGE_SCALED_ROOT / "episode_task"
    task_root.mkdir(parents=True, exist_ok=True)
    step_root.mkdir(parents=True, exist_ok=True)
    episode_root.mkdir(parents=True, exist_ok=True)
    copied_steps = 0
    for rec in selected:
        original = read_json(Path(str(rec["source_task_json"])))
        instruction = str(original.get("Task instruction") or rec.get("task_name") or rec["id"])
        level = str(rec["level"])
        task_dir = task_root / level / safe_name(instruction)
        trial_dir = task_dir / "success" / "trial_1"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_task = dict(original)
        config_task["Object"] = object_region_pairs(original)
        move_targets = [x for x in original.get("Subtask list", []) if "Move_to" in str(x)]
        config_task["gt_step"] = [1 for _ in move_targets] or [1 for _ in object_labels(original.get("Object"))] or [1]
        config_task["IDEN-VLN bridge"] = {"source_task_json": str(rec["source_task_json"]), "source_manifest_id": rec["id"], "bridge_version": "tiered_balanced_visual_bc"}
        (task_dir / "config.json").write_text(json.dumps(config_task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (trial_dir / "task.json").write_text(json.dumps(original, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for j, src_step in enumerate(step_index.get(str(rec["id"]), [])):
            step = read_json(src_step)
            step["trajectory path"] = f"task{BATCH}/{level}/{instruction}/success/trial_1"
            (step_root / f"{rec['id']}_step_{j:03d}.json").write_text(json.dumps(step, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            copied_steps += 1
    return {"status": "completed", "bridge_root": str(BRIDGE_SCALED_ROOT), "selected_high_level_tasks": len(selected), "copied_step_tasks": copied_steps}


def task_trial_dir(root: Path, config: dict[str, Any]) -> Path:
    return root / "task" / BATCH.strip("/") / str(len(config.get("Object", []))) / safe_name(str(config["Task instruction"])) / "success" / "trial_1"


def clear_action_dirs(trial_dir: Path) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    for child in trial_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def save_obs_folder(trial_dir: Path, step_index: int, action: str, target: str, obs: list[Any]) -> None:
    folder = trial_dir / f"{step_index}_{action}_for_{safe_target(target)}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, img in zip(["left.png", "front.png", "right.png"], obs):
        if img is not None:
            img.save(folder / name)


def yaw_from_rotation(rotation: Any) -> float:
    from utils.common_utils import rotation_matrix_to_euler_angles, transform_rotation

    try:
        return float(rotation_matrix_to_euler_angles(transform_rotation(rotation))[2])
    except Exception:
        return 0.0


def state_record(sim: Any) -> tuple[list[float], float]:
    pos, rot = sim.return_state()
    return [float(x) for x in np.asarray(pos).tolist()], yaw_from_rotation(rot)


def update_task_json(trial_dir: Path, positions: list[list[float]], yaws: list[float], rec: dict[str, Any]) -> None:
    task_path = trial_dir / "task.json"
    task = read_json(task_path) if task_path.exists() else {}
    task["trajectory path"] = str(trial_dir)
    task["trial"] = {"trial_1": {"pos": positions, "yaw": yaws}}
    task["expert_trajectory_generation"] = rec
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_one_expert(root: Path, args: SimpleNamespace, config: dict[str, Any], index: int) -> dict[str, Any]:
    from habitat_base.simulation import SceneSimulator

    trial_dir = task_trial_dir(root, config)
    clear_action_dirs(trial_dir)
    rec: dict[str, Any] = {"index": index, "instruction": config.get("Task instruction"), "scene": config.get("Scene"), "robot": config.get("Robot"), "trial_dir": str(trial_dir), "status": "not_run"}
    sim = None
    try:
        sim = SceneSimulator(args=args, config=config)
        sim.save_path = None
        obs, _done, info = sim.actor("stop")
        positions, yaws, actions = [], [], ["observe"]
        pos, yaw = state_record(sim)
        positions.append(pos)
        yaws.append(yaw)
        save_obs_folder(trial_dir, 0, "observe", sim.target[sim.stage], obs)
        step_index = 1
        while not sim.episode_over and step_index < MAX_EXPERT_STEPS:
            action = sim.get_next_action(info["target coord"]) if isinstance(info, dict) and "target coord" in info else "stop"
            action = str(action or "stop")
            if action not in ACTION_TO_IDX:
                action = "stop"
            target = sim.target[sim.stage] if sim.stage < len(sim.target) else "target"
            obs, _done, info = sim.actor(action)
            pos, yaw = state_record(sim)
            positions.append(pos)
            yaws.append(yaw)
            save_obs_folder(trial_dir, step_index, action, target, obs)
            actions.append(action)
            step_index += 1
        result = sim.return_results()
        success = bool(result.get("successes")) and all(bool(x) for x in result.get("successes", []))
        frame_dirs = [p for p in trial_dir.iterdir() if p.is_dir()]
        rec.update({"status": "completed" if len(frame_dirs) == len(positions) and len(frame_dirs) > 1 else "invalid_frame_count", "success": success, "frame_dir_count": len(frame_dirs), "position_count": len(positions), "yaw_count": len(yaws), "actions": actions, "sim_result": result})
        update_task_json(trial_dir, positions, yaws, rec)
    except Exception as exc:
        rec.update({"status": "failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]})
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass
    return rec


def generate_scaled_experts() -> dict[str, Any]:
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    from utils.dataset import TaskDataset

    out = {"stage": "tiered_bc_scale_generation", "generated_at": now(), "status": "not_run", "target_scaled_tasks": TARGET_SCALED, "results": []}
    bridge = build_scaled_bridge(TARGET_SCALED)
    out["bridge"] = bridge
    if bridge.get("status") != "completed":
        out["status"] = "blocked_bridge_failed"
        write_json("tiered_bc_scale_generation.json", out)
        return out
    install_habitat_sim_api_patch()
    args = make_args(BRIDGE_SCALED_ROOT, MAX_EXPERT_STEPS)
    ds = TaskDataset(args, "test")
    completed = valid = successes = 0
    for i in range(min(TARGET_SCALED, len(ds))):
        config, _steps = ds[i]
        rec = generate_one_expert(BRIDGE_SCALED_ROOT, args, config, i)
        out["results"].append(rec)
        completed += int(rec.get("status") == "completed")
        successes += int(bool(rec.get("success")))
        valid += int(rec.get("frame_dir_count", 0) > 1 and rec.get("frame_dir_count") == rec.get("position_count"))
        if (i + 1) % 10 == 0:
            out.update({"attempted": i + 1, "completed": completed, "valid_frame_trajectories": valid, "successes": successes})
            write_json("tiered_bc_scale_generation.progress.json", out)
            print(f"[{now()}] scale generation attempted={i+1} valid={valid} success={successes}", flush=True)
    out.update({"attempted": min(TARGET_SCALED, len(ds)), "completed": completed, "valid_frame_trajectories": valid, "successes": successes, "status": "completed" if valid >= 200 else "partial_or_blocked"})
    write_json("tiered_bc_scale_generation.json", out)
    return out


def inspect_sft_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, tuple) or len(item) != 2:
        return {"valid": False, "reason": f"unexpected_item_type:{type(item)}"}
    task, step_tasks = item
    if task is None or not isinstance(task, dict):
        return {"valid": False, "reason": "task_is_none"}
    traj = task.get("Trajectory")
    if not isinstance(traj, list) or not traj:
        return {"valid": False, "reason": "empty_or_missing_trajectory"}
    actions, obs_count = [], 0
    for step in traj:
        actions.append(step.get("action") if isinstance(step, dict) else None)
        obs = step.get("observation") if isinstance(step, dict) else None
        if isinstance(obs, list):
            obs_count += sum(1 for frame in obs if frame is not None)
    return {"valid": obs_count > 0 and all(a in ACTION_TO_IDX for a in actions), "reason": "ok" if obs_count > 0 else "trajectory_without_observation_frames", "trajectory_len": len(traj), "non_null_observation_frames": obs_count, "action_counts": dict(Counter(actions)), "step_task_count": len(step_tasks) if isinstance(step_tasks, list) else None}


def preflight(root: Path, name: str = "tiered_bc_preflight.json") -> dict[str, Any]:
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    from utils.dataset import SFTDataset, TaskDataset

    args = make_args(root, MAX_EXPERT_STEPS)
    out: dict[str, Any] = {"stage": name.replace(".json", ""), "generated_at": now(), "bridge_root": str(root), "status": "not_run"}
    ds = TaskDataset(args, "test")
    sft = SFTDataset(args, "test")
    reasons: Counter[str] = Counter()
    effective, all_effective, action_counts = 0, True, Counter()
    examples = []
    for i in range(min(len(sft), TARGET_SCALED)):
        try:
            info = inspect_sft_item(sft[i])
            reasons[info["reason"]] += 1
            if info["valid"]:
                effective += 1
                action_counts.update(info.get("action_counts", {}))
                if len(examples) < 5:
                    examples.append({"index": i, **info})
            else:
                all_effective = False
        except Exception as exc:
            reasons[repr(exc).splitlines()[0][:160]] += 1
            all_effective = False
    out.update(
        {
            "task_dataset": {"status": "passed", "len": len(ds), "step_task_groups": len(ds.step_tasks)},
            "sft_dataset": {"status": "instantiated", "len": len(sft), "inspected_count": min(len(sft), TARGET_SCALED), "effective_sft_sample_count": effective, "sampled_items_all_effective": all_effective, "invalid_reason_counts": dict(reasons), "action_counts": dict(action_counts), "valid_examples": examples},
            "status": "passed" if effective > 0 else "blocked_no_effective_samples",
        }
    )
    write_json(name, out)
    return out


def image_feature(obs: list[Any], position: Any, rotation: Any, prev_action_idx: int) -> list[float]:
    features: list[float] = []
    for img in obs[:3]:
        if img is None:
            features.extend([0.0] * 12)
            continue
        rgb = img.convert("RGB")
        stat = ImageStat.Stat(rgb)
        mean = [float(x) / 255.0 for x in stat.mean[:3]]
        std = [float(x) / 255.0 for x in stat.stddev[:3]]
        extrema = stat.extrema[:3]
        mins = [float(x[0]) / 255.0 for x in extrema]
        maxs = [float(x[1]) / 255.0 for x in extrema]
        features.extend(mean + std + mins + maxs)
    pos = list(position) if isinstance(position, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0]
    features.extend([float(x) for x in pos[:3]])
    try:
        features.append(float(rotation))
    except Exception:
        features.append(0.0)
    onehot = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        onehot[prev_action_idx] = 1.0
    features.extend(onehot)
    return features


def collect_samples(root: Path) -> tuple[list[list[float]], list[int], list[int], dict[str, Any]]:
    from utils.dataset import SFTDataset

    args = make_args(root, MAX_EXPERT_STEPS)
    sft = SFTDataset(args, "test")
    x, y, traj_id = [], [], []
    effective, action_counts = 0, Counter()
    for i in range(len(sft)):
        info = inspect_sft_item(sft[i])
        if not info.get("valid"):
            continue
        task, _step_tasks = sft[i]
        effective += 1
        prev = ACTION_TO_IDX["stop"]
        for step in task["Trajectory"]:
            action = step["action"]
            if action not in ACTION_TO_IDX:
                continue
            x.append(image_feature(step["observation"], step["position"], step["rotation"], prev))
            label = ACTION_TO_IDX[action]
            y.append(label)
            traj_id.append(i)
            action_counts[action] += 1
            prev = label
    return x, y, traj_id, {"effective_trajectories": effective, "sample_count": len(y), "action_counts": dict(action_counts)}


def split_by_traj(traj_ids: list[int]) -> dict[str, set[int]]:
    uniq = sorted(set(traj_ids))
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    n = len(uniq)
    return {"train": set(uniq[: int(0.7 * n)]), "val": set(uniq[int(0.7 * n) : int(0.85 * n)]), "test": set(uniq[int(0.85 * n) :])}


def metrics_from_preds(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    per_class = {}
    recalls = []
    f1s = []
    for idx, name in IDX_TO_ACTION.items():
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p == idx)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != idx and p == idx)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p != idx)
        support = sum(1 for y in y_true if y == idx)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {"precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6), "support": support}
        if support:
            recalls.append(rec)
            f1s.append(f1)
    acc = sum(1 for y, p in zip(y_true, y_pred) if y == p) / max(len(y_true), 1)
    return {"accuracy": round(acc, 6), "balanced_accuracy": round(sum(recalls) / len(recalls), 6) if recalls else 0.0, "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0, "per_class": per_class}


class BalancedMLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def train_balanced(root: Path, out_name: str, steps: int) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    x, y, traj_id, meta = collect_samples(root)
    out: dict[str, Any] = {"stage": out_name.replace(".json", ""), "generated_at": now(), "bridge_root": str(root), "status": "not_run", "sample_meta": meta, "requested_steps": steps}
    if len(y) < 100 or len(set(y)) < 2:
        out["status"] = "blocked_not_enough_samples"
        write_json(out_name, out)
        return out
    splits = split_by_traj(traj_id)
    idx_by_split = {k: [i for i, tid in enumerate(traj_id) if tid in v] for k, v in splits.items()}
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    train_idx = torch.tensor(idx_by_split["train"], dtype=torch.long)
    val_idx = torch.tensor(idx_by_split["val"], dtype=torch.long)
    test_idx = torch.tensor(idx_by_split["test"], dtype=torch.long)
    class_counts = Counter(y_t[train_idx].tolist())
    weights = torch.tensor([1.0 / max(class_counts.get(i, 0), 1) for i in range(4)], dtype=torch.float32)
    weights = weights / weights.sum() * 4
    sample_weights = torch.tensor([weights[int(label)].item() for label in y_t[train_idx]], dtype=torch.float32)
    model = BalancedMLP(x_t.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best = {"val_balanced_accuracy": -1.0, "step": 0, "state": None}
    losses = []
    for step in range(1, steps + 1):
        draw = torch.multinomial(sample_weights, num_samples=min(BATCH_SIZE, len(train_idx)), replacement=True)
        batch_idx = train_idx[draw]
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x_t[batch_idx]), y_t[batch_idx])
        loss.backward()
        opt.step()
        if step % 100 == 0:
            losses.append(float(loss.detach().cpu()))
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                pred = model(x_t[val_idx]).argmax(dim=1).tolist() if len(val_idx) else []
            m = metrics_from_preds(y_t[val_idx].tolist(), pred) if len(val_idx) else {"balanced_accuracy": 0.0}
            if m["balanced_accuracy"] > best["val_balanced_accuracy"]:
                best = {"val_balanced_accuracy": m["balanced_accuracy"], "step": step, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            if step % 2000 == 0:
                out.update({"status": "training", "step": step, "latest_loss": float(loss.detach().cpu()), "best_val_balanced_accuracy": best["val_balanced_accuracy"]})
                write_json(out_name + ".progress.json", out)
                print(f"[{now()}] {out_name} step={step} loss={float(loss.detach().cpu()):.4f} best_bal={best['val_balanced_accuracy']:.4f}", flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    with torch.no_grad():
        train_pred = model(x_t[train_idx]).argmax(dim=1).tolist()
        val_pred = model(x_t[val_idx]).argmax(dim=1).tolist() if len(val_idx) else []
        test_pred = model(x_t[test_idx]).argmax(dim=1).tolist() if len(test_idx) else []
    majority = Counter(y_t[train_idx].tolist()).most_common(1)[0][0]
    val_true = y_t[val_idx].tolist()
    test_true = y_t[test_idx].tolist()
    majority_val = [majority] * len(val_true)
    ckpt_name = "tiered_bc_best_100.pt" if "100" in out_name else "tiered_bc_best_scaled.pt"
    model_path = RESULTS / ckpt_name
    torch.save({"model_state_dict": model.state_dict(), "input_dim": x_t.shape[1], "feature": "rgb_stats_pose_prev_action", "action_to_idx": ACTION_TO_IDX}, model_path)
    out.update(
        {
            "status": "completed",
            "model_path": str(model_path),
            "steps_completed": steps,
            "split_counts": {k: len(v) for k, v in idx_by_split.items()},
            "train_metrics": metrics_from_preds(y_t[train_idx].tolist(), train_pred),
            "val_metrics": metrics_from_preds(val_true, val_pred),
            "test_metrics": metrics_from_preds(test_true, test_pred),
            "majority_label": IDX_TO_ACTION[majority],
            "majority_val_metrics": metrics_from_preds(val_true, majority_val),
            "beats_majority_balanced_accuracy": metrics_from_preds(val_true, val_pred)["balanced_accuracy"] > metrics_from_preds(val_true, majority_val)["balanced_accuracy"],
            "loss_tail": [round(v, 6) for v in losses[-20:]],
            "best_step": best["step"],
        }
    )
    write_json(out_name, out)
    return out


def eval_policy(train_out: dict[str, Any], root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"stage": "tiered_bc_eval", "generated_at": now(), "status": "not_run", "eval_root": str(root), "requested_eval_episodes": EVAL_EPISODES, "results": []}
    if train_out.get("status") != "completed" or not train_out.get("model_path"):
        out["status"] = "skipped_no_checkpoint"
        write_json("tiered_bc_eval.json", out)
        return out
    try:
        sys.path.insert(0, str(LH_ROOT))
        os.chdir(LH_ROOT)
        from habitat_base.simulation import SceneSimulator
        from utils.dataset import TaskDataset

        ckpt = torch.load(train_out["model_path"], map_location="cpu")
        model = BalancedMLP(int(ckpt["input_dim"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        install_habitat_sim_api_patch()
        args = make_args(root, MAX_EVAL_STEPS)
        ds = TaskDataset(args, "test")
        successes, oracle_successes, nav_errors, nav_steps = [], [], [], []
        for i in range(min(EVAL_EPISODES, len(ds))):
            config, _steps = ds[i]
            rec = {"index": i, "instruction": config.get("Task instruction"), "scene": config.get("Scene"), "robot": config.get("Robot")}
            sim = None
            try:
                sim = SceneSimulator(args=args, config=config)
                sim.save_path = None
                obs, _done, _info = sim.actor("stop")
                prev = ACTION_TO_IDX["stop"]
                step = 0
                while not sim.episode_over and step < MAX_EVAL_STEPS:
                    pos, yaw = state_record(sim)
                    feat = torch.tensor([image_feature(obs, pos, yaw, prev)], dtype=torch.float32)
                    with torch.no_grad():
                        pred = int(model(feat).argmax(dim=1).item())
                    action = IDX_TO_ACTION[pred]
                    obs, _done, _info = sim.actor(action)
                    prev = pred
                    step += 1
                result = sim.return_results()
                success = bool(result.get("successes")) and all(bool(x) for x in result.get("successes", []))
                oracle = bool(result.get("oracle_successes")) and all(bool(x) for x in result.get("oracle_successes", []))
                ne = result.get("navigation_errors", [])
                ns = result.get("navigation_steps", [])
                rec.update({"status": "completed", "success": success, "oracle_success": oracle, "navigation_error": float(sum(ne) / len(ne)) if ne else None, "navigation_steps": int(sum(ns)) if ns else None})
                successes.append(success)
                oracle_successes.append(oracle)
                if rec["navigation_error"] is not None:
                    nav_errors.append(rec["navigation_error"])
                if rec["navigation_steps"] is not None:
                    nav_steps.append(rec["navigation_steps"])
            except Exception as exc:
                rec.update({"status": "failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]})
            finally:
                if sim is not None:
                    try:
                        sim.close()
                    except Exception:
                        pass
            out["results"].append(rec)
            if (i + 1) % 5 == 0:
                write_json("tiered_bc_eval.progress.json", out)
        completed = sum(1 for r in out["results"] if r.get("status") == "completed")
        out.update({"status": "completed" if completed >= 20 else "partial", "episodes_completed": completed, "metrics": {"SR": round(sum(successes) / len(successes), 4) if successes else 0.0, "OSR": round(sum(oracle_successes) / len(oracle_successes), 4) if oracle_successes else 0.0, "NE": round(sum(nav_errors) / len(nav_errors), 4) if nav_errors else None, "mean_navigation_steps": round(sum(nav_steps) / len(nav_steps), 4) if nav_steps else None, "SPL": None}})
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc().splitlines()[-20:]
    write_json("tiered_bc_eval.json", out)
    return out


def summary(preflight_100: dict[str, Any], train_100: dict[str, Any], scale_gen: dict[str, Any], train_scaled: dict[str, Any], eval_out: dict[str, Any]) -> dict[str, Any]:
    best = train_scaled if train_scaled.get("status") == "completed" else train_100
    out = {
        "stage": "tiered_bc_summary",
        "generated_at": now(),
        "status": "completed" if train_100.get("status") == "completed" else "blocked",
        "run_root": str(RUN_ROOT),
        "results_root": str(RESULTS),
        "summary": {
            "preflight_status": preflight_100.get("status"),
            "effective_sft_sample_count": preflight_100.get("sft_dataset", {}).get("effective_sft_sample_count"),
            "train_100_status": train_100.get("status"),
            "train_100_val_balanced_accuracy": train_100.get("val_metrics", {}).get("balanced_accuracy"),
            "train_100_majority_val_balanced_accuracy": train_100.get("majority_val_metrics", {}).get("balanced_accuracy"),
            "train_100_beats_majority": train_100.get("beats_majority_balanced_accuracy"),
            "scale_generation_status": scale_gen.get("status"),
            "scaled_valid_frame_trajectories": scale_gen.get("valid_frame_trajectories"),
            "train_scaled_status": train_scaled.get("status"),
            "train_scaled_val_balanced_accuracy": train_scaled.get("val_metrics", {}).get("balanced_accuracy"),
            "best_model_path": best.get("model_path"),
            "eval_status": eval_out.get("status"),
            "eval_episodes_completed": eval_out.get("episodes_completed"),
            "eval_metrics": eval_out.get("metrics"),
        },
        "paper_framing": {"claim_strength": "tiered balanced visual BC feasibility", "safe_claim": "IDEN-VLN expert trajectories support offline visual action learning with class-balanced metrics.", "do_not_claim": "competitive navigation agent, PPO/RL, or robust embodied policy performance"},
    }
    write_json("tiered_bc_summary.json", out)
    return out


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] Starting tiered balanced visual BC at {RUN_ROOT}", flush=True)
    gpu_wait = wait_for_gpu()
    print(f"[{now()}] GPU gate status={gpu_wait.get('status')} free={gpu_wait.get('best_gpu_free_mib')} waited={gpu_wait.get('waited_sec')}", flush=True)
    copy = prepare_100_copy()
    pre = preflight(BRIDGE_100_ROOT, "tiered_bc_preflight.json")
    pre["gpu_wait"] = gpu_wait
    pre["bridge_copy"] = copy
    write_json("tiered_bc_preflight.json", pre)
    print(f"[{now()}] Preflight status={pre.get('status')} effective={pre.get('sft_dataset', {}).get('effective_sft_sample_count')}", flush=True)
    train100 = train_balanced(BRIDGE_100_ROOT, "tiered_bc_train_100.json", TRAIN_STEPS_100)
    print(f"[{now()}] Train100 status={train100.get('status')} val_bal={train100.get('val_metrics', {}).get('balanced_accuracy')} majority_bal={train100.get('majority_val_metrics', {}).get('balanced_accuracy')} beats={train100.get('beats_majority_balanced_accuracy')}", flush=True)
    scale_gen: dict[str, Any] = {"stage": "tiered_bc_scale_generation", "status": "skipped_100_not_above_majority", "reason": "train_100 did not beat majority balanced accuracy"}
    train_scaled: dict[str, Any] = {"stage": "tiered_bc_train_scaled", "status": "skipped_scale_generation_not_run"}
    if train100.get("beats_majority_balanced_accuracy"):
        scale_gen = generate_scaled_experts()
        if scale_gen.get("status") == "completed":
            pre_scaled = preflight(BRIDGE_SCALED_ROOT, "tiered_bc_preflight_scaled.json")
            if pre_scaled.get("status") == "passed":
                train_scaled = train_balanced(BRIDGE_SCALED_ROOT, "tiered_bc_train_scaled.json", TRAIN_STEPS_SCALED)
            else:
                train_scaled = {"stage": "tiered_bc_train_scaled", "status": "skipped_scaled_preflight_failed", "preflight": pre_scaled}
                write_json("tiered_bc_train_scaled.json", train_scaled)
        else:
            train_scaled = {"stage": "tiered_bc_train_scaled", "status": "skipped_scale_generation_failed"}
            write_json("tiered_bc_train_scaled.json", train_scaled)
    else:
        write_json("tiered_bc_scale_generation.json", scale_gen)
        write_json("tiered_bc_train_scaled.json", train_scaled)
    best_train = train_scaled if train_scaled.get("status") == "completed" else train100
    eval_root = BRIDGE_SCALED_ROOT if train_scaled.get("status") == "completed" else BRIDGE_100_ROOT
    eval_out = eval_policy(best_train, eval_root)
    print(f"[{now()}] Eval status={eval_out.get('status')} metrics={eval_out.get('metrics')}", flush=True)
    out = summary(pre, train100, scale_gen, train_scaled, eval_out)
    print(f"[{now()}] Summary status={out.get('status')} results={RESULTS}", flush=True)
    return 0 if out.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
