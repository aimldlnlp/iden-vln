#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageStat


LH_ROOT = Path("/home/aimldl/LH-VLN")
SOURCE_RUN = LH_ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706"
SOURCE_DATA = SOURCE_RUN / "data_expert_scaled"
SOURCE_RESULTS = SOURCE_RUN / "results"
RUN_ROOT = LH_ROOT / "runs/idenvln_longrun_target_conditioned_bc_level23_20260706"
DATA_ROOT = SOURCE_DATA
RESULTS = RUN_ROOT / "results"
BATCH = "/batch_generated_level23"

SEED = 52026
MIN_GPU_FREE_MIB = 6000
GPU_POLL_SEC = 60
GPU_WAIT_TIMEOUT_SEC = 24 * 60 * 60
TRAIN_STEPS = 30000
BATCH_SIZE = 128
EVAL_EPISODES = 60
MAX_EVAL_STEPS = 80
HASH_DIM = 128
ACTION_TO_IDX = {"stop": 0, "turn_left": 1, "move_forward": 2, "turn_right": 3}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def write_json(name: str, data: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        used, total, util = int(parts[2]), int(parts[3]), int(parts[4])
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": total - used,
                "utilization_gpu_percent": util,
            }
        )
    code, pout, _ = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    processes = []
    if code == 0:
        for line in pout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                processes.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return {"available": bool(gpus), "gpus": gpus, "processes": processes}


def wait_for_gpu() -> dict[str, Any]:
    started = time.time()
    attempt = 0
    history = []
    while True:
        attempt += 1
        snap = gpu_snapshot()
        best_free = max((g["memory_free_mib"] for g in snap.get("gpus", [])), default=0)
        rec = {
            "stage": "gpu_wait",
            "generated_at": now(),
            "attempt": attempt,
            "min_gpu_free_mib_required": MIN_GPU_FREE_MIB,
            "best_gpu_free_mib": best_free,
            "elapsed_sec": round(time.time() - started, 1),
            "snapshot": snap,
        }
        history.append(rec)
        history = history[-20:]
        write_json("longrun_gpu_wait.json", {"status": "waiting_for_gpu", "gpu_wait": rec, "history": history})
        log(f"GPU wait attempt={attempt} free={best_free} MiB required={MIN_GPU_FREE_MIB}")
        if best_free >= MIN_GPU_FREE_MIB:
            rec["status"] = "passed"
            rec["waited_sec"] = round(time.time() - started, 1)
            write_json("longrun_gpu_wait.json", {"status": "passed", "gpu_wait": rec, "history": history})
            return rec
        if time.time() - started > GPU_WAIT_TIMEOUT_SEC:
            rec["status"] = "timeout"
            rec["waited_sec"] = round(time.time() - started, 1)
            write_json("longrun_gpu_wait.json", {"status": "timeout", "gpu_wait": rec, "history": history})
            return rec
        time.sleep(GPU_POLL_SEC)


def ensure_data() -> dict[str, Any]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DATA.exists():
        out = {"stage": "data_setup", "status": "failed", "reason": f"missing source data: {SOURCE_DATA}"}
        write_json("longrun_data_setup.json", out)
        return out
    trials = sorted(DATA_ROOT.glob("task/batch_generated_level23/*/*/success/trial_1"))
    rec = {
        "stage": "data_setup",
        "status": "completed" if len(trials) >= 300 else "blocked_incomplete_source_data",
        "source": str(SOURCE_DATA),
        "data_root": str(DATA_ROOT),
        "copy_policy": "reuse_source_data_read_only",
        "trial_dirs": len(trials),
    }
    write_json("longrun_data_setup.json", rec)
    return rec


def parse_action_dir_name(name: str) -> tuple[int | None, str | None]:
    match = re.match(r"^(\d+)_(.+)$", name)
    if not match:
        return None, None
    idx = int(match.group(1))
    rest = match.group(2)
    for action in sorted(ACTION_TO_IDX, key=len, reverse=True):
        if rest == action or rest.startswith(action + "_"):
            return idx, action
    return idx, None


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
    text = str(value).strip().lower()
    return [text] if text else []


def normalize_regions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        value = [value]
    return [str(v).strip().lower() for v in value if str(v).strip()]


def stable_hash(text: str, modulo: int) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % modulo


def hash_features(tokens: list[str], dim: int = HASH_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        idx = stable_hash(token, dim)
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def load_task_context(task_json: Path) -> dict[str, Any]:
    try:
        task = read_json(task_json)
    except Exception:
        task = {}
    level = 0
    try:
        level = int(task_json.parents[2].name)
    except Exception:
        try:
            level = int(task.get("Generated level") or 0)
        except Exception:
            level = 0
    objects = object_labels(task.get("Object"))
    regions = normalize_regions(task.get("Region Name") or task.get("Region"))
    instruction = str(task.get("Task instruction") or task_json.parents[1].name)
    robot = str(task.get("Robot") or "").lower()
    scene = str(task.get("Scene") or "")
    trial = task.get("trial", {})
    if isinstance(trial, dict):
        trial = trial.get("trial_1", trial.get("success", trial))
    if not isinstance(trial, dict):
        trial = {}
    return {
        "task": task,
        "level": level,
        "robot": robot,
        "scene": scene,
        "instruction": instruction,
        "objects": objects,
        "regions": regions,
        "pos": trial.get("pos") if isinstance(trial.get("pos"), list) else [],
        "yaw": trial.get("yaw") if isinstance(trial.get("yaw"), list) else [],
    }


def image_feature(frame_dir: Path, position: Any, rotation: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float] | None:
    features: list[float] = []
    for camera in ["left", "front", "right"]:
        path = frame_dir / f"{camera}.png"
        if not path.exists():
            return None
        with Image.open(path) as img:
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
    prev = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        prev[prev_action_idx] = 1.0
    features.extend(prev)
    robot = [1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0]
    features.extend(robot)
    features.append(float(context.get("level") or 0) / 5.0)
    target_tokens = context.get("objects", []) + context.get("regions", [])
    instruction_tokens = tokenize(context.get("instruction"))[:80]
    features.extend(hash_features(target_tokens, HASH_DIM))
    features.extend(hash_features(instruction_tokens, HASH_DIM))
    return features


def collect_samples(root: Path) -> tuple[list[list[float]], list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    trials = sorted(root.glob("task/batch_generated_level23/*/*/success/trial_1"))
    x: list[list[float]] = []
    y: list[int] = []
    traj_id: list[int] = []
    metas: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    per_traj = []
    for tid, trial in enumerate(trials):
        context = load_task_context(trial / "task.json")
        dirs = []
        for child in trial.iterdir():
            if not child.is_dir():
                continue
            idx, action = parse_action_dir_name(child.name)
            if idx is None:
                invalid["unparseable_action_dir"] += 1
                continue
            if action is None:
                invalid["non_policy_action_dir"] += 1
                continue
            dirs.append((idx, action, child))
        dirs.sort(key=lambda item: item[0])
        before = len(y)
        prev = ACTION_TO_IDX["stop"]
        for idx, action, frame_dir in dirs:
            pos = context["pos"][idx] if idx < len(context["pos"]) else [0.0, 0.0, 0.0]
            yaw = context["yaw"][idx] if idx < len(context["yaw"]) else 0.0
            feat = image_feature(frame_dir, pos, yaw, prev, context)
            if feat is None:
                invalid["missing_or_unreadable_camera_png"] += 1
                continue
            label = ACTION_TO_IDX[action]
            x.append(feat)
            y.append(label)
            traj_id.append(tid)
            metas.append({"trial": str(trial.relative_to(root)), "step_index": idx, "action": action, "robot": context["robot"], "level": context["level"], "scene": context["scene"]})
            action_counts[action] += 1
            prev = label
        added = len(y) - before
        if added:
            per_traj.append({"trial": str(trial.relative_to(root)), "samples": added, "robot": context["robot"], "level": context["level"]})
        else:
            invalid["trajectory_without_samples"] += 1
    meta = {
        "collector": "target_conditioned_direct_action_frame_collector",
        "trial_dirs_found": len(trials),
        "effective_trajectories": len(per_traj),
        "sample_count": len(y),
        "action_counts": dict(action_counts),
        "invalid_reason_counts": dict(invalid),
        "example_trajectories": per_traj[:10],
    }
    return x, y, traj_id, metas, meta


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


class TargetConditionedMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 4),
        )

    def forward(self, x):
        return self.net(x)


def train_variant(name: str, x_t: torch.Tensor, y_t: torch.Tensor, idx_by_split: dict[str, list[int]], weight_power: float, stop_penalty: float) -> dict[str, Any]:
    torch.manual_seed(SEED + int(weight_power * 100) + int(stop_penalty * 10))
    train_idx = torch.tensor(idx_by_split["train"], dtype=torch.long)
    val_idx = torch.tensor(idx_by_split["val"], dtype=torch.long)
    test_idx = torch.tensor(idx_by_split["test"], dtype=torch.long)
    class_counts = Counter(y_t[train_idx].tolist())
    weights = torch.tensor([1.0 / (max(class_counts.get(i, 0), 1) ** weight_power) for i in range(4)], dtype=torch.float32)
    weights[ACTION_TO_IDX["stop"]] *= stop_penalty
    weights = weights / weights.sum() * 4
    sample_weights = torch.tensor([weights[int(label)].item() for label in y_t[train_idx]], dtype=torch.float32)
    model = TargetConditionedMLP(x_t.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best = {"score": -1.0, "step": 0, "state": None, "metrics": None}
    losses = []
    for step in range(1, TRAIN_STEPS + 1):
        draw = torch.multinomial(sample_weights, num_samples=min(BATCH_SIZE, len(train_idx)), replacement=True)
        batch_idx = train_idx[draw]
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x_t[batch_idx]), y_t[batch_idx])
        loss.backward()
        opt.step()
        if step % 200 == 0:
            losses.append(float(loss.detach().cpu()))
        if step % 1000 == 0 or step == TRAIN_STEPS:
            with torch.no_grad():
                pred = model(x_t[val_idx]).argmax(dim=1).tolist()
            metrics = metrics_from_preds(y_t[val_idx].tolist(), pred)
            per = metrics["per_class"]
            move_rec = per["move_forward"]["recall"]
            stop_prec = per["stop"]["precision"]
            score = metrics["balanced_accuracy"] + 0.10 * move_rec + 0.05 * stop_prec
            if score > best["score"]:
                best = {"score": score, "step": step, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metrics": metrics}
            if step % 5000 == 0:
                log(f"{name} step={step} loss={float(loss.detach().cpu()):.4f} bal={metrics['balanced_accuracy']:.4f} score={score:.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    with torch.no_grad():
        train_pred = model(x_t[train_idx]).argmax(dim=1).tolist()
        val_pred = model(x_t[val_idx]).argmax(dim=1).tolist()
        test_pred = model(x_t[test_idx]).argmax(dim=1).tolist()
    majority = Counter(y_t[train_idx].tolist()).most_common(1)[0][0]
    val_true = y_t[val_idx].tolist()
    result = {
        "variant": name,
        "status": "completed",
        "weight_power": weight_power,
        "stop_penalty": stop_penalty,
        "steps_completed": TRAIN_STEPS,
        "best_step": best["step"],
        "best_selection_score": round(float(best["score"]), 6),
        "train_metrics": metrics_from_preds(y_t[train_idx].tolist(), train_pred),
        "val_metrics": metrics_from_preds(val_true, val_pred),
        "test_metrics": metrics_from_preds(y_t[test_idx].tolist(), test_pred),
        "majority_label": IDX_TO_ACTION[majority],
        "majority_val_metrics": metrics_from_preds(val_true, [majority] * len(val_true)),
        "loss_tail": [round(v, 6) for v in losses[-20:]],
    }
    ckpt = RESULTS / f"target_conditioned_bc_{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "input_dim": x_t.shape[1], "action_to_idx": ACTION_TO_IDX, "variant": result}, ckpt)
    result["model_path"] = str(ckpt)
    return result


def train_target_conditioned() -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log("Collecting target-conditioned samples")
    x, y, traj_id, _metas, meta = collect_samples(DATA_ROOT)
    out: dict[str, Any] = {"stage": "target_conditioned_bc_train", "generated_at": now(), "status": "not_run", "sample_meta": meta}
    if len(y) < 1000:
        out["status"] = "blocked_not_enough_samples"
        write_json("target_conditioned_bc_train.json", out)
        return out
    splits = split_by_traj(traj_id)
    idx_by_split = {k: [i for i, tid in enumerate(traj_id) if tid in v] for k, v in splits.items()}
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    variants = [
        ("mild_weights", 0.45, 0.60),
        ("strong_weights", 0.80, 0.45),
        ("action_calibrated", 0.35, 0.30),
    ]
    results = []
    for name, weight_power, stop_penalty in variants:
        log(f"Training variant={name}")
        rec = train_variant(name, x_t, y_t, idx_by_split, weight_power, stop_penalty)
        results.append(rec)
        out.update({"status": "training", "completed_variants": results, "split_counts": {k: len(v) for k, v in idx_by_split.items()}})
        write_json("target_conditioned_bc_train.json.progress.json", out)
    best = max(results, key=lambda r: (r["val_metrics"]["balanced_accuracy"], r["val_metrics"]["macro_f1"]))
    out.update(
        {
            "status": "completed",
            "input_dim": int(x_t.shape[1]),
            "split_counts": {k: len(v) for k, v in idx_by_split.items()},
            "variants": results,
            "best_variant": best["variant"],
            "best_model_path": best["model_path"],
            "best_val_metrics": best["val_metrics"],
            "best_test_metrics": best["test_metrics"],
            "majority_val_metrics": best["majority_val_metrics"],
        }
    )
    write_json("target_conditioned_bc_train.json", out)
    return out


def rollout_failure_diagnostic() -> dict[str, Any]:
    eval_path = SOURCE_RESULTS / "tiered_bc_eval.json"
    train_path = SOURCE_RESULTS / "tiered_bc_train_scaled.json"
    out = {"stage": "rollout_failure_diagnostic_longrun", "generated_at": now(), "status": "not_run"}
    if eval_path.exists():
        ev = read_json(eval_path)
        results = ev.get("results", [])
        nav_errors = [r.get("navigation_error") for r in results if isinstance(r.get("navigation_error"), (int, float))]
        steps = [r.get("navigation_steps") for r in results if isinstance(r.get("navigation_steps"), (int, float))]
        by_robot: dict[str, Counter[str]] = defaultdict(Counter)
        for r in results:
            robot = str(r.get("robot") or "unknown")
            by_robot[robot]["episodes"] += 1
            by_robot[robot]["success"] += int(bool(r.get("success")))
            by_robot[robot]["oracle_success"] += int(bool(r.get("oracle_success")))
        out.update(
            {
                "status": "completed",
                "source_eval": str(eval_path),
                "episodes": len(results),
                "eval_metrics": ev.get("metrics"),
                "max_step_rate": round(sum(1 for s in steps if s >= MAX_EVAL_STEPS) / max(len(steps), 1), 6),
                "navigation_error": {
                    "mean": round(sum(nav_errors) / len(nav_errors), 6) if nav_errors else None,
                    "min": round(min(nav_errors), 6) if nav_errors else None,
                    "max": round(max(nav_errors), 6) if nav_errors else None,
                },
                "by_robot": {k: dict(v) for k, v in by_robot.items()},
            }
        )
    if train_path.exists():
        tr = read_json(train_path)
        out["offline_action_counts"] = tr.get("sample_meta", {}).get("action_counts")
        out["offline_val_per_class"] = tr.get("val_metrics", {}).get("per_class")
    out["interpretation"] = "Prior tiered BC improved offline balanced accuracy but failed online, indicating rollout compounding error and weak target/instruction conditioning."
    write_json("rollout_failure_diagnostic_longrun.json", out)
    return out


def expert_oracle_upper_bound() -> dict[str, Any]:
    gen_path = SOURCE_RESULTS / "tiered_bc_scale_generation.json"
    out = {"stage": "expert_oracle_upper_bound", "generated_at": now(), "status": "not_run"}
    trials = sorted(DATA_ROOT.glob("task/batch_generated_level23/*/*/success/trial_1"))
    sample_lengths = []
    for trial in trials[:1000]:
        count = 0
        for child in trial.iterdir():
            if child.is_dir():
                _idx, action = parse_action_dir_name(child.name)
                if action in ACTION_TO_IDX:
                    count += 1
        if count:
            sample_lengths.append(count)
    if gen_path.exists():
        gen = read_json(gen_path)
        attempted = gen.get("attempted") or gen.get("requested") or len(trials)
        successes = gen.get("successes")
        valid = gen.get("valid_frame_trajectories")
        out.update(
            {
                "status": "completed",
                "source_generation": str(gen_path),
                "attempted": attempted,
                "valid_frame_trajectories": valid,
                "expert_successes": successes,
                "valid_trajectory_rate": round(valid / attempted, 6) if attempted and valid is not None else None,
                "expert_success_rate": round(successes / attempted, 6) if attempted and successes is not None else None,
            }
        )
    else:
        out.update({"status": "completed_without_generation_summary", "valid_frame_trajectories": len(trials)})
    out["action_frame_trajectory_count"] = len(trials)
    out["trajectory_action_length"] = {
        "count": len(sample_lengths),
        "mean": round(sum(sample_lengths) / len(sample_lengths), 6) if sample_lengths else None,
        "min": min(sample_lengths) if sample_lengths else None,
        "max": max(sample_lengths) if sample_lengths else None,
    }
    out["interpretation"] = "This is an expert/generated-trajectory upper bound for data feasibility, not a learned-policy result."
    write_json("expert_oracle_upper_bound.json", out)
    return out


def longrun_expert_generation_note() -> dict[str, Any]:
    out = {
        "stage": "longrun_expert_generation",
        "generated_at": now(),
        "status": "not_run_reuse_existing_300",
        "reason": "Existing 300 valid expert trajectories are reused for this long run; additional generation is left optional to avoid destroying or duplicating known-good data in the first target-conditioned experiment.",
        "existing_data_root": str(DATA_ROOT),
    }
    write_json("longrun_expert_generation.json", out)
    return out


def make_args(max_step: int = MAX_EVAL_STEPS):
    from types import SimpleNamespace

    return SimpleNamespace(
        task_data=str(DATA_ROOT / "task"),
        step_task_data=str(DATA_ROOT / "step_task"),
        episode_data=str(DATA_ROOT / "episode_task"),
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


def install_habitat_patch():
    from habitat_base.simulation import SceneSimulator

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


def state_record(sim):
    state = sim.agent.get_state()
    pos = list(state.position)
    rot = state.rotation
    try:
        yaw = float(rot.angle())
    except Exception:
        yaw = 0.0
    return pos, yaw


def policy_action(logits: torch.Tensor, mode: str, recent: list[int], step: int) -> int:
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(float)
    if mode == "stop_threshold" and probs[ACTION_TO_IDX["stop"]] < 0.75 and step < 12:
        probs[ACTION_TO_IDX["stop"]] = 0.0
    if mode == "anti_turn_loop" and len(recent) >= 8:
        tail = recent[-8:]
        if all(a in (ACTION_TO_IDX["turn_left"], ACTION_TO_IDX["turn_right"]) for a in tail):
            return ACTION_TO_IDX["move_forward"]
    return int(np.argmax(probs))


def eval_policy(train_out: dict[str, Any]) -> dict[str, Any]:
    out = {"stage": "target_conditioned_bc_eval", "generated_at": now(), "status": "not_run", "requested_eval_episodes": EVAL_EPISODES, "variants": {}}
    if train_out.get("status") != "completed":
        out["status"] = "skipped_train_not_completed"
        write_json("target_conditioned_bc_eval.json", out)
        return out
    try:
        sys.path.insert(0, str(LH_ROOT))
        os.chdir(LH_ROOT)
        from habitat_base.simulation import SceneSimulator
        from utils.dataset import TaskDataset

        install_habitat_patch()
        ckpt = torch.load(train_out["best_model_path"], map_location="cpu")
        model = TargetConditionedMLP(int(ckpt["input_dim"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        args = make_args(MAX_EVAL_STEPS)
        ds = TaskDataset(args, "test")
        modes = ["raw", "anti_turn_loop", "stop_threshold"]
        for mode in modes:
            successes, oracle_successes, nav_errors, nav_steps = [], [], [], []
            action_counts: Counter[str] = Counter()
            repeated_turn_loop_episodes = 0
            results = []
            for i in range(min(EVAL_EPISODES, len(ds))):
                config, _steps = ds[i]
                trial_dir = DATA_ROOT / str(config.get("trajectory path", "")).strip("/")
                context = load_task_context(trial_dir / "task.json")
                rec = {"index": i, "instruction": config.get("Task instruction"), "scene": config.get("Scene"), "robot": config.get("Robot"), "mode": mode}
                sim = None
                recent: list[int] = []
                try:
                    sim = SceneSimulator(args=args, config=config)
                    sim.save_path = None
                    obs, _done, _info = sim.actor("stop")
                    prev = ACTION_TO_IDX["stop"]
                    step = 0
                    while not sim.episode_over and step < MAX_EVAL_STEPS:
                        pos, yaw = state_record(sim)
                        # Use live observations saved to temporary feature path via PIL objects.
                        # Reuse image stats directly from simulator observations when possible.
                        feat = live_feature(obs, pos, yaw, prev, context)
                        logits = model(torch.tensor([feat], dtype=torch.float32))[0]
                        pred = policy_action(logits, mode, recent, step)
                        action = IDX_TO_ACTION[pred]
                        obs, _done, _info = sim.actor(action)
                        action_counts[action] += 1
                        recent.append(pred)
                        recent = recent[-16:]
                        prev = pred
                        step += 1
                    if len(recent) >= 8 and all(a in (ACTION_TO_IDX["turn_left"], ACTION_TO_IDX["turn_right"]) for a in recent[-8:]):
                        repeated_turn_loop_episodes += 1
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
                results.append(rec)
                if (i + 1) % 10 == 0:
                    out["variants"][mode] = {"partial_results": results, "action_counts": dict(action_counts)}
                    write_json("target_conditioned_bc_eval.progress.json", out)
            completed = sum(1 for r in results if r.get("status") == "completed")
            out["variants"][mode] = {
                "status": "completed" if completed >= 20 else "partial",
                "episodes_completed": completed,
                "metrics": {
                    "SR": round(sum(successes) / len(successes), 4) if successes else 0.0,
                    "OSR": round(sum(oracle_successes) / len(oracle_successes), 4) if oracle_successes else 0.0,
                    "NE": round(sum(nav_errors) / len(nav_errors), 4) if nav_errors else None,
                    "mean_navigation_steps": round(sum(nav_steps) / len(nav_steps), 4) if nav_steps else None,
                    "SPL": None,
                },
                "action_counts": dict(action_counts),
                "repeated_turn_loop_episodes": repeated_turn_loop_episodes,
                "results": results,
            }
        out["status"] = "completed"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc().splitlines()[-30:]
    write_json("target_conditioned_bc_eval.json", out)
    return out


def live_feature(obs: list[Any], position: Any, rotation: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float]:
    features: list[float] = []
    for img in list(obs or [])[:3]:
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
    while len(features) < 36:
        features.extend([0.0] * 12)
    pos = list(position) if isinstance(position, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0]
    features.extend([float(x) for x in pos[:3]])
    try:
        features.append(float(rotation))
    except Exception:
        features.append(0.0)
    prev = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        prev[prev_action_idx] = 1.0
    features.extend(prev)
    features.extend([1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0])
    features.append(float(context.get("level") or 0) / 5.0)
    features.extend(hash_features(context.get("objects", []) + context.get("regions", []), HASH_DIM))
    features.extend(hash_features(tokenize(context.get("instruction"))[:80], HASH_DIM))
    return features


def summary(diag: dict[str, Any], oracle: dict[str, Any], gen: dict[str, Any], train: dict[str, Any], eval_out: dict[str, Any]) -> dict[str, Any]:
    out = {
        "stage": "longrun_agent_summary",
        "generated_at": now(),
        "status": "completed" if train.get("status") == "completed" else "blocked",
        "run_root": str(RUN_ROOT),
        "data_root": str(DATA_ROOT),
        "summary": {
            "source_300_reused": True,
            "expert_oracle_status": oracle.get("status"),
            "expert_success_rate": oracle.get("expert_success_rate"),
            "generation_status": gen.get("status"),
            "train_status": train.get("status"),
            "best_variant": train.get("best_variant"),
            "best_val_balanced_accuracy": train.get("best_val_metrics", {}).get("balanced_accuracy"),
            "best_test_balanced_accuracy": train.get("best_test_metrics", {}).get("balanced_accuracy"),
            "majority_val_balanced_accuracy": train.get("majority_val_metrics", {}).get("balanced_accuracy"),
            "eval_status": eval_out.get("status"),
            "eval_variants": {k: v.get("metrics") for k, v in eval_out.get("variants", {}).items()},
        },
        "paper_framing": {
            "claim_strength": "target-conditioned visual BC feasibility plus expert/oracle upper-bound and failure analysis",
            "safe_claim": "Long-run target-conditioned BC tests whether IDEN-VLN expert trajectories support stronger offline visual action learning and structured online-rollout diagnostics.",
            "do_not_claim": "competitive navigation-agent performance, robust RL/PPO/MGDM, or solved embodied navigation.",
        },
    }
    write_json("longrun_agent_summary.json", out)
    return out


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log(f"Starting long-run target-conditioned BC at {RUN_ROOT}")
    setup = ensure_data()
    if setup.get("status") != "completed":
        return 2
    gpu = wait_for_gpu()
    if gpu.get("status") != "passed":
        return 2
    diag = rollout_failure_diagnostic()
    oracle = expert_oracle_upper_bound()
    gen = longrun_expert_generation_note()
    train = train_target_conditioned()
    eval_out = eval_policy(train)
    out = summary(diag, oracle, gen, train, eval_out)
    log(f"Completed summary status={out.get('status')} results={RESULTS}")
    return 0 if out.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
