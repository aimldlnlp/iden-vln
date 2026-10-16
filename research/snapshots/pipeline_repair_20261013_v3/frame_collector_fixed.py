#!/usr/bin/env python3
from __future__ import annotations
from repair_contract import before_action, best_snapshot, signed_yaw

import importlib.util
import json
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
BASE_PATH = RUN_DIR / "frame_trainer_fixed.py"
spec = importlib.util.spec_from_file_location("tiered_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

base.MIN_GPU_FREE_MIB = 6000
base.GPU_POLL_SEC = 60
base.TRAIN_STEPS_SCALED = 20000

ACTION_NAMES = sorted(base.ACTION_TO_IDX.keys(), key=len, reverse=True)


def log(msg: str) -> None:
    print(f"[{base.now()}] {msg}", flush=True)


def wait_for_gpu_resume() -> dict[str, Any]:
    started = time.time()
    attempt = 0
    history = []
    while True:
        attempt += 1
        snap = base.gpu_snapshot()
        best_free = max((gpu["memory_free_mib"] for gpu in snap.get("gpus", [])), default=0)
        rec = {
            "stage": "tiered_bc_resume_gpu_wait",
            "generated_at": base.now(),
            "attempt": attempt,
            "min_gpu_free_mib_required": base.MIN_GPU_FREE_MIB,
            "best_gpu_free_mib": best_free,
            "elapsed_sec": round(time.time() - started, 1),
            "snapshot": snap,
        }
        history.append(rec)
        history = history[-20:]
        base.write_json(
            "tiered_bc_resume_gpu_wait.json",
            {"status": "waiting_for_gpu", "gpu_wait": rec, "recent_gpu_wait_history": history},
        )
        log(f"GPU wait attempt={attempt} best_free={best_free} MiB required={base.MIN_GPU_FREE_MIB}")
        if best_free >= base.MIN_GPU_FREE_MIB:
            rec["status"] = "passed"
            rec["waited_sec"] = round(time.time() - started, 1)
            base.write_json(
                "tiered_bc_resume_gpu_wait.json",
                {"status": "passed", "gpu_wait": rec, "recent_gpu_wait_history": history},
            )
            return rec
        if time.time() - started > base.GPU_WAIT_TIMEOUT_SEC:
            rec["status"] = "timeout"
            rec["waited_sec"] = round(time.time() - started, 1)
            base.write_json(
                "tiered_bc_resume_gpu_wait.json",
                {"status": "timeout", "gpu_wait": rec, "recent_gpu_wait_history": history},
            )
            return rec
        time.sleep(base.GPU_POLL_SEC)


def parse_action_dir_name(name: str) -> tuple[int | None, str | None]:
    match = re.match(r"^(\d+)_(.+)$", name)
    if not match:
        return None, None
    step_idx = int(match.group(1))
    rest = match.group(2)
    for action in ACTION_NAMES:
        if rest == action or rest.startswith(action + "_"):
            return step_idx, action
    return step_idx, None


def load_trial_pose(task_json: Path) -> tuple[list[Any], list[Any]]:
    try:
        task = json.loads(task_json.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    trial = task.get("trial", {})
    if isinstance(trial, dict):
        trial = trial.get("trial_1", trial.get("success", trial))
    if not isinstance(trial, dict):
        return [], []
    pos = trial.get("pos") or trial.get("position") or []
    yaw = trial.get("yaw") or trial.get("rotation") or []
    return pos if isinstance(pos, list) else [], yaw if isinstance(yaw, list) else []


def open_triplet(frame_dir: Path):
    obs = []
    for camera in ["left", "front", "right"]:
        path = frame_dir / f"{camera}.png"
        if not path.exists():
            return None
        with Image.open(path) as img:
            obs.append(img.convert("RGB").copy())
    return obs


def collect_samples_direct(root: Path):
    random.seed(base.SEED)
    trials = sorted(root.glob("task/batch_generated_level23/*/*/success/trial_1"))
    x, y, traj_id = [], [], []
    action_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    valid_trajectories = 0
    trajectory_summaries = []

    for tid, trial in enumerate(trials):
        task_json = trial / "task.json"
        pos_list, yaw_list = load_trial_pose(task_json)
        action_dirs = []
        for child in trial.iterdir():
            if not child.is_dir():
                continue
            step_idx, action = parse_action_dir_name(child.name)
            if step_idx is None:
                invalid_reasons["unparseable_action_dir"] += 1
                continue
            if action is None:
                invalid_reasons["non_policy_action_dir"] += 1
                continue
            if not all((child / f"{cam}.png").exists() for cam in ["left", "front", "right"]):
                invalid_reasons["missing_camera_png"] += 1
                continue
            action_dirs.append((step_idx, action, child))

        action_dirs.sort(key=lambda item: item[0])
        prev = base.ACTION_TO_IDX["stop"]
        sample_before = len(y)
        for step_idx, action, frame_dir in action_dirs:
            frame_dir, position, rotation = before_action(trial, step_idx, pos_list, yaw_list)
            obs = open_triplet(frame_dir)
            if obs is None:
                invalid_reasons["failed_open_triplet"] += 1
                continue
            label = base.ACTION_TO_IDX[action]
            x.append(base.image_feature(obs, position, rotation, prev))
            y.append(label)
            traj_id.append(tid)
            action_counts[action] += 1
            prev = label

        added = len(y) - sample_before
        if added > 0:
            valid_trajectories += 1
            if len(trajectory_summaries) < 10:
                trajectory_summaries.append({"trial": str(trial.relative_to(root)), "samples": added})
        else:
            invalid_reasons["trajectory_without_policy_samples"] += 1

    meta = {
        "collector": "direct_action_frame_collector_no_sftdataset_getitem",
        "trial_dirs_found": len(trials),
        "effective_trajectories": valid_trajectories,
        "sample_count": len(y),
        "action_counts": dict(action_counts),
        "invalid_reason_counts": dict(invalid_reasons),
        "example_trajectories": trajectory_summaries,
    }
    return x, y, traj_id, meta


base.collect_samples = collect_samples_direct


def read_result(name: str) -> dict[str, Any]:
    path = base.RESULTS / name
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    base.RESULTS.mkdir(parents=True, exist_ok=True)
    log("Starting resume scaled tiered balanced visual BC")

    train100 = read_result("tiered_bc_train_100.json")
    scale_gen = read_result("tiered_bc_scale_generation.json")
    pre = read_result("tiered_bc_preflight.json")
    if train100.get("status") != "completed":
        raise RuntimeError("tiered_bc_train_100.json is not completed")
    if scale_gen.get("status") != "completed":
        raise RuntimeError("tiered_bc_scale_generation.json is not completed")

    log(
        "Reusing completed train100 "
        f"val_bal={train100.get('val_metrics', {}).get('balanced_accuracy')} "
        f"and scale valid={scale_gen.get('valid_frame_trajectories')}"
    )
    gpu_wait = wait_for_gpu_resume()
    if gpu_wait.get("status") != "passed":
        base.write_json(
            "tiered_bc_train_scaled.json",
            {"stage": "tiered_bc_train_scaled", "status": "blocked_gpu_wait", "gpu_wait": gpu_wait},
        )
        return 2

    log("Collecting scaled direct action-frame samples")
    try:
        train_scaled = base.train_balanced(
            base.BRIDGE_SCALED_ROOT,
            "tiered_bc_train_scaled.json",
            base.TRAIN_STEPS_SCALED,
        )
    except Exception as exc:
        train_scaled = {
            "stage": "tiered_bc_train_scaled",
            "generated_at": base.now(),
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc().splitlines()[-40:],
            "collector": "direct_action_frame_collector_no_sftdataset_getitem",
        }
        base.write_json("tiered_bc_train_scaled.json", train_scaled)
        log(f"Scaled training failed: {exc!r}")

    log(
        f"Scaled training status={train_scaled.get('status')} "
        f"val_bal={train_scaled.get('val_metrics', {}).get('balanced_accuracy')}"
    )
    best_train = train_scaled if train_scaled.get("status") == "completed" else train100
    eval_root = base.BRIDGE_SCALED_ROOT if train_scaled.get("status") == "completed" else base.BRIDGE_100_ROOT
    eval_out = base.eval_policy(best_train, eval_root)
    log(f"Eval status={eval_out.get('status')} metrics={eval_out.get('metrics')}")

    summary = base.summary(pre, train100, scale_gen, train_scaled, eval_out)
    summary["summary"]["resume_gpu_gate_mib"] = base.MIN_GPU_FREE_MIB
    summary["summary"]["resume_gpu_wait"] = gpu_wait
    summary["paper_framing"]["claim_strength"] = (
        "tiered balanced visual BC feasibility; not competitive RL/navigation"
    )
    base.write_json("tiered_bc_summary.json", summary)
    log(f"Summary status={summary.get('status')} results={base.RESULTS}")
    return 0 if train_scaled.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit("Use run_repair_pilot.py; historical generation entry point is disabled")
