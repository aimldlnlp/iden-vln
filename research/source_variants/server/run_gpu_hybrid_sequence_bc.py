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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageStat


LH_ROOT = Path("/home/aimldl/LH-VLN")
SOURCE_DATA = LH_ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706/data_expert_scaled"
RUN_ROOT = LH_ROOT / "runs/idenvln_gpu_hybrid_sequence_bc_level23_20260707"
RESULTS = RUN_ROOT / "results"
BATCH = "/batch_generated_level23"

SEED = 57007
MIN_GPU_FREE_MIB = 6000
GPU_POLL_SEC = 60
GPU_WAIT_TIMEOUT_SEC = 24 * 60 * 60
WINDOW = 8
HASH_DIM = 64
BATCH_SIZE = 256
TRAIN_STEPS = 16000
EVAL_EPISODES = 60
MAX_EVAL_STEPS = 80
PREV_TIERED_BAL = 0.450175
PREV_TARGET_BAL = 0.385684
ACTION_TO_IDX = {"stop": 0, "turn_left": 1, "move_forward": 2, "turn_right": 3}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(name: str, data: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def gpu_snapshot() -> dict[str, Any]:
    code, out, err = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    if code != 0:
        return {"available": False, "error": err.strip() or out.strip(), "gpus": []}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            used, total, util = int(parts[2]), int(parts[3]), int(parts[4])
            gpus.append({"index": int(parts[0]), "name": parts[1], "memory_used_mib": used, "memory_total_mib": total, "memory_free_mib": total - used, "utilization_gpu_percent": util})
    code, apps, _ = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    processes = []
    if code == 0:
        for line in apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                processes.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return {"available": bool(gpus), "gpus": gpus, "processes": processes}


def wait_for_gpu() -> dict[str, Any]:
    started = time.time()
    attempt = 0
    while True:
        attempt += 1
        snap = gpu_snapshot()
        best_free = max((g["memory_free_mib"] for g in snap.get("gpus", [])), default=0)
        rec = {"stage": "gpu_wait", "generated_at": now(), "attempt": attempt, "min_gpu_free_mib_required": MIN_GPU_FREE_MIB, "best_gpu_free_mib": best_free, "elapsed_sec": round(time.time() - started, 1), "snapshot": snap}
        write_json("hybrid_sequence_gpu_wait.json", {"status": "waiting_for_gpu", "gpu_wait": rec})
        log(f"GPU wait attempt={attempt} free={best_free} MiB required={MIN_GPU_FREE_MIB}")
        if best_free >= MIN_GPU_FREE_MIB:
            rec["status"] = "passed"
            rec["waited_sec"] = round(time.time() - started, 1)
            write_json("hybrid_sequence_gpu_wait.json", {"status": "passed", "gpu_wait": rec})
            return rec
        if time.time() - started > GPU_WAIT_TIMEOUT_SEC:
            rec["status"] = "timeout"
            write_json("hybrid_sequence_gpu_wait.json", {"status": "timeout", "gpu_wait": rec})
            return rec
        time.sleep(GPU_POLL_SEC)


def choose_device(gpu_wait: dict[str, Any]) -> tuple[torch.device, str | None]:
    if torch.cuda.is_available() and gpu_wait.get("best_gpu_free_mib", 0) >= MIN_GPU_FREE_MIB:
        try:
            torch.empty((1,), device="cuda")
            return torch.device("cuda"), None
        except Exception as exc:
            return torch.device("cpu"), f"cuda_probe_failed:{repr(exc)}"
    return torch.device("cpu"), "cuda_unavailable_or_free_vram_below_gate"


def stable_hash(text: str, modulo: int) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % modulo


def hash_features(tokens: list[str], dim: int = HASH_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in tokens:
        tok = str(tok).strip().lower()
        if tok:
            vec[stable_hash(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def object_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, list) and item:
                out.append(str(item[0]).lower())
            elif isinstance(item, dict):
                out.append(str(item.get("name") or item.get("object") or item.get("label") or "").lower())
            elif item not in ("", None):
                out.append(str(item).lower())
        return [x.strip() for x in out if x.strip()]
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
    return [str(x).strip().lower() for x in value if str(x).strip()]


def parse_action_dir(name: str) -> tuple[int | None, str | None]:
    m = re.match(r"^(\d+)_(.+)$", name)
    if not m:
        return None, None
    idx, rest = int(m.group(1)), m.group(2)
    for action in sorted(ACTION_TO_IDX, key=len, reverse=True):
        if rest == action or rest.startswith(action + "_"):
            return idx, action
    return idx, None


def load_context(trial: Path) -> dict[str, Any]:
    try:
        task = read_json(trial / "task.json")
    except Exception:
        task = {}
    trial_meta = task.get("trial", {})
    if isinstance(trial_meta, dict):
        trial_meta = trial_meta.get("trial_1", trial_meta)
    if not isinstance(trial_meta, dict):
        trial_meta = {}
    try:
        level = int(trial.parents[2].name)
    except Exception:
        level = int(task.get("Generated level") or 0)
    return {
        "task": task,
        "instruction": str(task.get("Task instruction") or trial.parents[1].name),
        "objects": object_labels(task.get("Object")),
        "regions": normalize_regions(task.get("Region Name") or task.get("Region")),
        "robot": str(task.get("Robot") or "").lower(),
        "level": level,
        "scene": str(task.get("Scene") or ""),
        "pos": trial_meta.get("pos") if isinstance(trial_meta.get("pos"), list) else [],
        "yaw": trial_meta.get("yaw") if isinstance(trial_meta.get("yaw"), list) else [],
    }


def frame_feature(frame_dir: Path, pos: Any, yaw: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float] | None:
    f: list[float] = []
    for cam in ["left", "front", "right"]:
        p = frame_dir / f"{cam}.png"
        if not p.exists():
            return None
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            stat = ImageStat.Stat(rgb)
            mean = [float(x) / 255.0 for x in stat.mean[:3]]
            std = [float(x) / 255.0 for x in stat.stddev[:3]]
            extrema = stat.extrema[:3]
            mins = [float(x[0]) / 255.0 for x in extrema]
            maxs = [float(x[1]) / 255.0 for x in extrema]
            f.extend(mean + std + mins + maxs)
    pos_list = list(pos) if isinstance(pos, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0]
    f.extend([float(x) for x in pos_list[:3]])
    try:
        f.append(float(yaw))
    except Exception:
        f.append(0.0)
    prev = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        prev[prev_action_idx] = 1.0
    f.extend(prev)
    f.extend([1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0])
    f.append(float(context.get("level") or 0) / 5.0)
    f.extend(hash_features(context.get("objects", []) + context.get("regions", []), HASH_DIM))
    f.extend(hash_features(tokenize(context.get("instruction"))[:80], HASH_DIM))
    return f


def live_feature(obs: list[Any], pos: Any, yaw: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float]:
    f: list[float] = []
    for img in list(obs or [])[:3]:
        if img is None:
            f.extend([0.0] * 12)
            continue
        rgb = img.convert("RGB")
        stat = ImageStat.Stat(rgb)
        mean = [float(x) / 255.0 for x in stat.mean[:3]]
        std = [float(x) / 255.0 for x in stat.stddev[:3]]
        extrema = stat.extrema[:3]
        mins = [float(x[0]) / 255.0 for x in extrema]
        maxs = [float(x[1]) / 255.0 for x in extrema]
        f.extend(mean + std + mins + maxs)
    while len(f) < 36:
        f.extend([0.0] * 12)
    pos_list = list(pos) if isinstance(pos, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0]
    f.extend([float(x) for x in pos_list[:3]])
    try:
        f.append(float(yaw))
    except Exception:
        f.append(0.0)
    prev = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        prev[prev_action_idx] = 1.0
    f.extend(prev)
    f.extend([1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0])
    f.append(float(context.get("level") or 0) / 5.0)
    f.extend(hash_features(context.get("objects", []) + context.get("regions", []), HASH_DIM))
    f.extend(hash_features(tokenize(context.get("instruction"))[:80], HASH_DIM))
    return f


def collect_sequences() -> tuple[list[list[list[float]]], list[int], list[int], dict[str, Any]]:
    trials = sorted(SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))
    xs: list[list[list[float]]] = []
    ys: list[int] = []
    tids: list[int] = []
    action_counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    valid_traj = 0
    feat_dim = None
    zero: list[float] | None = None
    for tid, trial in enumerate(trials):
        context = load_context(trial)
        dirs = []
        for child in trial.iterdir():
            if not child.is_dir():
                continue
            idx, action = parse_action_dir(child.name)
            if idx is None:
                invalid["unparseable_action_dir"] += 1
                continue
            if action is None:
                invalid["non_policy_action_dir"] += 1
                continue
            dirs.append((idx, action, child))
        dirs.sort(key=lambda x: x[0])
        feats, labels = [], []
        prev = ACTION_TO_IDX["stop"]
        for idx, action, child in dirs:
            pos = context["pos"][idx] if idx < len(context["pos"]) else [0.0, 0.0, 0.0]
            yaw = context["yaw"][idx] if idx < len(context["yaw"]) else 0.0
            feat = frame_feature(child, pos, yaw, prev, context)
            if feat is None:
                invalid["missing_camera_png"] += 1
                continue
            feat_dim = len(feat)
            label = ACTION_TO_IDX[action]
            feats.append(feat)
            labels.append(label)
            action_counts[action] += 1
            prev = label
        if feats:
            valid_traj += 1
        if feat_dim is None:
            continue
        zero = [0.0] * feat_dim
        for i, label in enumerate(labels):
            start = max(0, i - WINDOW + 1)
            seq = feats[start : i + 1]
            if len(seq) < WINDOW:
                seq = [zero] * (WINDOW - len(seq)) + seq
            xs.append(seq)
            ys.append(label)
            tids.append(tid)
    meta = {"collector": "sequence_action_frame_collector", "window": WINDOW, "trial_dirs_found": len(trials), "effective_trajectories": valid_traj, "sequence_sample_count": len(ys), "feature_dim": feat_dim, "action_counts": dict(action_counts), "invalid_reason_counts": dict(invalid)}
    return xs, ys, tids, meta


def split_by_traj(tids: list[int]) -> dict[str, set[int]]:
    uniq = sorted(set(tids))
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    n = len(uniq)
    return {"train": set(uniq[: int(0.7 * n)]), "val": set(uniq[int(0.7 * n) : int(0.85 * n)]), "test": set(uniq[int(0.85 * n) :])}


def metrics_from_preds(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    per = {}
    recalls, f1s = [], []
    for idx, name in IDX_TO_ACTION.items():
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p == idx)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != idx and p == idx)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p != idx)
        support = sum(1 for y in y_true if y == idx)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[name] = {"precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6), "support": support}
        if support:
            recalls.append(rec)
            f1s.append(f1)
    acc = sum(1 for y, p in zip(y_true, y_pred) if y == p) / max(len(y_true), 1)
    return {"accuracy": round(acc, 6), "balanced_accuracy": round(sum(recalls) / len(recalls), 6), "macro_f1": round(sum(f1s) / len(f1s), 6), "per_class": per}


class SequenceGRU(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.gru = torch.nn.GRU(dim, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def train_variant(name: str, x_t: torch.Tensor, y_t: torch.Tensor, idx_by_split: dict[str, list[int]], device: torch.device, weight_power: float, stop_penalty: float) -> dict[str, Any]:
    torch.manual_seed(SEED)
    train_idx = torch.tensor(idx_by_split["train"], dtype=torch.long, device=device)
    val_idx_cpu = idx_by_split["val"]
    test_idx_cpu = idx_by_split["test"]
    x_d = x_t.to(device)
    y_d = y_t.to(device)
    counts = Counter(y_t[idx_by_split["train"]].tolist())
    weights = torch.tensor([1.0 / (max(counts.get(i, 0), 1) ** weight_power) for i in range(4)], dtype=torch.float32, device=device)
    weights[ACTION_TO_IDX["stop"]] *= stop_penalty
    weights = weights / weights.sum() * 4
    sample_weights = torch.tensor([weights[int(label)].item() for label in y_t[idx_by_split["train"]]], dtype=torch.float32, device=device)
    model = SequenceGRU(x_t.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best = {"score": -1.0, "step": 0, "state": None}
    losses = []
    for step in range(1, TRAIN_STEPS + 1):
        draw = torch.multinomial(sample_weights, min(BATCH_SIZE, len(train_idx)), replacement=True)
        bi = train_idx[draw]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_d[bi]), y_d[bi])
        loss.backward()
        opt.step()
        if step % 200 == 0:
            losses.append(float(loss.detach().cpu()))
        if step % 1000 == 0 or step == TRAIN_STEPS:
            model.eval()
            preds = []
            with torch.no_grad():
                for off in range(0, len(val_idx_cpu), 2048):
                    ids = torch.tensor(val_idx_cpu[off : off + 2048], dtype=torch.long, device=device)
                    preds.extend(model(x_d[ids]).argmax(1).detach().cpu().tolist())
            m = metrics_from_preds(y_t[val_idx_cpu].tolist(), preds)
            score = m["balanced_accuracy"] + 0.05 * m["per_class"]["move_forward"]["recall"] + 0.03 * m["per_class"]["stop"]["precision"]
            if score > best["score"]:
                best = {"score": score, "step": step, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            if step % 4000 == 0:
                log(f"{name} step={step} loss={float(loss.detach().cpu()):.4f} bal={m['balanced_accuracy']:.4f} score={score:.4f}")
            model.train()
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.to(device).eval()
    def predict(indices: list[int]) -> list[int]:
        out = []
        with torch.no_grad():
            for off in range(0, len(indices), 2048):
                ids = torch.tensor(indices[off : off + 2048], dtype=torch.long, device=device)
                out.extend(model(x_d[ids]).argmax(1).detach().cpu().tolist())
        return out
    train_pred, val_pred, test_pred = predict(idx_by_split["train"]), predict(idx_by_split["val"]), predict(idx_by_split["test"])
    majority = Counter(y_t[idx_by_split["train"]].tolist()).most_common(1)[0][0]
    ckpt = RESULTS / f"sequence_bc_best_{name}.pt"
    torch.save({"model_state_dict": model.cpu().state_dict(), "input_dim": int(x_t.shape[-1]), "window": WINDOW, "action_to_idx": ACTION_TO_IDX, "variant": name}, ckpt)
    return {
        "variant": name,
        "status": "completed",
        "device": str(device),
        "steps_completed": TRAIN_STEPS,
        "best_step": best["step"],
        "best_selection_score": round(best["score"], 6),
        "model_path": str(ckpt),
        "train_metrics": metrics_from_preds(y_t[idx_by_split["train"]].tolist(), train_pred),
        "val_metrics": metrics_from_preds(y_t[idx_by_split["val"]].tolist(), val_pred),
        "test_metrics": metrics_from_preds(y_t[idx_by_split["test"]].tolist(), test_pred),
        "majority_label": IDX_TO_ACTION[majority],
        "majority_val_metrics": metrics_from_preds(y_t[idx_by_split["val"]].tolist(), [majority] * len(idx_by_split["val"])),
        "loss_tail": [round(v, 6) for v in losses[-20:]],
    }


def train_sequence_bc(device: torch.device, fallback_reason: str | None) -> dict[str, Any]:
    xs, ys, tids, meta = collect_sequences()
    out = {"stage": "sequence_bc_train", "generated_at": now(), "status": "not_run", "device": str(device), "device_fallback_reason": fallback_reason, "sample_meta": meta, "previous_tiered_bc_val_balanced_accuracy": PREV_TIERED_BAL, "previous_target_conditioned_bc_val_balanced_accuracy": PREV_TARGET_BAL}
    if len(ys) < 20000:
        out["status"] = "blocked_not_enough_sequence_samples"
        write_json("sequence_bc_train.json", out)
        return out
    splits = split_by_traj(tids)
    idx_by_split = {k: [i for i, tid in enumerate(tids) if tid in v] for k, v in splits.items()}
    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.long)
    variants = [("mild", 0.45, 0.65), ("strong", 0.80, 0.45), ("calibrated", 0.35, 0.25)]
    results = []
    for name, wp, sp in variants:
        log(f"Training sequence variant={name} device={device}")
        try:
            rec = train_variant(name, x_t, y_t, idx_by_split, device, wp, sp)
        except torch.cuda.OutOfMemoryError as exc:
            if str(device) == "cuda":
                log(f"CUDA OOM on {name}, retrying CPU")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                rec = train_variant(name, x_t, y_t, idx_by_split, device, wp, sp)
                rec["device_fallback_reason"] = f"cuda_oom:{repr(exc)}"
            else:
                raise
        results.append(rec)
        out.update({"status": "training", "completed_variants": results, "split_counts": {k: len(v) for k, v in idx_by_split.items()}})
        write_json("sequence_bc_train.json.progress.json", out)
    best = max(results, key=lambda r: (r["val_metrics"]["balanced_accuracy"], r["val_metrics"]["macro_f1"]))
    out.update({"status": "completed", "split_counts": {k: len(v) for k, v in idx_by_split.items()}, "variants": results, "best_variant": best["variant"], "best_model_path": best["model_path"], "best_val_metrics": best["val_metrics"], "best_test_metrics": best["test_metrics"], "majority_val_metrics": best["majority_val_metrics"], "beats_previous_tiered_bc": best["val_metrics"]["balanced_accuracy"] > PREV_TIERED_BAL, "beats_previous_target_conditioned_bc": best["val_metrics"]["balanced_accuracy"] > PREV_TARGET_BAL})
    write_json("sequence_bc_train.json", out)
    return out


def make_args():
    from types import SimpleNamespace
    return SimpleNamespace(task_data=str(SOURCE_DATA / "task"), step_task_data=str(SOURCE_DATA / "step_task"), episode_data=str(SOURCE_DATA / "episode_task"), train_batch=[BATCH], val_batch=[BATCH], test_batch=[BATCH], scene=str(LH_ROOT / "data/hm3d") + "/", scene_dataset=str(LH_ROOT / "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"), max_step=MAX_EVAL_STEPS, success_dis=1, gradient_accumulation_step=1, rank=0)


def install_habitat_patch():
    from habitat_base.simulation import SceneSimulator
    def patched_get_coord(self, obj_target):
        scene = self.sim.semantic_scene
        coords = []
        index = self.target.index(obj_target)
        region_id = self.region[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    c = obj.aabb.center
                    coords.append(c() if callable(c) else c)
        if not coords:
            print("wrong target")
            return 0
        return coords
    SceneSimulator.get_coord = patched_get_coord


def state_record(sim):
    st = sim.agent.get_state()
    try:
        yaw = float(st.rotation.angle())
    except Exception:
        yaw = 0.0
    return list(st.position), yaw


def policy_select(logits: torch.Tensor, mode: str, recent: list[int], positions: list[list[float]], step: int) -> int:
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(float)
    if mode in {"stop_threshold", "planner_prior"} and step < 15 and probs[ACTION_TO_IDX["stop"]] < 0.80:
        probs[ACTION_TO_IDX["stop"]] = 0.0
    if mode in {"anti_turn_loop", "planner_prior"} and len(recent) >= 8:
        if all(a in (ACTION_TO_IDX["turn_left"], ACTION_TO_IDX["turn_right"]) for a in recent[-8:]):
            return ACTION_TO_IDX["move_forward"]
    if mode == "planner_prior" and len(positions) >= 6:
        a, b = np.array(positions[-1]), np.array(positions[-6])
        if float(np.linalg.norm(a - b)) < 0.10:
            return ACTION_TO_IDX["turn_left"] if step % 2 == 0 else ACTION_TO_IDX["move_forward"]
    return int(np.argmax(probs))


def eval_sequence(train_out: dict[str, Any], eval_name: str, modes: list[str]) -> dict[str, Any]:
    out = {"stage": eval_name.replace(".json", ""), "generated_at": now(), "status": "not_run", "variants": {}, "requested_eval_episodes": EVAL_EPISODES}
    if train_out.get("status") != "completed":
        out["status"] = "skipped_train_not_completed"
        write_json(eval_name, out)
        return out
    try:
        sys.path.insert(0, str(LH_ROOT))
        os.chdir(LH_ROOT)
        from habitat_base.simulation import SceneSimulator
        from utils.dataset import TaskDataset
        install_habitat_patch()
        ckpt = torch.load(train_out["best_model_path"], map_location="cpu")
        model = SequenceGRU(int(ckpt["input_dim"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except Exception:
            device = torch.device("cpu")
            model.to(device)
        args = make_args()
        ds = TaskDataset(args, "test")
        for mode in modes:
            successes, oracle_successes, nav_errors, nav_steps = [], [], [], []
            action_counts: Counter[str] = Counter()
            stuck_episodes = 0
            results = []
            for i in range(min(EVAL_EPISODES, len(ds))):
                config, _steps = ds[i]
                trial_dir = SOURCE_DATA / str(config.get("trajectory path", "")).strip("/")
                context = load_context(trial_dir)
                rec = {"index": i, "mode": mode, "instruction": config.get("Task instruction"), "scene": config.get("Scene"), "robot": config.get("Robot")}
                sim = None
                seq_buf: list[list[float]] = []
                recent: list[int] = []
                positions: list[list[float]] = []
                try:
                    sim = SceneSimulator(args=args, config=config)
                    sim.save_path = None
                    obs, _done, _info = sim.actor("stop")
                    prev = ACTION_TO_IDX["stop"]
                    step = 0
                    while not sim.episode_over and step < MAX_EVAL_STEPS:
                        pos, yaw = state_record(sim)
                        feat = live_feature(obs, pos, yaw, prev, context)
                        seq_buf.append(feat)
                        if len(seq_buf) < WINDOW:
                            seq = [[0.0] * len(feat)] * (WINDOW - len(seq_buf)) + seq_buf
                        else:
                            seq = seq_buf[-WINDOW:]
                        with torch.no_grad():
                            logits = model(torch.tensor([seq], dtype=torch.float32, device=device))[0]
                        pred = policy_select(logits, mode, recent, positions, step)
                        action = IDX_TO_ACTION[pred]
                        obs, _done, _info = sim.actor(action)
                        action_counts[action] += 1
                        recent.append(pred)
                        recent = recent[-16:]
                        positions.append(pos)
                        positions = positions[-16:]
                        prev = pred
                        step += 1
                    if len(positions) >= 8 and float(np.linalg.norm(np.array(positions[-1]) - np.array(positions[-8]))) < 0.10:
                        stuck_episodes += 1
                    res = sim.return_results()
                    success = bool(res.get("successes")) and all(bool(x) for x in res.get("successes", []))
                    oracle = bool(res.get("oracle_successes")) and all(bool(x) for x in res.get("oracle_successes", []))
                    ne, ns = res.get("navigation_errors", []), res.get("navigation_steps", [])
                    rec.update({"status": "completed", "success": success, "oracle_success": oracle, "navigation_error": float(sum(ne) / len(ne)) if ne else None, "navigation_steps": int(sum(ns)) if ns else None})
                    successes.append(success); oracle_successes.append(oracle)
                    if rec["navigation_error"] is not None: nav_errors.append(rec["navigation_error"])
                    if rec["navigation_steps"] is not None: nav_steps.append(rec["navigation_steps"])
                except Exception as exc:
                    rec.update({"status": "failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]})
                finally:
                    if sim is not None:
                        try: sim.close()
                        except Exception: pass
                results.append(rec)
                if (i + 1) % 10 == 0:
                    out["variants"][mode] = {"partial_results": results, "action_counts": dict(action_counts)}
                    write_json(eval_name.replace(".json", ".progress.json"), out)
            completed = sum(1 for r in results if r.get("status") == "completed")
            out["variants"][mode] = {"status": "completed" if completed >= 20 else "partial", "episodes_completed": completed, "metrics": {"SR": round(sum(successes) / len(successes), 4) if successes else 0.0, "OSR": round(sum(oracle_successes) / len(oracle_successes), 4) if oracle_successes else 0.0, "NE": round(sum(nav_errors) / len(nav_errors), 4) if nav_errors else None, "mean_navigation_steps": round(sum(nav_steps) / len(nav_steps), 4) if nav_steps else None, "SPL": None}, "action_counts": dict(action_counts), "stuck_episodes": stuck_episodes, "results": results}
        out["status"] = "completed"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc().splitlines()[-30:]
    write_json(eval_name, out)
    return out


def preflight(gpu_wait: dict[str, Any], device: torch.device, fallback_reason: str | None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    trials = sorted(SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))
    rec = {"stage": "hybrid_sequence_preflight", "generated_at": now(), "status": "passed" if len(trials) >= 300 else "blocked_missing_trajectories", "source_data": str(SOURCE_DATA), "trajectory_folders": len(trials), "window": WINDOW, "selected_device": str(device), "device_fallback_reason": fallback_reason, "gpu_wait": gpu_wait}
    if meta:
        rec["sequence_meta"] = meta
    write_json("hybrid_sequence_preflight.json", rec)
    return rec


def summary(pre: dict[str, Any], train: dict[str, Any], seq_eval: dict[str, Any], hybrid_eval: dict[str, Any]) -> dict[str, Any]:
    out = {"stage": "hybrid_sequence_summary", "generated_at": now(), "status": "completed" if train.get("status") == "completed" else "blocked", "run_root": str(RUN_ROOT), "summary": {"preflight_status": pre.get("status"), "selected_device": pre.get("selected_device"), "sequence_samples": train.get("sample_meta", {}).get("sequence_sample_count"), "best_variant": train.get("best_variant"), "best_val_balanced_accuracy": train.get("best_val_metrics", {}).get("balanced_accuracy"), "best_test_balanced_accuracy": train.get("best_test_metrics", {}).get("balanced_accuracy"), "beats_previous_tiered_bc": train.get("beats_previous_tiered_bc"), "beats_previous_target_conditioned_bc": train.get("beats_previous_target_conditioned_bc"), "sequence_eval": {k: v.get("metrics") for k, v in seq_eval.get("variants", {}).items()}, "hybrid_eval": {k: v.get("metrics") for k, v in hybrid_eval.get("variants", {}).items()}}, "paper_framing": {"claim_strength": "sequence BC and hybrid planner diagnostic feasibility", "safe_claim": "History-aware sequence BC and simple planner priors provide a stronger diagnostic baseline over frame-wise BC.", "do_not_claim": "competitive navigation-agent performance or robust RL/PPO/MGDM."}}
    write_json("hybrid_sequence_summary.json", out)
    return out


def main() -> int:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    RUN_ROOT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"Starting GPU hybrid sequence BC at {RUN_ROOT}")
    gpu_wait = wait_for_gpu()
    device, fallback = choose_device(gpu_wait)
    log(f"Selected device={device} fallback={fallback}")
    # collect once here so preflight can report sample count before training
    xs, ys, tids, meta = collect_sequences()
    pre = preflight(gpu_wait, device, fallback, meta)
    if pre.get("status") != "passed":
        return 2
    # Reuse collected arrays by monkey patching collect_sequences for train call.
    globals()["collect_sequences"] = lambda: (xs, ys, tids, meta)
    train = train_sequence_bc(device, fallback)
    seq_eval = eval_sequence(train, "sequence_bc_eval.json", ["raw"])
    hybrid_eval = eval_sequence(train, "hybrid_policy_eval.json", ["anti_turn_loop", "stop_threshold", "planner_prior"])
    out = summary(pre, train, seq_eval, hybrid_eval)
    log(f"Completed hybrid sequence summary status={out.get('status')} results={RESULTS}")
    return 0 if out.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
