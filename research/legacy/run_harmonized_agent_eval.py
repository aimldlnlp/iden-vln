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
from PIL import ImageStat


LH_ROOT = Path("/home/aimldl/LH-VLN")
RUN_ROOT = LH_ROOT / "runs/idenvln_harmonized_agent_eval_level23_20260707"
RESULTS = RUN_ROOT / "results"
SOURCE_DATA = LH_ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706/data_expert_scaled"
TIERED_RESULTS = LH_ROOT / "runs/idenvln_tiered_balanced_visual_bc_level23_20260706/results"
TARGET_RESULTS = LH_ROOT / "runs/idenvln_longrun_target_conditioned_bc_level23_20260706/results"
SEQ_RESULTS = LH_ROOT / "runs/idenvln_gpu_hybrid_sequence_bc_level23_20260707/results"
BATCH = "/batch_generated_level23"

SEED = 61007
REQUESTED_EPISODES = 100
MAX_STEPS = 150
WINDOW = 8
FIXED_ONLY = os.environ.get("HARMONIZED_FIXED_ONLY", "0") == "1"
OUT_SUFFIX = "_fixed" if FIXED_ONLY else ""
HASH_DIM_TARGET = 128
HASH_DIM_SEQ = 64
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
    return {"available": bool(gpus), "gpus": gpus}


def stable_hash(text: str, modulo: int) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % modulo


def hash_features(tokens: list[str], dim: int) -> list[float]:
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
    return [str(v).strip().lower() for v in value if str(v).strip()]


def load_context(config: dict[str, Any]) -> dict[str, Any]:
    trial_dir = SOURCE_DATA / str(config.get("trajectory path", "")).strip("/")
    try:
        task = read_json(trial_dir / "task.json")
    except Exception:
        task = config
    try:
        level = int(trial_dir.parents[2].name)
    except Exception:
        level = int(task.get("Generated level") or 0)
    return {
        "task": task,
        "instruction": str(task.get("Task instruction") or config.get("Task instruction") or ""),
        "objects": object_labels(task.get("Object")),
        "regions": normalize_regions(task.get("Region Name") or task.get("Region")),
        "robot": str(task.get("Robot") or config.get("Robot") or "").lower(),
        "level": level,
        "scene": str(task.get("Scene") or config.get("Scene") or ""),
    }


def image_stats_from_obs(obs: list[Any]) -> list[float]:
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
    return features


def common_pose_prev_features(pos: Any, yaw: Any, prev_action_idx: int) -> list[float]:
    features: list[float] = []
    pos_list = list(pos) if isinstance(pos, (list, tuple, np.ndarray)) else [0.0, 0.0, 0.0]
    features.extend([float(x) for x in pos_list[:3]])
    try:
        features.append(float(yaw))
    except Exception:
        features.append(0.0)
    prev = [0.0] * 4
    if 0 <= prev_action_idx < 4:
        prev[prev_action_idx] = 1.0
    features.extend(prev)
    return features


def tiered_feature(obs: list[Any], pos: Any, yaw: Any, prev_action_idx: int, _context: dict[str, Any]) -> list[float]:
    return image_stats_from_obs(obs) + common_pose_prev_features(pos, yaw, prev_action_idx)


def target_feature(obs: list[Any], pos: Any, yaw: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float]:
    features = tiered_feature(obs, pos, yaw, prev_action_idx, context)
    features.extend([1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0])
    features.append(float(context.get("level") or 0) / 5.0)
    features.extend(hash_features(context.get("objects", []) + context.get("regions", []), HASH_DIM_TARGET))
    features.extend(hash_features(tokenize(context.get("instruction"))[:80], HASH_DIM_TARGET))
    return features


def seq_feature(obs: list[Any], pos: Any, yaw: Any, prev_action_idx: int, context: dict[str, Any]) -> list[float]:
    features = tiered_feature(obs, pos, yaw, prev_action_idx, context)
    features.extend([1.0 if context.get("robot") == "stretch" else 0.0, 1.0 if context.get("robot") == "spot" else 0.0])
    features.append(float(context.get("level") or 0) / 5.0)
    features.extend(hash_features(context.get("objects", []) + context.get("regions", []), HASH_DIM_SEQ))
    features.extend(hash_features(tokenize(context.get("instruction"))[:80], HASH_DIM_SEQ))
    return features


class BalancedMLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


class TargetConditionedMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 4))

    def forward(self, x):
        return self.net(x)


class SequenceGRU(torch.nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.gru = torch.nn.GRU(dim, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def load_policy(policy: str) -> dict[str, Any]:
    try:
        if policy == "tiered_bc":
            ckpt_path = TIERED_RESULTS / "tiered_bc_best_scaled.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = BalancedMLP(int(ckpt["input_dim"]))
            model.load_state_dict(ckpt["model_state_dict"])
            return {"status": "loaded", "policy": policy, "model": model.eval(), "feature": "tiered", "ckpt": str(ckpt_path)}
        if policy == "target_conditioned_bc":
            train = read_json(TARGET_RESULTS / "target_conditioned_bc_train.json")
            ckpt_path = Path(train["best_model_path"])
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = TargetConditionedMLP(int(ckpt["input_dim"]))
            model.load_state_dict(ckpt["model_state_dict"])
            return {"status": "loaded", "policy": policy, "model": model.eval(), "feature": "target", "ckpt": str(ckpt_path)}
        if policy == "sequence_bc":
            train = read_json(SEQ_RESULTS / "sequence_bc_train.json")
            ckpt_path = Path(train["best_model_path"])
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = SequenceGRU(int(ckpt["input_dim"]))
            model.load_state_dict(ckpt["model_state_dict"])
            return {"status": "loaded", "policy": policy, "model": model.eval(), "feature": "seq", "ckpt": str(ckpt_path), "window": int(ckpt.get("window", WINDOW))}
    except Exception as exc:
        return {"status": "blocked_checkpoint_missing_or_incompatible", "policy": policy, "error": repr(exc), "traceback": traceback.format_exc().splitlines()[-12:]}
    return {"status": "blocked_unknown_policy", "policy": policy}


def make_args():
    from types import SimpleNamespace

    return SimpleNamespace(
        task_data=str(SOURCE_DATA / "task"),
        step_task_data=str(SOURCE_DATA / "step_task"),
        episode_data=str(SOURCE_DATA / "episode_task"),
        train_batch=[BATCH],
        val_batch=[BATCH],
        test_batch=[BATCH],
        scene=str(LH_ROOT / "data/hm3d") + "/",
        scene_dataset=str(LH_ROOT / "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"),
        max_step=MAX_STEPS,
        success_dis=1,
        gradient_accumulation_step=1,
        rank=0,
    )


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
                    center = obj.aabb.center
                    coords.append(center() if callable(center) else center)
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


def select_action_from_logits(logits: torch.Tensor, mode: str, recent: list[int], positions: list[list[float]], step: int) -> int:
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(float)
    if mode in {"stop_threshold", "planner_prior"} and step < 15 and probs[ACTION_TO_IDX["stop"]] < 0.80:
        probs[ACTION_TO_IDX["stop"]] = 0.0
    if mode in {"anti_turn_loop", "planner_prior"} and len(recent) >= 8 and all(a in (ACTION_TO_IDX["turn_left"], ACTION_TO_IDX["turn_right"]) for a in recent[-8:]):
        return ACTION_TO_IDX["move_forward"]
    if mode == "planner_prior" and len(positions) >= 6:
        if float(np.linalg.norm(np.array(positions[-1]) - np.array(positions[-6]))) < 0.10:
            return ACTION_TO_IDX["turn_left"] if step % 2 == 0 else ACTION_TO_IDX["move_forward"]
    return int(np.argmax(probs))


def bootstrap_ci(values: list[float], reducer, rounds: int = 1000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(SEED)
    stats = []
    n = len(values)
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(float(reducer(sample)))
    stats.sort()
    return [round(stats[int(0.025 * rounds)], 6), round(stats[int(0.975 * rounds) - 1], 6)]


def summarize_results(results: list[dict[str, Any]], action_counts: Counter[str], stuck_episodes: int) -> dict[str, Any]:
    completed = [r for r in results if r.get("status") == "completed"]
    successes = [1.0 if r.get("success") else 0.0 for r in completed]
    oracle = [1.0 if r.get("oracle_success") else 0.0 for r in completed]
    ne = [float(r["navigation_error"]) for r in completed if isinstance(r.get("navigation_error"), (int, float))]
    steps = [float(r["navigation_steps"]) for r in completed if isinstance(r.get("navigation_steps"), (int, float))]
    by_robot: dict[str, Counter[str]] = defaultdict(Counter)
    for r in completed:
        robot = str(r.get("robot") or "unknown")
        by_robot[robot]["episodes"] += 1
        by_robot[robot]["successes"] += int(bool(r.get("success")))
        by_robot[robot]["oracle_successes"] += int(bool(r.get("oracle_success")))
    return {
        "episodes_completed": len(completed),
        "metrics": {
            "SR": round(sum(successes) / len(successes), 6) if successes else 0.0,
            "OSR": round(sum(oracle) / len(oracle), 6) if oracle else 0.0,
            "NE": round(sum(ne) / len(ne), 6) if ne else None,
            "mean_navigation_steps": round(sum(steps) / len(steps), 6) if steps else None,
            "SPL": None,
        },
        "ci95": {
            "SR": bootstrap_ci(successes, lambda x: sum(x) / len(x)) if successes else None,
            "OSR": bootstrap_ci(oracle, lambda x: sum(x) / len(x)) if oracle else None,
            "NE": bootstrap_ci(ne, lambda x: sum(x) / len(x)) if ne else None,
        },
        "action_counts": dict(action_counts),
        "stuck_rate": round(stuck_episodes / max(len(completed), 1), 6),
        "max_step_rate": round(sum(1 for s in steps if s >= MAX_STEPS) / max(len(steps), 1), 6),
        "by_robot": {k: dict(v) for k, v in by_robot.items()},
    }


def eval_policy(name: str, policy_state: dict[str, Any], episode_indices: list[int], mode: str = "raw") -> dict[str, Any]:
    out = {"stage": f"harmonized_eval_{name}", "generated_at": now(), "policy": name, "mode": mode, "requested_episodes": REQUESTED_EPISODES, "max_steps": MAX_STEPS, "status": "not_run", "results": []}
    if name != "random" and policy_state.get("status") != "loaded":
        out.update({"status": policy_state.get("status"), "blocker": policy_state})
        return out
    try:
        from habitat_base.simulation import SceneSimulator
        from utils.dataset import TaskDataset

        args = make_args()
        ds = TaskDataset(args, "test")
        model = policy_state.get("model")
        feature_kind = policy_state.get("feature")
        seq_buf: list[list[float]]
        action_counts: Counter[str] = Counter()
        stuck_episodes = 0
        for count, i in enumerate(episode_indices):
            config, _steps = ds[i]
            context = load_context(config)
            rec = {"index": i, "ordinal": count, "instruction": config.get("Task instruction"), "scene": config.get("Scene"), "robot": config.get("Robot")}
            sim = None
            try:
                sim = SceneSimulator(args=args, config=config)
                sim.save_path = None
                obs, _done, _info = sim.actor("stop")
                prev = ACTION_TO_IDX["stop"]
                recent: list[int] = []
                positions: list[list[float]] = []
                seq_buf = []
                step = 0
                while not sim.episode_over and step < MAX_STEPS:
                    pos, yaw = state_record(sim)
                    if name == "random":
                        pred = random.Random(SEED + i * 1009 + step).choice(list(ACTION_TO_IDX.values()))
                    elif feature_kind == "tiered":
                        feat = tiered_feature(obs, pos, yaw, prev, context)
                        with torch.no_grad():
                            logits = model(torch.tensor([feat], dtype=torch.float32))[0]
                        pred = select_action_from_logits(logits, mode, recent, positions, step)
                    elif feature_kind == "target":
                        feat = target_feature(obs, pos, yaw, prev, context)
                        with torch.no_grad():
                            logits = model(torch.tensor([feat], dtype=torch.float32))[0]
                        pred = select_action_from_logits(logits, mode, recent, positions, step)
                    elif feature_kind == "seq":
                        feat = seq_feature(obs, pos, yaw, prev, context)
                        seq_buf.append(feat)
                        if len(seq_buf) < WINDOW:
                            seq = [[0.0] * len(feat)] * (WINDOW - len(seq_buf)) + seq_buf
                        else:
                            seq = seq_buf[-WINDOW:]
                        with torch.no_grad():
                            logits = model(torch.tensor([seq], dtype=torch.float32))[0]
                        pred = select_action_from_logits(logits, mode, recent, positions, step)
                    else:
                        pred = ACTION_TO_IDX["stop"]
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
                osr = bool(res.get("oracle_successes")) and all(bool(x) for x in res.get("oracle_successes", []))
                ne = res.get("navigation_errors", [])
                ns = res.get("navigation_steps", [])
                rec.update({"status": "completed", "success": success, "oracle_success": osr, "navigation_error": float(sum(ne) / len(ne)) if ne else None, "navigation_steps": int(sum(ns)) if ns else None})
            except Exception as exc:
                rec.update({"status": "failed", "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-8:]})
            finally:
                if sim is not None:
                    try:
                        sim.close()
                    except Exception:
                        pass
            out["results"].append(rec)
            if (count + 1) % 10 == 0:
                tmp = dict(out)
                tmp.update(summarize_results(out["results"], action_counts, stuck_episodes))
                tmp["status"] = "running"
                write_json(f"harmonized_eval_{name}{OUT_SUFFIX}.progress.json", tmp)
        out.update(summarize_results(out["results"], action_counts, stuck_episodes))
        out["status"] = "completed" if out["episodes_completed"] >= min(20, len(episode_indices)) else "partial"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc().splitlines()[-30:]
    return out


def expert_oracle(episode_indices: list[int]) -> dict[str, Any]:
    out = {"stage": "harmonized_eval_expert_oracle", "generated_at": now(), "status": "metadata_upper_bound", "requested_episodes": REQUESTED_EPISODES, "max_steps": MAX_STEPS, "episode_indices": episode_indices}
    gen_path = TIERED_RESULTS / "tiered_bc_scale_generation.json"
    if gen_path.exists():
        gen = read_json(gen_path)
        attempted = gen.get("attempted")
        valid = gen.get("valid_frame_trajectories")
        successes = gen.get("successes")
        out.update({"source_generation": str(gen_path), "attempted": attempted, "valid_frame_trajectories": valid, "expert_successes": successes, "valid_trajectory_rate": round(valid / attempted, 6) if attempted and valid is not None else None, "expert_success_rate": round(successes / attempted, 6) if attempted and successes is not None else None})
    out["note"] = "Metadata expert/oracle upper-bound, separated from learned-policy online eval."
    return out


def protocol() -> tuple[dict[str, Any], list[int]]:
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    from utils.dataset import TaskDataset

    args = make_args()
    ds = TaskDataset(args, "test")
    n = len(ds)
    episode_indices = list(range(min(REQUESTED_EPISODES, n)))
    out = {"stage": "harmonized_eval_protocol", "generated_at": now(), "status": "completed", "source_data_root": str(SOURCE_DATA), "requested_episodes": REQUESTED_EPISODES, "available_episodes": n, "selected_episodes": len(episode_indices), "episode_selection": "first_n_deterministic_taskdataset_order", "episode_indices": episode_indices, "max_steps": MAX_STEPS, "metrics": ["SR", "OSR", "NE", "mean_navigation_steps", "SPL", "action_distribution", "stuck_rate", "max_step_rate", "per_robot_breakdown", "bootstrap_ci95_for_SR_OSR_NE"], "learned_policies": ["tiered_bc", "target_conditioned_bc", "sequence_bc_raw", "sequence_bc_planner_prior"], "upper_bound": "metadata_expert_oracle"}
    write_json(f"harmonized_eval_protocol{OUT_SUFFIX}.json", out)
    return out, episode_indices


def summary(protocol_out: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for name, rec in results.items():
        rows[name] = {"status": rec.get("status"), "episodes_completed": rec.get("episodes_completed"), "metrics": rec.get("metrics"), "stuck_rate": rec.get("stuck_rate"), "max_step_rate": rec.get("max_step_rate"), "action_counts": rec.get("action_counts")}
        if name == "expert_oracle" or "expert_oracle" in name:
            rows[name] = {"status": rec.get("status"), "expert_success_rate": rec.get("expert_success_rate"), "valid_trajectory_rate": rec.get("valid_trajectory_rate"), "note": rec.get("note")}
    out = {"stage": "harmonized_agent_eval_summary", "generated_at": now(), "status": "completed", "protocol": {"requested_episodes": REQUESTED_EPISODES, "selected_episodes": protocol_out.get("selected_episodes"), "max_steps": MAX_STEPS, "source_data_root": str(SOURCE_DATA)}, "policies": rows, "paper_framing": {"claim_strength": "controlled downstream diagnostic evaluation", "safe_claim": "Policies are compared under a shared episode/max-step protocol, while expert/oracle upper-bound is reported separately.", "do_not_claim": "competitive navigation-agent performance, robust RL/PPO/MGDM, or solved embodied navigation."}}
    write_json(f"harmonized_agent_eval_summary{OUT_SUFFIX}.json", out)
    return out


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"Starting harmonized agent eval at {RUN_ROOT}")
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    install_habitat_patch()
    protocol_out, episode_indices = protocol()
    policies = {
        "random": {"status": "loaded", "policy": "random"},
        "tiered_bc": load_policy("tiered_bc"),
        "target_conditioned_bc": load_policy("target_conditioned_bc"),
        "sequence_bc": load_policy("sequence_bc"),
    }
    results: dict[str, dict[str, Any]] = {}
    if FIXED_ONLY:
        for original_name, summary_name in [
            ("harmonized_eval_random.json", "random_existing"),
            ("harmonized_eval_tiered_bc.json", "tiered_bc_existing"),
            ("harmonized_eval_expert_oracle.json", "expert_oracle_existing"),
        ]:
            p = RESULTS / original_name
            if p.exists():
                results[summary_name] = read_json(p)
        eval_plan = [
            ("target_conditioned_bc", policies["target_conditioned_bc"], "raw", "harmonized_eval_target_conditioned_bc_fixed.json"),
            ("sequence_bc", policies["sequence_bc"], "raw", "harmonized_eval_sequence_bc_fixed.json"),
            ("sequence_hybrid", policies["sequence_bc"], "planner_prior", "harmonized_eval_sequence_hybrid_fixed.json"),
        ]
    else:
        eval_plan = [
            ("random", policies["random"], "raw", "harmonized_eval_random.json"),
            ("tiered_bc", policies["tiered_bc"], "raw", "harmonized_eval_tiered_bc.json"),
            ("target_conditioned_bc", policies["target_conditioned_bc"], "raw", "harmonized_eval_target_conditioned_bc.json"),
            ("sequence_bc", policies["sequence_bc"], "raw", "harmonized_eval_sequence_bc.json"),
            ("sequence_hybrid", policies["sequence_bc"], "planner_prior", "harmonized_eval_sequence_hybrid.json"),
        ]
    for label, state, mode, out_name in eval_plan:
        log(f"Evaluating {label} mode={mode}")
        rec = eval_policy(label, state, episode_indices, mode)
        write_json(out_name, rec)
        results[label] = rec
        log(f"{label} status={rec.get('status')} metrics={rec.get('metrics')}")
    if not FIXED_ONLY:
        expert = expert_oracle(episode_indices)
        write_json("harmonized_eval_expert_oracle.json", expert)
        results["expert_oracle"] = expert
    out = summary(protocol_out, results)
    log(f"Completed harmonized eval status={out.get('status')} results={RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
