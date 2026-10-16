#!/usr/bin/env python3
"""Ordered-subgoal VLN evaluator for the IDEN-VLN Habitat release.

This runner deliberately lives beside the existing evaluator instead of
modifying it.  It reuses the already published policy feature/model code,
but records a per-episode rollout trace and computes ordered-subgoal metrics
from that trace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


POLICIES = [
    ("random", "raw"),
    ("tiered_bc", "raw"),
    ("target_conditioned_bc", "raw"),
    ("sequence_bc", "raw"),
    ("sequence_hybrid", "planner_prior"),
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def bootstrap_ci(values: Sequence[float], seed: int, rounds: int = 1000) -> Optional[List[float]]:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    return [round(estimates[int(0.025 * rounds)], 6), round(estimates[int(0.975 * rounds) - 1], 6)]


def load_legacy_module(lh_root: Path):
    legacy_path = (
        lh_root
        / "runs"
        / "idenvln_harmonized_agent_eval_level23_20260707"
        / "run_harmonized_agent_eval_fixed.py"
    )
    spec = importlib.util.spec_from_file_location("idenvln_legacy_eval", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy evaluator: {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_index_aware_target_patch():
    """Patch target lookup to use the active ordered target's region.

    The original helper uses ``list.index(label)``.  Using the active stage
    avoids ambiguity when two ordered targets share the same category label.
    """

    from habitat_base.simulation import SceneSimulator

    def get_coord(self, obj_target):
        if not self.target:
            return []
        index = min(max(int(self.stage), 0), len(self.target) - 1)
        if self.target[index] != obj_target:
            matches = [i for i, label in enumerate(self.target) if label == obj_target]
            index = matches[0] if matches else index
        region_id = str(self.region[index])
        coords = []
        for region in self.sim.semantic_scene.regions:
            if str(region.id)[1:] != region_id:
                continue
            for obj in region.objects:
                try:
                    name = obj.category.name()
                except Exception:
                    continue
                if name != obj_target:
                    continue
                center = obj.aabb.center
                center = center() if callable(center) else center
                coords.append(center)
        return coords

    SceneSimulator.get_coord = get_coord


def capture_state(sim) -> Dict[str, Any]:
    state = sim.agent.get_state()
    rotation = state.rotation
    return {
        "position": [float(x) for x in state.position],
        "rotation": [
            float(rotation.w),
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
        ],
    }


def reset_simulator(sim, state: Dict[str, Any]) -> None:
    sim.set_state(state["position"], state["rotation"])
    sim.observations = sim.sim.get_sensor_observations()
    sim.step = -1
    sim.stage = 0
    sim.nav_steps = []
    sim.successes = [False for _ in sim.target]
    sim.oracle_successes = [False for _ in sim.target]
    sim.nav_errors = []
    sim.target_num = len(sim.target)
    sim.info = sim.get_info()
    sim.gt_step = None
    sim.gt_path = [sim.info["geo dis"]]
    sim.done = False
    sim.episode_over = False


def target_coords_for_index(sim, index: int) -> List[np.ndarray]:
    label = sim.target[index]
    region_id = str(sim.region[index])
    coords: List[np.ndarray] = []
    for region in sim.sim.semantic_scene.regions:
        if str(region.id)[1:] != region_id:
            continue
        for obj in region.objects:
            try:
                name = obj.category.name()
            except Exception:
                continue
            if name != label:
                continue
            center = obj.aabb.center
            center = center() if callable(center) else center
            coords.append(np.asarray(sim.pathfinder.snap_point(center), dtype=np.float32))
    return coords


def geodesic_between(pathfinder, start: Sequence[float], goals: Iterable[Sequence[float]]) -> Tuple[float, Optional[np.ndarray]]:
    best_distance = math.inf
    best_goal = None
    for goal in goals:
        path = pathfinder.find_path if hasattr(pathfinder, "find_path") else None
        del path
        shortest = __import__("habitat_sim").nav.ShortestPath()
        shortest.requested_start = np.asarray(start, dtype=np.float32)
        shortest.requested_end = np.asarray(goal, dtype=np.float32)
        if pathfinder.find_path(shortest) and finite(shortest.geodesic_distance):
            distance = float(shortest.geodesic_distance)
            if distance < best_distance:
                best_distance = distance
                best_goal = np.asarray(goal, dtype=np.float32)
    return best_distance, best_goal


def all_target_distances(sim, target_coords: Sequence[Sequence[np.ndarray]]) -> List[float]:
    state = sim.agent.get_state()
    distances = []
    for coords in target_coords:
        distance, _ = geodesic_between(sim.pathfinder, state.position, coords)
        distances.append(distance)
    return distances


def reference_length(sim, start: Sequence[float], target_coords: Sequence[Sequence[np.ndarray]]) -> Tuple[Optional[float], List[Optional[List[float]]]]:
    current = np.asarray(start, dtype=np.float32)
    total = 0.0
    selected: List[Optional[List[float]]] = []
    for coords in target_coords:
        distance, goal = geodesic_between(sim.pathfinder, current, coords)
        if not finite(distance) or goal is None:
            return None, selected
        total += distance
        selected.append([float(x) for x in goal])
        current = goal
    return total, selected


def serial_distances(values: Sequence[float]) -> List[Optional[float]]:
    return [round(float(value), 6) if finite(value) else None for value in values]


def position(sim) -> List[float]:
    return [float(x) for x in sim.agent.get_state().position]


def path_length(positions: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for first, second in zip(positions, positions[1:]):
        total += float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
    return total


def choose_action(legacy, name: str, policy_state: Dict[str, Any], obs, pos, yaw, previous: int, sequence_buffer, context, mode: str, recent, positions, step: int) -> Tuple[int, List[List[float]]]:
    if name == "random":
        return random.Random(legacy.SEED + context["index"] * 1009 + step).choice(list(legacy.ACTION_TO_IDX.values())), sequence_buffer
    model = policy_state["model"]
    feature_kind = policy_state["feature"]
    if feature_kind == "tiered":
        feat = legacy.tiered_feature(obs, pos, yaw, previous, context)
        with legacy.torch.no_grad():
            logits = model(legacy.torch.tensor([feat], dtype=legacy.torch.float32))[0]
        return legacy.select_action_from_logits(logits, mode, recent, positions, step), sequence_buffer
    if feature_kind == "target":
        feat = legacy.target_feature(obs, pos, yaw, previous, context)
        with legacy.torch.no_grad():
            logits = model(legacy.torch.tensor([feat], dtype=legacy.torch.float32))[0]
        return legacy.select_action_from_logits(logits, mode, recent, positions, step), sequence_buffer
    if feature_kind == "seq":
        feat = legacy.seq_feature(obs, pos, yaw, previous, context)
        sequence_buffer = sequence_buffer + [feat]
        window = int(policy_state.get("window", legacy.WINDOW))
        if len(sequence_buffer) < window:
            sequence = [[0.0] * len(feat)] * (window - len(sequence_buffer)) + sequence_buffer
        else:
            sequence = sequence_buffer[-window:]
        with legacy.torch.no_grad():
            logits = model(legacy.torch.tensor([sequence], dtype=legacy.torch.float32))[0]
        return legacy.select_action_from_logits(logits, mode, recent, positions, step), sequence_buffer
    return legacy.ACTION_TO_IDX["stop"], sequence_buffer


def summarize_policy(records: Sequence[Dict[str, Any]], max_steps: int, seed: int) -> Dict[str, Any]:
    valid = [record for record in records if record.get("status") == "completed"]
    n = len(valid)
    if not n:
        return {"N": len(records), "N_effective": 0, "status": "no_completed_episodes"}

    sr = [float(record["full_chain_success"]) for record in valid]
    osr = [float(record["oracle_full_chain_success"]) for record in valid]
    ocs = [float(record["ocs"]) for record in valid]
    ne = [float(record["navigation_error"]) for record in valid if finite(record.get("navigation_error"))]
    spl_values = [record.get("spl") for record in valid]
    spl_available = [float(value) for value in spl_values if finite(value)]
    order = [float(record["order_violation"]) for record in valid]
    premature = [float(record["premature_stop"]) for record in valid]
    max_step = [float(record["max_step_termination"]) for record in valid]
    prefix = [float(record["completed_prefix_ratio"]) for record in valid]
    mean_steps = [float(record["navigation_steps"]) for record in valid if finite(record.get("navigation_steps"))]

    def mean(values):
        return round(sum(values) / len(values), 6) if values else None

    metrics = {
        "SR_full_chain": mean(sr),
        "Full_chain_success": mean(sr),
        "OSR": mean(osr),
        "NE": mean(ne),
        "OCS": mean(ocs),
        "SPL": mean(spl_available) if len(spl_available) == n else None,
        "out_of_order_visit_rate": mean(order),
        "premature_subgoal_stop_rate": mean(premature),
        "max_step_termination_rate": mean(max_step),
        "mean_completed_prefix_ratio": mean(prefix),
        "mean_navigation_steps": mean(mean_steps),
    }
    ci = {
        "SR_full_chain": bootstrap_ci(sr, seed + 1),
        "OSR": bootstrap_ci(osr, seed + 2),
        "OCS": bootstrap_ci(ocs, seed + 3),
        "SPL": bootstrap_ci(spl_available, seed + 4) if len(spl_available) == n else None,
        "NE": bootstrap_ci(ne, seed + 5),
    }
    return {
        "N": len(records),
        "N_effective": n,
        "failed_episodes": len(records) - n,
        "spl_coverage": round(len(spl_available) / n, 6),
        "metrics": metrics,
        "bootstrap_ci95": ci,
        "status": "completed" if len(records) == n else "incomplete",
    }


def evaluate_policy(legacy, dataset, name: str, mode: str, policy_state: Dict[str, Any], episode_indices: Sequence[int], start_states: Dict[str, Any], output_trace: Path, max_steps: int, success_dis: float, seed: int, write_progress: Path, target_cache: Dict[int, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if name != "random" and policy_state.get("status") != "loaded":
        raise RuntimeError(f"policy {name} unavailable: {policy_state}")

    output_trace.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    with output_trace.open("w", encoding="utf-8") as trace_handle:
        for ordinal, index in enumerate(episode_indices):
            config, _step_tasks = dataset[index]
            context = legacy.load_context(config)
            context["index"] = int(index)
            record: Dict[str, Any] = {
                "status": "started",
                "policy": name,
                "mode": mode,
                "index": int(index),
                "ordinal": int(ordinal),
                "instruction": config.get("Task instruction"),
                "scene": config.get("Scene"),
                "robot": config.get("Robot"),
                "targets": list(config.get("Object", [])),
                "regions": list(config.get("Region", [])),
            }
            sim = None
            try:
                from habitat_base.simulation import SceneSimulator

                args = legacy.make_args()
                args.max_step = max_steps
                args.success_dis = success_dis
                sim = SceneSimulator(args=args, config=config)
                key = str(index)
                if key not in start_states:
                    start_states[key] = capture_state(sim)
                    write_json_atomic(write_progress.parent / "start_states.json", start_states)
                reset_simulator(sim, start_states[key])

                target_coords = target_cache.get(index)
                if target_coords is None:
                    target_coords = [target_coords_for_index(sim, k) for k in range(len(sim.target))]
                    target_cache[index] = target_coords
                reference, reference_targets = reference_length(sim, start_states[key]["position"], target_coords)
                record["start_state"] = start_states[key]
                record["target_count"] = len(sim.target)
                record["reference_path_length"] = round(reference, 6) if finite(reference) else None
                record["reference_target_points"] = reference_targets

                obs, _done, _info = sim.actor("stop")
                previous = legacy.ACTION_TO_IDX["stop"]
                recent: List[int] = []
                sequence_buffer: List[List[float]] = []
                positions: List[List[float]] = [position(sim)]
                steps: List[Dict[str, Any]] = []
                visited = [False for _ in sim.target]
                order_violation = False
                premature_events = 0
                stop_events = 0
                pre_distances = all_target_distances(sim, target_coords)
                for target_index, distance in enumerate(pre_distances):
                    if distance <= success_dis:
                        visited[target_index] = True

                step = 0
                while not sim.episode_over and step < max_steps:
                    active_stage = int(sim.stage)
                    before = position(sim)
                    before_distances = all_target_distances(sim, target_coords)
                    for target_index, distance in enumerate(before_distances):
                        if distance <= success_dis:
                            if target_index > active_stage and not bool(sim.successes[active_stage]):
                                order_violation = True
                            visited[target_index] = True
                    yaw = 0.0
                    try:
                        yaw = float(sim.agent.get_state().rotation.angle())
                    except Exception:
                        pass
                    pred, sequence_buffer = choose_action(
                        legacy,
                        name,
                        policy_state,
                        obs,
                        before,
                        yaw,
                        previous,
                        sequence_buffer,
                        context,
                        mode,
                        recent,
                        positions[-16:],
                        step,
                    )
                    action = legacy.IDX_TO_ACTION[pred]
                    obs, _done, _info = sim.actor(action)
                    action_counts[action] += 1
                    after = position(sim)
                    positions.append(after)
                    after_distances = all_target_distances(sim, target_coords)
                    if action == "stop":
                        stop_events += 1
                        if active_stage < len(sim.successes) and not bool(sim.successes[active_stage]):
                            premature_events += 1
                        for target_index, distance in enumerate(after_distances):
                            if distance <= success_dis:
                                if target_index > active_stage and not bool(sim.successes[active_stage]):
                                    order_violation = True
                                visited[target_index] = True
                    for target_index, distance in enumerate(after_distances):
                        if distance <= success_dis:
                            visited[target_index] = True
                    steps.append({
                        "t": int(step),
                        "active_target": active_stage,
                        "action": action,
                        "position_before": before,
                        "position_after": after,
                        "distances_before": serial_distances(before_distances),
                        "distances_after": serial_distances(after_distances),
                        "stop": action == "stop",
                        "active_target_success_after": bool(sim.successes[active_stage]) if active_stage < len(sim.successes) else False,
                    })
                    recent.append(pred)
                    recent = recent[-16:]
                    previous = pred
                    step += 1

                result = sim.return_results()
                successes = [bool(value) for value in result.get("successes", [])]
                oracle_successes = [bool(value) for value in result.get("oracle_successes", [])]
                prefix_length = 0
                for value in successes:
                    if not value:
                        break
                    prefix_length += 1
                full_chain = bool(successes) and prefix_length == len(successes) and bool(sim.done)
                navigation_errors = [float(value) for value in result.get("navigation_errors", []) if finite(value)]
                navigation_steps = [float(value) for value in result.get("navigation_steps", []) if finite(value)]
                nav_error = sum(navigation_errors) / len(navigation_errors) if navigation_errors else None
                executed = path_length(positions)
                reference_value = record.get("reference_path_length")
                spl = None
                if finite(reference_value):
                    spl = (float(reference_value) / max(executed, float(reference_value))) if full_chain and executed > 0 else 0.0
                termination = "completed_chain" if sim.done else ("max_steps" if sim.episode_over else "loop_exit")
                record.update({
                    "status": "completed",
                    "actions": [step_record["action"] for step_record in steps],
                    "steps": steps,
                    "visited_targets": visited,
                    "successes": successes,
                    "oracle_successes": oracle_successes,
                    "completed_prefix": prefix_length,
                    "completed_prefix_ratio": (prefix_length / len(successes)) if successes else 0.0,
                    "ocs": (prefix_length / len(successes)) if successes else 0.0,
                    "full_chain_success": full_chain,
                    "oracle_full_chain_success": bool(oracle_successes) and all(oracle_successes),
                    "navigation_error": nav_error,
                    "navigation_steps": sum(navigation_steps) if navigation_steps else step,
                    "executed_path_length": round(executed, 6),
                    "spl": round(float(spl), 6) if finite(spl) else None,
                    "order_violation": bool(order_violation),
                    "premature_stop": bool(premature_events > 0),
                    "premature_stop_events": int(premature_events),
                    "stop_events": int(stop_events),
                    "max_step_termination": bool(termination == "max_steps"),
                    "termination_reason": termination,
                })
            except Exception as exc:
                record.update({
                    "status": "failed",
                    "error": repr(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                })
            finally:
                if sim is not None:
                    try:
                        sim.close()
                    except Exception:
                        pass
            records.append(record)
            trace_handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            trace_handle.flush()
            if (ordinal + 1) % 5 == 0 or ordinal + 1 == len(episode_indices):
                summary = summarize_policy(records, max_steps, seed)
                summary.update({"policy": name, "mode": mode, "updated_at": utc_now(), "action_counts": dict(action_counts)})
                write_json_atomic(write_progress, summary)
    return records, summarize_policy(records, max_steps, seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lh-root", default="/home/aimldl/LH-VLN")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--success-distance", type=float, default=1.0)
    args = parser.parse_args()

    lh_root = Path(args.lh_root)
    run_dir = Path(args.output_dir)
    results_dir = run_dir / "results"
    traces_dir = results_dir / "traces"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"

    status = {
        "status": "starting",
        "started_at": utc_now(),
        "run_dir": str(run_dir),
        "policies": [name for name, _mode in POLICIES],
        "requested_episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_distance": args.success_distance,
    }
    write_json_atomic(status_path, status)

    try:
        sys.path.insert(0, str(lh_root))
        os.chdir(lh_root)
        legacy = load_legacy_module(lh_root)
        import torch
        from utils.dataset import TaskDataset

        install_index_aware_target_patch()
        random.seed(legacy.SEED)
        np.random.seed(legacy.SEED)
        torch.manual_seed(legacy.SEED)
        dataset = TaskDataset(legacy.make_args(), "test")
        episode_count = min(args.episodes, len(dataset))
        if episode_count <= 0:
            raise RuntimeError("no test episodes available")

        protocol = {
            "status": "completed",
            "created_at": utc_now(),
            "source_data_root": str(legacy.SOURCE_DATA),
            "episode_selection": "first_n_deterministic_taskdataset_order",
            "episode_indices": list(range(episode_count)),
            "policies": [name for name, _mode in POLICIES],
            "max_steps": args.max_steps,
            "success_distance": args.success_distance,
            "metrics": [
                "SR_full_chain",
                "Full_chain_success",
                "OSR",
                "NE",
                "OCS",
                "SPL",
                "out_of_order_visit_rate",
                "premature_subgoal_stop_rate",
                "max_step_termination_rate",
                "mean_completed_prefix_ratio",
            ],
            "definitions": {
                "OCS": "ordered prefix length divided by ordered target count",
                "order_violation": "later target within success radius before active target success",
                "premature_stop": "stop emitted while active target is outside success radius",
                "SPL": "full-chain success weighted executed path over shortest geodesic route through ordered targets",
                "NE": "mean per-stage navigation error retained from existing simulator",
            },
        }
        write_json_atomic(results_dir / "protocol.json", protocol)

        # Smoke test is intentionally part of the same long-lived orchestrator.
        smoke_count = min(max(args.smoke_episodes, 0), episode_count)
        if smoke_count:
            status.update({"status": "smoke_running", "smoke_episodes": smoke_count})
            write_json_atomic(status_path, status)
            smoke_dir = results_dir / "smoke"
            smoke_starts: Dict[str, Any] = {}
            smoke_cache: Dict[int, Any] = {}
            smoke_indices = list(range(smoke_count))
            for name, mode in POLICIES:
                policy_state = {"status": "loaded", "policy": "random"} if name == "random" else legacy.load_policy("sequence_bc" if name == "sequence_hybrid" else name)
                records, summary = evaluate_policy(
                    legacy, dataset, name, mode, policy_state, smoke_indices, smoke_starts,
                    smoke_dir / "traces" / f"{name}.jsonl", args.max_steps, args.success_distance,
                    legacy.SEED, smoke_dir / "progress" / f"{name}.json", smoke_cache,
                )
                write_json_atomic(smoke_dir / f"{name}_summary.json", summary)
                if len(records) != smoke_count or any(record.get("status") != "completed" for record in records):
                    raise RuntimeError(f"smoke test failed for {name}")
            write_json_atomic(smoke_dir / "status.json", {"status": "passed", "episodes": smoke_count, "updated_at": utc_now()})

        status.update({"status": "full_running", "episodes": episode_count})
        write_json_atomic(status_path, status)
        full_starts: Dict[str, Any] = {}
        target_cache: Dict[int, Any] = {}
        summaries: Dict[str, Any] = {}
        episode_indices = list(range(episode_count))
        for name, mode in POLICIES:
            policy_state = {"status": "loaded", "policy": "random"} if name == "random" else legacy.load_policy("sequence_bc" if name == "sequence_hybrid" else name)
            records, summary = evaluate_policy(
                legacy, dataset, name, mode, policy_state, episode_indices, full_starts,
                traces_dir / f"{name}.jsonl", args.max_steps, args.success_distance,
                legacy.SEED, results_dir / "progress" / f"{name}.json", target_cache,
            )
            summaries[name] = summary
            write_json_atomic(results_dir / f"{name}_summary.json", summary)
            status.update({"status": "full_running", "completed_policies": list(summaries), "last_policy": name})
            write_json_atomic(status_path, status)

        write_json_atomic(results_dir / "summary.json", summaries)
        write_json_atomic(results_dir / "start_states.json", full_starts)
        manifest_files = {}
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                manifest_files[str(path.relative_to(run_dir))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        write_json_atomic(run_dir / "manifest.json", {"created_at": utc_now(), "files": manifest_files})
        write_json_atomic(run_dir / "checksums.json", {key: value["sha256"] for key, value in manifest_files.items()})
        status.update({"status": "completed", "finished_at": utc_now(), "completed_policies": list(summaries)})
        write_json_atomic(status_path, status)
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        status.update({"status": "failed", "finished_at": utc_now(), "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-30:]})
        write_json_atomic(status_path, status)
        print(json.dumps(status, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
