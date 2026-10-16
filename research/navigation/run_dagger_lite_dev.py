#!/usr/bin/env python3
"""One DAgger-lite round on train tasks; compare policies on separate dev tasks."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import random
import sys
import textwrap
import traceback
from pathlib import Path

ROOT = Path("/home/aimldl/LH-VLN")
RUN = Path(os.environ["IDENVLN_RUN_DIR"])
SOURCE = ROOT / "runs/idenvln_policy_sweep_20260805_014026"
BASE_CHECKPOINT = ROOT / "runs/idenvln_heldout_20260925T093354Z_3688599/training/sequence_bc_best_causal_shifted.pt"
SEQUENCE = ROOT / "runs/idenvln_gpu_hybrid_sequence_bc_level23_20260707/run_gpu_hybrid_sequence_bc.py"
ORDERED = ROOT / "runs/idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py"
HARMONIZED = ROOT / "runs/idenvln_harmonized_agent_eval_level23_20260707/run_harmonized_agent_eval_fixed.py"
SEED = 61007


def save(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def task_key(config):
    return (str(config["Task instruction"]), str(config["Scene"]), str(config["Robot"]).lower())


def diagnose(records):
    minima, stop_distances = [], []
    for record in records:
        distances = []
        for step in record["steps"]:
            stage = step["active_target"]
            if stage < len(step["distances_before"]):
                distance = step["distances_before"][stage]
                if distance is not None:
                    distances.append(distance)
                    if step["stop"]:
                        stop_distances.append(distance)
        minima.append(min(distances) if distances else None)
    return {
        "episodes": len(records),
        "first_subgoal_success": sum(bool(record["successes"]) and record["successes"][0] for record in records),
        "reached_active_1m": sum(distance is not None and distance <= 1.0 for distance in minima),
        "reached_active_2m": sum(distance is not None and distance <= 2.0 for distance in minima),
        "minimum_active_distances_m": minima,
        "stop_events": len(stop_distances),
        "stop_distances_m": stop_distances,
        "premature_stop_episodes": sum(record["premature_stop"] for record in records),
    }


def main():
    import numpy as np
    import torch

    RUN.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).parent))
    import train_causal_sequence_bc as causal

    sequence = load(SEQUENCE, "dagger_train_source")
    ordered = load(ORDERED, "dagger_ordered_eval")
    harmonized = load(HARMONIZED, "dagger_harmonized_eval")
    sweep = load(SOURCE / "run_policy_sweep.py", "dagger_policy_sweep")
    for name in ("load_context", "make_args", "tiered_feature", "target_feature", "seq_feature", "select_action_from_logits", "ACTION_TO_IDX", "IDX_TO_ACTION", "SEED", "WINDOW", "SOURCE_DATA", "torch", "BalancedMLP", "TargetConditionedMLP", "SequenceGRU"):
        setattr(ordered, name, getattr(harmonized, name))
    ordered.install_index_aware_target_patch()
    from utils.dataset import TaskDataset

    dataset = TaskDataset(harmonized.make_args(), "test")
    trials = sorted(sequence.SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))
    index_by_key = {task_key(dataset[i][0]): i for i in range(len(dataset))}
    trial_keys = [task_key(sequence.load_context(trial)["task"]) for trial in trials]
    assert len(dataset) == len(trials) == len(index_by_key) == len(set(trial_keys)) == 300
    assert set(trial_keys) == set(index_by_key)

    bank_path = SOURCE / "episode_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    canonical = {int(i) for i in bank["episode_indices"]}
    assert canonical == set(range(100))
    excluded = {tid for tid, key in enumerate(trial_keys) if index_by_key[key] in canonical}
    assert len(excluded) == 100
    retained = sorted(set(range(len(trials))) - excluded)
    split = sequence.split_by_traj(retained)
    assert {key: len(value) for key, value in split.items()} == {"train": 140, "val": 30, "test": 30}
    train_task_indices = [index_by_key[trial_keys[tid]] for tid in sorted(split["train"])]
    val_task_indices = [index_by_key[trial_keys[tid]] for tid in sorted(split["val"])]
    test_task_indices = [index_by_key[trial_keys[tid]] for tid in sorted(split["test"])]
    assert not canonical.intersection(train_task_indices + val_task_indices + test_task_indices)
    assert len(set(train_task_indices + val_task_indices + test_task_indices)) == 200

    rng = random.Random(SEED)
    collect_indices = rng.sample(train_task_indices, 30)
    dev_indices = rng.sample(val_task_indices, 10)
    save("protocol.json", {
        "canonical_bank_sha256": causal.sha(bank_path),
        "baseline_checkpoint": str(BASE_CHECKPOINT),
        "baseline_checkpoint_sha256": causal.sha(BASE_CHECKPOINT),
        "excluded_canonical_trajectories": len(excluded),
        "split_task_indices": {"train": train_task_indices, "val": val_task_indices, "test": test_task_indices},
        "dagger_collection_indices": collect_indices,
        "dev_evaluation_indices": dev_indices,
        "dagger_beta": 0.5,
        "dagger_max_steps": 100,
        "dev_max_steps": 150,
        "stop_penalty": 0.25,
        "teacher_target_coordinate_used_for_training_labels_only": True,
    })

    base_state = sweep.load_policy_from_checkpoint(ordered, {"name": "causal_sequence_bc", "feature": "seq", "checkpoint": str(BASE_CHECKPOINT)})
    assert base_state["status"] == "loaded"

    eval_source = textwrap.dedent(inspect.getsource(ordered.evaluate_policy))
    old_yaw = "yaw = float(sim.agent.get_state().rotation.angle())"
    assert old_yaw in eval_source
    eval_namespace = dict(vars(ordered), signed_yaw=causal.signed_yaw)
    exec(compile(eval_source.replace(old_yaw, "yaw = signed_yaw(sim.agent.get_state().rotation)"), "<signed-dev-evaluator>", "exec"), eval_namespace)
    evaluate = eval_namespace["evaluate_policy"]

    starts = {}
    baseline_records, baseline_summary = evaluate(ordered, dataset, "causal_sequence_bc", "raw", base_state, dev_indices, starts, RUN / "results/baseline.jsonl", 150, 1.0, SEED, RUN / "results/baseline_progress.json", {})
    assert len(baseline_records) == baseline_summary["N_effective"] == 10
    save("results/dev_start_states.json", starts)
    save("results/baseline_summary.json", baseline_summary)
    save("results/baseline_diagnostic.json", diagnose(baseline_records))
    print(f"baseline dev: {diagnose(baseline_records)}", flush=True)

    class TrainSubset:
        def __len__(self):
            return len(collect_indices)

        def __getitem__(self, ordinal):
            return dataset[collect_indices[ordinal]]

    collector_source = textwrap.dedent(inspect.getsource(sweep.collect_dagger_round))
    old_yaw = "yaw = float(state.rotation.angle())"
    assert old_yaw in collector_source
    collector_namespace = dict(vars(sweep), signed_yaw=causal.signed_yaw)
    exec(compile(collector_source.replace(old_yaw, "yaw = signed_yaw(state.rotation)"), "<signed-dagger-collector>", "exec"), collector_namespace)
    collect = collector_namespace["collect_dagger_round"]
    online_x, online_y, _, collection = collect(ordered, TrainSubset(), base_state, RUN / "training/dagger_rollouts.jsonl", 30, 100, 0.5, 1)
    assert collection["episodes_completed"] == 30 and collection["episodes_failed"] == 0 and len(online_y) > 0
    save("training/dagger_collection.json", collection)
    print(f"DAgger samples: {len(online_y)}", flush=True)

    expert_x, expert_y, expert_tids, meta = causal.collect_causal_sequences(sequence)
    kept = [i for i, tid in enumerate(expert_tids) if tid not in excluded]
    expert_x = [expert_x[i] for i in kept]
    expert_y = [expert_y[i] for i in kept]
    expert_tids = [expert_tids[i] for i in kept]
    assert len(expert_x) == len(expert_y) and len(set(expert_tids)) == 200
    assert len(expert_x[0][-1]) == len(online_x[0][-1])
    save("training/data.json", {"expert_samples": len(expert_y), "dagger_samples": len(online_y), "source_cache_sha256": meta["source_cache_sha256"], "train_trajectories": len(split["train"]), "val_trajectories": len(split["val"]), "test_trajectories": len(split["test"])})
    split_indices = {key: [i for i, tid in enumerate(expert_tids) if tid in ids] for key, ids in split.items()}
    split_indices["train"].extend(range(len(expert_y), len(expert_y) + len(online_y)))
    x = torch.tensor(np.asarray(expert_x + online_x, dtype=np.float32))
    y = torch.tensor(expert_y + online_y, dtype=torch.long)
    sequence.RESULTS = RUN / "training"
    sequence.RESULTS.mkdir(exist_ok=True)
    trained = sequence.train_variant("dagger_lite_trainonly", x, y, split_indices, torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0.35, 0.25)
    save("training/result.json", trained)
    state = sweep.load_policy_from_checkpoint(ordered, {"name": "dagger_lite", "feature": "seq", "checkpoint": trained["model_path"]})
    assert state["status"] == "loaded"
    records, summary = evaluate(ordered, dataset, "dagger_lite", "raw", state, dev_indices, starts, RUN / "results/dagger.jsonl", 150, 1.0, SEED, RUN / "results/dagger_progress.json", {})
    assert len(records) == summary["N_effective"] == 10
    save("results/dagger_summary.json", summary)
    save("results/dagger_diagnostic.json", diagnose(records))
    print(f"DAgger dev: {diagnose(records)}", flush=True)
    save("status.json", {"status": "completed", "baseline": baseline_summary, "dagger": summary, "baseline_diagnostic": diagnose(baseline_records), "dagger_diagnostic": diagnose(records)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            save("status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
