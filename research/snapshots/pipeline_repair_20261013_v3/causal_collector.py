#!/usr/bin/env python3
"""Train sequence BC on pre-action frames and evaluate on the canonical bank."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import pickletools
import sys
import textwrap
import traceback
from collections import Counter
from pathlib import Path

LH_ROOT = Path("/home/aimldl/LH-VLN")
RUN = Path(os.environ.get("IDENVLN_RUN_DIR", str(LH_ROOT / "runs/idenvln_causal_sequence_bc_cache_20260925")))
SOURCE = LH_ROOT / "runs/idenvln_policy_sweep_20260805_014026"
SEQ = LH_ROOT / "runs/idenvln_gpu_hybrid_sequence_bc_level23_20260707/run_gpu_hybrid_sequence_bc.py"
LEGACY = LH_ROOT / "runs/idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py"
HARMONIZED = LH_ROOT / "runs/idenvln_harmonized_agent_eval_level23_20260707/run_harmonized_agent_eval_fixed.py"


def save(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_causal_sequences(module):
    cache = SOURCE / "training/dagger/expert_sequence_features.pkl"
    allowed = {"BINFLOAT", "LONG_BINGET", "MARK", "APPENDS", "MEMOIZE", "EMPTY_LIST", "BININT1", "BININT2", "BINGET", "FRAME", "SHORT_BINUNICODE", "EMPTY_DICT", "SETITEMS", "PROTO", "SETITEM", "STOP"}
    with cache.open("rb") as handle:
        unexpected = {op.name for op, _, _ in pickletools.genops(handle) if op.name not in allowed}
    if unexpected:
        raise ValueError(f"Cache contains unsafe or unexpected pickle opcodes: {unexpected}")

    class DataOnlyUnpickler(pickle.Unpickler):
        def find_class(self, module_name, class_name):
            raise ValueError(f"Cache references executable class: {module_name}.{class_name}")

        def persistent_load(self, identifier):
            raise ValueError(f"Cache references external data: {identifier}")

    with cache.open("rb") as handle:
        old = DataOnlyUnpickler(handle).load()
    old_x, ys, tids = old["xs"], old["ys"], old["tids"]
    assert len(old_x) == len(ys) == len(tids) and len(ys) > 1000
    trials = sorted(module.SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))
    xs = []
    history = []
    last_tid = None
    zero = [0.0] * len(old_x[0][-1])
    for i, (label, tid) in enumerate(zip(ys, tids)):
        first = tid != last_tid
        previous = module.ACTION_TO_IDX["stop"] if first else ys[i - 1]
        if first:
            history = []
            trial = trials[tid]
            context = module.load_context(trial)
            observe = next((p for p in trial.iterdir() if p.name.startswith("0_observe")), None)
            if observe is None:
                raise FileNotFoundError(f"Initial observation missing: {trial}")
            feature = module.frame_feature(observe, context["pos"][0], context["yaw"][0], previous, context)
            if feature is None:
                raise FileNotFoundError(f"Initial camera frame missing: {observe}")
            if tid % 50 == 0:
                print(f"initial observations: {tid + 1}/{len(trials)}", flush=True)
        else:
            feature = list(old_x[i - 1][-1])
        feature[40:44] = [float(action == previous) for action in range(4)]
        history.append(feature)
        window = history[-module.WINDOW :]
        xs.append([zero] * (module.WINDOW - len(window)) + window)
        last_tid = tid
    assert len(xs) == len(ys) and len(set(tids)) == len(trials)
    return xs, ys, tids, {"trial_count": len(trials), "samples": len(ys), "actions": dict(Counter(module.IDX_TO_ACTION[y] for y in ys)), "source_cache": str(cache), "source_cache_sha256": sha(cache), "safe_opcodes": sorted(allowed), "alignment": "pre-action frame and pose predict next action; previous action one-hot rebuilt"}


def signed_yaw(rotation):
    w, x, y, z = (float(getattr(rotation, field)) for field in ("w", "x", "y", "z"))
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 0.0
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return math.atan2(2 * (w*y + x*z), 1 - 2 * (y*y + z*z))


def main():
    import torch

    RUN.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(LH_ROOT))
    os.chdir(LH_ROOT)
    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    sequence = load(SEQ, "causal_sequence_training")
    legacy = load(LEGACY, "causal_sequence_evaluator")
    harmonized = load(HARMONIZED, "causal_sequence_harmonized")
    policy_sweep = load(SOURCE / "run_policy_sweep.py", "causal_sequence_sweep")
    for name in ("load_context", "make_args", "tiered_feature", "target_feature", "seq_feature", "select_action_from_logits", "ACTION_TO_IDX", "IDX_TO_ACTION", "SEED", "WINDOW", "SOURCE_DATA", "torch", "BalancedMLP", "TargetConditionedMLP", "SequenceGRU"):
        setattr(legacy, name, getattr(harmonized, name))
    legacy.install_index_aware_target_patch()
    from utils.dataset import TaskDataset

    dataset = TaskDataset(harmonized.make_args(), "test")
    bank_path = SOURCE / "episode_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    indices = [int(i) for i in bank["episode_indices"]]
    assert len(indices) == 100 and indices == list(range(100))
    exclude_bank = os.environ.get("IDENVLN_EXCLUDE_BANK") == "1"
    stop_penalty = float(os.environ.get("IDENVLN_STOP_PENALTY", "0.25"))
    if not math.isfinite(stop_penalty) or stop_penalty <= 0:
        raise ValueError("IDENVLN_STOP_PENALTY must be a positive finite number")
    smoke_only = os.environ.get("IDENVLN_SMOKE_ONLY") == "1"
    save("protocol.json", {"source_bank": str(bank_path), "source_bank_sha256": sha(bank_path), "indices": indices, "max_steps": 150, "success_distance": 1.0, "training_source": str(sequence.SOURCE_DATA), "exclude_bank_from_training": exclude_bank, "stop_penalty": stop_penalty, "smoke_only": smoke_only, "change": "pre-action frame/pose alignment and signed live yaw"})

    # In the first trajectory action 1 is a right turn, so its input must use
    # observe frame 0 and yaw[0], not the resulting frame 1 and yaw[1].
    first = sorted(sequence.SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))[0]
    context = sequence.load_context(first)
    assert any(p.name.startswith("0_observe") for p in first.iterdir())
    assert context["yaw"][0] != context["yaw"][1]

    xs, ys, tids, meta = collect_causal_sequences(sequence)
    if exclude_bank:
        bank_instructions = {str(dataset[i][0]["Task instruction"]) for i in indices}
        trials = sorted(sequence.SOURCE_DATA.glob("task/batch_generated_level23/*/*/success/trial_1"))
        excluded = {tid for tid, trial in enumerate(trials) if sequence.load_context(trial)["instruction"] in bank_instructions}
        assert len(excluded) == len(bank_instructions) == 100
        kept = [i for i, tid in enumerate(tids) if tid not in excluded]
        xs, ys, tids = [xs[i] for i in kept], [ys[i] for i in kept], [tids[i] for i in kept]
        meta.update({"excluded_bank_trajectories": len(excluded), "remaining_trajectories": len(set(tids)), "remaining_samples": len(ys)})
        assert len(set(tids)) == len(trials) - len(excluded)
    save("training/data.json", meta)
    import numpy as np
    x = torch.tensor(np.asarray(xs, dtype=np.float32))
    y = torch.tensor(ys, dtype=torch.long)
    split = sequence.split_by_traj(tids)
    split_indices = {key: [i for i, tid in enumerate(tids) if tid in ids] for key, ids in split.items()}
    sequence.RESULTS = RUN / "training"
    sequence.RESULTS.mkdir(exist_ok=True)
    result = sequence.train_variant("causal_shifted", x, y, split_indices, torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0.35, stop_penalty)
    save("training/result.json", result)
    print(f"validation stop: {result['val_metrics']['per_class']['stop']}", flush=True)
    checkpoint = Path(result["model_path"])
    state = policy_sweep.load_policy_from_checkpoint(legacy, {"name": "causal_sequence_bc", "feature": "seq", "checkpoint": str(checkpoint)})
    assert state["status"] == "loaded"

    source = textwrap.dedent(inspect.getsource(legacy.evaluate_policy))
    old = "yaw = float(sim.agent.get_state().rotation.angle())"
    assert old in source
    namespace = dict(vars(legacy), signed_yaw=signed_yaw)
    exec(compile(source.replace(old, "yaw = signed_yaw(sim.agent.get_state().rotation)"), "<causal-signed-evaluator>", "exec"), namespace)
    evaluator = namespace["evaluate_policy"]
    summaries = {}
    evaluations = (("smoke", indices[:10]),) if smoke_only else (("smoke", indices[:10]), ("full", indices))
    for label, selected in evaluations:
        print(f"{label}: {len(selected)} episodes", flush=True)
        starts = {str(i): bank["start_states"][str(i)] for i in selected}
        records, summary = evaluator(legacy, dataset, "causal_sequence_bc", "raw", state, selected, starts, RUN / "results" / f"{label}.jsonl", 150, 1.0, 61007, RUN / "results" / f"{label}_progress.json", {})
        assert len(records) == len(selected) and summary["N_effective"] == len(selected)
        summaries[label] = summary
        save(f"results/{label}_summary.json", summary)
        print(f"{label}: {summary['metrics']}; stop_events={sum(record['stop_events'] for record in records)}", flush=True)
    save("status.json", {"status": "completed", "checkpoint_sha256": sha(checkpoint), "summaries": summaries})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            save("status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
