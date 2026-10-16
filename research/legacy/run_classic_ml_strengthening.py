#!/usr/bin/env python3
"""
Classic ML Strengthening for IDEN-VLN IEEE Access
==================================================
CPU-only script. Three improvements over the existing classic_ml_baselines.json:

1. STRATIFIED ABLATION — breakdown by level, CMI, instruction length, subtask count
2. FEATURE + HYPERPARAMETER ABLATION — unigram/bigram/combined × C values, val-selected
3. FEATURE IMPORTANCE — top-N tokens per class for best LogReg model

Outputs (written to ./results/):
  classic_ml_ablation.json
  classic_ml_tuning.json
  classic_ml_feature_importance.json
"""

import json, hashlib, os, sys, time, warnings
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TASKS_PATH = "/home/aimldl/LH-VLN/nav_gen/runs/ieee_access_release_export_20260708/iden_vln_tasks.jsonl"
CODE_MIXING_PATH = "/home/aimldl/LH-VLN/nav_gen/runs/ieee_access_experiments_2500_20260704/results/code_mixing_token_metrics.jsonl"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────

def load_tasks():
    """Load all 2500 tasks from IDEN-VLN task JSONL.
       Fields: task_id, task_instruction, objects, region_names, level, robot, scene,
               subtask_list, random_split, scene_heldout_split"""
    tasks = []
    with open(TASKS_PATH) as f:
        for line in f:
            row = json.loads(line)
            # objects field is list of dicts; extract names
            obj_raw = row.get("objects", [])
            if obj_raw and isinstance(obj_raw[0], dict):
                obj_names = [o["name"] for o in obj_raw]
            else:
                obj_names = row.get("object_names", obj_raw)

            tasks.append({
                "task_id": row["task_id"],
                "instruction": row["task_instruction"],
                "objects": obj_names,
                "rooms": row.get("region_names", []),
                "level": row.get("level"),
                "robot": row.get("robot"),
                "scene": row.get("scene"),
                "subtask_count": len(row.get("subtask_list", [])),
                "random_split": row.get("random_split", ""),
                "scene_heldout_split": row.get("scene_heldout_split", ""),
            })
    return tasks


def load_code_mixing():
    """Load per-task code-mixing metrics. Field 'id' is the task_id."""
    cm_map = {}
    with open(CODE_MIXING_PATH) as f:
        for line in f:
            row = json.loads(line)
            cm_map[row["id"]] = {
                "cmi": row.get("cmi", 0),
                "n_tokens": row.get("token_count", 0),
                "switch_point_rate": row.get("switch_point_rate", 0),
            }
    return cm_map


def split_maps(tasks):
    """Use existing split columns from the dataset (not re-hashing)."""
    r_train, r_val, r_test = set(), set(), set()
    s_train, s_val, s_test = set(), set(), set()

    for t in tasks:
        tid = t["task_id"]
        rs = t.get("random_split", "")
        if rs == "train": r_train.add(tid)
        elif rs == "val": r_val.add(tid)
        elif rs == "test": r_test.add(tid)

        ss = t.get("scene_heldout_split", "")
        if ss == "train": s_train.add(tid)
        elif ss == "val": s_val.add(tid)
        elif ss == "test": s_test.add(tid)

    return {
        "random": {"train": r_train, "val": r_val, "test": r_test},
        "scene_heldout": {"train": s_train, "val": s_val, "test": s_test},
    }


def stable_unique(values):
    """Canonicalize labels with first-occurrence-preserving set semantics."""
    output = []
    seen = set()
    for value in values:
        item = str(value).strip().casefold()
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def target_value(task, target):
    if target == "first_object":
        objects = stable_unique(task["objects"])
        return objects[0] if objects else "none"
    elif target == "first_room":
        rooms = stable_unique(task["rooms"])
        return rooms[0] if rooms else "none"
    elif target == "object_sequence":
        return " > ".join(stable_unique(task["objects"]))
    else:
        raise ValueError(target)


def compute_metrics(y_true, y_pred, n_classes, y_proba=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            m["top3_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=min(3, n_classes), labels=list(range(n_classes))))
        except Exception:
            m["top3_accuracy"] = None
    return m


def build_pipeline(model, ngram_range):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=ngram_range, min_df=1)),
        ("clf", model),
    ])


# ── part 1: stratified ablation ──────────────────────────────────────

STRATA_BINS = {
    "level": lambda t: t["level"],
    "cmi_median": lambda t: "high" if (
        cm_data.get(t["task_id"], {}).get("cmi", 0) > cmi_split_val
    ) else "low",
    "instr_len_median": lambda t: "long" if len(t["instruction"].split()) > len_median else "short",
    "subtask_median": lambda t: "high" if t.get("subtask_count", 0) > subtask_median else "low",
}


def run_stratified_ablation(tasks, splits, cm_data):
    print("=" * 60)
    print("PART 1: Stratified Ablation")
    print("=" * 60)

    global cmi_split_val, len_median, subtask_median
    cmis = [cm_data.get(t["task_id"], {}).get("cmi", 0) for t in tasks]
    cmi_split_val = float(np.median(cmis))
    lengths = [len(t["instruction"].split()) for t in tasks]
    len_median = int(np.median(lengths))
    substasks = [t.get("subtask_count", 0) for t in tasks]
    subtask_median = int(np.median(substasks))

    print(f"CMI median split: {cmi_split_val:.2f}")
    print(f"Instruction length median: {len_median}")
    print(f"Subtask count median: {subtask_median}")

    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1),
        "linear_svm": lambda: LinearSVC(class_weight="balanced", dual="auto", max_iter=5000),
    }
    targets = ["first_object", "first_room"]
    split_names = ["random", "scene_heldout"]

    results = []

    for split_name in split_names:
        s = splits[split_name]
        train_ids, test_ids = s["train"], s["test"]
        x_train = [t["instruction"] for t in tasks if t["task_id"] in train_ids]
        x_test_all = [t for t in tasks if t["task_id"] in test_ids]

        for model_name, model_fn in models.items():
            for target in targets:
                print(f"  {split_name} | {model_name} | {target}  ", end="", flush=True)

                y_train = [target_value(t, target) for t in tasks if t["task_id"] in train_ids]
                pipe = build_pipeline(model_fn(), (1, 2))
                pipe.fit(x_train, y_train)

                # overall (all test)
                x_test_texts = [t["instruction"] for t in x_test_all]
                y_test = [target_value(t, target) for t in x_test_all]
                y_pred = pipe.predict(x_test_texts)

                overall = compute_metrics(y_test, y_pred, len(set(y_test)))
                overall["stratum"] = "all"
                overall["split"] = split_name
                overall["model"] = model_name
                overall["target"] = target
                overall["n"] = len(x_test_texts)
                results.append(overall)

                # by level
                for lvl in [2, 3, 4, 5]:
                    subset = [t for t in x_test_all if t["level"] == lvl]
                    if len(subset) < 10:
                        continue
                    y_sub = [target_value(t, target) for t in subset]
                    yp = pipe.predict([t["instruction"] for t in subset])
                    m = compute_metrics(y_sub, yp, len(set(y_sub)))
                    m["stratum"] = f"level_{lvl}"
                    m["split"] = split_name
                    m["model"] = model_name
                    m["target"] = target
                    m["n"] = len(subset)
                    results.append(m)

                # by CMI
                for bin_name in ["low", "high"]:
                    subset = [t for t in x_test_all if STRATA_BINS["cmi_median"](t) == bin_name]
                    if len(subset) < 10:
                        continue
                    y_sub = [target_value(t, target) for t in subset]
                    yp = pipe.predict([t["instruction"] for t in subset])
                    m = compute_metrics(y_sub, yp, len(set(y_sub)))
                    m["stratum"] = f"cmi_{bin_name}"
                    m["split"] = split_name
                    m["model"] = model_name
                    m["target"] = target
                    m["n"] = len(subset)
                    results.append(m)

                # by instruction length
                for bin_name in ["short", "long"]:
                    subset = [t for t in x_test_all if STRATA_BINS["instr_len_median"](t) == bin_name]
                    if len(subset) < 10:
                        continue
                    y_sub = [target_value(t, target) for t in subset]
                    yp = pipe.predict([t["instruction"] for t in subset])
                    m = compute_metrics(y_sub, yp, len(set(y_sub)))
                    m["stratum"] = f"instrlen_{bin_name}"
                    m["split"] = split_name
                    m["model"] = model_name
                    m["target"] = target
                    m["n"] = len(subset)
                    results.append(m)

                # by subtask count
                for bin_name in ["low", "high"]:
                    subset = [t for t in x_test_all if STRATA_BINS["subtask_median"](t) == bin_name]
                    if len(subset) < 10:
                        continue
                    y_sub = [target_value(t, target) for t in subset]
                    yp = pipe.predict([t["instruction"] for t in subset])
                    m = compute_metrics(y_sub, yp, len(set(y_sub)))
                    m["stratum"] = f"subtask_{bin_name}"
                    m["split"] = split_name
                    m["model"] = model_name
                    m["target"] = target
                    m["n"] = len(subset)
                    results.append(m)

                print(f"→ {len(results)} rows")

    out_path = os.path.join(RESULTS_DIR, "classic_ml_ablation.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmi_median": float(cmi_split_val),
            "instr_len_median": len_median,
            "subtask_median": subtask_median,
            "task_count": len(tasks),
            "results": results,
        }, f, indent=2)
    print(f"  saved → {out_path}  ({len(results)} ablation rows)")
    return results


# ── part 2: feature + hyperparameter ablation ────────────────────────

def run_tuning_ablation(tasks, splits):
    print("\n" + "=" * 60)
    print("PART 2: Feature + Hyperparameter Ablation")
    print("=" * 60)

    ngram_options = [(1, 1), (1, 2)]          # unigram-only, combined
    c_values = [0.1, 1.0, 10.0]
    targets = ["first_object", "first_room"]
    split_name = "scene_heldout"  # use scene-held-out for tuning (harder, more informative)

    s = splits[split_name]
    train_ids, val_ids, test_ids = s["train"], s["val"], s["test"]

    x_train_texts = [t["instruction"] for t in tasks if t["task_id"] in train_ids]
    x_val_texts = [t["instruction"] for t in tasks if t["task_id"] in val_ids]
    x_test_texts = [t["instruction"] for t in tasks if t["task_id"] in test_ids]

    results = []

    for target in targets:
        y_train = [target_value(t, target) for t in tasks if t["task_id"] in train_ids]
        y_val = [target_value(t, target) for t in tasks if t["task_id"] in val_ids]
        y_test = [target_value(t, target) for t in tasks if t["task_id"] in test_ids]

        best_logreg = {"score": -1, "config": None}
        best_svm = {"score": -1, "config": None}

        for ngram in ngram_options:
            for C in c_values:
                ng_label = "unigram" if ngram == (1, 1) else "combined"
                print(f"  {target} | {ng_label} | C={C:.1f}  ", end="", flush=True)

                # Logistic Regression
                pipe_lr = build_pipeline(
                    LogisticRegression(max_iter=2000, class_weight="balanced", C=C, n_jobs=1),
                    ngram
                )
                pipe_lr.fit(x_train_texts, y_train)
                val_pred_lr = pipe_lr.predict(x_val_texts)
                val_f1_lr = f1_score(y_val, val_pred_lr, average="weighted", zero_division=0)
                test_pred_lr = pipe_lr.predict(x_test_texts)
                test_m_lr = compute_metrics(y_test, test_pred_lr, len(set(y_test)))

                results.append({
                    "target": target, "model": "logreg", "ngram": ng_label, "C": C,
                    "val_weighted_f1": float(val_f1_lr),
                    "test_accuracy": test_m_lr["accuracy"],
                    "test_weighted_f1": test_m_lr["weighted_f1"],
                    "test_macro_f1": test_m_lr["macro_f1"],
                })
                if val_f1_lr > best_logreg["score"]:
                    best_logreg = {"score": val_f1_lr, "config": (ngram, C, "logreg")}

                # Linear SVM
                pipe_svm = build_pipeline(
                    LinearSVC(class_weight="balanced", dual="auto", max_iter=5000, C=C),
                    ngram
                )
                pipe_svm.fit(x_train_texts, y_train)
                val_pred_svm = pipe_svm.predict(x_val_texts)
                val_f1_svm = f1_score(y_val, val_pred_svm, average="weighted", zero_division=0)
                test_pred_svm = pipe_svm.predict(x_test_texts)
                test_m_svm = compute_metrics(y_test, test_pred_svm, len(set(y_test)))

                results.append({
                    "target": target, "model": "linear_svm", "ngram": ng_label, "C": C,
                    "val_weighted_f1": float(val_f1_svm),
                    "test_accuracy": test_m_svm["accuracy"],
                    "test_weighted_f1": test_m_svm["weighted_f1"],
                    "test_macro_f1": test_m_svm["macro_f1"],
                })
                if val_f1_svm > best_svm["score"]:
                    best_svm = {"score": val_f1_svm, "config": (ngram, C, "svm")}

                print(f"LR={val_f1_lr:.3f}/{test_m_lr['accuracy']:.3f}  SVM={val_f1_svm:.3f}/{test_m_svm['accuracy']:.3f}")

        print(f"  Best LogReg {target}: ngram={best_logreg['config'][0]} C={best_logreg['config'][1]:.1f} valF1={best_logreg['score']:.4f}")
        print(f"  Best SVM    {target}: ngram={best_svm['config'][0]} C={best_svm['config'][1]:.1f} valF1={best_svm['score']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "classic_ml_tuning.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": split_name,
            "ngram_options": ["unigram", "combined"],
            "c_values": c_values,
            "results": results,
        }, f, indent=2)
    print(f"  saved → {out_path}  ({len(results)} tuning rows)")
    return results


# ── part 3: feature importance ───────────────────────────────────────

def run_feature_importance(tasks, splits):
    print("\n" + "=" * 60)
    print("PART 3: Feature Importance (Top Tokens)")
    print("=" * 60)

    split_name = "random"
    s = splits[split_name]
    train_ids, test_ids = s["train"], s["test"]
    targets = ["first_object", "first_room"]

    results = {}

    for target in targets:
        print(f"  {target}  ", end="", flush=True)
        x_train = [t["instruction"] for t in tasks if t["task_id"] in train_ids]
        x_test_texts = [t["instruction"] for t in tasks if t["task_id"] in test_ids]
        y_train = [target_value(t, target) for t in tasks if t["task_id"] in train_ids]
        y_test = [target_value(t, target) for t in tasks if t["task_id"] in test_ids]

        pipe = build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, n_jobs=1),
            (1, 2)
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test_texts)
        overall = compute_metrics(y_test, y_pred, len(set(y_test)))
        print(f"acc={overall['accuracy']:.3f} f1(w)={overall['weighted_f1']:.3f}  ", end="", flush=True)

        # Extract top tokens per class
        coef = pipe.named_steps["clf"].coef_
        vocab = pipe.named_steps["tfidf"].get_feature_names_out()
        classes = pipe.named_steps["clf"].classes_

        top_per_class = {}
        for i, cls in enumerate(classes):
            if len(classes) <= 1:
                break
            top_indices = np.argsort(coef[i])[-5:][::-1]
            top_tokens = [(vocab[j], float(coef[i][j])) for j in top_indices if coef[i][j] > 0]
            if top_tokens:
                top_per_class[str(cls)] = top_tokens

        # Limit to top 15 classes by frequency
        from collections import Counter
        class_freq = Counter(y_test)
        top_classes = [cls for cls, _ in class_freq.most_common(15)]

        results[target] = {
            "test_accuracy": overall["accuracy"],
            "test_weighted_f1": overall["weighted_f1"],
            "n_classes": len(classes),
            "top_classes": {cls: top_per_class.get(str(cls), []) for cls in top_classes if str(cls) in top_per_class},
        }
        print(f"→ {sum(1 for v in results[target]['top_classes'].values() if v)} classes with tokens")

    out_path = os.path.join(RESULTS_DIR, "classic_ml_feature_importance.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Top-5 tokens per class (positive coefficients only) for TF-IDF Logistic Regression on random split",
            **results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading tasks...")
    tasks = load_tasks()
    print(f"  {len(tasks)} tasks loaded")

    print("Loading code-mixing data...")
    global cm_data
    cm_data = load_code_mixing()
    print(f"  {len(cm_data)} CM entries")

    print("Computing splits...")
    splits = split_maps(tasks)
    print(f"  random: train={len(splits['random']['train'])} val={len(splits['random']['val'])} test={len(splits['random']['test'])}")
    print(f"  heldout: train={len(splits['scene_heldout']['train'])} val={len(splits['scene_heldout']['val'])} test={len(splits['scene_heldout']['test'])}")

    results_ablation = run_stratified_ablation(tasks, splits, cm_data)
    results_tuning = run_tuning_ablation(tasks, splits)
    run_feature_importance(tasks, splits)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Outputs in {RESULTS_DIR}/:")
    print(f"  classic_ml_ablation.json")
    print(f"  classic_ml_tuning.json")
    print(f"  classic_ml_feature_importance.json")


if __name__ == "__main__":
    main()
