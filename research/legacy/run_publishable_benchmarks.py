#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


NAV_GEN = Path("/home/aimldl/LH-VLN/nav_gen")
RUN_ROOT = NAV_GEN / "runs/ieee_access_experiments_2500_20260704"
LOGS = RUN_ROOT / "logs"
RESULTS = RUN_ROOT / "results"
DOWNSTREAM = NAV_GEN / "runs/audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704"
STEP_TASK = DOWNSTREAM / "step_task"
API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY", "dummy")
TIMEOUT = int(os.environ.get("IEEE_EXP_TIMEOUT", "180"))

SOURCES = [
    ("old", NAV_GEN / "runs/audit_regenerate_cs_aligned_v5_levels2_5_split_step_success1206/success_pool/task", "config.json"),
    ("v2", NAV_GEN / "runs/audit_regenerate_cs_aligned_v5_levels2_5_habitat_validate_2300_qwen7b_20260701_v2/combined_task", "success/trial_*/task.json"),
    ("topup", NAV_GEN / "runs/audit_regenerate_cs_aligned_v5_levels2_5_habitat_validate_topup2500_qwen7b_20260703/combined_task", "success/trial_*/task.json"),
]

ID_WORDS = {
    "pergi", "ke", "lalu", "ambil", "dengan", "perlahan", "kemudian", "dan", "letakkan",
    "secara", "aman", "cek", "akhir", "mulai", "menuju", "ruang", "di", "dari", "sampai",
    "setelah", "sebelum", "kembali", "ambilkan", "taruh", "bawa", "masuk", "keluar",
    "dekat", "jauh", "sebelah", "atas", "bawah", "depan", "belakang", "kanan", "kiri",
}
EN_WORDS = {
    "move", "on", "target", "end", "start", "bedroom", "bathroom", "kitchen", "living",
    "room", "office", "hall", "cabinet", "door", "lamp", "table", "chair", "sink",
    "toilet", "shelf", "bed", "picture", "sofa", "stool", "counter", "closet",
}
REGION_NAMES = [
    "bedroom", "bathroom", "kitchen", "living room", "office", "hall", "laundry room",
    "dining room", "garage", "rec room", "mixed room",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def norm(value: Any) -> str:
    return str(value).strip().casefold() if value is not None else ""


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def stable_unique(values: list[Any]) -> list[str]:
    """Canonicalize labels with first-occurrence-preserving set semantics."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = norm(value)
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def object_labels(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return stable_unique(out)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list) and item:
                out.append(norm(item[0]).lower())
            elif isinstance(item, dict):
                out.append(norm(item.get("name") or item.get("object") or item.get("label") or item).lower())
            elif item not in (None, ""):
                out.append(norm(item).lower())
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(object_labels(item))
        return stable_unique(out)
    return stable_unique([value])


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [norm(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [norm(item) for item in value.values() if item not in (None, "")]
    return [norm(value)]


def task_hash(source: str, path: Path, data: dict[str, Any]) -> str:
    payload = str(path) + norm(data.get("Scene")) + norm(data.get("Task instruction"))
    return f"{source}_{hashlib.sha1(payload.encode()).hexdigest()[:16]}"


def discover_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, root, pattern in SOURCES:
        paths = sorted(root.glob("*/*/config.json")) if source == "old" else sorted(root.glob(f"*/*/{pattern}"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            level = norm(data.get("Generated level")) or next((part for part in path.parts if part in {"2", "3", "4", "5"}), "")
            audit = data.get("Audit") if isinstance(data.get("Audit"), dict) else {}
            rows.append(
                {
                    "id": task_hash(source, path, data),
                    "source": source,
                    "path": str(path),
                    "level": level,
                    "robot": norm(data.get("Robot")).lower(),
                    "scene": norm(data.get("Scene")),
                    "instruction": norm(data.get("Task instruction")),
                    "objects": object_labels(data.get("Object")),
                    "rooms": as_list(data.get("Region Name") or data.get("Region")),
                    "subtasks": as_list(data.get("Subtask list")),
                    "audit": audit,
                }
            )
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f1_list(pred: list[Any], gold: list[Any]) -> float:
    pred_set = set(stable_unique(pred))
    gold_set = set(stable_unique(gold))
    inter = len(pred_set & gold_set)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def task_tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def token_lang(token: str) -> str:
    token = token.lower()
    if token in ID_WORDS:
        return "id"
    if token in EN_WORDS:
        return "en"
    if re.fullmatch(r"[a-z]+", token):
        return "en"
    return "unk"


def token_langs(text: str) -> list[str]:
    return [token_lang(token) for token in task_tokens(text)]


def code_mixing_metrics() -> dict[str, Any]:
    ensure()
    tasks = discover_tasks()
    per_row = []
    total_lang = Counter()
    switch_rates = []
    i_indices = []
    cmi_values = []
    m_values = []
    span_lengths = []
    for row in tasks:
        langs_all = token_langs(row["instruction"])
        langs = [lang for lang in langs_all if lang in {"id", "en"}]
        counts = Counter(langs)
        total_lang.update(counts)
        total = sum(counts.values())
        switches = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
        switch_rate = switches / (len(langs) - 1) if len(langs) > 1 else 0.0
        cmi = 100 * (1 - (max(counts.values()) / total)) if total else 0.0
        probs = [counts[lang] / total for lang in ("id", "en") if counts[lang] > 0 and total]
        m_index = -sum(p * math.log(p) for p in probs) / math.log(2) if len(probs) > 1 else 0.0
        i_index = switch_rate
        spans = []
        if langs:
            current = langs[0]
            length = 1
            for lang in langs[1:]:
                if lang == current:
                    length += 1
                else:
                    spans.append(length)
                    current = lang
                    length = 1
            spans.append(length)
        span_lengths.extend(spans)
        switch_rates.append(switch_rate)
        i_indices.append(i_index)
        cmi_values.append(cmi)
        m_values.append(m_index)
        per_row.append(
            {
                "id": row["id"],
                "level": row["level"],
                "robot": row["robot"],
                "token_count": len(langs),
                "id_count": counts["id"],
                "en_count": counts["en"],
                "cmi": cmi,
                "m_index": m_index,
                "i_index": i_index,
                "switch_point_rate": switch_rate,
                "mixed_span_mean": statistics.mean(spans) if spans else 0.0,
            }
        )
    total_tokens = total_lang["id"] + total_lang["en"]
    jsonl = RESULTS / "code_mixing_token_metrics.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    for row in per_row:
        append_jsonl(jsonl, row)
    out = {
        "generated_at": now(),
        "task_count": len(tasks),
        "token_count_labeled": total_tokens,
        "indonesian_ratio": round(total_lang["id"] / total_tokens, 4) if total_tokens else 0.0,
        "english_ratio": round(total_lang["en"] / total_tokens, 4) if total_tokens else 0.0,
        "cmi_mean": round(statistics.mean(cmi_values), 4),
        "m_index_mean": round(statistics.mean(m_values), 4),
        "i_index_mean": round(statistics.mean(i_indices), 4),
        "switch_point_rate_mean": round(statistics.mean(switch_rates), 4),
        "mixed_span_length_mean": round(statistics.mean(span_lengths), 4) if span_lengths else 0.0,
        "jsonl": str(jsonl),
        "method_note": "Heuristic token-level ID/EN labeling; object/action English tokens are counted as EN.",
    }
    write_json(RESULTS / "code_mixing_token_metrics.json", out)
    return out


def split_maps(tasks: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    sorted_tasks = sorted(tasks, key=lambda row: stable_hash(row["id"]))
    random_map: dict[str, str] = {}
    for index, row in enumerate(sorted_tasks):
        frac = index / len(sorted_tasks)
        random_map[row["id"]] = "train" if frac < 0.8 else ("val" if frac < 0.9 else "test")
    scene_count = Counter(row["scene"] for row in tasks)
    scenes = sorted(scene_count, key=stable_hash)
    scene_split: dict[str, str] = {}
    running = 0
    total = len(tasks)
    for scene in scenes:
        split = "train" if running < 0.8 * total else ("val" if running < 0.9 * total else "test")
        scene_split[scene] = split
        running += scene_count[scene]
    scene_map = {row["id"]: scene_split[row["scene"]] for row in tasks}
    return {"random": random_map, "scene_heldout": scene_map}


def split_summary(tasks: list[dict[str, Any]], split_map: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in tasks if split_map[row["id"]] == split]
        out[split] = {
            "count": len(rows),
            "scene_count": len(set(row["scene"] for row in rows)),
            "level": dict(sorted(Counter(row["level"] for row in rows).items())),
            "robot": dict(sorted(Counter(row["robot"] for row in rows).items())),
        }
    split_scenes = {split: {row["scene"] for row in tasks if split_map[row["id"]] == split} for split in ("train", "val", "test")}
    out["scene_overlap"] = {
        "train_val": len(split_scenes["train"] & split_scenes["val"]),
        "train_test": len(split_scenes["train"] & split_scenes["test"]),
        "val_test": len(split_scenes["val"] & split_scenes["test"]),
    }
    return out


def target_value(row: dict[str, Any], target: str) -> str:
    if target == "first_object":
        objects = stable_unique(row["objects"])
        return objects[0] if objects else ""
    if target == "first_room":
        rooms = stable_unique(row["rooms"])
        return rooms[0] if rooms else ""
    if target == "object_sequence":
        return " > ".join(stable_unique(row["objects"]))
    raise ValueError(target)


def topk_accuracy(model: Pipeline, texts: list[str], gold: list[str], k: int = 3) -> float | None:
    try:
        classes = list(model.named_steps["clf"].classes_)
        if hasattr(model.named_steps["clf"], "predict_proba"):
            scores = model.predict_proba(texts)
        else:
            scores = model.decision_function(texts)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
        hits = 0
        for row_scores, label in zip(scores, gold):
            top = [classes[index] for index in np.argsort(row_scores)[::-1][:k]]
            hits += int(label in top)
        return hits / len(gold) if gold else None
    except Exception:
        return None


def train_eval_classic(tasks: list[dict[str, Any]], split_name: str, split_map: dict[str, str], target: str, model_name: str) -> dict[str, Any]:
    rows = [row for row in tasks if target_value(row, target)]
    train = [row for row in rows if split_map[row["id"]] == "train"]
    val = [row for row in rows if split_map[row["id"]] == "val"]
    test = [row for row in rows if split_map[row["id"]] == "test"]
    y_train = [target_value(row, target) for row in train]
    y_test = [target_value(row, target) for row in test]
    x_train = [row["instruction"] for row in train]
    x_test = [row["instruction"] for row in test]
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    elif model_name == "linear_svm":
        clf = LinearSVC(class_weight="balanced", dual="auto", max_iter=5000)
    else:
        raise ValueError(model_name)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=1)),
            ("clf", clf),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = list(pipe.predict(x_test))
    majority = Counter(y_train).most_common(1)[0][0]
    labels = sorted(set(y_train) | set(y_test))
    return {
        "split": split_name,
        "model": f"tfidf_{model_name}",
        "target": target,
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "class_count": len(set(y_train)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4) if y_test else 0.0,
        "macro_f1": round(float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)), 4) if y_test else 0.0,
        "weighted_f1": round(float(f1_score(y_test, pred, labels=labels, average="weighted", zero_division=0)), 4) if y_test else 0.0,
        "top3_accuracy": None if topk_accuracy(pipe, x_test, y_test, 3) is None else round(float(topk_accuracy(pipe, x_test, y_test, 3)), 4),
        "majority_accuracy": round(sum(1 for y in y_test if y == majority) / len(y_test), 4) if y_test else 0.0,
    }


def classic_ml_baselines(limit: int | None = None) -> dict[str, Any]:
    ensure()
    tasks = discover_tasks()
    if limit:
        tasks = sorted(tasks, key=lambda row: stable_hash(row["id"]))[:limit]
    maps = split_maps(tasks)
    records = []
    for split_name, split_map in maps.items():
        for target in ("first_object", "first_room", "object_sequence"):
            for model_name in ("logistic_regression", "linear_svm"):
                records.append(train_eval_classic(tasks, split_name, split_map, target, model_name))
    out = {
        "generated_at": now(),
        "task_count": len(tasks),
        "split_summary": {name: split_summary(tasks, smap) for name, smap in maps.items()},
        "results": records,
    }
    write_json(RESULTS / "classic_ml_baselines.json", out)
    write_json(RESULTS / "scene_heldout_benchmark.json", {"generated_at": now(), "split_summary": out["split_summary"], "classic_ml_results": records})
    return out


def parse_jsonish(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"raw": text}


def post_chat(model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    response = requests.post(
        f"{API_BASE.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": 256},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def llm_compare(model: str, label: str, sample_size: int = 2500, resume: bool = True) -> dict[str, Any]:
    ensure()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    summary_path = RESULTS / f"baseline_llm_instruction_parse_{safe}.json"
    jsonl_path = RESULTS / f"baseline_llm_instruction_parse_{safe}.jsonl"
    tasks = sorted(discover_tasks(), key=lambda row: stable_hash(row["id"]))[:sample_size]
    done: dict[str, dict[str, Any]] = {}
    if resume and jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            done[row.get("id", "")] = row
    records = [done[row["id"]] for row in tasks if row["id"] in done]
    for index, row in enumerate(tasks):
        if row["id"] in done:
            continue
        prompt = (
            "Extract the ordered target objects and room/region names from this Indonesian-English embodied "
            "navigation instruction. Return only JSON with keys objects and rooms, each a list of strings.\n\n"
            "Instruction: " + row["instruction"]
        )
        try:
            pred = parse_jsonish(
                post_chat(
                    model,
                    [
                        {"role": "system", "content": "You are a strict information extraction evaluator."},
                        {"role": "user", "content": prompt},
                    ],
                    0.0,
                )
            )
            pred_objects = pred.get("objects", []) if isinstance(pred, dict) else []
            pred_rooms = pred.get("rooms", []) if isinstance(pred, dict) else []
            rec = {
                "index": index,
                "id": row["id"],
                "level": row["level"],
                "robot": row["robot"],
                "scene": row["scene"],
                "status": "ok",
                "object_f1": f1_list(pred_objects, row["objects"]),
                "room_f1": f1_list(pred_rooms, row["rooms"]),
                "object_order_exact": [str(item).lower().strip() for item in pred_objects] == [item.lower().strip() for item in row["objects"]],
                "pred_objects": pred_objects,
                "pred_rooms": pred_rooms,
            }
        except Exception as exc:
            rec = {
                "index": index,
                "id": row["id"],
                "level": row["level"],
                "robot": row["robot"],
                "scene": row["scene"],
                "status": "error",
                "error": repr(exc),
                "object_f1": 0.0,
                "room_f1": 0.0,
                "object_order_exact": False,
            }
        append_jsonl(jsonl_path, rec)
        records.append(rec)
        if len(records) % 50 == 0:
            write_json(summary_path.with_suffix(".progress.json"), {"label": label, "model": model, "completed": len(records), "sample_size": len(tasks), "updated_at": now()})
    summary = summarize_llm_records(label, model, tasks, records, jsonl_path)
    write_json(summary_path, summary)
    return summary


def summarize_llm_records(label: str, model: str, tasks: list[dict[str, Any]], records: list[dict[str, Any]], jsonl_path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "sample_size": len(tasks),
        "completed_count": len(records),
        "object_f1_mean": round(statistics.mean([row.get("object_f1", 0.0) for row in records]), 4) if records else 0.0,
        "room_f1_mean": round(statistics.mean([row.get("room_f1", 0.0) for row in records]), 4) if records else 0.0,
        "object_order_exact_rate": round(sum(1 for row in records if row.get("object_order_exact")) / len(records), 4) if records else 0.0,
        "error_count": sum(1 for row in records if row.get("status") != "ok"),
        "completed_at": now(),
        "jsonl": str(jsonl_path),
    }


def metric_group(rows: list[dict[str, Any]], keyfn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(keyfn(row))].append(row)
    return {
        key: {
            "count": len(values),
            "object_f1": round(statistics.mean([row.get("object_f1", 0.0) for row in values]), 4),
            "room_f1": round(statistics.mean([row.get("room_f1", 0.0) for row in values]), 4),
            "order_exact": round(sum(1 for row in values if row.get("object_order_exact")) / len(values), 4),
        }
        for key, values in groups.items()
    }


def llm_scene_reports() -> dict[str, Any]:
    tasks = discover_tasks()
    task_by_id = {row["id"]: row for row in tasks}
    maps = split_maps(tasks)
    candidates = {
        "qwen2.5_7b_full2500": RESULTS / "baseline_llm_instruction_parse_full2500.jsonl",
        "llama3.1_8b_full2500": RESULTS / "baseline_llm_instruction_parse_llama3.1_8b_full2500.jsonl",
        "mistral_7b_full2500": RESULTS / "baseline_llm_instruction_parse_mistral_7b_full2500.jsonl",
        "rule_full2500": RESULTS / "baseline_rule_extractor_full2500.jsonl",
    }
    reports: dict[str, Any] = {}
    for name, path in candidates.items():
        rows = read_jsonl(path)
        if not rows:
            continue
        for row in rows:
            task = task_by_id.get(row["id"], {})
            row.setdefault("level", task.get("level", ""))
            row.setdefault("robot", task.get("robot", ""))
            row["random_split"] = maps["random"].get(row["id"], "")
            row["scene_heldout_split"] = maps["scene_heldout"].get(row["id"], "")
            row["instruction_words"] = len(task_tokens(task.get("instruction", "")))
            row["subtask_count"] = len(task.get("subtasks", []))
        reports[name] = {
            "overall": {
                "count": len(rows),
                "object_f1": round(statistics.mean([row.get("object_f1", 0.0) for row in rows]), 4),
                "room_f1": round(statistics.mean([row.get("room_f1", 0.0) for row in rows]), 4),
                "order_exact": round(sum(1 for row in rows if row.get("object_order_exact")) / len(rows), 4),
            },
            "by_random_split": metric_group(rows, lambda row: row["random_split"]),
            "by_scene_heldout_split": metric_group(rows, lambda row: row["scene_heldout_split"]),
            "by_level": metric_group(rows, lambda row: row.get("level")),
            "by_robot": metric_group(rows, lambda row: row.get("robot")),
        }
    return reports


def downstream_summary() -> dict[str, Any]:
    path = DOWNSTREAM / "logs/latest_summary.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data["trail_list_count_observed"] = sum(1 for _ in (DOWNSTREAM / "split/trail_list.txt").open(encoding="utf-8")) if (DOWNSTREAM / "split/trail_list.txt").exists() else 0
    data["step_task_json_count_observed"] = len(list(STEP_TASK.glob("*.json")))
    return data


def failure_mode_breakdown() -> dict[str, Any]:
    patterns = {
        "json_parse_or_format": ["json", "parse", "format"],
        "object_or_region_mismatch": ["object", "region", "room", "mismatch", "not found"],
        "trajectory_or_unreachable": ["trajectory", "unreachable", "path", "navmesh"],
        "simulator_or_timeout": ["timeout", "simulator", "habitat", "exception", "error"],
        "split_or_step": ["split", "step_task", "step"],
    }
    files: list[Path] = []
    run_roots = sorted((NAV_GEN / "runs").glob("audit_regenerate_cs_aligned_v5_levels2_5*"))
    for root in run_roots:
        candidates = []
        candidates.extend(root.glob("logs/*"))
        candidates.extend(root.glob("*summary*.json"))
        candidates.extend(root.glob("**/*summary*.json"))
        candidates.extend(root.glob("fail/**/*.json"))
        candidates.extend(root.glob("failed/**/*.json"))
        candidates.extend(root.glob("pending/**/*.json"))
        for path in candidates:
            if not path.is_file() or path in files:
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".log", ".out", ".txt"}:
                continue
            try:
                if path.stat().st_size > 5_000_000:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= 2500:
                break
        if len(files) >= 2500:
            break
    counts = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if not any(word in text for word in ["fail", "error", "timeout", "unreachable", "mismatch", "pending"]):
            continue
        scanned += 1
        matched = False
        for category, words in patterns.items():
            if any(word in text for word in words):
                counts[category] += 1
                matched = True
                if len(examples[category]) < 5:
                    examples[category].append(str(path))
        if not matched:
            counts["other_or_unclassified"] += 1
            if len(examples["other_or_unclassified"]) < 5:
                examples["other_or_unclassified"].append(str(path))
    downstream = downstream_summary()
    out = {
        "generated_at": now(),
        "method_note": "File-level heuristic scan over available run logs/results; counts indicate files mentioning failure categories, not mutually exclusive event counts.",
        "files_scanned_with_failure_terms": scanned,
        "category_file_counts": dict(counts),
        "examples": dict(examples),
        "final_downstream": {
            "split_done": downstream.get("split_done"),
            "split_pending": downstream.get("split_pending"),
            "split_failed": downstream.get("split_failed"),
            "trail_list_count": downstream.get("trail_list_count"),
            "step_task_json_count": downstream.get("step_task_json_count"),
            "status": downstream.get("status"),
        },
    }
    write_json(RESULTS / "failure_mode_breakdown.json", out)
    return out


def publishable_summary() -> dict[str, Any]:
    summary = {
        "generated_at": now(),
        "downstream": downstream_summary(),
        "files": {
            "classic_ml_baselines": str(RESULTS / "classic_ml_baselines.json"),
            "scene_heldout_benchmark": str(RESULTS / "scene_heldout_benchmark.json"),
            "code_mixing_token_metrics": str(RESULTS / "code_mixing_token_metrics.json"),
            "failure_mode_breakdown": str(RESULTS / "failure_mode_breakdown.json"),
            "llama_full": str(RESULTS / "baseline_llm_instruction_parse_llama3.1_8b_full2500.json"),
            "mistral_full": str(RESULTS / "baseline_llm_instruction_parse_mistral_7b_full2500.json"),
        },
    }
    if (RESULTS / "classic_ml_baselines.json").exists():
        summary["classic_ml"] = json.loads((RESULTS / "classic_ml_baselines.json").read_text(encoding="utf-8"))
    if (RESULTS / "code_mixing_token_metrics.json").exists():
        summary["code_mixing"] = json.loads((RESULTS / "code_mixing_token_metrics.json").read_text(encoding="utf-8"))
    if (RESULTS / "failure_mode_breakdown.json").exists():
        summary["failure_modes"] = json.loads((RESULTS / "failure_mode_breakdown.json").read_text(encoding="utf-8"))
    summary["llm_scene_reports"] = llm_scene_reports()
    write_json(RESULTS / "publishable_benchmark_summary.json", summary)
    return summary


def run_cpu(limit: int | None = None) -> dict[str, Any]:
    ensure()
    classic = classic_ml_baselines(limit)
    code = code_mixing_metrics()
    failure = failure_mode_breakdown()
    summary = publishable_summary()
    out = {"classic": classic, "code_mixing": code, "failure_modes": failure, "summary": summary}
    print(json.dumps({"status": "completed", "generated_at": now(), "task_count": classic["task_count"]}, indent=2))
    return out


def run_llm_full(models: list[str]) -> dict[str, Any]:
    ensure()
    outputs = {}
    mapping = {
        "llama": ("llama3.1:8b", "llama3.1_8b_full2500"),
        "mistral": ("mistral:7b-instruct", "mistral_7b_full2500"),
    }
    for key in models:
        model, label = mapping[key]
        outputs[label] = llm_compare(model, label, 2500, True)
    publishable_summary()
    print(json.dumps({"status": "completed", "outputs": outputs}, indent=2))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cpu = sub.add_parser("cpu")
    cpu.add_argument("--limit", type=int)
    llm = sub.add_parser("llm-full")
    llm.add_argument("--models", nargs="+", choices=["llama", "mistral"], default=["llama", "mistral"])
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.cmd == "cpu":
        run_cpu(args.limit)
    elif args.cmd == "llm-full":
        run_llm_full(args.models)
    elif args.cmd == "summary":
        print(json.dumps(publishable_summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
