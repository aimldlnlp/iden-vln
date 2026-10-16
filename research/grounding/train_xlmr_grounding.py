"""Train and aggregate a reproducible XLM-R textual-grounding baseline.

The runner deliberately consumes only the portable task JSONL and official split
files. Habitat is not imported because this experiment evaluates text grounding,
not online simulator rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
OBJECT_OFFSET = 3
METRIC_CONTRACT = {
    "label_normalization": "casefold, whitespace-normalize, and stable-unique by first occurrence",
    "grounding": "canonical set-F1; duplicate raw label mentions are not counted twice",
    "ordered_sequence": "ordered canonical object-label sequence after stable-unique normalization",
    "prefix": "Prefix@k is evaluated only on examples with at least k canonical gold objects",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def canonical_label(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def unique_labels(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = canonical_label(value)
        if label and label not in seen:
            output.append(label)
            seen.add(label)
    return output


def task_objects(task: dict) -> list[str]:
    """Return canonical set-like object labels for grounding heads and sequence targets."""
    names = task.get("object_names")
    if names is None:
        names = [item.get("name", "") for item in task.get("objects", [])]
    return unique_labels(names or [])


def task_rooms(task: dict) -> list[str]:
    names = task.get("region_names")
    return unique_labels(names or [])


def load_split_ids(path: Path, split_name: str) -> list[str]:
    split = load_json(path)
    return list(split[split_name]["task_ids"])


class GroundingDataset(Dataset):
    def __init__(
        self,
        tasks: list[dict],
        tokenizer,
        object_to_id: dict[str, int],
        room_to_id: dict[str, int],
        max_length: int,
        sequence_max_length: int,
    ) -> None:
        self.tasks = tasks
        self.object_to_id = object_to_id
        self.room_to_id = room_to_id
        self.sequence_max_length = sequence_max_length
        encoded = tokenizer(
            [task.get("task_instruction", "") for task in tasks],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict:
        task = self.tasks[index]
        objects = task_objects(task)
        rooms = task_rooms(task)
        object_target = torch.zeros(len(self.object_to_id), dtype=torch.float32)
        room_target = torch.zeros(len(self.room_to_id), dtype=torch.float32)
        for label in objects:
            if label in self.object_to_id:
                object_target[self.object_to_id[label]] = 1.0
        for label in rooms:
            if label in self.room_to_id:
                room_target[self.room_to_id[label]] = 1.0

        object_ids = [self.object_to_id[label] for label in objects if label in self.object_to_id]
        sequence_ids = [OBJECT_OFFSET + item for item in object_ids[: self.sequence_max_length]]
        sequence_input = [BOS_ID] + sequence_ids
        sequence_target = sequence_ids + [EOS_ID]
        sequence_input = (sequence_input + [PAD_ID] * (self.sequence_max_length + 1))[: self.sequence_max_length + 1]
        sequence_target = (sequence_target + [PAD_ID] * (self.sequence_max_length + 1))[: self.sequence_max_length + 1]

        first_object = self.object_to_id.get(objects[0], 0) if objects else 0
        first_room = self.room_to_id.get(rooms[0], 0) if rooms else 0
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "object_target": object_target,
            "room_target": room_target,
            "first_object": torch.tensor(first_object, dtype=torch.long),
            "first_room": torch.tensor(first_room, dtype=torch.long),
            "sequence_input": torch.tensor(sequence_input, dtype=torch.long),
            "sequence_target": torch.tensor(sequence_target, dtype=torch.long),
            "task_id": task.get("task_id", str(index)),
            "gold_objects": objects,
            "gold_rooms": rooms,
        }


def collate_batch(batch: list[dict]) -> dict:
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "object_target",
        "room_target",
        "first_object",
        "first_room",
        "sequence_input",
        "sequence_target",
    ]
    output = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    output["task_id"] = [item["task_id"] for item in batch]
    output["gold_objects"] = [item["gold_objects"] for item in batch]
    output["gold_rooms"] = [item["gold_rooms"] for item in batch]
    return output


class XLMRGroundingModel(nn.Module):
    def __init__(self, model_name: str, object_count: int, room_count: int, sequence_vocab_size: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.object_head = nn.Linear(hidden, object_count)
        self.room_head = nn.Linear(hidden, room_count)
        self.first_object_head = nn.Linear(hidden, object_count)
        self.first_room_head = nn.Linear(hidden, room_count)
        self.sequence_embedding = nn.Embedding(sequence_vocab_size, hidden, padding_idx=PAD_ID)
        self.sequence_decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.sequence_head = nn.Linear(hidden, sequence_vocab_size)

    def forward(self, input_ids, attention_mask, sequence_input):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(encoded.last_hidden_state[:, 0])
        sequence_hidden = pooled.contiguous().unsqueeze(0)
        sequence_embedded = self.sequence_embedding(sequence_input)
        sequence_output, _ = self.sequence_decoder(sequence_embedded, sequence_hidden)
        return {
            "object_logits": self.object_head(pooled),
            "room_logits": self.room_head(pooled),
            "first_object_logits": self.first_object_head(pooled),
            "first_room_logits": self.first_room_head(pooled),
            "sequence_logits": self.sequence_head(sequence_output),
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    for key in [
        "input_ids",
        "attention_mask",
        "object_target",
        "room_target",
        "first_object",
        "first_room",
        "sequence_input",
        "sequence_target",
    ]:
        batch[key] = batch[key].to(device)
    return batch


def loss_for_batch(outputs: dict, batch: dict) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    sequence_loss = ce(
        outputs["sequence_logits"].reshape(-1, outputs["sequence_logits"].shape[-1]),
        batch["sequence_target"].reshape(-1),
    )
    return (
        bce(outputs["object_logits"], batch["object_target"])
        + bce(outputs["room_logits"], batch["room_target"])
        + ce(outputs["first_object_logits"], batch["first_object"])
        + ce(outputs["first_room_logits"], batch["first_room"])
        + sequence_loss
    )


def decode_sequence(model: XLMRGroundingModel, pooled: torch.Tensor, max_length: int) -> list[list[int]]:
    hidden = pooled.contiguous().unsqueeze(0)
    current = torch.full((pooled.shape[0], 1), BOS_ID, dtype=torch.long, device=pooled.device)
    output: list[list[int]] = [[] for _ in range(pooled.shape[0])]
    finished = torch.zeros(pooled.shape[0], dtype=torch.bool, device=pooled.device)
    for _ in range(max_length + 1):
        embedded = model.sequence_embedding(current[:, -1:])
        decoded, hidden = model.sequence_decoder(embedded, hidden)
        next_token = model.sequence_head(decoded[:, -1]).argmax(dim=-1)
        current = torch.cat([current, next_token[:, None]], dim=1)
        for row, token in enumerate(next_token.tolist()):
            if not finished[row]:
                if token == EOS_ID:
                    finished[row] = True
                elif token >= OBJECT_OFFSET:
                    output[row].append(token - OBJECT_OFFSET)
        if bool(finished.all()):
            break
    return output


def sample_f1(predicted: list[str], gold: list[str]) -> float:
    """Task-averaged F1 under the shared canonical set-label contract."""
    pred = set(predicted)
    truth = set(gold)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    overlap = len(pred & truth)
    return 2.0 * overlap / (len(pred) + len(truth))


def multilabel_scores(predicted: list[list[str]], gold: list[list[str]], vocabulary: list[str]) -> tuple[float, float, list[float]]:
    """Compute micro/macro label scores after canonical set normalization."""
    counts = {label: [0, 0, 0] for label in vocabulary}
    sample_scores = []
    for pred, truth in zip(predicted, gold):
        pred_set = set(pred)
        truth_set = set(truth)
        sample_scores.append(sample_f1(pred, truth))
        for label in vocabulary:
            counts[label][0] += int(label in pred_set and label in truth_set)
            counts[label][1] += int(label in pred_set and label not in truth_set)
            counts[label][2] += int(label not in pred_set and label in truth_set)
    tp = sum(item[0] for item in counts.values())
    fp = sum(item[1] for item in counts.values())
    fn = sum(item[2] for item in counts.values())
    micro = 2 * tp / max(2 * tp + fp + fn, 1)
    per_label = []
    for tp_label, fp_label, fn_label in counts.values():
        per_label.append(2 * tp_label / max(2 * tp_label + fp_label + fn_label, 1))
    return micro, float(np.mean(per_label)), sample_scores


def levenshtein_similarity(predicted: list[str], gold: list[str]) -> float:
    rows = len(predicted) + 1
    cols = len(gold) + 1
    distance = list(range(cols))
    for row in range(1, rows):
        previous = distance[0]
        distance[0] = row
        for col in range(1, cols):
            current = distance[col]
            distance[col] = min(
                distance[col] + 1,
                distance[col - 1] + 1,
                previous + (predicted[row - 1] != gold[col - 1]),
            )
            previous = current
    return 1.0 - distance[-1] / max(len(predicted), len(gold), 1)


def bootstrap(values: list[float], seed: int, samples: int = 1000) -> dict:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def evaluate(model, loader, device, object_labels, room_labels, id_to_object, sequence_max_length, amp_enabled, max_batches=None):
    model.eval()
    object_predicted: list[list[str]] = []
    room_predicted: list[list[str]] = []
    object_gold: list[list[str]] = []
    room_gold: list[list[str]] = []
    sequence_predicted: list[list[str]] = []
    sequence_gold: list[list[str]] = []
    records = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            cpu_gold_objects = batch["gold_objects"]
            cpu_gold_rooms = batch["gold_rooms"]
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["sequence_input"])
                encoded = model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                pooled = model.dropout(encoded.last_hidden_state[:, 0])
            object_probs = torch.sigmoid(outputs["object_logits"]).cpu()
            room_probs = torch.sigmoid(outputs["room_logits"]).cpu()
            first_objects = outputs["first_object_logits"].argmax(dim=-1).cpu().tolist()
            first_rooms = outputs["first_room_logits"].argmax(dim=-1).cpu().tolist()
            sequence_indices = decode_sequence(model, pooled, sequence_max_length)
            for row in range(len(cpu_gold_objects)):
                predicted_objects = [object_labels[index] for index, value in enumerate(object_probs[row].tolist()) if value >= 0.5]
                predicted_rooms = [room_labels[index] for index, value in enumerate(room_probs[row].tolist()) if value >= 0.5]
                predicted_first_object = object_labels[first_objects[row]]
                predicted_first_room = room_labels[first_rooms[row]]
                predicted_sequence = [id_to_object[index] for index in sequence_indices[row] if index < len(id_to_object)]
                gold_objects = cpu_gold_objects[row]
                gold_rooms = cpu_gold_rooms[row]
                gold_sequence = gold_objects[:sequence_max_length]
                object_predicted.append(predicted_objects)
                room_predicted.append(predicted_rooms)
                object_gold.append(gold_objects)
                room_gold.append(gold_rooms)
                sequence_predicted.append(predicted_sequence)
                sequence_gold.append(gold_sequence)
                records.append(
                    {
                        "task_id": batch["task_id"][row],
                        "gold_objects": gold_objects,
                        "pred_objects": predicted_objects,
                        "gold_rooms": gold_rooms,
                        "pred_rooms": predicted_rooms,
                        "gold_sequence": gold_sequence,
                        "pred_sequence": predicted_sequence,
                        "pred_first_object": predicted_first_object,
                        "pred_first_room": predicted_first_room,
                        "first_object_correct": bool(gold_objects and predicted_first_object == gold_objects[0]),
                        "first_room_correct": bool(gold_rooms and predicted_first_room == gold_rooms[0]),
                    }
                )
    object_micro, object_macro, object_sample_f1 = multilabel_scores(object_predicted, object_gold, object_labels)
    room_micro, room_macro, room_sample_f1 = multilabel_scores(room_predicted, room_gold, room_labels)
    sequence_exact_values = [float(pred == truth) for pred, truth in zip(sequence_predicted, sequence_gold)]
    edit_values = [levenshtein_similarity(pred, truth) for pred, truth in zip(sequence_predicted, sequence_gold)]
    prefix_values = {}
    for k in [1, 2, 3]:
        values = [float(len(truth) >= k and pred[:k] == truth[:k]) for pred, truth in zip(sequence_predicted, sequence_gold) if len(truth) >= k]
        prefix_values[k] = values
    metrics = {
        "metric_contract": METRIC_CONTRACT,
        "object_f1": object_micro,
        "object_macro_f1": object_macro,
        "room_f1": room_micro,
        "room_macro_f1": room_macro,
        "first_object_accuracy": float(np.mean([item["first_object_correct"] for item in records])) if records else 0.0,
        "first_room_accuracy": float(np.mean([item["first_room_correct"] for item in records])) if records else 0.0,
        "sequence_exact": float(np.mean(sequence_exact_values)) if sequence_exact_values else 0.0,
        "edit_similarity": float(np.mean(edit_values)) if edit_values else 0.0,
        "prefix1": float(np.mean(prefix_values[1])) if prefix_values[1] else 0.0,
        "prefix2": float(np.mean(prefix_values[2])) if prefix_values[2] else 0.0,
        "prefix3": float(np.mean(prefix_values[3])) if prefix_values[3] else 0.0,
    }
    sample_metrics = {
        "object_f1": object_sample_f1,
        "room_f1": room_sample_f1,
        "sequence_exact": sequence_exact_values,
        "edit_similarity": edit_values,
        "prefix1": prefix_values[1],
        "prefix2": prefix_values[2],
        "prefix3": prefix_values[3],
    }
    return metrics, sample_metrics, records


def build_tasks(data_root: Path, split_name: str):
    tasks = {task["task_id"]: task for task in load_jsonl(data_root / "iden_vln_tasks.jsonl")}
    split_file = data_root / f"splits_{'scene_heldout' if split_name == 'scene_heldout' else 'random'}.json"
    split = load_json(split_file)
    split_tasks = {part: [tasks[task_id] for task_id in split[part]["task_ids"] if task_id in tasks] for part in ["train", "val", "test"]}
    if any(len(split_tasks[part]) != split[part]["count"] for part in split_tasks):
        raise RuntimeError(f"Split count mismatch for {split_name}: {[(part, len(split_tasks[part]), split[part]['count']) for part in split_tasks]}")
    return split_tasks


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = args.run_root / args.split / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    done_marker = output_dir / "done.marker"
    if done_marker.exists() and not args.force:
        print(f"already complete: {output_dir}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    split_tasks = build_tasks(args.data_root, args.split)
    all_tasks = [task for part in split_tasks.values() for task in part]
    object_labels = sorted(unique_labels([label for task in all_tasks for label in task_objects(task)]))
    room_labels = sorted(unique_labels([label for task in all_tasks for label in task_rooms(task)]))
    object_to_id = {label: index for index, label in enumerate(object_labels)}
    room_to_id = {label: index for index, label in enumerate(room_labels)}
    datasets = {
        part: GroundingDataset(split_tasks[part], tokenizer, object_to_id, room_to_id, args.max_length, args.sequence_max_length)
        for part in ["train", "val", "test"]
    }
    loaders = {
        part: DataLoader(datasets[part], batch_size=args.batch_size, shuffle=part == "train", num_workers=0, collate_fn=collate_batch)
        for part in datasets
    }
    model = XLMRGroundingModel(args.model_name, len(object_labels), len(room_labels), len(object_labels) + OBJECT_OFFSET)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    epochs = 1 if args.smoke else args.epochs
    max_train_batches = args.smoke_batches if args.smoke else None
    max_eval_batches = 1 if args.smoke else None
    best_score = -math.inf
    best_epoch = -1
    patience = 0
    history = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(loaders["train"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and args.amp):
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["sequence_input"])
                loss = loss_for_batch(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        val_metrics, _, _ = evaluate(
            model,
            loaders["val"],
            device,
            object_labels,
            room_labels,
            object_labels,
            args.sequence_max_length,
            device.type == "cuda" and args.amp,
            max_eval_batches,
        )
        score = val_metrics["sequence_exact"] + val_metrics["edit_similarity"] + val_metrics["object_f1"] + val_metrics["room_f1"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)) if train_losses else None, "val": val_metrics})
        print(json.dumps({"epoch": epoch + 1, "train_loss": history[-1]["train_loss"], "val": val_metrics}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            patience = 0
            torch.save({"model": model.state_dict(), "object_labels": object_labels, "room_labels": room_labels, "config": vars(args)}, output_dir / "checkpoint_best.pt")
        else:
            patience += 1
            if not args.smoke and patience >= args.patience:
                break

    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, sample_metrics, records = evaluate(
        model,
        loaders["test"],
        device,
        object_labels,
        room_labels,
        object_labels,
        args.sequence_max_length,
        device.type == "cuda" and args.amp,
        max_eval_batches,
    )
    ci = {key: bootstrap(values, args.seed + index) for index, (key, values) in enumerate(sample_metrics.items())}
    metrics = {
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "count": len(records),
        "metrics": test_metrics,
        "bootstrap_ci95": ci,
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
        "config": {
            "max_length": args.max_length,
            "sequence_max_length": args.sequence_max_length,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": epochs,
            "object_vocab_size": len(object_labels),
            "room_vocab_size": len(room_labels),
            "device": str(device),
        },
        "test_samples": sample_metrics,
    }
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    done_marker.write_text("completed\n", encoding="utf-8")
    print(f"completed {args.split} seed={args.seed} output={output_dir}", flush=True)


def aggregate_runs(args: argparse.Namespace) -> None:
    aggregate = {
        "model": args.model_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_contract": METRIC_CONTRACT,
        "aggregate": {},
    }
    for split in ["random", "scene_heldout"]:
        run_metrics = []
        for path in sorted((args.run_root / split).glob("seed_*/metrics.json")):
            run_metrics.append(load_json(path))
        if len(run_metrics) < 3:
            raise RuntimeError(f"Expected three completed seeds for {split}, found {len(run_metrics)}")
        metric_keys = ["object_f1", "object_macro_f1", "room_f1", "room_macro_f1", "first_object_accuracy", "first_room_accuracy", "sequence_exact", "edit_similarity", "prefix1", "prefix2", "prefix3"]
        counts = [int(item["count"]) for item in run_metrics]
        if len(set(counts)) != 1:
            raise RuntimeError(f"Seed test counts differ for {split}: {counts}")
        split_entry = {
            "count": counts[0],
            "seed_count": len(run_metrics),
            "n_per_seed": counts[0],
            "seed_metrics": [item["metrics"] for item in run_metrics],
            "per_seed_success_counts": {},
            "aggregation": {
                "point_estimate": "mean_of_seed_metrics",
                "ci_estimate": "pooled_seed_task_bootstrap_when_test_samples_available",
                "effective_denominators": {},
            },
            "metrics": {},
        }
        for key in metric_keys:
            values = [item["metrics"][key] for item in run_metrics]
            per_seed_samples = [item.get("test_samples", {}).get(key) for item in run_metrics]
            sample_values = [value for samples in per_seed_samples if isinstance(samples, list) for value in samples]
            point_estimate = float(np.mean(values))
            interval = bootstrap(sample_values or values, args.seed)
            interval["mean"] = point_estimate
            split_entry["metrics"][key] = interval
            if sample_values:
                split_entry["aggregation"]["effective_denominators"][key] = len(sample_values)
                if all(isinstance(samples, list) for samples in per_seed_samples):
                    split_entry["per_seed_success_counts"][key] = {
                        "successes": [int(round(sum(float(value) for value in samples))) for samples in per_seed_samples],
                        "denominators": [len(samples) for samples in per_seed_samples],
                    }
            else:
                split_entry["aggregation"]["effective_denominators"][key] = len(values)
        aggregate["aggregate"][split] = split_entry
    save_json(args.run_root / "aggregate_metrics.json", aggregate)
    print(f"wrote {args.run_root / 'aggregate_metrics.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=["random", "scene_heldout"])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sequence-max-length", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not args.aggregate and not args.split:
        parser.error("--split is required unless --aggregate is used")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.aggregate:
        aggregate_runs(arguments)
    else:
        run_training(arguments)
