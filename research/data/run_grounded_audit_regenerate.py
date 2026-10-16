#!/usr/bin/env python3
"""Run grounded audit-regenerate generation for LH-VLN contexts.

This script is intentionally self-contained so it can run on a fresh RunPod
instance with only contexts.jsonl and a Transformers model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ID_LEXICON = {
    "agen", "ambil", "atas", "bawa", "berangkat", "berhenti", "berikutnya",
    "dalam", "dan", "dari", "dekat", "dengan", "di", "dorong", "gerak",
    "hati", "instruksi", "jangan", "ke", "kembali", "keluar", "kemudian",
    "kiri", "kanan", "lalu", "lanjut", "letak", "letakkan", "masuk",
    "menuju", "pada", "pindah", "pergi", "ruang", "sana", "satu", "sini",
    "setelah", "simpan", "taruh", "tempat", "terdekat", "tujuan", "untuk",
}

COMMON_EN_ALLOWED = {
    "near", "next", "to", "straight", "left", "right", "front", "behind",
    "beside", "room", "office", "kitchen", "bedroom", "bathroom", "living",
    "garage", "laundry", "rec", "dining", "hall",
}

KNOWN_INDOOR_TERMS = {
    "alarm clock", "armchair", "bag", "basket", "bathroom counter",
    "bathroom towel", "bathtub", "bed", "bench", "blanket", "book",
    "book rack", "bottle", "box", "cabinet", "candle", "chair", "clock",
    "clothes", "coffee machine", "coffee table", "counter", "countertop",
    "couch", "cup", "desk", "device", "doormat", "door", "dresser",
    "dumbbell", "exercise ball", "fireplace", "floor", "flower",
    "flower vase", "glass", "kettle", "kitchen counter", "kitchen island",
    "knife holder", "ladder", "lamp", "laptop", "laundry basket",
    "laundry machine", "ledge", "magazine", "mirror", "monitor", "mouse",
    "pad", "painting", "papers", "picture", "plant", "plush toy", "pot",
    "printer", "record player", "refrigerator", "shelf", "sink",
    "soap dispenser", "soapbox", "stack of papers", "stuffed animal",
    "table", "tissue box", "toilet", "towel", "toy", "trashcan",
    "typewriter", "vacuum cleaner", "vase", "wall", "window",
    "buku", "dinding", "handuk", "jendela", "kabinet", "karpet", "kasur",
    "keranjang", "kursi", "lantai", "lemari", "meja", "nampan", "pintu",
    "rak", "sofa", "vas", "wastafel",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.strip('"').strip("'").strip()
    text = re.sub(r"^(Instruksi|Instruction)\s*:\s*", "", text, flags=re.I)
    text = " ".join(text.splitlines()).strip()
    return text


def phrase_present(phrase: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(norm(phrase)) + r"\b", norm(text)) is not None


def tokenize_with_phrases(text: str, phrases: list[str]) -> tuple[list[str], list[str]]:
    lowered = norm(text)
    phrase_tokens: list[str] = []
    for phrase in sorted({norm(p) for p in phrases if p}, key=len, reverse=True):
        if phrase_present(phrase, lowered):
            placeholder = "__PHRASE_%d__" % len(phrase_tokens)
            lowered = re.sub(r"\b" + re.escape(phrase) + r"\b", placeholder, lowered)
            phrase_tokens.append(phrase)
    raw = re.findall(r"__PHRASE_\d+__|[a-z]+(?:[-'][a-z]+)?", lowered)
    tokens = []
    for tok in raw:
        if tok.startswith("__PHRASE_"):
            idx = int(tok.replace("__PHRASE_", "").replace("__", ""))
            tokens.append(phrase_tokens[idx])
        else:
            tokens.append(tok)
    return tokens, phrase_tokens


def audit_instruction(text: str, context: dict[str, Any]) -> dict[str, Any]:
    objects = [norm(x) for x in context.get("objects", [])]
    rooms = [norm(x) for x in context.get("rooms", [])]
    phrases = objects + rooms + sorted(COMMON_EN_ALLOWED, key=len, reverse=True)
    tokens, _ = tokenize_with_phrases(text, phrases)

    id_count = 0
    en_count = 0
    unk_count = 0
    allowed_en = set(objects + rooms) | COMMON_EN_ALLOWED
    for tok in tokens:
        simple = tok.replace("-", " ")
        if simple in allowed_en:
            en_count += 1
        elif simple in ID_LEXICON:
            id_count += 1
        else:
            unk_count += 1

    labelled = id_count + en_count
    id_ratio = id_count / labelled if labelled else 0.0
    en_ratio = en_count / labelled if labelled else 0.0
    wc = len(words(text))
    rooms_found = sorted({r for r in rooms if phrase_present(r, text)})

    failures: list[str] = []
    r1 = True
    if "{" in text or "}" in text or "[" in text or "]" in text:
        r1 = False
    if "'" in text or '"' in text or "_" in text:
        r1 = False
    if "```" in text or "\n" in text.strip():
        r1 = False
    if re.search(r"^\s*[-*]|\b\d+\s*[\.)]", text):
        r1 = False
    if r1 is False:
        failures.append("R1 failed: output must be one plain sentence.")

    r2 = not re.search(r"\b(grab|release)\b", text, flags=re.I)
    if not r2:
        failures.append("R2 failed: literal grab/release appears.")

    r3 = 18 <= wc <= 35
    if not r3:
        failures.append(f"R3 failed: word count {wc}, target 18-35.")

    r4 = 0.60 <= id_ratio <= 0.80 and 0.20 <= en_ratio <= 0.40
    if not r4:
        failures.append(
            f"R4 failed: ID ratio {id_ratio:.2f}, EN ratio {en_ratio:.2f}; "
            "target ID 0.60-0.80 and EN 0.20-0.40."
        )

    r5 = 1 <= len(rooms_found) <= 2
    if not r5:
        failures.append(f"R5 failed: {len(rooms_found)} rooms found, target 1-2.")

    object_mentions = sorted({o for o in objects if phrase_present(o, text)})
    room_mentions = sorted({r for r in rooms if phrase_present(r, text)})
    allowed_terms = set(objects + rooms)
    allowed_object_terms = set(objects)
    hallucinated = []
    for term in sorted(KNOWN_INDOOR_TERMS, key=len, reverse=True):
        term_n = norm(term)
        if term_n in allowed_terms:
            continue
        if any(term_n in allowed.split() for allowed in allowed_object_terms):
            continue
        if phrase_present(term_n, text):
            hallucinated.append(term)

    entity_consistency = bool(object_mentions) and not hallucinated
    room_consistency = bool(room_mentions) and all(r in rooms for r in room_mentions)
    hallucination_proxy = bool(hallucinated)

    if not entity_consistency:
        failures.append("Grounding failed: object mentions are missing or hallucinated terms appear.")
    if not room_consistency:
        failures.append("Grounding failed: room mentions are missing or outside context.")
    if hallucination_proxy:
        failures.append("Grounding failed: hallucinated terms = " + ", ".join(hallucinated))

    return {
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "R5": r5,
        "word_count": wc,
        "id_count": id_count,
        "en_count": en_count,
        "unk_count": unk_count,
        "id_ratio": id_ratio,
        "en_ratio": en_ratio,
        "room_count": len(rooms_found),
        "rooms_found": rooms_found,
        "strict_pass": all([r1, r2, r3, r4, r5]),
        "object_mentions": object_mentions,
        "room_mentions": room_mentions,
        "hallucinated_terms": hallucinated,
        "entity_consistency": entity_consistency,
        "room_consistency": room_consistency,
        "hallucination_proxy": hallucination_proxy,
        "grounded_strict_pass": all([r1, r2, r3, r4, r5, entity_consistency, room_consistency])
        and not hallucination_proxy,
        "failures": failures,
    }


def build_prompt(context: dict[str, Any], feedback: list[str] | None = None) -> list[dict[str, str]]:
    selected_rooms = list(context.get("rooms", []))[:2] or list(context.get("rooms", []))
    selected_objects = list(context.get("objects", []))[:4] or list(context.get("objects", []))
    objects = "; ".join(selected_objects)
    rooms = "; ".join(selected_rooms)
    object_a = selected_objects[0] if selected_objects else "object"
    object_b = selected_objects[1] if len(selected_objects) > 1 else object_a
    room_a = selected_rooms[0] if selected_rooms else "room"
    room_b = selected_rooms[1] if len(selected_rooms) > 1 else room_a
    scaffold = (
        f"Pergi ke {room_a} lalu ambil {object_a} dekat {object_a} "
        f"kemudian bawa ke {room_b} dan letakkan di {object_b} secara perlahan."
    )
    feedback_text = ""
    if feedback:
        feedback_text = "\nFix these failures from the previous attempt:\n- " + "\n- ".join(feedback)

    system = (
        "You generate one grounded Indonesian-English code-switching indoor navigation "
        "instruction. Follow the constraints exactly. Do not explain. Use Indonesian "
        "verbs and connectors; keep object and room names in English."
    )
    user = f"""Task context:
Objects allowed for this instruction: {objects}
Rooms allowed for this instruction: {rooms}
Selected grounded plan: move {object_a} from {room_a} to {object_b} in {room_b}.
Grounded scaffold to follow: {scaffold}

Write exactly one new instruction.
Rules:
R1: one plain sentence only; no JSON, bullets, quotes, labels, or code block.
R2: never use the literal words grab or release.
R3: 18-35 words.
R4: use 60-80 percent Indonesian tokens and 20-40 percent English tokens.
R5: mention only rooms from "Rooms allowed for this instruction".
Grounding: every English object, room, or landmark you mention must be copied exactly from the allowed lists above. Do not invent wall, window, sink, bed, desk, table, shelf, door, floor, cabinet, countertop, or any other landmark unless it appears in Objects allowed.
Scope: cover only the Selected grounded plan. Do not add other subtasks, objects, rooms, numeric suffixes, or IDs from the original task.
Language policy: use Indonesian for actions/connectors such as pergi, ambil, bawa, letakkan, lalu, kemudian, dekat. Use English only for allowed object names and allowed room names. Do not translate object or room names. Do not use Move_to, numeric suffixes, underscores, or quotation marks.
Recommended action: copy the Grounded scaffold exactly unless feedback requires a small correction.
{feedback_text}
Return only the instruction."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class Generator:
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[:, inputs.input_ids.shape[-1] :]
        return clean_generation(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0])


def load_contexts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    def rate(key: str) -> float:
        return sum(1 for r in rows if r["audit"].get(key)) / total if total else 0.0
    accepted = [r for r in rows if r["status"] == "accepted"]
    return {
        "variant": "grounded_audit_regenerate_cs",
        "total_contexts": total,
        "accepted": len(accepted),
        "failed_after_max_attempts": total - len(accepted),
        "R1_pass_rate": rate("R1"),
        "R2_pass_rate": rate("R2"),
        "R3_pass_rate": rate("R3"),
        "R4_pass_rate": rate("R4"),
        "R5_pass_rate": rate("R5"),
        "strict_pass_rate": rate("strict_pass"),
        "grounded_strict_pass_rate": rate("grounded_strict_pass"),
        "entity_consistency_rate": rate("entity_consistency"),
        "room_consistency_rate": rate("room_consistency"),
        "hallucination_proxy_rate": rate("hallucination_proxy"),
        "mean_word_count": statistics.mean(r["audit"]["word_count"] for r in rows) if rows else 0,
        "id_ratio_mean": statistics.mean(r["audit"]["id_ratio"] for r in rows) if rows else 0,
        "en_ratio_mean": statistics.mean(r["audit"]["en_ratio"] for r in rows) if rows else 0,
        "mean_attempts": statistics.mean(r["attempts_used"] for r in rows) if rows else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts(args.contexts, args.limit)
    gen = Generator(args.model, args.max_new_tokens, args.temperature, args.top_p)

    attempts_path = args.output_dir / "attempts.jsonl"
    accepted_path = args.output_dir / "accepted.jsonl"
    final_rows: list[dict[str, Any]] = []
    with attempts_path.open("w", encoding="utf-8") as attempts_f, accepted_path.open("w", encoding="utf-8") as accepted_f:
        for idx, ctx in enumerate(contexts, start=1):
            feedback = None
            final = None
            for attempt in range(1, args.max_attempts + 1):
                instruction = gen.generate(build_prompt(ctx, feedback))
                audit = audit_instruction(instruction, ctx)
                row = {
                    "variant": "grounded_audit_regenerate_cs",
                    "context_id": ctx.get("context_id", idx),
                    "source_task_path": ctx.get("source_path", ""),
                    "original_instruction": ctx.get("original_instruction", ""),
                    "objects": ctx.get("objects", []),
                    "rooms": ctx.get("rooms", []),
                    "subtask_list": ctx.get("subtask_list", []),
                    "instruction": instruction,
                    "attempt": attempt,
                    "attempts_used": attempt,
                    "audit": audit,
                    "status": "accepted" if audit["grounded_strict_pass"] else "retry",
                }
                attempts_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                attempts_f.flush()
                if audit["grounded_strict_pass"]:
                    final = row
                    final["status"] = "accepted"
                    accepted_f.write(json.dumps(final, ensure_ascii=False) + "\n")
                    accepted_f.flush()
                    break
                feedback = audit["failures"][:6]
            if final is None:
                final = row
                final["status"] = "failed_after_max_attempts"
            final_rows.append(final)
            print(
                f"[{idx}/{len(contexts)}] {final['status']} attempts={final['attempts_used']} "
                f"grounded={final['audit']['grounded_strict_pass']}",
                flush=True,
            )

    summary = summarize(final_rows)
    metadata = {
        "contexts": str(args.contexts),
        "model": args.model,
        "max_attempts": args.max_attempts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (args.output_dir / "final_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
