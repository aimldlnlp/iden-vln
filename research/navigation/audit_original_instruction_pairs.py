"""Audit frozen original instructions against the existing navigation-only demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = "4b582fbef28803e68fc384318d9037eb505bd77e5664a2e69fcbace591b7adcb"
NOTES = {
    0: "Original Office label not independently established by RGB; revision adds fireplace and disambiguates bathroom with shower tub/curtain.",
    1: "Original room labels retained, but folding-chair landmark and cloth location are added; original room wording alone does not establish a unique instance.",
    2: "Original says pick up fireplace; demonstration only approaches fireplace. Original Kitchen label not independently supported by RGB; revision omits it.",
    3: "Original says pick up desk; demonstration only approaches desk. Bedroom label for screw box not independently supported by RGB; revision omits it.",
    4: "Office+box is ambiguous because another room has tie: bedroom & office. Revision uses global box+desk+shelf conjunction.",
}

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)

def main(bank, out, verify_sources=False, freeze_projection=False):
    bank, out = Path(bank), Path(out)
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == EXPECTED
    rows = [json.loads(line) for line in bank.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and {r["index"] for r in rows} == set(NOTES)
    audit = []
    for row in rows:
        text = row["instruction_original"]
        assert "ambil" in text.lower() and "letakkan" in text.lower()
        actions = sorted({a["action"] for a in row["teacher_actions"]})
        assert set(actions) <= {"move_forward", "turn_left", "turn_right", "stop"}
        assert sum(a["action"] == "stop" for a in row["teacher_actions"]) == 2
        verified = False
        if verify_sources:
            source = Path(row["source_task_json"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
            task = json.loads(source.read_text(encoding="utf-8"))
            assert any(text == value for value in strings(task)), "original text must be present verbatim in source task"
            verified = True
        audit.append({"index": row["index"], "split": row["split"], "scene": row["scene"],
            "instruction_original": text, "instruction_revised": row["instruction_revised"],
            "source_sha256": row["source_sha256"], "source_verified": verified,
            "demonstration_action_set": actions, "original_explicit_stop_count": text.count("STOP"),
            "demonstration_stop_count": 2, "literal_instruction_pair_ready": False,
            "navigation_projection_only": True,
            "hold_reason": "pick_and_place_requested_but_demonstration_has_navigation_only",
            "room_landmark_review": NOTES[row["index"]]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "original_pair_audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in audit), encoding="utf-8")
    summary = {"protocol": "original_instruction_literal_pair_audit_v1", "manifest_sha256": EXPECTED,
        "audited_train": 4, "audited_dev": 1, "literal_ready_train": 0, "literal_ready_dev": 0,
        "sources_verified": verify_sources, "training_launched": False,
        "scope": "five reviewed pilot episodes only; not an audit of all 2500 source tasks",
        "comparison_status": "No valid literal original-vs-revised training comparison. Navigation projection would require an explicit separate protocol.",
        "technical_note": "Existing context derives level from STOP count in base instruction. Originals have zero explicit STOP; naive substitution would alter a non-text feature as well."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    assert len(audit) == 5 and not any(r["literal_instruction_pair_ready"] for r in audit)
    if freeze_projection:
        assert verify_sources, "verify original source files before freezing projection"
        for row in rows:
            row["policy_context"]["instruction"] = row["instruction_original"]
            row["instruction_version"] = "original_verbatim_navigation_projection_v1"
            row["navigation_projection_stages"] = 2
            row["navigation_projection_ready"] = True
            row["pair_ready"] = row["overfit_pilot_ready"] = False
            row["literal_instruction_pair_ready"] = False
            row["scope"] = "Navigation-only projection with original verbatim text; pick/place not executed or scored; original ambiguities retained."
        projection = out / "manifest_original_projection.jsonl"
        projection.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        assert all(r["policy_context"]["instruction"] == r["instruction_original"] for r in rows)
        assert all(r["navigation_projection_stages"] == 2 for r in rows)
        print("projection_sha256", hashlib.sha256(projection.read_bytes()).hexdigest())
    print(json.dumps(summary))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--freeze-projection", action="store_true")
    args = parser.parse_args()
    main(args.bank, args.out, args.verify_sources, args.freeze_projection)
