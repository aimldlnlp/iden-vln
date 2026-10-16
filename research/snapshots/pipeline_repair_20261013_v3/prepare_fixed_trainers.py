"""Create corrected copies; original scripts and checkpoints remain immutable."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old), count)
    return source.replace(old, new)


def main():
    manifest = {}
    for family in ('frame_trainer', 'frame_collector', 'target_trainer', 'sequence_trainer'):
        original = HERE/(family+'_original.py')
        target = HERE/(family+'_fixed.py')
        assert not target.exists(), target
        source = original.read_text(encoding='utf-8')
        source = source.replace('from __future__ import annotations', 'from __future__ import annotations\nfrom repair_contract import before_action, best_snapshot, signed_yaw')
        marker = '\nif __name__ == "__main__":'
        assert marker in source
        source = source.split(marker)[0] + marker + '\n    raise SystemExit("Use run_repair_pilot.py; historical generation entry point is disabled")\n'
        if family == 'frame_collector':
            source = replace(source, 'RUN_DIR / "run_tiered_balanced_visual_bc.py"', 'RUN_DIR / "frame_trainer_fixed.py"')
            source = replace(source, '            obs = open_triplet(frame_dir)', '            frame_dir, position, rotation = before_action(trial, step_idx, pos_list, yaw_list)\n            obs = open_triplet(frame_dir)')
            source = replace(source, '            position = pos_list[step_idx] if step_idx < len(pos_list) else [0.0, 0.0, 0.0]\n            rotation = yaw_list[step_idx] if step_idx < len(yaw_list) else 0.0\n', '')
        else:
            # Imported trainers may write only below this new version directory.
            source = replace(source, 'RESULTS = RUN_ROOT / "results"', 'RESULTS = Path(__file__).resolve().parent / "'+family+'_results"')
            source = replace(source, '{k: v.detach().cpu() for k, v in model.state_dict().items()}', 'best_snapshot(model)', count=1)
            if family == 'frame_trainer':
                source = replace(source, '            with torch.no_grad():\n                pred = model(x_t[val_idx])', '            model.eval()\n            with torch.no_grad():\n                pred = model(x_t[val_idx])')
                source = replace(source, '            if m["balanced_accuracy"] >', '            model.train()\n            if m["balanced_accuracy"] >')
                source = replace(source, '    with torch.no_grad():\n        train_pred', '    model.eval()\n    with torch.no_grad():\n        train_pred')
                source += '\n# The resume collector is the effective historical training caller.\ndef collect_samples(root):\n    from frame_collector_fixed import collect_samples_direct\n    return collect_samples_direct(root)\n'
            elif family == 'target_trainer':
                source = replace(source, '            pos = context["pos"][idx] if idx < len(context["pos"]) else [0.0, 0.0, 0.0]\n            yaw = context["yaw"][idx] if idx < len(context["yaw"]) else 0.0', '            frame_dir, pos, yaw = before_action(trial, idx, context["pos"], context["yaw"])')
                source = replace(source, '        yaw = float(rot.angle())', '        yaw = signed_yaw(rot)')
                source = replace(source, '            with torch.no_grad():\n                pred = model(x_t[val_idx])', '            model.eval()\n            with torch.no_grad():\n                pred = model(x_t[val_idx])')
                source = replace(source, '            per = metrics["per_class"]', '            model.train()\n            per = metrics["per_class"]')
                source = replace(source, '    with torch.no_grad():\n        train_pred', '    model.eval()\n    with torch.no_grad():\n        train_pred')
            else:
                source = replace(source, '            pos = context["pos"][idx] if idx < len(context["pos"]) else [0.0, 0.0, 0.0]\n            yaw = context["yaw"][idx] if idx < len(context["yaw"]) else 0.0', '            child, pos, yaw = before_action(trial, idx, context["pos"], context["yaw"])')
                source = replace(source, '        yaw = float(st.rotation.angle())', '        yaw = signed_yaw(st.rotation)')
        ast.parse(source)
        target.write_text(source, encoding='utf-8', newline='\n')
        manifest[family] = {'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'fixed_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (HERE/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Prepared four corrected source copies')


if __name__ == '__main__':
    main()
