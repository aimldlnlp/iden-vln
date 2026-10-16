"""Export instance masks and room metadata for visual review; never demonstration labels."""
import contextlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis

import diagnose_held_train_targets as audit


def main():
    out = audit.ROOT / 'approach_expansion_audit_20261011_v1'
    evidence = out / 'review_evidence'
    if evidence.exists():
        assert not any(evidence.iterdir()), 'refuse to overwrite evidence'
        evidence.rename(out / 'review_evidence_failed_startup')
    evidence.mkdir()
    manifest = out / 'frozen_diagnostic_goal_manifest.json'
    original_hash = audit.sha(manifest)
    args = audit.driver.probe.load_legacy_module(audit.driver.probe.ROOT).make_args()
    args.success_dis, args.max_step = 1., 400
    audit.resolver.install(audit.evaluator, 'obb_v1')
    records = []
    for row in json.loads((out / 'construction_ledger.json').read_text()):
        source = Path(row['source_task_json'])
        assert audit.sha(source) == row['source_sha256']
        task = json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            sim = audit.driver.probe.SceneSimulator(args=args, config=task)
            sim.save_path = None
        regions = []
        try:
            for stage in row['stages']:
                region = next(r for r in sim.sim.semantic_scene.regions if str(r.id)[1:] == stage['region'])
                regions.append({'stage': stage['stage'], 'id': str(region.id),
                    'category': region.category.name() if region.category else None,
                    'objects': dict(Counter(o.category.name() for o in region.objects if o.category))})
                tiles = []
                for goal in stage['goals']:
                    sim.set_state(goal['position'], quat_from_angle_axis(math.radians(goal['yaw_degrees']), np.array([0., 1., 0.])))
                    obs = sim.sim.get_sensor_observations()
                    pixels = np.asarray(obs['color_sensor_f'])[..., :3].copy()
                    mask = obs['semantic_sensor'] == goal['instance_id']
                    assert int(mask.sum()) == goal['pixels']
                    pixels[mask] = (.65*pixels[mask] + .35*np.array([255, 0, 255])).astype(np.uint8)
                    annotated = Image.fromarray(pixels)
                    annotated.save(evidence / f"{row['source_index']:03d}_{stage['stage']}_{goal['instance_id']}_mask.png")
                    tiles.append((f"{stage['target']} instance {goal['instance_id']}", annotated))
                for code in ('l', 'f', 'r'):
                    stop = out / f"{row['source_index']:03d}_{stage['stage']}_STOP_{code}.png"
                    if stop.exists():
                        with Image.open(stop) as image:
                            tiles.append((f'STOP {code}', image.copy()))
                if tiles:
                    sheet = Image.new('RGB', (768, math.ceil(len(tiles)/3)*280), 'white')
                    draw = ImageDraw.Draw(sheet)
                    for i, (label, image) in enumerate(tiles):
                        x, y = i % 3 * 256, i // 3 * 280
                        sheet.paste(image.resize((256, 256)), (x, y+24))
                        draw.text((x+4, y+4), label, fill='black')
                    sheet.save(evidence / f"{row['source_index']:03d}_{stage['stage']}_sheet.png")
            records.append({'source_index': row['source_index'], 'split': row['split'], 'scene': row['scene'],
                            'instruction_original': row['instruction_original'],
                            'source_room_names': task['Region Name'], 'regions': regions})
            print(json.dumps({'source_index': row['source_index'], 'evidence_complete': True}), flush=True)
        finally:
            sim.close()
    assert audit.sha(manifest) == original_hash
    (evidence / 'metadata.json').write_text(json.dumps({'records': records, 'goal_manifest_sha256': original_hash,
        'counterfactual_pose_resets': 'diagnostic images only; not demonstration labels',
        'training_performed': False, 'script_sha256': audit.sha(Path(__file__))}, indent=2)+'\n')


if __name__ == '__main__':
    main()
