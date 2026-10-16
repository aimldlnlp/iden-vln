#!/usr/bin/env python3
"""Replay frozen train rollouts, then audit front-camera OWL-ViT proposals."""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/aimldl/LH-VLN')
SOURCE = ROOT / 'runs/idenvln_rgbd_stop_verifier_20260926'
RUN = Path(os.environ['IDENVLN_RUN_DIR'])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs'))
import run_rgbd_stop_verifier as verifier


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instances(sim, stage):
    target, region = sim.target[stage], str(sim.region[stage])
    target_ids, distractor_ids = [], []
    for semantic_region in sim.sim.semantic_scene.regions:
        for obj in semantic_region.objects:
            if obj.category is None or obj.category.name() != target:
                continue
            obj_id = int(obj.id.rsplit('_', 1)[1])
            (target_ids if str(semantic_region.id)[1:] == region else distractor_ids).append(obj_id)
    return target_ids, distractor_ids


def boxes(mask, ids):
    result = []
    for obj_id in ids:
        ys, xs = np.where(mask == obj_id)
        if len(xs):
            result.append({'instance_id': obj_id, 'pixels': int(len(xs)),
                           'box': [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]})
    return result


def add_side_semantics():
    import habitat_sim
    import magnum as mn
    from habitat_base import simulation

    original = simulation.make_cfg

    def with_side_semantics(settings):
        cfg = original(settings)
        for code, yaw in (('l', math.pi / 3), ('r', -math.pi / 3)):
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid = 'semantic_sensor_' + code
            sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
            sensor.resolution = [settings['height'], settings['width']]
            sensor.position = mn.Vector3(0, settings['sensor_height'], 0)
            sensor.orientation = mn.Vector3(0, yaw, 0)
            cfg.agents[settings['default_agent']].sensor_specifications.append(sensor)
        return cfg

    simulation.make_cfg = with_side_semantics


def capture(three_camera=False):
    from habitat_base.simulation import SceneSimulator

    torch, first, causal, nav, ordered, harmonized, dataset, train_indices, dev_indices, checkpoint, state = verifier.setup()
    if three_camera:
        add_side_semantics()
    protocol = read(SOURCE / 'protocol.json')
    starts = read(SOURCE / 'training/start_states.json')
    indices = protocol['train_indices']
    assert len(indices) == 30 and set(map(int, starts)) == set(indices)
    assert set(indices).isdisjoint(dev_indices) and set(indices).isdisjoint(range(100))
    reference = {(row['task_index'], row['step']): row for line in
                 (SOURCE / 'training/rows.jsonl').open(encoding='utf-8')
                 if (row := json.loads(line))['mode'] == 'learned'}
    assert len(reference) == 2488
    manifest = {'status': 'train_only_replay', 'indices': indices,
                'source_start_states_sha256': sha(SOURCE / 'training/start_states.json'),
                'source_rows_sha256': sha(SOURCE / 'training/rows.jsonl'),
                'source_protocol_sha256': sha(SOURCE / 'protocol.json'),
                'source_checkpoint_sha256': protocol['checkpoint_sha256'],
                'sampling': 'every target-visible camera view and every tenth target-absent state per camera',
                'visibility': 'semantic target pixels sampled every 8 pixels >= 4',
                'cameras': ['left', 'front', 'right'] if three_camera else ['front'],
                'prompt': 'gold first/active target object category; no region or target coordinate',
                'gold_only_for_scoring': ['semantic instance mask', 'target region identity', 'geodesic distance'],
                'not_final_holdout': True}
    save(RUN / 'protocol.json', manifest)
    (RUN / 'images').mkdir(parents=True, exist_ok=True)
    output, visited = [], set()
    for ordinal, index in enumerate(indices, 1):
        config, _ = dataset[index]
        context = harmonized.load_context(config)
        sim = SceneSimulator(harmonized.make_args(), config)
        try:
            sim.save_path = None
            ordered.reset_simulator(sim, starts[str(index)])
            obs, _, info = sim.actor('stop')
            previous = ordered.ACTION_TO_IDX['stop']
            policy_history, recent = [], []
            positions = [ordered.position(sim)]
            for step in range(100):
                if sim.episode_over:
                    break
                stage = int(sim.stage)
                distance = float(info['geo dis'])
                yaw = causal.signed_yaw(sim.agent.get_state().rotation)
                pred, policy_history = ordered.choose_action(
                    ordered, 'paired_plain', state, obs, ordered.position(sim), yaw,
                    previous, policy_history, context, 'raw', recent, positions[-16:], step)
                target_pixels, distractor_pixels = verifier.visibility(sim, stage)
                old = reference[(index, step)]
                assert abs(distance - old['distance_m']) < 0.05, (index, step, distance, old['distance_m'])
                assert target_pixels == old['target_pixels_front_sampled'], (index, step, target_pixels, old)
                assert distractor_pixels == old['distractor_pixels_front_sampled'], (index, step)
                assert int(pred == 0) == old['policy_stop'], (index, step, pred, old)
                visited.add((index, step))
                target_ids, distractor_ids = instances(sim, stage)
                cameras = (('left', 'l'), ('front', 'f'), ('right', 'r')) if three_camera else (('front', 'f'),)
                for camera, code in cameras:
                    semantic_key = 'semantic_sensor' if code == 'f' else 'semantic_sensor_' + code
                    mask = np.asarray(sim.observations[semantic_key])
                    visible = int(np.isin(mask[::8, ::8], target_ids).sum())
                    if code == 'f':
                        assert visible == target_pixels
                    if visible >= 4 or step % 10 == 0:
                        frame = f'{index}_{step:03d}_{camera}'
                        Image.fromarray(np.asarray(sim.observations['color_sensor_' + code])[..., :3]).save(
                            RUN / 'images' / f'{frame}.png')
                        output.append({'task_index': index, 'step': step, 'stage': stage,
                                       'camera': camera, 'image': f'images/{frame}.png',
                                       'target': str(sim.target[stage]), 'target_visible': visible >= 4,
                                       'target_pixels_sampled': visible,
                                       'distractor_pixels_sampled': int(np.isin(mask[::8, ::8], distractor_ids).sum()),
                                       'target_boxes': boxes(mask, target_ids),
                                       'distractor_boxes': boxes(mask, distractor_ids)})
                obs, _, info = sim.actor(ordered.IDX_TO_ACTION[pred])
                previous = pred
                recent = (recent + [pred])[-16:]
                positions.append(ordered.position(sim))
        finally:
            sim.close()
        print(f'replayed {ordinal}/30 task={index} captured={len(output)}', flush=True)
    assert visited == set(reference), (len(visited), len(reference))
    (RUN / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in output), encoding='utf-8')
    save(RUN / 'capture_summary.json', {'replayed_states': len(visited), 'captured_frames': len(output),
                                        'target_visible_frames': sum(row['target_visible'] for row in output),
                                        'tasks_with_visible_target': len({row['task_index'] for row in output if row['target_visible']}),
                                        'states_with_visible_target': len({(row['task_index'], row['step']) for row in output if row['target_visible']})})


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / area if area else 0.0


def bearing(box, width):
    return math.degrees(math.atan(((box[0] + box[2]) / 2 - width / 2) / (width / 2)))


def median(values):
    return round(statistics.median(values), 3) if values else None


def score():
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    manifest = read(RUN / 'protocol.json')
    assert manifest['source_start_states_sha256'] == sha(SOURCE / 'training/start_states.json')
    frames = [json.loads(line) for line in (RUN / 'frames.jsonl').open(encoding='utf-8')]
    assert len({(row['task_index'], row['step'], row['camera']) for row in frames}) == len(frames)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'google/owlvit-base-patch32'
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(frames), 4):
            batch = frames[start:start + 4]
            images = [Image.open(RUN / row['image']).convert('RGB') for row in batch]
            prompts = [[f'a photo of a {row["target"]}'] for row in batch]
            inputs = processor(text=prompts, images=images, return_tensors='pt')
            predictions = processor.post_process_object_detection(
                model(**{key: value.to(device) for key, value in inputs.items()}),
                threshold=0.0, target_sizes=[image.size[::-1] for image in images])
            for row, prediction, image in zip(batch, predictions, images):
                ranks = prediction['scores'].argsort(descending=True)[:20]
                proposals = [{'score': round(float(prediction['scores'][rank]), 6),
                              'box': [round(float(value), 3) for value in prediction['boxes'][rank]]}
                             for rank in ranks]
                assert proposals and image.width == 512 and image.height == 512
                gold = [box['box'] for box in row['target_boxes'] if box['pixels'] >= 25]
                distractors = [box['box'] for box in row['distractor_boxes'] if box['pixels'] >= 25]
                top = proposals[0]['box']
                hit_rank = next((rank for rank, proposal in enumerate(proposals, 1)
                                 if any(iou(proposal['box'], box) >= 0.5 for box in gold)), None)
                top_hit = any(iou(top, box) >= 0.5 for box in gold)
                output.append({'task_index': row['task_index'], 'step': row['step'], 'camera': row['camera'],
                               'target_visible': row['target_visible'],
                               'target_boxes': len(gold), 'distractor_boxes': len(distractors),
                               'top1_score': proposals[0]['score'], 'top1_box': top,
                               'top1_hit': top_hit, 'first_hit_rank': hit_rank,
                               'top1_distractor_hit': any(iou(top, box) >= 0.5 for box in distractors),
                               'min_abs_bearing_error_deg': round(min(abs(bearing(top, 512) - bearing(box, 512))
                                                                   for box in gold), 3) if gold else None,
                               'proposals': proposals})
            for image in images:
                image.close()
            if start % 40 == 0 or start + 4 >= len(frames):
                print(f'scored {min(start + 4, len(frames))}/{len(frames)}', flush=True)
    (RUN / 'proposals.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in output), encoding='utf-8')
    visible = [row for row in output if row['target_visible']]
    absent = [row for row in output if not row['target_visible']]
    by_task = defaultdict(list)
    for row in visible:
        by_task[row['task_index']].append(row)
    summary = {'status': 'train_only_replay_diagnostic', 'tasks': len(manifest['indices']),
               'replayed_states': read(RUN / 'capture_summary.json')['replayed_states'],
               'scored_frames': len(output), 'visible_frames': len(visible), 'sampled_absent_frames': len(absent),
               'visible_tasks': len(by_task),
               'top1_hit_visible_frames': sum(row['top1_hit'] for row in visible),
               'top20_hit_visible_frames': sum(row['first_hit_rank'] is not None for row in visible),
               'top1_hit_visible_tasks': sum(any(row['top1_hit'] for row in rows) for rows in by_task.values()),
               'top20_hit_visible_tasks': sum(any(row['first_hit_rank'] is not None for row in rows) for rows in by_task.values()),
               'median_bearing_error_visible_deg': median([row['min_abs_bearing_error_deg'] for row in visible]),
               'median_bearing_error_on_top1_hits_deg': median([row['min_abs_bearing_error_deg'] for row in visible if row['top1_hit']]),
               'top1_distractor_hit_sampled_absent_frames': sum(row['top1_distractor_hit'] for row in absent),
               'median_top1_score_visible': median([row['top1_score'] for row in visible]),
               'median_top1_score_sampled_absent': median([row['top1_score'] for row in absent]),
               'not_measured': ['depth error', 'navigation success', 'language-to-target extraction']}
    summary['by_camera'] = {}
    for camera in manifest['cameras']:
        subset = [row for row in output if row['camera'] == camera]
        seen = [row for row in subset if row['target_visible']]
        summary['by_camera'][camera] = {
            'scored_frames': len(subset), 'visible_frames': len(seen),
            'visible_tasks': len({row['task_index'] for row in seen}),
            'top1_hit_visible_frames': sum(row['top1_hit'] for row in seen),
            'top20_hit_visible_frames': sum(row['first_hit_rank'] is not None for row in seen)}
    save(RUN / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    assert iou([0, 0, 2, 2], [1, 1, 3, 3]) == 1 / 7
    assert bearing([256, 0, 256, 10], 512) == 0
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['capture', 'score'])
    parser.add_argument('--three-camera', action='store_true')
    args = parser.parse_args()
    capture(args.three_camera) if args.phase == 'capture' else score()
