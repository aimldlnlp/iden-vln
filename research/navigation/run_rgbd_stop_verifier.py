#!/usr/bin/env python3
"""Train-only RGB-D STOP verifier and fixed dev30 rollout diagnostic."""

import argparse
import inspect
import json
import os
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/aimldl/LH-VLN')
SOURCE = ROOT / 'runs/idenvln_active_target_ablation_20260926'
RUN = Path(os.environ['IDENVLN_RUN_DIR'])
WINDOW = 4
SEED = 61007


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = RUN / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf-8')


def window(history):
    dim = len(history[-1])
    return np.asarray([[0.0] * dim] * max(0, WINDOW - len(history)) + history[-WINDOW:],
                      dtype=np.float32)


def feature(sim, context, stage, previous, harmonized):
    values = []
    for camera in ('l', 'f', 'r'):
        color = np.asarray(sim.observations['color_sensor_' + camera])[::16, ::16, :3]
        depth = np.asarray(sim.observations['depth_sensor_' + camera])[::16, ::16]
        depth = np.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.0)
        depth = np.clip(depth, 0, 10)
        for ys in np.array_split(np.arange(len(depth)), 3):
            for xs in np.array_split(np.arange(depth.shape[1]), 3):
                patch = color[np.ix_(ys, xs)]
                values.extend((patch.mean(axis=(0, 1)) / 255.0).tolist())
                values.append(float(np.median(depth[np.ix_(ys, xs)]) / 10.0))
    objects, regions = context['objects'], context['regions']
    target = objects[stage] if stage < len(objects) else ''
    region = regions[stage] if stage < len(regions) else ''
    values.extend(harmonized.hash_features([target, region], 16))
    values.extend([float(stage) / max(1, len(objects)), float(context['robot'] == 'spot')])
    values.extend(float(previous == i) for i in range(4))
    assert len(values) == 130
    return values


def visibility(sim, stage):
    mask = np.asarray(sim.observations['semantic_sensor'])[::8, ::8]
    target, region = sim.target[stage], str(sim.region[stage])
    target_ids, distractor_ids = [], []
    for semantic_region in sim.sim.semantic_scene.regions:
        for obj in semantic_region.objects:
            if obj.category is None or obj.category.name() != target:
                continue
            obj_id = int(obj.id.rsplit('_', 1)[1])
            if str(semantic_region.id)[1:] == region:
                target_ids.append(obj_id)
            else:
                distractor_ids.append(obj_id)
    return (int(np.isin(mask, target_ids).sum()), int(np.isin(mask, distractor_ids).sum()))


def snapshot(sim, name):
    folder = RUN / 'snapshots' / name
    folder.mkdir(parents=True, exist_ok=True)
    color = np.asarray(sim.observations['color_sensor_f'])[..., :3]
    depth = np.nan_to_num(np.asarray(sim.observations['depth_sensor_f']),
                          nan=10.0, posinf=10.0, neginf=0.0)
    Image.fromarray(color).save(folder / 'front.png')
    Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16)).save(folder / 'depth_mm.png')
    return str(folder)


def setup():
    import torch
    import run_dagger_lite_dev as first
    import train_causal_sequence_bc as causal
    import run_oracle_navigation_dev30 as nav
    from utils.dataset import TaskDataset

    ordered = first.load(first.ORDERED, 'rgbd_ordered')
    harmonized = first.load(first.HARMONIZED, 'rgbd_harmonized')
    sweep = first.load(first.SOURCE / 'run_policy_sweep.py', 'rgbd_sweep')
    for name in ('load_context', 'make_args', 'tiered_feature', 'target_feature', 'seq_feature',
                 'select_action_from_logits', 'ACTION_TO_IDX', 'IDX_TO_ACTION', 'SEED', 'WINDOW',
                 'SOURCE_DATA', 'torch', 'BalancedMLP', 'TargetConditionedMLP', 'SequenceGRU'):
        setattr(ordered, name, getattr(harmonized, name))
    ordered.install_index_aware_target_patch()
    nav.ordered = ordered
    dataset = TaskDataset(harmonized.make_args(), 'test')
    protocol = read(SOURCE / 'protocol.json')
    train_indices = read(ROOT / 'runs/idenvln_dagger_dev_20260925T131842Z_4042083/protocol.json')['split_task_indices']['train']
    dev_indices = protocol['validation_indices']
    assert len(dataset) == 300 and len(train_indices) == 140 and len(dev_indices) == 30
    assert set(train_indices).isdisjoint(dev_indices) and not set(train_indices + dev_indices).intersection(range(100))
    checkpoint = Path(read(SOURCE / 'training/paired_plain.json')['model_path'])
    state = sweep.load_policy_from_checkpoint(ordered, {'name': 'paired_plain', 'feature': 'seq',
                                                         'checkpoint': str(checkpoint)})
    assert state['status'] == 'loaded'
    return torch, first, causal, nav, ordered, harmonized, dataset, train_indices, dev_indices, checkpoint, state


def collect(limit):
    from habitat_base.simulation import SceneSimulator
    torch, first, causal, nav, ordered, harmonized, dataset, train_indices, dev_indices, checkpoint, state = setup()
    assert 0 <= limit <= 30
    indices = read(ROOT / 'runs/idenvln_dagger_dev_20260925T131842Z_4042083/protocol.json')['dagger_collection_indices']
    assert len(indices) == 30 and set(indices).issubset(train_indices)
    starts = {}
    xs, ys, task_ids, modes, rows = [], [], [], [], []
    snapshots = Counter()
    for ordinal, index in enumerate(indices[:limit or 30], 1):
        config, _ = dataset[index]
        context = harmonized.load_context(config)
        for mode in ('learned', 'follower'):
            sim = SceneSimulator(harmonized.make_args(), config)
            try:
                sim.save_path = None
                starts.setdefault(str(index), ordered.capture_state(sim))
                ordered.reset_simulator(sim, starts[str(index)])
                if mode == 'follower':
                    nav.make_oracle_follower(sim)
                target_coords = [ordered.target_coords_for_index(sim, k) for k in range(len(sim.target))]
                obs, _, info = sim.actor('stop')
                previous = ordered.ACTION_TO_IDX['stop']
                policy_history, verifier_history, recent = [], [], []
                positions = [ordered.position(sim)]
                for step in range(100):
                    if sim.episode_over:
                        break
                    stage = int(sim.stage)
                    distance = float(info['geo dis'])
                    assert np.isfinite(distance)
                    if step % 20 == 0:
                        check = ordered.all_target_distances(sim, target_coords)[stage]
                        assert abs(check - distance) < 0.05, (index, step, distance, check)
                    yaw = causal.signed_yaw(sim.agent.get_state().rotation)
                    pred, policy_history = ordered.choose_action(
                        ordered, 'paired_plain', state, obs, ordered.position(sim), yaw,
                        previous, policy_history, context, 'raw', recent, positions[-16:], step)
                    frame = feature(sim, context, stage, previous, harmonized)
                    verifier_history.append(frame)
                    target_pixels, distractor_pixels = visibility(sim, stage)
                    label = int(distance < 1.0)
                    hard_negative = int(pred == ordered.ACTION_TO_IDX['stop'] and not label)
                    category = ('near' if label else 'wrong_stop' if hard_negative else
                                'distractor' if distractor_pixels >= 4 and target_pixels < 4 else 'far')
                    key = (index, mode, category)
                    saved = None
                    if snapshots[key] < 1:
                        saved = snapshot(sim, f'{index}_{mode}_{category}')
                        snapshots[key] += 1
                    xs.append(window(verifier_history))
                    ys.append(label)
                    task_ids.append(index)
                    modes.append(mode)
                    rows.append({'task_index': index, 'mode': mode, 'step': step, 'stage': stage,
                                 'distance_m': distance, 'label': label, 'policy_stop': int(pred == 0),
                                 'target_pixels_front_sampled': target_pixels,
                                 'distractor_pixels_front_sampled': distractor_pixels,
                                 'snapshot': saved})
                    if mode == 'learned':
                        action = ordered.IDX_TO_ACTION[pred]
                    else:
                        pred = nav.oracle_navigation_action(
                            ordered, sim, target_coords, stage, distance, pred, 1.0, True,
                            index, step, [])
                        action = ordered.IDX_TO_ACTION[pred]
                    obs, _, info = sim.actor(action)
                    previous = pred
                    recent = (recent + [pred])[-16:]
                    positions.append(ordered.position(sim))
            finally:
                sim.close()
        print(f'collected {ordinal}/{limit or 30} task={index} rows={len(ys)} positives={sum(ys)}',
              flush=True)
    (RUN / 'training').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUN / 'training/states.npz', x=np.asarray(xs, dtype=np.float32),
                        y=np.asarray(ys, dtype=np.uint8), task=np.asarray(task_ids, dtype=np.int16),
                        mode=np.asarray(modes))
    (RUN / 'training/rows.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    save('training/summary.json', {'task_count': len(indices[:limit or 30]), 'states': len(ys),
                                   'positive': sum(ys), 'mode_counts': dict(Counter(modes)),
                                   'hard_negative': sum(row['policy_stop'] and not row['label'] for row in rows),
                                   'target_visible_front_sampled': sum(row['target_pixels_front_sampled'] >= 4 for row in rows),
                                   'distractor_visible_front_sampled': sum(row['distractor_pixels_front_sampled'] >= 4 for row in rows),
                                   'snapshots': sum(snapshots.values())})
    save('training/start_states.json', starts)
    save('protocol.json', {'train_indices': indices[:limit or 30], 'validation_indices': dev_indices,
                           'canonical_indices_excluded': list(range(100)),
                           'checkpoint_sha256': causal.sha(checkpoint),
                           'success_distance_m': 1.0, 'feature': 'three-camera RGB-D grids, active target hash, stage, previous action',
                           'privileged_distance_used_for_labels_only': True,
                           'follower_navigation_privileged_for_training_only': True})
    print(read(RUN / 'training/summary.json'), flush=True)


def ap(scores, labels):
    total = int(sum(labels))
    if not total:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    hits = 0
    return sum(((hits := hits + 1) / rank) for rank, (_, label) in enumerate(ranked, 1)
               if label) / total


def train():
    import torch
    from torch import nn
    data = np.load(RUN / 'training/states.npz')
    x, y, tasks = data['x'], data['y'], data['task']
    assert x.ndim == 3 and x.shape[1:] == (WINDOW, 130) and len(x) == len(y) == len(tasks)
    unique = sorted(set(tasks.tolist()))
    assert len(unique) == 30 and int(y.sum()) > 0
    random.Random(SEED).shuffle(unique)
    held = set(unique[:6])
    train_mask = np.asarray([task not in held for task in tasks])
    val_mask = ~train_mask
    assert y[train_mask].sum() > 0 and y[val_mask].sum() > 0
    flat = x.reshape(len(x), -1)
    mean = flat[train_mask].mean(0)
    std = np.maximum(flat[train_mask].std(0), 0.1)
    flat = ((flat - mean) / std).astype(np.float32)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = nn.Sequential(nn.Linear(flat.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 1)).to(device)
    tx = torch.from_numpy(flat[train_mask]).to(device)
    ty = torch.from_numpy(y[train_mask].astype(np.float32)).to(device)
    vx = torch.from_numpy(flat[val_mask]).to(device)
    vy = y[val_mask].astype(int).tolist()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
    pos_weight = min(50.0, float((len(ty) - ty.sum()) / ty.sum()))
    best = {'ap': -1.0}
    for epoch in range(1, 16):
        net.train()
        for batch in torch.randperm(len(tx), device=device).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(tx[batch]).flatten(), ty[batch],
                pos_weight=torch.tensor(pos_weight, device=device))
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(net(vx).flatten()).cpu().tolist()
        score = ap(scores, vy)
        print(f'epoch={epoch} internal_val_ap={score:.5f}', flush=True)
        if score > best['ap']:
            best = {'ap': score, 'epoch': epoch, 'scores': scores,
                    'state': {k: v.cpu().clone() for k, v in net.state_dict().items()}}
    candidates = sorted(set(best['scores']))
    ranked = []
    for threshold in candidates:
        selected = [score >= threshold for score in best['scores']]
        tp = sum(p and y for p, y in zip(selected, vy))
        fp = sum(p and not y for p, y in zip(selected, vy))
        precision = tp / (tp + fp) if tp + fp else 0.0
        ranked.append((tp if precision >= 0.3 else -1, -fp, threshold, precision))
    tp, neg_fp, threshold, precision = max(ranked)
    if tp < 0:
        threshold = 1.0
        precision = 0.0
    checkpoint = RUN / 'training/verifier.pt'
    torch.save({'state': best['state'], 'mean': mean, 'std': std, 'threshold': threshold,
                'feature_dim': 130, 'window': WINDOW}, checkpoint)
    save('training/model.json', {'internal_val_tasks': sorted(held), 'train_tasks': sorted(set(unique) - held),
                                 'train_states': int(train_mask.sum()), 'internal_val_states': int(val_mask.sum()),
                                 'train_positive': int(y[train_mask].sum()), 'internal_val_positive': int(y[val_mask].sum()),
                                 'best_epoch': best['epoch'], 'internal_val_ap': best['ap'],
                                 'internal_val_positive_rate': float(y[val_mask].mean()),
                                 'threshold': threshold, 'minimum_internal_precision': 0.3,
                                 'precision_at_threshold': precision,
                                 'recall_at_threshold': max(tp, 0) / sum(vy),
                                 'checkpoint': str(checkpoint)})
    print(read(RUN / 'training/model.json'), flush=True)


def evaluate():
    import torch
    torch, first, causal, nav, ordered, harmonized, dataset, train_indices, dev_indices, checkpoint, state = setup()
    data = torch.load(RUN / 'training/verifier.pt', map_location='cpu')
    net = torch.nn.Sequential(torch.nn.Linear(WINDOW * 130, 64), torch.nn.ReLU(),
                              torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))
    net.load_state_dict(data['state'])
    net.eval()
    mean, std, threshold = data['mean'], data['std'], float(data['threshold'])
    starts_path = Path(read(SOURCE / 'protocol.json')['source_start_states'])
    assert causal.sha(starts_path) == read(SOURCE / 'protocol.json')['source_start_states_sha256']
    starts = read(starts_path)
    assert set(starts) == {str(i) for i in dev_indices}
    source = textwrap.dedent(inspect.getsource(ordered.evaluate_policy))
    source = nav.replace_once(source, 'yaw = float(sim.agent.get_state().rotation.angle())',
                              'yaw = causal.signed_yaw(sim.agent.get_state().rotation)')
    source = nav.replace_once(source, 'sequence_buffer: List[List[float]] = []',
                              'sequence_buffer: List[List[float]] = []\n                verifier_history = []')
    source = nav.replace_once(source, 'action = legacy.IDX_TO_ACTION[pred]',
                              '''pred = decide(sim, context, active_stage, previous, verifier_history,
                                      sequence_buffer, policy_state, net, mean, std, threshold,
                                      harmonized, torch)
                    action = legacy.IDX_TO_ACTION[pred]''')
    save('results/protocol.json', {'dev_indices': dev_indices,
                                   'start_states_sha256': causal.sha(starts_path),
                                   'motion_checkpoint_sha256': causal.sha(checkpoint),
                                   'verifier_checkpoint': str(RUN / 'training/verifier.pt'),
                                   'threshold': threshold, 'max_steps': 150, 'success_distance_m': 1.0,
                                   'distance_used_at_inference': False})
    for mode in ('learned', 'follower'):
        body = source
        if mode == 'follower':
            body = nav.replace_once(body, 'reset_simulator(sim, start_states[key])',
                                    'reset_simulator(sim, start_states[key])\n                nav.make_oracle_follower(sim)')
            body = nav.replace_once(body, 'action = legacy.IDX_TO_ACTION[pred]',
                                    '''pred = nav.oracle_navigation_action(
                        legacy, sim, target_coords, active_stage, before_distances[active_stage],
                        pred, success_dis, False, index, step, [])
                    action = legacy.IDX_TO_ACTION[pred]''')
        namespace = dict(vars(ordered), causal=causal, nav=nav, decide=decide, net=net,
                         mean=mean, std=std, threshold=threshold, harmonized=harmonized,
                         torch=torch)
        exec(compile(body, '<rgbd-verifier-evaluator>', 'exec'), namespace)
        records, summary = namespace['evaluate_policy'](
            ordered, dataset, 'paired_plain_rgbd_verifier_' + mode, 'raw', state,
            dev_indices, dict(starts), RUN / f'results/{mode}.jsonl', 150, 1.0, SEED,
            RUN / f'results/{mode}_progress.json', {})
        assert len(records) == summary['N_effective'] == 30
        result = nav.diagnose(records, first)
        save(f'results/{mode}_summary.json', {'diagnostic': result, 'summary': summary})
        print(f'{mode}: first={result["first_subgoal_success"]}/30 full={result["full_chain_success"]}/30 '
              f'valid_stop={result["valid_stop_events"]} invalid_stop={result["invalid_stop_events"]}', flush=True)
    save('status.json', {'status': 'completed'})


def decide(sim, context, stage, previous, history, policy_history, policy_state, net,
           mean, std, threshold, harmonized, torch):
    current = feature(sim, context, stage, previous, harmonized)
    history.append(current)
    x = ((window(history).reshape(-1) - mean) / std).astype(np.float32)
    with torch.inference_mode():
        probability = float(torch.sigmoid(net(torch.from_numpy(x).unsqueeze(0)))[0, 0])
    if probability >= threshold:
        return 0
    policy_window = int(policy_state.get('window', 8))
    padded = [[0.0] * len(policy_history[-1])] * max(0, policy_window - len(policy_history)) + policy_history[-policy_window:]
    with torch.inference_mode():
        logits = policy_state['model'](torch.tensor([padded], dtype=torch.float32))[0]
    return int(logits[1:].argmax()) + 1


def main():
    assert window([[1.0, 2.0]]).tolist() == [[0.0, 0.0]] * 3 + [[1.0, 2.0]]
    assert abs(ap([0.9, 0.8, 0.1], [1, 0, 1]) - 5 / 6) < 1e-9
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('collect', 'train', 'evaluate'))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
    RUN.mkdir(parents=True, exist_ok=True)
    if args.phase == 'collect':
        collect(args.limit)
    elif args.phase == 'train':
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()
