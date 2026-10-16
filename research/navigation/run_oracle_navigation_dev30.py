#!/usr/bin/env python3
"""Privileged navigation diagnostic with learned and oracle STOP on fixed dev30."""

import inspect
import json
import os
import sys
import textwrap
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path('/home/aimldl/LH-VLN')
SOURCE = ROOT / 'runs/idenvln_active_target_ablation_20260926'
RUN = Path(os.environ['IDENVLN_RUN_DIR'])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def make_oracle_follower(sim):
    import habitat_sim

    sim.follower = habitat_sim.nav.GreedyGeodesicFollower(
        pathfinder=sim.pathfinder, agent=sim.agent, goal_radius=0.75,
        stop_key='stop', forward_key='move_forward', left_key='turn_left', right_key='turn_right')


def oracle_navigation_action(legacy, sim, target_coords, active_stage, distance, prediction,
                             success_dis, oracle_stop, index, step, events):
    if oracle_stop and distance < success_dis:
        action = 'stop'
        kind = 'forced_stop'
    elif not oracle_stop and prediction == legacy.ACTION_TO_IDX['stop']:
        action = 'stop'
        kind = 'learned_stop'
    else:
        _, goal = ordered.geodesic_between(sim.pathfinder, sim.agent.get_state().position,
                                           target_coords[active_stage])
        if goal is None:
            raise RuntimeError(f'no_reachable_goal index={index} stage={active_stage}')
        action = sim.get_next_action(goal)
        if action == 'stop':
            action = 'turn_left'
            kind = 'hold_at_goal'
        elif action in ('move_forward', 'turn_left', 'turn_right'):
            kind = 'oracle_move'
        else:
            raise RuntimeError(f'unexpected_follower_action={action!r}')
    events.append({'index': int(index), 'step': int(step), 'stage': int(active_stage),
                   'kind': kind, 'distance_m': float(distance),
                   'learned_action': legacy.IDX_TO_ACTION[prediction], 'executed_action': action})
    return legacy.ACTION_TO_IDX[action]


def diagnose(records, first):
    result = first.diagnose(records)
    result['reached_first_subgoal_1m'] = sum(row['visited_targets'][0] for row in records)
    result['reached_first_while_active_1m'] = sum(any(
        step['active_target'] == 0 and step['distances_after'][0] is not None
        and step['distances_after'][0] < 1.0 for step in row['steps']) for row in records)
    result['full_chain_success'] = sum(row['full_chain_success'] for row in records)
    result['invalid_stop_events'] = sum(step['stop'] and not step['active_target_success_after']
                                        for row in records for step in row['steps'])
    result['valid_stop_events'] = sum(step['stop'] and step['active_target_success_after']
                                      for row in records for step in row['steps'])
    return result


def main():
    assert replace_once('ab', 'a', 'x') == 'xb'
    RUN.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
    import run_dagger_lite_dev as first
    import train_causal_sequence_bc as causal
    from utils.dataset import TaskDataset
    global ordered

    protocol = read(SOURCE / 'protocol.json')
    comparison = read(SOURCE / 'results/comparison.json')
    val_indices = protocol['validation_indices']
    starts_path = Path(protocol['source_start_states'])
    assert causal.sha(starts_path) == protocol['source_start_states_sha256']
    starts = read(starts_path)
    assert len(val_indices) == len(starts) == 30 and set(val_indices).isdisjoint(range(100))
    checkpoint = Path(read(SOURCE / 'training/paired_plain.json')['model_path'])
    assert checkpoint.exists() and comparison['status'] == 'completed'

    ordered = first.load(first.ORDERED, 'oracle_navigation_ordered')
    harmonized = first.load(first.HARMONIZED, 'oracle_navigation_harmonized')
    sweep = first.load(first.SOURCE / 'run_policy_sweep.py', 'oracle_navigation_sweep')
    for name in ('load_context', 'make_args', 'tiered_feature', 'target_feature', 'seq_feature',
                 'select_action_from_logits', 'ACTION_TO_IDX', 'IDX_TO_ACTION', 'SEED', 'WINDOW',
                 'SOURCE_DATA', 'torch', 'BalancedMLP', 'TargetConditionedMLP', 'SequenceGRU'):
        setattr(ordered, name, getattr(harmonized, name))
    ordered.install_index_aware_target_patch()
    dataset = TaskDataset(harmonized.make_args(), 'test')
    state = sweep.load_policy_from_checkpoint(ordered, {'name': 'paired_plain', 'feature': 'seq',
                                                        'checkpoint': str(checkpoint)})
    assert state['status'] == 'loaded' and len(dataset) == 300
    limit = int(os.environ.get('IDENVLN_LIMIT', '0'))
    indices = val_indices[:limit] if limit else val_indices
    save('protocol.json', {'label': 'DIAGNOSTIC_PRIVILEGED', 'source_run': str(SOURCE),
                           'checkpoint_sha256': causal.sha(checkpoint),
                           'start_states_sha256': causal.sha(starts_path),
                           'validation_indices': indices, 'max_steps': 150, 'success_distance_m': 1.0,
                           'follower_goal_radius_m': 0.75,
                           'learned_stop_rule': 'preserve predicted STOP; oracle overrides only movement',
                           'oracle_stop_rule': 'force STOP within 1m; otherwise oracle movement',
                           'at_goal_without_stop': 'turn_left', 'privileged_navigation': True})

    source = textwrap.dedent(inspect.getsource(ordered.evaluate_policy))
    source = replace_once(source, 'yaw = float(sim.agent.get_state().rotation.angle())',
                          'yaw = causal.signed_yaw(sim.agent.get_state().rotation)')
    source = replace_once(source, 'reset_simulator(sim, start_states[key])',
                          'reset_simulator(sim, start_states[key])\n                make_oracle_follower(sim)')
    source = replace_once(source, 'action = legacy.IDX_TO_ACTION[pred]', '''pred = oracle_navigation_action(
                        legacy, sim, target_coords, active_stage, before_distances[active_stage],
                        pred, success_dis, oracle_stop, index, step, events)
                    action = legacy.IDX_TO_ACTION[pred]''')
    results = {}
    for name, oracle_stop in [('learned_stop', False), ('oracle_stop', True)]:
        events = []
        namespace = dict(vars(ordered), causal=causal, make_oracle_follower=make_oracle_follower,
                         oracle_navigation_action=oracle_navigation_action,
                         oracle_stop=oracle_stop, events=events)
        exec(compile(source, '<oracle-navigation-evaluator>', 'exec'), namespace)
        records, summary = namespace['evaluate_policy'](
            ordered, dataset, 'paired_plain_oracle_navigation_' + name, 'raw', state,
            indices, dict(starts), RUN / f'results/{name}.jsonl', 150, 1.0, first.SEED,
            RUN / f'results/{name}_progress.json', {})
        assert len(records) == summary['N_effective'] == len(indices)
        diagnostic = diagnose(records, first)
        results[name] = {'diagnostic': diagnostic, 'summary': summary,
                         'action_kinds': dict(Counter(event['kind'] for event in events))}
        save(f'results/{name}_events.json', events)
        save('results/comparison.json', {'status': 'running', **results})
        print(json.dumps({'condition': name, 'result': {key: diagnostic[key] for key in
                         ('first_subgoal_success', 'reached_first_subgoal_1m', 'full_chain_success',
                          'valid_stop_events', 'invalid_stop_events')},
                          'action_kinds': results[name]['action_kinds']}), flush=True)
    save('results/comparison.json', {'status': 'completed', 'label': 'DIAGNOSTIC_PRIVILEGED',
                                     'baseline': comparison['baseline'], **results})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            save('status.json', {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
