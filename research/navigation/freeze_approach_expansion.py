"""Freeze the reviewed expansion, preserving parent banks and goal geometry."""
import copy
import json
from pathlib import Path

from build_route_instruction_pilot import sha
from frozen_approach_resolver import fingerprint
from audit_original_instruction_pairs import strings

ROOT = Path('/home/aimldl/LH-VLN/runs')
BATCH = ROOT / 'approach_expansion_audit_20261011_v1'
OUT = ROOT / 'visible_approach_framed_v2_expanded_bank_20261011_v1'
PARENTS = {
    'train': ROOT / 'visible_approach_framed_v2_recovered_train4_20261011',
    'dev': ROOT / 'visible_approach_framed_v2_dev_20261011',
}
# Decisions cover all retained instances, not only the instance reached by teacher.
REASONS = {
    2: 'accepted: oven/stove and bedroom chairs recognizable; room contexts supported',
    3: 'held: stool masks and terminal view severely fragmented; several boxes unclear',
    4: 'held: toilet substantially occluded; handrail instance 203 is difficult to distinguish from shutter structure',
    5: 'held: picture instance 456 hidden behind scan geometry; book appearance ambiguous',
    6: 'accepted: office trashcan and bathroom rug recognizable with supported room context',
    7: 'held: sofa severely fragmented; living-room context insufficiently established',
    9: 'held: vacuum cleaner narrowly exposed; RGB identity insufficiently established',
    10: 'held: lamp instance 156 narrowly visible behind partition; all-instance review not satisfied',
    11: 'held: vent occupies a tiny ceiling patch, not recognizable from these views',
    15: 'held: flower scan primarily resembles damaged pot/geometry; identity insufficient',
    16: 'held: towel bar mask appears on a ring; annotation requires human review',
    17: 'accepted: living-room side table and bathroom cabinets recognizable',
    18: 'accepted: framed hall wall artwork and bathroom toilet recognizable',
    19: 'held: bowl not recognizable in supplied mask/view; garage context also uncertain',
    20: 'held: fan recognizable but supplied Office view shows foosball room, office context unconfirmed',
    21: 'accepted: bedroom flowers and living-room footrests recognizable; scan damage noted',
}


def read(path):
    return json.loads(path.read_text())


def main():
    assert not OUT.exists(), 'refuse to overwrite frozen bank'
    ledger = read(BATCH / 'construction_ledger.json')
    candidates = {r['source_index']: r for r in read(BATCH / 'demonstration_candidates.json')}
    runtime = read(BATCH / 'runtime_verification.json')
    assert runtime['status'] == 'passed' and runtime['tasks'] == 16 and runtime['runtime_STOPs_verified'] == 32
    verified = {r['source_index']: r for r in runtime['records']}
    goals = read(BATCH / 'frozen_diagnostic_goal_manifest.json')
    goal_hash = sha(BATCH / 'frozen_diagnostic_goal_manifest.json')
    assert goal_hash == '1b571250ffd18223729f4600e28226c0f299b35c1a62c26d97af0841ddfd6b12'
    decisions = []
    for row in ledger:
        i = row['source_index']
        reason = REASONS.get(i, 'held: incomplete goal construction' if i != 8 else 'held: no_visible_STOP_view_after_follower')
        accepted = reason.startswith('accepted:')
        if accepted:
            assert i in verified and candidates[i]['independent_replay_pass']
        decisions.append({'source_index': i, 'task_id': row['task_id'], 'split': row['split'],
                          'accepted': accepted, 'reason': reason, 'human_validated': False,
                          'reviewer': 'Codex RGB review',
                          'evidence': [f'review_evidence/{i:03d}_{s}_sheet.png' for s in range(2)] if i in REASONS else []})
    assert len(decisions) == 23 and {r['source_index'] for r in decisions if r['accepted']} == {2, 6, 17, 18, 21}
    parent_hashes = {}
    banks, goal_banks = {}, {}
    for split, folder in PARENTS.items():
        manifest = folder / 'reviewed_demonstrations_frozen.jsonl'
        goal_path = folder / 'reviewed_goal_manifest_frozen.json'
        parent_hashes[str(manifest)] = sha(manifest)
        parent_hashes[str(goal_path)] = sha(goal_path)
        banks[split] = [json.loads(line) for line in manifest.read_text().splitlines()]
        goal_banks[split] = read(goal_path)
    assert len(banks['train']) == 4 and len(banks['dev']) == 1
    mapping = []
    for decision in decisions:
        if not decision['accepted']:
            continue
        i, split = decision['source_index'], decision['split']
        row = copy.deepcopy(candidates[i])
        verify = verified[i]
        source = Path(row['source_task_json'])
        task = read(source)
        assert sha(source) == row['source_sha256'] and fingerprint(task) == row['config_fingerprint']
        assert row['instruction_original'] in list(strings(task))
        assert verify['max_position_error_m'] < 1e-5
        assert [r['stage'] for r in row['teacher_actions'] if r['action'] == 'stop'] == [0, 1]
        assert len(row['teacher_actions']) <= 400 and len(row['stops']) == 2
        assert all(s['runtime_success'] and s['distance_m'] < 1 for s in verify['stops'])
        assert all(s['framing_gate_pass'] and s['approach_geo_m'] < 1 and 25 <= s['front_target_pixels'] <= 52428.8 for s in row['stops'])
        goal = copy.deepcopy(next(g for g in goals if g['source_index'] == i))
        row.update(source_index=1000+i, expansion_source_index=i, index=len(banks[split]),
                   training_approved=split == 'train', evaluation_approved=split == 'dev',
                   benchmark_training_ready=False, human_validated=False,
                   policy_context={'instruction': row['instruction_original'], 'robot': str(task['Robot']).lower()},
                   start_rotation_simulator_argument=[1., 0., 0., 0.],
                   visual_review_reason=decision['reason'], RGB_review='Codex; not human annotation')
        goal.update(source_index=1000+i, expansion_source_index=i, RGB_review=decision['reason'])
        mapping.append({'batch_source_index': i, 'bank_source_index': 1000+i, 'split': split, 'task_id': row['task_id']})
        banks[split].append(row)
        goal_banks[split].append(goal)
    full_dev_goals = read(PARENTS['dev'] / 'frozen_diagnostic_goal_manifest.json')
    train_scenes = {r['scene'] for r in banks['train']}
    dev_scenes = {r['scene'] for r in full_dev_goals} | {r['scene'] for r in ledger if r['split'] == 'dev'}
    assert not train_scenes & dev_scenes
    all_rows = banks['train'] + banks['dev']
    assert len({r['task_id'] for r in all_rows}) == len(all_rows)
    assert all(r['training_approved'] for r in banks['train'])
    assert all(not r['training_approved'] for r in banks['dev'])
    OUT.mkdir()
    (BATCH / 'visual_review_decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
    summaries = {}
    for split in ('train', 'dev'):
        gp = OUT / f'{split}_goal_manifest_frozen.json'
        gp.write_text(json.dumps(goal_banks[split], indent=2)+'\n')
        for row in banks[split]:
            row['reviewed_goal_manifest_sha256'] = sha(gp)
        mp = OUT / f'{split}_demonstrations_frozen.jsonl'
        mp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in banks[split]))
        summaries[split] = {'tasks': len(banks[split]), 'scenes': len({r['scene'] for r in banks[split]}),
                            'actions': sum(len(r['teacher_actions']) for r in banks[split]),
                            'STOPs': sum(sum(a['action'] == 'stop' for a in r['teacher_actions']) for r in banks[split]),
                            'manifest_sha256': sha(mp), 'goal_sha256': sha(gp)}
    assert summaries['train']['tasks'] == 8 and summaries['dev']['tasks'] == 2
    assert summaries['train']['actions'] == 1357 and summaries['dev']['actions'] == 386
    assert all(sha(Path(p)) == digest for p, digest in parent_hashes.items())
    result = {'status': 'reviewed expansion frozen', 'bank_version': 'expanded_bank_v1',
              'resolver': 'visible_approach_framed_v2', 'scoring_changed': False,
              'new_policy_trained': False, 'human_validated': False,
              'batch_selected': {'train': 20, 'dev': 3}, 'batch_replay_pass': {'train': 14, 'dev': 2},
              'batch_accepted': {'train': 4, 'dev': 1}, 'full_dev_cohort': 5, 'accepted_dev': 2,
              'train_target': 25, 'target_reached': False, 'train_dev_scene_overlap': 0,
              'banks': summaries, 'source_index_mapping': mapping, 'parent_hashes': parent_hashes,
              'frozen_batch_goal_sha256': goal_hash,
              'decisions_sha256': sha(BATCH / 'visual_review_decisions.json'), 'script_sha256': sha(Path(__file__))}
    (OUT / 'reviewed_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'protocol.json').write_text(json.dumps({**read(BATCH / 'protocol.json'), **result}, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
