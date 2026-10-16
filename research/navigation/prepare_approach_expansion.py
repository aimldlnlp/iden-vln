"""Freeze a score-independent, scene-disjoint train/dev candidate batch."""
import hashlib
import json
from pathlib import Path

from build_route_instruction_pilot import sha

ROOT = Path('/home/aimldl/LH-VLN/runs')
OUT = ROOT / 'approach_expansion_selection_20261011_v1'


def main():
    assert not OUT.exists(), 'refuse to overwrite selection'
    pool_path = ROOT / 'route_instruction_pilot_v1_20260930/original_instruction_audit.jsonl'
    pool = [json.loads(l) for l in pool_path.read_text().splitlines()]
    canonical = ROOT / 'spatial_clip_v3_canonical100_aabb_20261011/episode_bank.json'
    forbidden = {r['scene'] for r in json.loads(canonical.read_text())}
    known_tasks = set()
    parents = {}
    for folder in ('visible_approach_framed_v2_20261011',
                   'visible_approach_framed_v2_train_expand_20261011',
                   'visible_approach_framed_v2_dev_20261011'):
        path = ROOT / folder / 'construction_ledger.json'
        records = json.loads(path.read_text())
        parents[str(path)] = sha(path)
        forbidden.update(r['scene'] for r in records)
        known_tasks.update(r['task_id'] for r in records)
    train_scenes = {r['scene'] for r in pool if r['split'] == 'train'}
    dev_scenes = {r['scene'] for r in pool if r['split'] == 'val'}
    assert train_scenes.isdisjoint(dev_scenes)
    selected = []
    availability = {}
    for split, wanted in (('train', 20), ('val', 6)):
        candidates = [r for r in pool if r['level'] == 2 and r['split'] == split
                      and r['valid_room_labels'] and r['object_mentions_in_order']
                      and r['room_mentions_in_order'] and r['task_id'] not in known_tasks
                      and r['scene'] not in forbidden]
        candidates.sort(key=lambda r: hashlib.sha256(('approach_expansion_v1:'+r['task_id']).encode()).digest())
        availability[split] = {'tasks': len(candidates), 'scenes': len({r['scene'] for r in candidates})}
        limit = min(wanted, availability[split]['scenes'])
        assert limit > 0
        used = set()
        for row in candidates:
            if row['scene'] in used:
                continue
            source = Path(row['source_task_json'])
            task = json.loads(source.read_text())
            assert len(task['Object']) == len(task['Region Name']) == len(task['trial']) == 2
            selected.append({**row, 'index': len(selected), 'source_task_sha256': sha(source),
                             'split': 'train' if split == 'train' else 'dev',
                             'start': task['trial']['trial_0']['pos'][0]})
            used.add(row['scene'])
            if len(used) == limit:
                break
        assert len(used) == limit
    assert sum(r['split']=='train' for r in selected) == 20
    assert len({r['scene'] for r in selected}) == len({r['task_id'] for r in selected}) == len(selected)
    OUT.mkdir()
    manifest = OUT / 'selection_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    protocol = {'version': 'approach_expansion_selection_v1', 'goal_resolver': 'visible_approach_framed_v2',
                'selection_rule': 'up to 20 train and 6 val-derived dev, limited by available scenes; fixed task hash order; one task per new scene; level 2; original ordered object/room mentions and valid room labels',
                'selection_before_simulator_results': True, 'availability': availability,
                'excluded_scenes': sorted(forbidden), 'parent_ledgers_sha256': parents,
                'pool_sha256': sha(pool_path), 'canonical_bank_sha256': sha(canonical),
                'selection_sha256': sha(manifest), 'script_sha256': sha(Path(__file__)),
                'train_candidates': 20, 'dev_candidates': sum(r['split']=='dev' for r in selected), 'existing_dev_cohort_preserved': 2,
                'dev_not_training': True, 'policy_scores_used': False,
                'scope': 'development expansion; original scene grouping retained; no untouched-test claim'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
