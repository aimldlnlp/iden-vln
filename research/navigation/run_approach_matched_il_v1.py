"""Freeze a matched bank4/bank8 comparison before fitting either policy."""
import copy
import json
from pathlib import Path
import train_approach_matched_il_v1 as il

def main():
    root = il.ROOT / 'runs'
    parent = root / 'visible_approach_framed_v2_expanded_bank_20261011_v2'
    out = root / 'approach_matched_il_20261011_v1'
    assert not out.exists(), 'refuse overwrite'
    paths = [parent / f'{split}_{kind}_frozen.{ext}' for split in ('train','dev')
             for kind,ext in (('demonstrations','jsonl'),('goal_manifest','json'))]
    hashes = {str(p):il.review.sha(p) for p in paths}
    train = list(map(json.loads,paths[0].read_text().splitlines()))
    tg = json.loads(paths[1].read_text())
    dev = list(map(json.loads,paths[2].read_text().splitlines()))
    dg = json.loads(paths[3].read_text())
    assert len(train)==len(tg)==8 and len(dev)==len(dg)==2
    assert [r['source_index'] for r in train[:4]]==[100,104,108,0]
    assert [r['source_index'] for r in dev]==[207,1021]
    assert {r['scene'] for r in train}.isdisjoint({r['scene'] for r in dev})
    assert sum(len(r['teacher_actions']) for r in train[:4])==692
    assert sum(len(r['teacher_actions']) for r in train)==1357
    out.mkdir()
    runs = []
    for n in (4,8):
        folder = out / f'bank{n}'
        folder.mkdir()
        rows = copy.deepcopy(train[:n]+dev)
        goals = copy.deepcopy(tg[:n]+dg)
        assert {r['config_fingerprint'] for r in rows}=={r['config_fingerprint'] for r in goals}
        for index,row in enumerate(rows):
            assert row['independent_replay_pass'] and row['teacher_failure'] is None
            assert row['policy_context']['instruction']==row['instruction_original']
            assert row['training_approved']==(row['split']=='train')
            assert [a['stage'] for a in row['teacher_actions'] if a['action']=='stop']==[0,1]
            row.update(index=index,navigation_projection_ready=True,navigation_projection_stages=2)
        bank = folder/'model_manifest_frozen.jsonl'
        goal = folder/'goal_manifest_frozen.json'
        bank.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        goal.write_text(json.dumps(goals,indent=2)+'\n')
        runs.append({'train_tasks':n,'bank':str(bank),'goals':str(goal),
                     'bank_sha256':il.review.sha(bank),'goals_sha256':il.review.sha(goal)})
    protocol = {'version':'approach_matched_il_v1','parents':hashes,'runs':runs,
        'model':'frozen spatial CLIP ViT-B/16; SequenceGRU hidden128; window16',
        'seed':20260930,'optimizer':'AdamW lr0.0012 weight_decay0.0001',
        'optimizer_updates':1200,'early_stopping':False,'checkpoint_selection':'last update',
        'budget_note':'equal optimizer updates; full-batch sizes differ, compute is not equal',
        'controls':['normal','empty_text','swapped_text'],
        'dev_source_indices':[207,1021],'full_dev_cohort':5,'held_dev_source_indices':[203,1020,1022],
        'scoring':'two ordered STOPs; geodesic strictly <1m; horizon400; frozen approach-v2',
        'policy_inputs':'RGB L/F/R, instruction, robot, own relative pose/yaw and previous action',
        'forbidden_inputs':['target metadata','gold goal coordinates','semantic masks','NavMesh','teacher stage'],
        'human_validated':False,'scope':'small development pilot; no generalization claim or Table4 replacement',
        'launcher_sha256':il.review.sha(Path(__file__)),
        'trainer_sha256':il.review.sha(Path(il.__file__))}
    (out/'comparison_protocol_frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    assert json.loads((out/'bank4/model_manifest_frozen.jsonl').read_text().splitlines()[-1])['policy_context']==dev[-1]['policy_context']
    for run in runs:
        il.main(bank=run['bank'],out=str(Path(run['bank']).parent),expected=run['bank_sha256'],
                clip=True,approach_manifest=run['goals'])
        summary=json.loads((Path(run['bank']).parent/'summary.json').read_text())
        assert summary['training_last']['step']==1200
        assert summary['dev_episodes']==2
    assert all(il.review.sha(Path(p))==h for p,h in hashes.items())
    summaries={f"bank{r['train_tasks']}":json.loads((Path(r['bank']).parent/'summary.json').read_text()) for r in runs}
    (out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    main()
