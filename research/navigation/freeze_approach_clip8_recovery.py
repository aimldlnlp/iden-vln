"""Freeze reviewed recovery alongside unchanged originals; no policy fitting."""
import json
from pathlib import Path
from build_route_instruction_pilot import sha

ROOT=Path('/home/aimldl/LH-VLN/runs')
OUT=ROOT/'approach_clip8_recovery_collection_20261013_v2'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
protocol=json.loads((OUT/'protocol_frozen.json').read_text())
assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
verification=json.loads((OUT/'independent_verification.json').read_text())
assert verification['passed'] and verification['candidate_count']==7 and verification['verified_STOPs']==14
checks={r['index']:r for r in verification['records']}
decisions={r['index']:r for r in json.loads((OUT/'visual_review_decisions.json').read_text())}
ledger=json.loads((OUT/'candidate_ledger.json').read_text())
assert set(decisions)=={r['index'] for r in ledger}==set(range(8))
original=list(map(json.loads,(PARENT/'model_manifest_frozen.jsonl').read_text().splitlines()))
candidates=json.loads((OUT/'candidate_rows.json').read_text())
assert len(candidates)==7
bank=list(original);accepted=[]
for row in candidates:
    if not decisions[row['index']]['accepted']:continue
    parent=next(r for r in original if r['index']==row['index'])
    assert parent['split']==row['split']=='train'
    assert all(row[k]==parent[k] for k in ('task_id','scene','start','source_sha256','instruction_original','policy_context','config_fingerprint'))
    assert checks[row['index']]['max_position_error_m']<1e-5
    actions=row['teacher_actions']
    assert len(actions)<=400 and [a['stage'] for a in actions if a['action']=='stop']==[0,1]
    assert not any(a['supervise'] for a in actions[:row['correction_at']])
    assert not any(a['supervise'] for a in actions if a['kind'] in ('prefix','perturbation','learner_prefix','teacher_view_search'))
    assert all(a['supervise'] for a in actions if a['action']=='stop')
    reviewed={**row,'index':len(bank),'parent_index':row['index'],'training_approved':True,
        'recovery_replay_verified':True,'visual_review_pending':False,'human_validated':False,
        'RGB_review':'Codex; not human annotation','visual_review_reason':decisions[row['index']]['reason'],
        'benchmark_training_ready':False,'goal_manifest_sha256':sha(PARENT/'goal_manifest_frozen.json')}
    bank.append(reviewed);accepted.append(reviewed)
assert len(accepted)==7 and bank[:len(original)]==original
assert [r for r in bank if r['split']=='dev']==[r for r in original if r['split']=='dev']
assert {r['scene'] for r in bank if r['split']=='train'}.isdisjoint(r['scene'] for r in bank if r['split']=='dev')
manifest=OUT/'model_manifest_with_recovery_frozen.jsonl'
assert not manifest.exists(),'refuse overwrite'
manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bank))
summary={'status':'reviewed recovery bank frozen','accepted_recovery_demonstrations':len(accepted),'held_parent_indices':[3],
    'unique_train_tasks':8,'train_demonstrations':15,'dev_tasks_unchanged':2,
    'recovery_supervised_actions':sum(sum(a['supervise'] for a in r['teacher_actions']) for r in accepted),
    'total_supervised_train_actions':sum(sum(a.get('supervise',True) for a in r['teacher_actions']) for r in bank if r['split']=='train'),
    'recovery_STOPs':14,'total_supervised_train_STOPs':30,'parent_rows_retained_identically':True,
    'manifest_sha256':sha(manifest),'goal_manifest_path':str(PARENT/'goal_manifest_frozen.json'),
    'goal_manifest_sha256':sha(PARENT/'goal_manifest_frozen.json'),'review_sha256':sha(OUT/'visual_review_decisions.json'),
    'protocol_sha256':sha(OUT/'protocol_frozen.json'),'human_validated':False,'training_performed':False,
    'holdout_used_for_collection':False,'scope':'reviewed train branch recovery pilot, not generalization or Table4'}
assert summary['recovery_supervised_actions']==697 and summary['total_supervised_train_actions']==2054
(OUT/'reviewed_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
