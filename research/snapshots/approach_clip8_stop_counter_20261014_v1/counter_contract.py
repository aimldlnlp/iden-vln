"""Freeze counter ablation and verify its paired outcomes without changing parents."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/aimldl/LH-VLN/runs')
BASE=ROOT/'approach_clip8_human_recovery_lr_drop_20261014_v1'
OUT=HERE/'experiment'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    assert not p.exists(),'refuse overwrite'
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
    assert not OUT.exists() and not (HERE/'contract_frozen.json').exists()
    assert read(BASE/'experiment/final_verification.json')['passed']
    old=read(BASE/'contract_frozen.json')
    parents={**old['original_dependencies'],**old['dependencies']}
    assert all(sha(Path(p))==h for p,h in parents.items()),'parent changed'
    prep=read(HERE/'preparation.json')
    assert all(sha(HERE/n)==h for n,h in prep['generated_scripts_sha256'].items())
    collection=ROOT/'approach_clip8_recovery_collection_20261013_v2'
    cache=ROOT/'approach_clip8_human_recovery_cache_20261014_v1'
    paths=list(HERE.glob('*.py'))+[HERE/'launch.sh',HERE/'preparation.json',
        collection/'model_manifest_human_reviewed_frozen.jsonl',cache/'recovery_sample_refs.json']
    paths += [BASE/'experiment'/n for n in ['protocol_frozen.json','evaluation_protocol_frozen.json','final_verification.json','training_summary.json','BC_recovery.pt']]
    save(HERE/'contract_frozen.json',{'version':HERE.name,'created_utc':time.time(),
        'question':'Does persistent own-STOP count improve second-stage transitions without harming first-stage competence?',
        'arms':{'counter_zero':{'samples':2054,'counter_enabled':False},'counter_active':{'samples':2054,'counter_enabled':True}},
        'intervention':'one appended scalar per history frame: zero control versus count of STOP actions executed before that frame',
        'input_dim':6768,'counter_mean':0.,'counter_std':1.,'window':16,
        'initializer':'same archived seeded6767 weights mapped to6768; added input column zero; shared between arms',
        'updates':3600,'seed':20260930,'batch_size':128,'lr_schedule':{'switch_after_updates':1200,'initial':.0012,'after':.00012},
        'unchanged':['2054 examples and labels','legacy6767 features and normalization','AdamW and weight decay','sampling draws','loss weights','original text','goal sets','scoring','horizon400','all18 cases per arm'],
        'checkpoint_selection':'final3600 only; no dev selection',
        'policy_counter_source':'own executed STOP actions, including invalid STOPs; no sim.stage or successful-stage feedback',
        'primary':'paired original second-stage valid STOP and chain success; retention of stage0; perturbations and dev separate',
        'guard':'do not prohibit consecutive STOPs; all failures retained; fresh actor replay36rollouts',
        'scope':'BC+recovery only, one seed, development perturbations; not balanced100 or Table4',
        'original_dependencies':parents,'dependencies':{str(p):sha(p) for p in paths}})
    print('frozen',sha(HERE/'contract_frozen.json'),flush=True)


def finish():
    c=read(HERE/'contract_frozen.json')
    assert all(sha(Path(p))==h for p,h in {**c['original_dependencies'],**c['dependencies']}.items())
    protocol=read(OUT/'protocol_frozen.json');train=read(OUT/'training_summary.json')
    assert protocol['arms']=={'counter_zero':2054,'counter_active':2054}
    assert protocol['lr_schedule']==c['lr_schedule']
    assert len({a['initial_state_sha256'] for a in train.values()})==1
    assert len({a['uniform_draws_sha256'] for a in train.values()})==1
    assert all(a['samples']==2054 and a['updates']==3600 for a in train.values())
    assert read(OUT/'counter_cache_verification.json')['passed']
    fitting=read(OUT/'fitting_verification.json')
    assert fitting['same_training_samples'] and fitting['counter_input_is_only_arm_difference']
    old_eval=read(BASE/'experiment/evaluation_protocol_frozen.json');new_eval=read(OUT/'evaluation_protocol_frozen.json')
    assert old_eval['cases']==new_eval['cases']
    for key in ['scoring','scope','dev_scope']:assert old_eval[key]==new_eval[key]
    result=read(OUT/'final_verification.json');assert result['passed'] and result['rollouts_verified']==36
    records=read(OUT/'evaluation_rollouts.json');boundary={}
    for cohort in result['metrics']:
        boundary[cohort]={}
        for arm in train:
            group=[r for r in records if r['cohort']==cohort and r['arm']==arm]
            immediate=[];entered=0;valid=0;stage0=0
            for row in group:
                count=0
                for event in row['actions']:
                    assert event['own_STOP_count_before']==count
                    assert event['counter_feature']==(float(count) if arm=='counter_active' else 0.)
                    count+=int(event['action']=='stop')
                events=[a for a in row['actions'] if a['stage']==1]
                entered+=bool(events)
                if events and events[0]['action']=='stop':immediate.append({'index':row['index'],'valid':events[0]['pre_geo_m'] is not None and events[0]['pre_geo_m']<1})
                valid+=sum(s['valid'] for s in row['stops'] if s['stage']==1)
                stage0+=sum(s['valid'] for s in row['stops'] if s['stage']==0)
            boundary[cohort][arm]={'episodes':len(group),'stage0_valid_STOPs':stage0,'stage1_entered':entered,
                'stage1_valid_STOPs':valid,'immediate_stage1_STOPs':immediate,
                'immediate_invalid_stage1_STOPs':sum(not i['valid'] for i in immediate)}
    for arm in train:
        curve=read(OUT/(arm+'_progress.json'))
        assert len(curve)==36 and all(v['lr']==(.0012 if v['step']<=1200 else .00012) for v in curve)
        assert [v['step'] for v in read(OUT/(arm+'_fixed_set_diagnostics.json'))]==[1200,2400,3600]
    save(HERE/'counter_comparison.json',{'verified':True,'contract_sha256':sha(HERE/'contract_frozen.json'),
        'metrics':result['metrics'],'stage_transition_metrics':boundary,'paired_same_cases_scoring_and_samples':True,
        'baseline_sources_unchanged':True,'archived_LR_recovery_context_only':read(BASE/'experiment/final_verification.json')['metrics'],
        'scope':'one-seed development ablation; no population or generalization claim'})
    print(json.dumps({'verified':True,'metrics':result['metrics'],'boundary':boundary}),flush=True)


if __name__=='__main__':
    assert sys.argv[1:] in (['--freeze'],['--finish'])
    freeze() if sys.argv[1]=='--freeze' else finish()
