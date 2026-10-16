"""Freeze budget intervention, verify matched inputs and compare final results."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/aimldl/LH-VLN/runs')
OLD=ROOT/'approach_clip8_human_recovery_pair_20261014_v1'
OUT=HERE/'experiment'
SCRIPTS=ROOT/'idenvln_ordered_eval_20260804_234338'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    assert not path.exists(),'refuse overwrite'
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
    assert not OUT.exists() and not (HERE/'contract_frozen.json').exists()
    old=read(OLD/'protocol_frozen.json')
    assert old['updates']==1200 and read(OLD/'final_verification.json')['passed']
    paths=[OLD/'protocol_frozen.json',OLD/'training_summary.json',OLD/'final_verification.json',
        OLD/'evaluation_protocol_frozen.json',OLD/'BC.pt',OLD/'BC_recovery.pt',
        ROOT/'human_recovery_regression_research_20261014_v1/diagnostic_results.json']
    original_hashes={**old['source_hashes'],**read(OLD/'evaluation_protocol_frozen.json')['source_hashes']}
    assert all(sha(Path(p))==h for p,h in original_hashes.items()),'baseline dependency changed'
    prep=read(HERE/'preparation.json')
    assert all(sha(HERE/n)==h for n,h in prep['generated_scripts_sha256'].items())
    paths += list(HERE.glob('*.py'))+[HERE/'launch.sh',HERE/'preparation.json']
    paths += [SCRIPTS/'train_two_reviewed_il_overfit.py',SCRIPTS/'spatial_clip_features.py',SCRIPTS/'frozen_approach_resolver.py']
    save(HERE/'contract_frozen.json',{'version':HERE.name,'created_utc':time.time(),
        'question':'Does increasing optimization budget improve original competence and recovery under the same training package?',
        'intervention':'1200 archived vs3600 new updates; fresh common initializer; no warm start',
        'baseline':str(OLD),'new_output':str(OUT),'updates':3600,'seed':20260930,'batch_size':128,
        'checkpoint_selection':'final3600 only; no outcome-based early stop or dev selection',
        'unchanged':['data','model','initialization','sampling rule','loss weights','normalization','optimizer','goal sets','horizon400','strict STOP<1m','evaluation cases'],
        'initializer_and_sampling_prefix':'assert initializer state hash and first1200 uniform draw hash match archived run',
        'completion':'both arms trained;36actor rollouts with fresh replay;final verifier;common original/recovery inference;all failures retained',
        'scope':'one-seed development screening; train8/8 is operational competence target, not a single-label accuracy gate or generalization proof',
        'branches':'same previously inspected development perturbations; not untouched test',
        'original_dependencies':original_hashes,'dependencies':{str(p):sha(p) for p in paths}})
    print('frozen',sha(HERE/'contract_frozen.json'),flush=True)


def finish():
    contract=read(HERE/'contract_frozen.json')
    assert all(sha(Path(p))==h for p,h in {**contract['original_dependencies'],**contract['dependencies']}.items())
    old_protocol=read(OLD/'protocol_frozen.json');new_protocol=read(OUT/'protocol_frozen.json')
    for key in old_protocol:
        if key not in ['version','updates','source_hashes']:assert old_protocol[key]==new_protocol[key],key
    assert new_protocol['updates']==3600
    old_eval=read(OLD/'evaluation_protocol_frozen.json');new_eval=read(OUT/'evaluation_protocol_frozen.json')
    assert old_eval['cases']==new_eval['cases'] and old_eval['arms']==new_eval['arms']
    for key in ['scoring','primary','scope','dev_scope']:assert old_eval[key]==new_eval[key]
    old_train=read(OLD/'training_summary.json');new_train=read(OUT/'training_summary.json')
    for arm in new_train:
        assert old_train[arm]['initial_state_sha256']==new_train[arm]['initial_state_sha256']
        assert old_train[arm]['samples']==new_train[arm]['samples'] and new_train[arm]['updates']==3600
    old_final=read(OLD/'final_verification.json');new_final=read(OUT/'final_verification.json')
    assert old_final['passed'] and new_final['passed'] and new_final['rollouts_verified']==36
    diagnostic=read(OUT/'diagnosis/diagnostic_results.json')
    old_diag=read(ROOT/'human_recovery_regression_research_20261014_v1/diagnostic_results.json')
    pairs={}
    for cohort,arms in new_final['metrics'].items():
        pairs[cohort]={arm:{'1200':old_final['metrics'][cohort][arm],'3600':new,
            'delta_SR':new['SR']-old_final['metrics'][cohort][arm]['SR'],
            'delta_ISR':new['ISR']-old_final['metrics'][cohort][arm]['ISR']} for arm,new in arms.items()}
    rollouts=read(OUT/'evaluation_rollouts.json');stop_divergences=[]
    for item in diagnostic['first_divergences']:
        t=item['first_action_difference']
        if t is None:continue
        record=next(r for r in rollouts if r['cohort']=='train_original' and r['index']==item['index'] and r['arm']==item['arm'])
        event=record['actions'][t]
        if event['action']=='stop':stop_divergences.append({'index':item['index'],'arm':item['arm'],'step':t,
            'geodesic_m':event['pre_geo_m'],'STOP_valid':event['pre_geo_m'] is not None and event['pre_geo_m']<1.})
    save(OUT/'diagnosis/first_STOP_divergences.json',stop_divergences)
    save(HERE/'budget_comparison.json',{'verified':True,'rollout_comparison':pairs,
        'common_set_metrics1200':old_diag['common_set_prediction_metrics'],
        'common_set_metrics3600':diagnostic['common_set_prediction_metrics'],
        'sampling_exposure1200':old_diag['sampling_exposure'],'sampling_exposure3600':diagnostic['sampling_exposure'],
        'same_cases_and_training_contract_except_budget':True,'baseline_and_sources_unchanged':True,
        'no_population_causal_claim':True,'contract_sha256':sha(HERE/'contract_frozen.json')})
    print(json.dumps({'verified':True,'metrics3600':new_final['metrics']}),flush=True)


if __name__=='__main__':
    assert sys.argv[1:] in (['--freeze'],['--finish'])
    freeze() if sys.argv[1]=='--freeze' else finish()
