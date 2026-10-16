"""Matched original starts/dev and reserved train perturbations; actual actor and replay."""
import contextlib,importlib.util,io,json,math
from pathlib import Path
import numpy as np
import torch
import train_two_reviewed_il_overfit as il
from spatial_clip_features import SpatialCLIPFeatures
from frozen_approach_resolver import install
from run_single_goal_route_pilot import reset

ROOT=il.ROOT/'runs';OUT=ROOT/'approach_clip8_human_recovery_lr_drop_20261014_v1/experiment'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
SOURCE=ROOT/'approach_clip8_recovery_collection_20261013_v2'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
    assert not (OUT/'evaluation_protocol_frozen.json').exists(),'refuse duplicate evaluation'
    fitting=json.loads((OUT/'fitting_verification.json').read_text());assert fitting['passed']
    bank=list(map(json.loads,(PARENT/'model_manifest_frozen.jsonl').read_text().splitlines()))
    branches={b['index']:b for b in json.loads((SOURCE/'protocol_frozen.json').read_text())['branches']}
    assert len(bank)==10 and len(branches)==8
    cases=[{'index':r['index'],'cohort':'train_original' if r['split']=='train' else 'dev_original','prefix':0,'forced':[]} for r in bank]
    cases += [{'index':i,'cohort':'train_holdout','prefix':b['holdout_prefix_actions'],'forced':b['holdout_forced_actions']} for i,b in branches.items()]
    assert len(cases)==18 and all(c['prefix']!=branches[c['index']]['collection_prefix_actions'] for c in cases if c['cohort']=='train_holdout')
    paths=[PARENT/'model_manifest_frozen.jsonl',PARENT/'goal_manifest_frozen.json',PARENT/'causal_features.npz',SOURCE/'protocol_frozen.json',OUT/'protocol_frozen.json',Path(__file__)]+[OUT/(a+'.pt') for a in fitting['arms']]
    hashes={str(p):il.review.sha(p) for p in paths}
    save('evaluation_protocol_frozen.json',{'cases':cases,'arms':list(fitting['arms']),'expected_rollouts':36,'source_hashes':hashes,
        'scoring':'two actual actor STOPs; strict geodesic<1m; horizon400 includes teacher prefix and perturbation',
        'primary':'paired train_holdout success; train_original retention and dev_original separate',
        'scope':'train_holdout is teacher-prefix robustness on train scenes, not autonomous unseen-scene SR',
        'dev_scope':'two accepted development tasks; full candidate cohort5 with3held; no test claim'})
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    spec=importlib.util.spec_from_file_location('recovery_eval_base',il.TRAINER);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    module=SpatialCLIPFeatures(base,device)
    assert module.protocol(bank)['checkpoint_files_sha256']==json.loads((PARENT/'protocol.json').read_text())['feature_encoder']['checkpoint_files_sha256']
    policies={}
    for arm in fitting['arms']:
        cp=torch.load(OUT/(arm+'.pt'),map_location='cpu');assert cp['protocol_sha256']==il.review.sha(OUT/'protocol_frozen.json')
        model=module.SequenceGRU(6767).to(device);model.load_state_dict(cp['state_dict']);model.eval()
        policies[arm]=(model,np.asarray(cp['mean'],dtype=np.float32),np.asarray(cp['std'],dtype=np.float32))
    cache=np.load(PARENT/'causal_features.npz')
    install(il.evaluator,PARENT/'goal_manifest_frozen.json','visible_approach_framed_v2')
    args=il.review.driver.probe.load_legacy_module(il.review.driver.probe.ROOT).make_args();args.success_dis,args.max_step=1.,400
    records=[];replay_checks=[]
    for case in cases:
        row=next(r for r in bank if r['index']==case['index']);task=json.loads(Path(row['source_task_json']).read_text())
        assert il.review.sha(Path(row['source_task_json']))==row['source_sha256']
        for arm,(model,mean,std) in policies.items():
            with contextlib.redirect_stdout(io.StringIO()):sim=il.review.driver.probe.SceneSimulator(args=args,config=task)
            sim.save_path=None;trace=[];stops=[];history=[];previous=-1;prefix_error=None
            try:
                with contextlib.redirect_stdout(io.StringIO()):reset(sim,row['start'])
                start=list(map(float,sim.agent.get_state().position));yaw=il.signed_yaw(sim.agent.get_state().rotation)
                def act(action,kind):
                    nonlocal previous
                    stage=sim.stage
                    with contextlib.redirect_stdout(io.StringIO()):geo=float(sim.get_info()['geo dis']);sim.actor(action)
                    trace.append({'action':action,'stage':stage,'position':list(map(float,sim.agent.get_state().position)),
                        'pre_geo_m':geo if math.isfinite(geo) else None,'kind':kind})
                    previous=module.ACTION_TO_IDX[action]
                    if action=='stop':stops.append({'stage':stage,'geo_m':geo if math.isfinite(geo) else None,'valid':bool(sim.successes[stage])})
                if case['prefix']:
                    cached=cache['x'][cache['tids']==row['index']]
                    for t,item in enumerate(row['teacher_actions'][:case['prefix']]):
                        assert item['action']!='stop' and item['stage']==0
                        history.append(cached[t,-1].tolist());act(item['action'],'teacher_prefix')
                        assert math.dist(sim.agent.get_state().position,item['position'])<1e-5
                    feature=il.feature(module,sim,start,yaw,previous,il.context(row))
                    prefix_error=float(np.max(np.abs(np.asarray(feature)-cached[case['prefix'],-1])));assert prefix_error<1e-5
                for action in case['forced']:
                    history.append(il.feature(module,sim,start,yaw,previous,il.context(row)));act(action,'holdout_perturbation')
                branch_position=list(map(float,sim.agent.get_state().position))
                while len(trace)<400 and not sim.done:
                    history.append(il.feature(module,sim,start,yaw,previous,il.context(row)))
                    array=(np.asarray(il.window(history),dtype=np.float32)-mean)/std
                    with torch.inference_mode():action=module.IDX_TO_ACTION[int(model(torch.tensor(array[None],device=device)).argmax(-1))]
                    act(action,'policy')
                record={'index':row['index'],'source_index':row['source_index'],'cohort':case['cohort'],'arm':arm,'stops':stops,
                    'steps':len(trace),'success':len(stops)==2 and all(s['valid'] for s in stops),
                    'stage_success':sum(s['valid'] for s in stops),'branch_position':branch_position,
                    'prefix_feature_error':prefix_error,'actions':trace}
            finally:sim.close()
            # Fresh actor replay covers prefixes, injected actions and every policy decision.
            with contextlib.redirect_stdout(io.StringIO()):sim=il.review.driver.probe.SceneSimulator(args=args,config=task);reset(sim,row['start'])
            sim.save_path=None;errors=[];actual_stops=[]
            try:
                for item in trace:
                    assert sim.stage==item['stage'] and not sim.done
                    with contextlib.redirect_stdout(io.StringIO()):geo=float(sim.get_info()['geo dis']);sim.actor(item['action'])
                    if item['pre_geo_m'] is not None:assert abs(geo-item['pre_geo_m'])<1e-5
                    errors.append(math.dist(sim.agent.get_state().position,item['position']))
                    if item['action']=='stop':actual_stops.append(bool(sim.successes[item['stage']]))
                assert max(errors)<1e-5 and actual_stops==[s['valid'] for s in stops]
                assert record['success']==(len(actual_stops)==2 and all(actual_stops))
            finally:sim.close()
            records.append(record);replay_checks.append({'index':row['index'],'cohort':case['cohort'],'arm':arm,'position_error_m':max(errors)})
            save('evaluation_rollouts.json',records);save('status.json',{'phase':'evaluation','completed':len(records),'requested':36})
            print(json.dumps({k:v for k,v in record.items() if k!='actions'}),flush=True)
    assert len(records)==36 and len({(r['cohort'],r['arm'],r['index']) for r in records})==36
    metrics={}
    for cohort,n in (('train_original',8),('train_holdout',8),('dev_original',2)):
        metrics[cohort]={}
        for arm in policies:
            rs=[r for r in records if r['cohort']==cohort and r['arm']==arm];assert len(rs)==n
            metrics[cohort][arm]={'episodes':n,'chain_success':sum(r['success'] for r in rs),'stage_success':sum(r['stage_success'] for r in rs),
                'SR':sum(r['success'] for r in rs)/n,'ISR':sum(r['stage_success'] for r in rs)/(2*n)}
    for c in cases:
        pair=[r for r in records if r['cohort']==c['cohort'] and r['index']==c['index']]
        assert len(pair)==2 and math.dist(pair[0]['branch_position'],pair[1]['branch_position'])<1e-5
    assert all(il.review.sha(Path(p))==h for p,h in hashes.items())
    save('evaluation_summary.json',metrics)
    save('independent_evaluation_verification.json',{'passed':True,'replayed_rollouts':36,'max_position_error_m':max(r['position_error_m'] for r in replay_checks),
        'same_branch_positions':True,'sources_unchanged':True,'records':replay_checks,'scope':'fresh actor replay and arithmetic; no unseen-test claim'})
    save('status.json',{'phase':'completed','rollouts':36,'verification_passed':True})
if __name__=='__main__':main()
