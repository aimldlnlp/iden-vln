"""Versioned feature-contract preflight and actual-actor teacher feasibility audit."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/home/aimldl/LH-VLN')
RUNS = ROOT/'runs'
OUT = RUNS/'balanced100_teacher_contract_20261013_v1'
BANK = RUNS/'balanced100_eval_20261013_v1/bank_frozen.json'
REPAIR = RUNS/'pipeline_repair_20261013_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def smoke_indices(bank):
    return [i for level in range(2, 6) for i in [j for j, row in enumerate(bank) if row['level']==level][:2]]


def self_check():
    bank = [{'level':k} for k in range(2,6) for _ in range(25)]
    assert smoke_indices(bank)==[0,1,25,26,50,51,75,76]
    assert len(smoke_indices(bank))==8


def main(phase):
    import numpy as np
    import torch
    import habitat_sim
    self_check()
    torch.set_num_threads(2)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(REPAIR))
    ordered_path = RUNS/'idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py'
    ordered = load(ordered_path, 'teacher_contract_ordered')
    legacy = ordered.load_legacy_module(ROOT)
    ordered.install_index_aware_target_patch()
    sequence = load(REPAIR/'sequence_trainer_fixed.py', 'teacher_contract_sequence')
    causal = load(REPAIR/'causal_collector.py', 'teacher_contract_causal')
    contract = load(REPAIR/'repair_contract.py', 'teacher_contract_helpers')
    from habitat_base.simulation import SceneSimulator
    bank = json.loads(BANK.read_text())
    assert len(bank)==100 and all(sum(r['level']==k for r in bank)==25 for k in range(2,6))
    assert sha(BANK)=='9f74e219b74adc3b9807bbde230092415b0b2f5c8fe7e6149162874d34dfeec2'
    checkpoint = REPAIR/'experiment/training/sequence_repaired_gate.pt'
    paths = [Path(__file__), BANK, ordered_path, REPAIR/'sequence_trainer_fixed.py', REPAIR/'causal_collector.py', REPAIR/'repair_contract.py', REPAIR/'experiment/protocol_frozen.json', checkpoint]
    dependencies = {str(p):sha(p) for p in paths}
    args = legacy.make_args()
    args.episode_data='source_gt_metadata'
    args.max_step=150
    args.success_dis=1.
    def context(config):
        return {'instruction':config['Task instruction'], 'objects':legacy.object_labels(config['Object']), 'regions':legacy.normalize_regions(config['Region Name']), 'robot':config['Robot'].lower(), 'level':config['Generated level'], 'scene':config['Scene']}
    def simulator(config, start):
        sim = SceneSimulator(args=args, config=config)
        ordered.reset_simulator(sim, start)
        actual=ordered.capture_state(sim)
        assert np.allclose(actual['position'],start['position'],atol=1e-5)
        assert abs(abs(np.dot(actual['rotation'],start['rotation']))-1)<1e-5
        return sim
    null = open(os.devnull,'w')
    if phase=='preflight':
        OUT.mkdir(exist_ok=False)
        save('protocol_frozen.json', {'version':OUT.name, 'started_utc':time.time(), 'dependencies':dependencies, 'bank_sha256':sha(BANK), 'smoke_indices':smoke_indices(bank), 'selection':'first two indices per level in frozen bank, before outcomes', 'horizon':150, 'strict_STOP_m':1., 'resolver':'unchanged historical index-aware snapped AABB', 'teacher':'GreedyGeodesicFollower, goal_radius=0.8m, closest reachable snapped target at each stage activation; fixed goal within stage', 'actor':'all actions including scored STOP use sim.actor; initial observation STOP unscored', 'feature_gate':'cache-source <=1e-6; same-observation disk/live and evaluator feature <=1e-6; prefix position <=1e-4; circular yaw <=1e-5; raw live-cache feature <=1e-4', 'scope':'diagnostic teacher, privileged target/pose; not learned SR or proof of impossibility when follower fails', 'normalization':'repair v3 checkpoint has no external mean/std normalization', 'training_executed':False})
        save('status.json', {'phase':'feature_preflight'})
        xs,ys,tids,meta=causal.collect_causal_sequences(sequence)
        cohort=json.loads((REPAIR/'experiment/protocol_frozen.json').read_text())['cohort']['train']
        blob=torch.load(checkpoint,map_location='cpu')
        model=sequence.SequenceGRU(175).eval()
        model.load_state_dict(blob['model_state_dict'])
        assert blob['input_dim']==175 and blob['window']==8 and not model.training
        rows=[]
        for row in cohort:
            trial=Path(row['trial']);c=sequence.load_context(trial)
            start={'position':c['pos'][0], 'rotation':[math.cos(c['yaw'][0]/2),0,math.sin(c['yaw'][0]/2),0]}
            indices=[i for i,tid in enumerate(tids) if tid==row['tid']]
            sim=None
            try:
                with contextlib.redirect_stdout(null):
                    sim=simulator(c['task'],start)
                    obs,_,_=sim.actor('stop')
                previous=0;history=[];checks=[]
                for offset,i in enumerate(indices):
                    state=sim.agent.get_state();pose=list(state.position);yaw=contract.signed_yaw(state.rotation)
                    frame,source_pose,source_yaw=contract.before_action(trial,offset+1,c['pos'],c['yaw'])
                    fresh=sequence.frame_feature(frame,source_pose,source_yaw,previous,c)
                    live=sequence.live_feature(obs,pose,yaw,previous,c)
                    evaluator=legacy.seq_feature(obs,pose,yaw,previous,c)
                    snapshot=OUT/'runtime_frame'
                    snapshot.mkdir(exist_ok=True)
                    assert len(obs)>=3
                    for camera,img in zip(('left','front','right'),obs):img.save(snapshot/(camera+'.png'))
                    disk=sequence.frame_feature(snapshot,pose,yaw,previous,c)
                    history.append(live)
                    window=[[0.]*175]*max(0,8-len(history))+history[-8:]
                    with torch.no_grad():
                        a=model(torch.tensor([window],dtype=torch.float32))
                        b=model(torch.tensor([xs[i]],dtype=torch.float32))
                    checks.append({'offset':offset,'cache_source_error':float(np.max(np.abs(np.array(fresh)-xs[i][-1]))), 'disk_live_error':float(np.max(np.abs(np.array(disk)-live))), 'evaluator_live_error':float(np.max(np.abs(np.array(evaluator)-live))), 'position_error':float(np.linalg.norm(np.array(pose)-source_pose)), 'circular_yaw_error':abs(math.atan2(math.sin(yaw-source_yaw),math.cos(yaw-source_yaw))), 'raw_live_cache_error':float(np.max(np.abs(np.array(live)-xs[i][-1]))), 'window_error':float(np.max(np.abs(np.array(window)-xs[i]))), 'logit_error':float((a-b).abs().max()), 'cache_pred':int(b.argmax()), 'live_pred':int(a.argmax()), 'teacher_label':ys[i]})
                    with contextlib.redirect_stdout(null):obs,_,_=sim.actor(sequence.IDX_TO_ACTION[ys[i]])
                    previous=ys[i]
                rows.append({'tid':row['tid'],'index':row['index'],'checks':checks,'successes':list(map(bool,sim.successes))})
            finally:
                if sim is not None:sim.sim.close()
        flat=[c for r in rows for c in r['checks']]
        maxima={k:max(c[k] for c in flat) for k in flat[0] if k.endswith('error')}
        passed=all(maxima[k]<=tol for k,tol in [('cache_source_error',1e-6),('disk_live_error',1e-6),('evaluator_live_error',1e-6),('position_error',1e-4),('circular_yaw_error',1e-5),('raw_live_cache_error',1e-4)])
        save('feature_verification.json', {'passed':passed,'samples':len(flat),'tasks':len(rows),'maxima':maxima,'predictions_match':all(c['cache_pred']==c['live_pred'] for c in flat),'rows':rows,'cache':meta})
        save('status.json', {'phase':'preflight_passed' if passed else 'preflight_failed','feature_gate_passed':passed})
        print('Feature gate',passed,maxima,flush=True)
        return 0 if passed else 2
    protocol=json.loads((OUT/'protocol_frozen.json').read_text())
    assert all(sha(Path(p))==h for p,h in protocol['dependencies'].items())
    assert json.loads((OUT/'feature_verification.json').read_text())['passed']
    if phase=='full':assert json.loads((OUT/'smoke/verification.json').read_text())['passed']
    folder=OUT/phase
    folder.mkdir(exist_ok=False)
    selected=protocol['smoke_indices'] if phase=='smoke' else list(range(100))
    results=[]
    with (folder/'traces.jsonl').open('w') as handle:
        for index in selected:
            row=bank[index]
            assert sha(Path(row['source']))==row['source_sha256']
            result={'index':index,'task_id':row['task_id'],'level':row['level'],'scene':row['scene'],'steps':[],'stages':[],'runtime_ok':True,'failure':None}
            sim=None
            try:
                with contextlib.redirect_stdout(null):
                    sim=simulator(row['config'],row['start_state'])
                    goals=[ordered.target_coords_for_index(sim,k) for k in range(row['level'])]
                    ref,_=ordered.reference_length(sim,row['start_state']['position'],goals)
                    result['reference_finite']=ordered.finite(ref)
                    sim.actor('stop')
                last_stage=None;goal=None;follower=None
                for step in range(150):
                    if sim.episode_over:break
                    stage=int(sim.stage)
                    if stage!=last_stage:
                        distance,goal=ordered.geodesic_between(sim.pathfinder,sim.agent.get_state().position,goals[stage])
                        result['stages'].append({'stage':stage,'start_distance_m':distance if math.isfinite(distance) else None,'goal':list(map(float,goal)) if goal is not None else None})
                        if goal is None:
                            result['failure']='unreachable_active_target';break
                        follower=habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder,sim.agent,goal_radius=.8,stop_key='stop',forward_key='move_forward',left_key='turn_left',right_key='turn_right')
                        last_stage=stage
                    try:action=follower.next_action_along(goal)
                    except habitat_sim.errors.GreedyFollowerError:
                        result['failure']='follower_failure';break
                    assert action in legacy.ACTION_TO_IDX
                    before=list(map(float,sim.agent.get_state().position))
                    distance,_=ordered.geodesic_between(sim.pathfinder,before,goals[stage])
                    with contextlib.redirect_stdout(null):sim.actor(action)
                    after=list(map(float,sim.agent.get_state().position))
                    result['steps'].append({'t':step,'stage':stage,'action':action,'position_before':before,'position_after':after,'distance_before_m':distance if math.isfinite(distance) else None,'success_after':bool(sim.successes[stage])})
                    if action=='stop':
                        expected=distance<1.
                        assert bool(sim.successes[stage])==expected
                        result['stages'][-1].update({'STOP_step':step,'STOP_distance_m':distance,'STOP_valid':expected})
                        if not expected:result['failure']='invalid_STOP';break
                result['successes']=list(map(bool,sim.successes))
                assert len(result['successes'])==row['level']
                result['full_chain_success']=all(result['successes'])
                if not result['full_chain_success'] and result['failure'] is None:result['failure']='horizon_exhausted'
            except Exception as exc:
                result.update(runtime_ok=False,failure='runtime_contract_error',error=repr(exc),traceback=traceback.format_exc())
                result.setdefault('successes',[False]*row['level'])
                result.setdefault('full_chain_success',False)
            finally:
                if sim is not None:sim.sim.close()
            handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            results.append(result)
            save('status.json',{'phase':phase,'completed':len(results),'requested':len(selected),'last_index':index,'last_failure':result['failure']})
            print(phase,index,result['successes'],len(result['steps']),result['failure'],flush=True)
    verification={'passed':len(results)==len(selected) and all(r['runtime_ok'] for r in results), 'episodes':len(results),'chains':sum(r['full_chain_success'] for r in results),'stages':sum(sum(r['successes']) for r in results),'stage_count':sum(r['level'] for r in results),'by_level':{str(k):{'episodes':sum(r['level']==k for r in results),'chains':sum(r['full_chain_success'] for r in results if r['level']==k)} for k in range(2,6)},'failures':{f:sum(r['failure']==f for r in results) for f in sorted({r['failure'] for r in results if r['failure']})},'dependencies_unchanged':all(sha(Path(p))==h for p,h in dependencies.items())}
    assert verification['dependencies_unchanged']
    save(phase+'/verification.json',verification)
    save('status.json',{'phase':phase+'_completed','verification':verification,'finished_utc':time.time()})
    print('Verification',verification,flush=True)
    return 0 if verification['passed'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['preflight','smoke','full','self-check'])
    phase=parser.parse_args().phase
    if phase=='self-check':self_check();print('self-check passed')
    else:
        try:raise SystemExit(main(phase))
        except Exception as exc:
            if OUT.exists():save('failure.json',{'phase':phase,'error':repr(exc),'traceback':traceback.format_exc()})
            raise
