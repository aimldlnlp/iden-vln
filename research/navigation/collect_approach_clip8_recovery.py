"""Train-only recovery candidates under frozen approach-v2; review precedes training."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import train_two_reviewed_il_overfit as il
from spatial_clip_features import SpatialCLIPFeatures
from frozen_approach_resolver import install
from build_visible_approach_pilot import framed_box
from run_single_goal_route_pilot import reset
from pilot_level_balanced_route import geodesic

ROOT=il.ROOT/'runs'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
OUT=ROOT/'approach_clip8_recovery_collection_20261013_v2'
EXPECTED={'model_manifest_frozen.jsonl':'1b00762d780a70a991d9fd816f8d7fd75f7d62aea2a3caa1675d8475fe5bf83a',
          'goal_manifest_frozen.json':'f0e139942765a3c4a88d8531242cec7c239dbd39cd351249946cc5d1a0e0344d',
          'checkpoint.pt':'0f3dd7d38ab568d166ca32808897b4cb4a74cc11f7284dfff75ef7788331c8d6'}

def save(name,value):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def branches(row):
    forward=[i for i,a in enumerate(row['teacher_actions']) if a['stage']==0 and a['action']=='move_forward']
    assert len(forward)>=3
    collect=forward[len(forward)//2]+1
    holdout=forward[len(forward)//3]+1
    if collect==holdout:holdout=forward[0]+1
    assert collect!=holdout
    assert all(a['stage']==0 and a['action']!='stop' for a in row['teacher_actions'][:max(collect,holdout)])
    return {'index':row['index'],'collection_prefix_actions':collect,'collection_forced_actions':['turn_left','turn_left','move_forward'],
            'holdout_prefix_actions':holdout,'holdout_forced_actions':['turn_right','turn_right','move_forward'],
            'learner_budget':24}

def boxes_at(sim):
    ids,_=il.review.driver.probe.prior.instances(sim,sim.stage)
    obs=sim.sim.get_sensor_observations()
    return [b for b in il.review.driver.probe.prior.boxes(obs['semantic_sensor'],ids)
            if 25<=b['pixels']<=.2*obs['semantic_sensor'].size and framed_box(b['box'],512,512)]

def verify(rows):
    protocol=json.loads((OUT/'protocol_frozen.json').read_text())
    assert all(il.review.sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    install(il.evaluator,PARENT/'goal_manifest_frozen.json','visible_approach_framed_v2')
    args=il.review.driver.probe.load_legacy_module(il.review.driver.probe.ROOT).make_args()
    args.success_dis,args.max_step=1.,400
    checks=[]
    for row in rows:
        assert row['split']=='train' and not row['training_approved']
        task=json.loads(Path(row['source_task_json']).read_text())
        with contextlib.redirect_stdout(io.StringIO()):sim=il.review.driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None
        try:
            with contextlib.redirect_stdout(io.StringIO()):reset(sim,row['start'])
            errors=[];stops=[]
            for item in row['teacher_actions']:
                assert sim.stage==item['stage'] and not sim.done
                if item['action']=='stop':
                    with contextlib.redirect_stdout(io.StringIO()):geo=float(sim.get_info()['geo dis'])
                    assert geo<1 and boxes_at(sim) and item['supervise']
                    stops.append({'stage':sim.stage,'geo_m':geo})
                with contextlib.redirect_stdout(io.StringIO()):sim.actor(item['action'])
                errors.append(math.dist(sim.agent.get_state().position,item['position']))
            assert max(errors)<1e-5 and sim.done and all(sim.successes) and [s['stage'] for s in stops]==[0,1]
            assert not any(a['supervise'] for a in row['teacher_actions'][:row['correction_at']])
            checks.append({'index':row['index'],'max_position_error_m':max(errors),'stops':stops})
        finally:sim.close()
    save('independent_verification.json',{'passed':True,'candidate_count':len(rows),'verified_STOPs':2*len(rows),
        'source_hashes_unchanged':True,'records':checks,'no_dev_supervision':True,'training_performed':False})

def main():
    assert not OUT.exists(),'refuse overwrite; inspect previous run'
    for name,h in EXPECTED.items():assert il.review.sha(PARENT/name)==h,name
    bank=list(map(json.loads,(PARENT/'model_manifest_frozen.jsonl').read_text().splitlines()))
    rows=[r for r in bank if r['split']=='train'];dev=[r for r in bank if r['split']=='dev']
    assert len(rows)==8 and len(dev)==2 and {r['scene'] for r in rows}.isdisjoint(r['scene'] for r in dev)
    selected=[branches(r) for r in rows]
    assert all(b['collection_prefix_actions']!=b['holdout_prefix_actions'] for b in selected)
    OUT.mkdir()
    source_paths=[PARENT/n for n in EXPECTED]+[PARENT/'protocol.json',PARENT/'causal_features.npz',Path(__file__),
        Path(il.__file__),Path(il.TRAINER),Path(__file__).with_name('spatial_clip_features.py'),
        Path(__file__).with_name('frozen_approach_resolver.py')]+[Path(r['source_task_json']) for r in rows]
    hashes={str(p):il.review.sha(p) for p in source_paths}
    save('protocol_frozen.json',{'version':'approach_clip8_recovery_collection_v2','source_hashes':hashes,'branches':selected,
        'prior_attempt':'v1 held all candidates: compared live vector to entire cached window; v2 reads last frame without relaxing tolerance',
        'selection':'all eight reviewed train tasks, middle stage0 forward prefix, left60+forward, max24learner actions or proposed STOP',
        'holdout':'reserved different stage0 prefix and right60+forward; never collected as training in this run',
        'teacher':'frozen stage goal set, GreedyGeodesicFollower radius0.15, real-action yaw sweep, original ordered stages',
        'supervision':'prefix, injected actions, learner prefix and teacher yaw search masked; teacher continuation only',
        'scoring':'unchanged approach-v2 strict geodesic<1m; two actual actor STOPs; total horizon400',
        'proposal':'paired original BC vs BC+accepted recovery; same seed/init/model/budget; protocol for fitting must be frozen separately',
        'seed':20260930,'review_gate':'source/hash, independent actor replay, framed target view and Codex RGB/room review; not human validation',
        'dev_used':False,'policy_inputs':'original text, robot, RGB L/F/R, own relative pose/yaw, own previous action',
        'forbidden_policy_inputs':['goal coordinates','NavMesh','semantic masks','target metadata','teacher stage'],
        'training_performed':False,'scope':'teacher-prefix train recovery diagnostic; not autonomous SR, balanced100 or Table4'})
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('clip8_recovery_base',il.TRAINER)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    module=SpatialCLIPFeatures(base,device);module.check()
    parent_protocol=json.loads((PARENT/'protocol.json').read_text())
    assert module.protocol(rows)['checkpoint_files_sha256']==parent_protocol['feature_encoder']['checkpoint_files_sha256']
    checkpoint=torch.load(PARENT/'checkpoint.pt',map_location='cpu')
    assert checkpoint['protocol_sha256']==il.review.sha(PARENT/'protocol.json')
    model=module.SequenceGRU(6767).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
    mean=np.asarray(checkpoint['mean'],dtype=np.float32);std=np.asarray(checkpoint['std'],dtype=np.float32)
    cache=np.load(PARENT/'causal_features.npz')
    goals=install(il.evaluator,PARENT/'goal_manifest_frozen.json','visible_approach_framed_v2')
    args=il.review.driver.probe.load_legacy_module(il.review.driver.probe.ROOT).make_args();args.success_dis,args.max_step=1.,400
    ledger=[];candidates=[]
    for row,branch in zip(rows,selected):
        task=json.loads(Path(row['source_task_json']).read_text());assert il.review.sha(Path(row['source_task_json']))==row['source_sha256']
        with contextlib.redirect_stdout(io.StringIO()):sim=il.review.driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None;trace=[];stops=[];learner=[];failure=None;history=[];previous=-1
        def act(action,supervise,kind):
            nonlocal previous
            assert len(trace)<400 and not sim.done,'horizon_or_done'
            stage=sim.stage
            with contextlib.redirect_stdout(io.StringIO()):sim.actor(action)
            trace.append({'action':action,'stage':stage,'position':list(map(float,sim.agent.get_state().position)),
                          'supervise':supervise,'kind':kind})
            previous=module.ACTION_TO_IDX[action]
        try:
            with contextlib.redirect_stdout(io.StringIO()):reset(sim,row['start'])
            yaw=il.signed_yaw(sim.agent.get_state().rotation);start=list(map(float,sim.agent.get_state().position))
            windows=cache['x'][cache['tids']==row['index']]
            assert windows.ndim==3 and windows.shape[1:]==(16,6767)
            cached=windows[:,-1,:];at=branch['collection_prefix_actions']
            assert len(cached)==len(row['teacher_actions'])
            for t,item in enumerate(row['teacher_actions'][:at]):
                history.append(cached[t].tolist());act(item['action'],False,'prefix')
                assert math.dist(sim.agent.get_state().position,item['position'])<1e-5
            first=il.feature(module,sim,start,yaw,previous,il.context(row))
            error=float(np.max(np.abs(np.asarray(first)-cached[at])));assert error<1e-5,('prefix_feature_mismatch',error)
            before=list(map(float,sim.agent.get_state().position))
            for action in branch['collection_forced_actions']:
                history.append(il.feature(module,sim,start,yaw,previous,il.context(row)));act(action,False,'perturbation')
            displacement=math.dist(before,sim.agent.get_state().position)
            save(f'{row["index"]:02d}_branch.json',{'prefix_feature_error':error,'displacement_m':displacement,
                'position':list(map(float,sim.agent.get_state().position))})
            for step in range(branch['learner_budget']):
                history.append(il.feature(module,sim,start,yaw,previous,il.context(row)))
                array=(np.asarray(il.window(history),dtype=np.float32)-mean)/std
                with torch.inference_mode():prediction=int(model(torch.tensor(array[None],device=device)).argmax(-1))
                action=module.IDX_TO_ACTION[prediction]
                learner.append(action)
                if action=='stop':break  # Proposed STOP is not executed; teacher still corrects active stage0.
                act(action,False,'learner_prefix')
            correction_at=len(trace)
            for stage in (0,1):
                assert sim.stage==stage
                waypoint=min(goals(sim,stage),key=lambda p:geodesic(sim.pathfinder,sim.agent.get_state().position,p))
                follower=il.review.driver.habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder,sim.agent,goal_radius=.15,
                    stop_key='stop',forward_key='move_forward',left_key='turn_left',right_key='turn_right')
                while True:
                    action=follower.next_action_along(waypoint)
                    if action=='stop':break
                    act(action,True,'recovery_motion')
                views=[]
                for turn in range(12):
                    boxes=boxes_at(sim)
                    if boxes:views.append((min(abs(b['pixels']-.05*512*512) for b in boxes),turn))
                    act('turn_right',False,'teacher_view_search')
                assert views,'no_framed_recovery_STOP_view'
                turn=min(views)[1]
                for _ in range(min(turn,12-turn)):act('turn_right' if turn<=6 else 'turn_left',True,'recovery_alignment')
                with contextlib.redirect_stdout(io.StringIO()):geo=float(sim.get_info()['geo dis'])
                boxes=boxes_at(sim);assert geo<1 and boxes,'recovery_STOP_gate'
                obs=sim.sim.get_sensor_observations()
                for code in ('l','f','r'):
                    Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3]).save(OUT/f'{row["index"]:02d}_{stage}_STOP_{code}.png')
                stops.append({'stage':stage,'geo_m':geo,'boxes':boxes})
                act('stop',True,'recovery_STOP');assert sim.successes[stage]
            assert sim.done and all(sim.successes)
            candidates.append({**row,'teacher_actions':trace,'correction_at':correction_at,'recovery_supervision_mask_required':True,
                'recovery_replay_verified':False,'training_approved':False,'visual_review_pending':True,
                'episode_id':row['task_id']+':clip8_recovery_v2','augmentation':'clip8_recovery_v2','parent_index':row['index']})
        except (AssertionError,RuntimeError,ValueError) as exc:failure=type(exc).__name__+': '+str(exc)
        finally:sim.close()
        entry={'index':row['index'],'source_index':row['source_index'],'failure':failure,'actions':len(trace),
            'supervised_actions':sum(a['supervise'] for a in trace),'learner_actions':learner,'stops':stops}
        ledger.append(entry);save('candidate_ledger.json',ledger);save('candidate_rows.json',candidates)
        print(json.dumps(entry),flush=True)
    verify(candidates)
    save('collection_summary.json',{'selected':8,'candidates_replayed':len(candidates),'held_indices':[r['index'] for r in ledger if r['failure']],
        'RGB_review_pending':True,'training_performed':False,'holdout_executed':False,'dev_used':False})

if __name__=='__main__':main()
