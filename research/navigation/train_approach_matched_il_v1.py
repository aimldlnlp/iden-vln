"""Two reviewed train episodes: reuse SequenceGRU, causal RGB-stat/text features, bounded overfit and controls."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import review_three_train_context as review
import goal_resolver as resolver
import run_ordered_metrics_eval as evaluator
from train_causal_sequence_bc import signed_yaw

ROOT=review.ROOT
BANK=review.OUT/'manifest_frozen.jsonl'
OUT=ROOT/'runs/two_train_reviewed_il_overfit_20260930'
EXPECTED='742a5fbe54f6edd7717798a0e63ac71d772f47e55aa2c0f28fec01594b9386e2'
TRAINER=ROOT/'runs/idenvln_gpu_hybrid_sequence_bc_level23_20260707/run_gpu_hybrid_sequence_bc.py'
WINDOW=16
STEPS=1200

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def feature(module,sim,start,start_yaw,previous,context):
    obs=sim.sim.get_sensor_observations()
    images=[Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3]) for code in ('l','f','r')]
    state=sim.agent.get_state()
    relative=np.asarray(state.position)-np.asarray(start)
    yaw=signed_yaw(state.rotation)-start_yaw
    yaw=math.atan2(math.sin(yaw),math.cos(yaw))
    value=module.live_feature(images,relative,yaw,previous,context)
    assert len(value)==getattr(module,'INPUT_DIM',175) and np.isfinite(value).all()
    assert not any(value[47:47+module.HASH_DIM]), 'gold metadata hash must be zero'
    return value

def window(history):
    return [[0.]*len(history[0])]*max(0,WINDOW-len(history))+history[-WINDOW:]

def context(row,text=None):
    base=row['policy_context']
    assert set(base)=={'instruction','robot'}
    return {'instruction':base['instruction'] if text is None else text,'robot':base['robot'],
        'objects':[],'regions':[],'level':row.get('navigation_projection_stages',base['instruction'].count('STOP'))}

def main(bank=None, out=None, expected=None, comparison_cache=None, clip=False, paired_controls=False, recovery_bank=False, new_scene_bank=False, combined_bank=False, approach_manifest=None):
    global BANK, OUT, EXPECTED
    expanded = bank is not None
    if expanded:
        assert out and expected
        BANK, OUT, EXPECTED = Path(bank), Path(out), expected
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'checkpoint.pt').exists(), 'refuse to overwrite a trained run'
    assert review.sha(BANK)==EXPECTED
    accepted=[r for r in map(json.loads,BANK.read_text().splitlines()) if r.get('navigation_projection_ready',False) or r.get('pair_ready',r.get('overfit_pilot_ready',False))]
    rows=[r for r in accepted if r['split']=='train']
    dev_rows=[r for r in accepted if r['split']=='dev']
    recovery_summary=None
    new_scene_summary=None
    combined_summary=None
    corrective_v2=EXPECTED=='ff680504a544baec3158688e36653a4c2a42a07f40c4da435d6c5b7106e5a486'
    corrective_v3=EXPECTED=='694b262915d3bba32940c6d94c1701da241bd2436bc85d8f1d4a3fb5c10a8184'
    fixed_steps=bool(approach_manifest) or combined_bank and (corrective_v2 or corrective_v3 or EXPECTED=='cf355d52948af14e4772aded764665b5abc421f2abdaa77facc9c4585e2680a8')
    assert sum((recovery_bank,new_scene_bank,combined_bank))<=1
    if combined_bank:
        assert paired_controls and clip and expanded
        assert EXPECTED in ('a504605560a353874ace33ea3eb179d6702aba4e2f02e83a0ff84217f5a197c0','cf355d52948af14e4772aded764665b5abc421f2abdaa77facc9c4585e2680a8','ff680504a544baec3158688e36653a4c2a42a07f40c4da435d6c5b7106e5a486','694b262915d3bba32940c6d94c1701da241bd2436bc85d8f1d4a3fb5c10a8184')
        combined_summary=json.loads((BANK.parent/'frozen_summary.json').read_text())
        verified=json.loads((BANK.parent/'verification.json').read_text())
        assert combined_summary['manifest_sha256']==verified['manifest_sha256']==EXPECTED
        assert verified['status']=='passed'
        if corrective_v2 or corrective_v3:
            bank_protocol=json.loads((BANK.parent/'protocol.json').read_text())
            assert bank_protocol['version']==('combined_learner_corrective_v3' if corrective_v3 else 'combined_early_corrective_v2') and bank_protocol['dev_unchanged']
            assert verified['source_rows_retained_identically'] and verified['dev_rows_identical']
            correction_verification=ROOT/'runs'/('reviewed_learner_corrective_v3_20261010' if corrective_v3 else 'reviewed_early_corrective_v2_20261010')/'verification.json'
            assert review.sha(correction_verification)==verified['new_demonstrations_verification_sha256']
            assert json.loads(correction_verification.read_text())['complete_demonstrations_replayed']==(4 if corrective_v3 else 5)
            corrective_count=10 if corrective_v3 else 6
            if corrective_v3:
                parent=ROOT/'runs/combined_early_corrective_bank_v2_20261010/model_manifest_frozen.jsonl'
                assert review.sha(parent)==bank_protocol['parent_manifest_sha256']=='ff680504a544baec3158688e36653a4c2a42a07f40c4da435d6c5b7106e5a486'
                assert accepted[:22]==list(map(json.loads,parent.read_text().splitlines()))
        else:
            assert verified['source_task_hashes_verified']
            assert combined_summary['dev_unchanged'] and combined_summary['held_parent_indices']==[0]
            corrective_count=combined_summary['corrective_demonstrations']
            assert corrective_count==(0 if EXPECTED=='cf355d52948af14e4772aded764665b5abc421f2abdaa77facc9c4585e2680a8' else 3)
        assert len({r['task_id'] for r in rows})==9 and len({r['scene'] for r in rows})==7
        assert len({r['index'] for r in accepted})==len(accepted)==combined_summary['train_demonstrations']+2 and all(r['index']!=0 for r in accepted)
        augmentations=[r for r in rows if r.get('recovery_supervision_mask_required')]
        assert len(augmentations)==5+corrective_count
        for r in augmentations:
            assert r['recovery_replay_verified']
            assert all(type(a.get('supervise')) is bool for a in r['teacher_actions'])
            assert all(not a['supervise'] for a in r['teacher_actions'] if a['kind'] in ('prefix','perturbation','policy_prefix'))
    if new_scene_bank:
        assert paired_controls and expanded and not recovery_bank
        assert EXPECTED=='592008b7c799dcdd3f7b570a3d94ccbb8f53c4bca90bd04e9e2d2af99dcc7c2c'
        new_scene_summary=json.loads((BANK.parent/'frozen_summary.json').read_text())
        parent=ROOT/'runs/remaining_original_navigation_bank_20261001/model_manifest_original_projection_frozen.jsonl'
        assert review.sha(parent)==new_scene_summary['parent_manifest_sha256']=='1ccaec2d9bef13a4f14f5655b71a1d59620cc67d2acb749bbd1cf63e8018ebb4'
        parent_rows=list(map(json.loads,parent.read_text().splitlines()))
        assert accepted[:len(parent_rows)]==parent_rows
        assert dev_rows==[r for r in parent_rows if r['split']=='dev']
        assert new_scene_summary['frozen_manifest_sha256']==EXPECTED and new_scene_summary['dev_unchanged']
        assert len({r['task_id'] for r in accepted})==len(accepted)
        assert {r['scene'] for r in accepted[len(parent_rows):]}.isdisjoint({r['scene'] for r in parent_rows})
    if recovery_bank:
        assert paired_controls and expanded
        recovery_summary=json.loads((BANK.parent/'frozen_summary.json').read_text())
        assert recovery_summary['parent_manifest_sha256']=='1ccaec2d9bef13a4f14f5655b71a1d59620cc67d2acb749bbd1cf63e8018ebb4'
        assert recovery_summary['frozen_manifest_sha256']==EXPECTED and recovery_summary['dev_unchanged']
        recovery_rows=[r for r in rows if r.get('recovery_supervision_mask_required')]
        assert len(recovery_rows)==recovery_summary['accepted_recovery_episodes']>0
        for r in recovery_rows:
            assert r['split']=='train' and r['recovery_replay_verified']
            assert all(not a['supervise'] for a in r['teacher_actions'] if a['kind'] in ('prefix','perturbation'))
            assert any(a['supervise'] for a in r['teacher_actions'])
    if expanded and not approach_manifest:
        assert (len(rows),len(dev_rows))==((combined_summary['train_demonstrations'],2) if combined_bank else (10,2) if new_scene_bank else (8+len(recovery_rows),2) if recovery_bank else (8,2) if paired_controls else (4,1))
        assert not ({r['scene'] for r in rows} & {r['scene'] for r in dev_rows})
    if approach_manifest:
        assert expanded and clip and not any((paired_controls,recovery_bank,new_scene_bank,combined_bank))
        assert len(rows) in (4,8) and len(dev_rows)==2
        assert not ({r['scene'] for r in rows} & {r['scene'] for r in dev_rows})
        assert all(r['goal_resolver']=='visible_approach_framed_v2' for r in accepted)
        assert all(r['training_approved'] for r in rows) and not any(r['training_approved'] for r in dev_rows)
    else:
        assert [r['index'] for r in rows]==[4,5] and not dev_rows
    projection=all(r.get('navigation_projection_ready',False) for r in accepted)
    if projection:
        assert (comparison_cache or paired_controls or approach_manifest) and all(r['navigation_projection_stages']==2 for r in accepted)
        assert all(r['policy_context']['instruction']==r['instruction_original'] for r in accepted)
    swaps={}
    if paired_controls:
        assert clip and (combined_bank or new_scene_bank or recovery_bank or expected=='1ccaec2d9bef13a4f14f5655b71a1d59620cc67d2acb749bbd1cf63e8018ebb4')
        pair=[r for r in rows if r.get('complete_reviewed_same_start_pair')]
        assert len(pair)==2 and pair[0]['pair_id']==pair[1]['pair_id']
        assert all(pair[0][k]==pair[1][k] for k in ('scene','start','start_rotation_simulator_argument'))
        assert pair[0]['policy_context']['robot']==pair[1]['policy_context']['robot']
        swaps={pair[0]['index']:pair[1],pair[1]['index']:pair[0]}
    torch.set_num_threads(4)
    torch.manual_seed(20260930)
    np.random.seed(20260930)
    spec=importlib.util.spec_from_file_location('reviewed_overfit_trainer',TRAINER)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if clip:
        from spatial_clip_features import SpatialCLIPFeatures
        module=SpatialCLIPFeatures(module,device); module.check()
    input_dim=getattr(module,'INPUT_DIM',175)
    text_start=getattr(module,'TEXT_START',111)
    save('protocol.json',{'version':'spatial_clip_sequence_gru_v1' if clip else 'original_text_navigation_projection_gru_v1' if projection else 'expanded_instruction_rgb_stats_gru_v1' if expanded else 'two_train_instruction_rgb_stats_gru_v1','data_manifest_sha256':EXPECTED,
        'navigation_projection':projection,'stage_count_feature':2,'manipulation_scored':False,'source_protocol':'level2_context_view_obb_v1','instruction_version':'original_verbatim_navigation_projection_v1' if projection else 'mixed_frozen_v2_v3' if expanded else 'assistant_landmark_navigation_revision_v2',
        'model':'existing SequenceGRU hidden 128; window 16','features':'RGB stats plus spatial CLIP cells; relative pose/yaw; own action; robot; ordered CLIP text vector' if clip else 'RGB mean/std/min/max per L/F/R camera; relative pose and yaw; own previous action; robot; text token hash',
        'limitations':'Frozen CLIP; small cohort permits memorization; English pretraining and unknown corpus overlap; larger input increases policy capacity' if clip else 'RGB statistics lack spatial layout; text token hash ignores word order; small cohort permits memorization without language grounding',
        'forbidden_inputs':['gold category/region lists','goal coordinates','scene/task IDs','semantic masks','navmesh','success flags','teacher step/stage'],
        'seed':20260930,'max_optimizer_steps':STEPS,'fixed_optimizer_steps':fixed_steps,'window':WINDOW,'goal_resolver':'obb_v1','radius_strict_m':1.,'horizon':400,
        'training_indices':[r['index'] for r in rows],'dev_indices':[r['index'] for r in dev_rows],'controls':['normal','empty_text','swapped_text'],
        'swapped_text_rule':'only complete reviewed same-start pair; score against original targets' if paired_controls else 'first other instruction',
        'recovery_supervision_mask':recovery_bank or combined_bank,'recovery_summary':recovery_summary,'new_scene_summary':new_scene_summary,'combined_summary':combined_summary,
        'scope':f'{len(rows)} train; {len(dev_rows)} scene-separated reviewed development episodes' if expanded else 'train-only memorization/wiring check; no validation/test/generalization score',
        'feature_encoder':module.protocol(accepted) if clip else 'baseline RGB stats and text hash','trainer_sha256':review.sha(TRAINER),'script_sha256':review.sha(Path(__file__))})
    args=review.driver.probe.load_legacy_module(review.driver.probe.ROOT).make_args()
    args.success_dis,args.max_step=1.,400
    if approach_manifest:
        from frozen_approach_resolver import install
        install(evaluator,Path(approach_manifest),'visible_approach_framed_v2')
        protocol=json.loads((OUT/'protocol.json').read_text())
        protocol.update({'source_protocol':'visible_approach_framed_v2','goal_resolver':'visible_approach_framed_v2',
                         'goal_manifest_sha256':review.sha(Path(approach_manifest)),
                         'full_dev_cohort':5,'held_dev_source_indices':[203,1020,1022],'dev_scope':'accepted source 207 and 1021 only; three held excluded with coverage reported',
                         'stage_transition':'actual actor; failed STOP advances as in original simulator, full chain requires both valid'})
        save('protocol.json',protocol)
    else:resolver.install(evaluator,'obb_v1')
    xs,ys,tids=[],[],[]
    for row in rows:
        source=Path(row['source_task_json']); assert review.sha(source)==row['source_sha256']
        task=json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            sim=review.driver.probe.SceneSimulator(args=args,config=task)
            review.reset(sim,task['trial']['trial_0']['pos'][0])
        sim.save_path=None
        try:
            start=list(map(float,sim.agent.get_state().position)); start_yaw=signed_yaw(sim.agent.get_state().rotation)
            history=[]; previous=-1
            for item in row['teacher_actions']:
                sim.stage=item['stage']
                history.append(feature(module,sim,start,start_yaw,previous,context(row)))
                if item.get('supervise',True):
                    xs.append(window(history)); ys.append(module.ACTION_TO_IDX[item['action']]); tids.append(row['index'])
                if item['action']=='stop':
                    with contextlib.redirect_stdout(io.StringIO()): info=sim.get_info()
                    assert info['geo dis']<1
                if approach_manifest:
                    with contextlib.redirect_stdout(io.StringIO()):sim.actor(item['action'])
                    if item['action']=='stop':assert sim.successes[item['stage']]
                elif item['action']!='stop':sim.agent.act(item['action'])
                assert math.dist(sim.agent.get_state().position,item['position'])<1e-5
                previous=module.ACTION_TO_IDX[item['action']]
        finally: sim.close()
        print('collected',row['index'],len(row['teacher_actions']),flush=True)
    assert len(xs)==len(ys)==sum(a.get('supervise',True) for r in rows for a in r['teacher_actions'])
    assert sum(y==module.ACTION_TO_IDX['stop'] for y in ys)==sum(a['action']=='stop' and a.get('supervise',True) for r in rows for a in r['teacher_actions'])
    if recovery_bank:assert len(xs)==recovery_summary['total_supervised_train_actions']
    if new_scene_bank:assert len(xs)==new_scene_summary['train_action_labels']==1479
    if combined_bank:
        assert len(xs)==combined_summary['supervised_train_actions']==(2555 if corrective_v3 else 2207 if corrective_v2 else 1717 if corrective_count==0 else 2002)
        assert sum(y==module.ACTION_TO_IDX['stop'] for y in ys)==combined_summary['supervised_STOP_labels']
    raw=np.asarray(xs,dtype=np.float32)
    assert not np.any(raw[:,:,47:47+module.HASH_DIM])
    if comparison_cache:
        prior=np.load(comparison_cache)
        difference=float(np.max(np.abs(raw[:,:,:111]-prior['x'][:,:,:111])))
        assert (raw.shape[:2]==prior['x'].shape[:2] if clip else raw.shape==prior['x'].shape) and np.array_equal(ys,prior['y']) and np.array_equal(tids,prior['tids'])
        assert difference<=1e-6, 'non-text features changed'
        save('paired_feature_check.json',{'compared_feature_block':'baseline RGB stats/causal state/empty gold metadata; excludes new visual and text blocks' if clip else 'all nontext','max_nontext_difference':difference,'tolerance':1e-6,'labels_and_train_ids_identical':True,'comparison_cache_sha256':review.sha(Path(comparison_cache))})
    mean=raw.reshape(-1,input_dim).mean(0); std=np.maximum(raw.reshape(-1,input_dim).std(0),1e-3)
    np.savez_compressed(OUT/'causal_features.npz',x=raw,y=ys,tids=tids,mean=mean,std=std)
    x=torch.tensor((raw-mean)/std,device=device); y=torch.tensor(ys,dtype=torch.long,device=device)
    model=module.SequenceGRU(input_dim).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1.2e-3,weight_decay=1e-4)
    counts=Counter(ys)
    weights=torch.tensor([len(ys)/(4*max(counts[i],1)) for i in range(4)],device=device)
    weights=weights/weights.mean()
    history=[]; stable=0
    for step in range(1,STEPS+1):
        model.train(); optimizer.zero_grad()
        logits=model(x); loss=torch.nn.functional.cross_entropy(logits,y,weight=weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if step%50==0:
            model.eval()
            with torch.no_grad(): pred=model(x).argmax(-1)
            accuracy=float((pred==y).float().mean())
            stop=(y==module.ACTION_TO_IDX['stop'])
            recall=float((pred[stop]==y[stop]).float().mean())
            metric={'step':step,'loss':float(loss),'teacher_forced_action_accuracy':accuracy,'stop_recall':recall}
            history.append(metric); save('training_progress.json',history); print(json.dumps(metric),flush=True)
            stable=stable+1 if accuracy==1. else 0
            if stable>=3 and not fixed_steps: break
    model.eval()
    torch.save({'state_dict':model.cpu().state_dict(),'input_dim':input_dim,'window':WINDOW,'mean':mean.tolist(),'std':std.tolist(),'protocol_sha256':review.sha(OUT/'protocol.json')},OUT/'checkpoint.pt')
    model.to(device)
    controls=[]
    for variant in ('normal','empty_text','swapped_text'):
        other=raw.copy()
        selected_samples=np.asarray([tid in swaps for tid in tids]) if paired_controls and variant=='swapped_text' else np.ones(len(tids),dtype=bool)
        if variant=='empty_text': other[:,:,text_start:]=0
        if variant=='swapped_text':
            for i,tid in enumerate(tids):
                if paired_controls and tid not in swaps:continue
                alternate=swaps[tid] if paired_controls else next(r for r in rows if r['index']!=tid)
                block=module.encode_text(alternate['policy_context']['instruction']) if clip else module.hash_features(module.tokenize(alternate['policy_context']['instruction'])[:80],module.HASH_DIM)
                # Preserve zero padding; replace text on real history frames only.
                active=np.any(raw[i]!=0,axis=1)
                other[i,active,text_start:]=block
        assert np.array_equal(other[:,:,:text_start],raw[:,:,:text_start])
        with torch.no_grad(): pred=model(torch.tensor((other-mean)/std,device=device)).argmax(-1)
        mask=torch.tensor(selected_samples,device=device)
        stop_mask=(y==module.ACTION_TO_IDX['stop']) & mask
        controls.append({'variant':variant,'teacher_forced_action_accuracy':float((pred[mask]==y[mask]).float().mean()),
            'evaluated_samples':int(selected_samples.sum()),
            'stop_recall':float((pred[stop_mask]==y[stop_mask]).float().mean())})
    save('teacher_forced_controls.json',controls)
    rollouts=[]
    # Recovery copies share source tasks: evaluate the original bank once, not inflated duplicate rollouts.
    evaluation_rows=[r for r in rows+dev_rows if not r.get('recovery_supervision_mask_required')]
    for variant in ('normal','empty_text','swapped_text'):
        for row in evaluation_rows:
            if paired_controls and variant=='swapped_text' and row['index'] not in swaps:continue
            task=json.loads(Path(row['source_task_json']).read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                sim=review.driver.probe.SceneSimulator(args=args,config=task)
                review.reset(sim,task['trial']['trial_0']['pos'][0])
            sim.save_path=None
            try:
                start=list(map(float,sim.agent.get_state().position)); yaw=signed_yaw(sim.agent.get_state().rotation)
                text=row['policy_context']['instruction'] if variant=='normal' else ('' if variant=='empty_text' else swaps[row['index']]['policy_context']['instruction'] if paired_controls else next(r['policy_context']['instruction'] for r in evaluation_rows if r['index']!=row['index']))
                ctx=context(row,text); past=[]; previous=-1; stage=0; trace=[]; stops=[]; minimum=[math.inf,math.inf]
                for step in range(400):
                    sim.stage=stage
                    distance_before=None
                    if approach_manifest:
                        with contextlib.redirect_stdout(io.StringIO()):distance_before=float(sim.get_info()['geo dis'])
                        minimum[stage]=min(minimum[stage],distance_before)
                    past.append(feature(module,sim,start,yaw,previous,ctx))
                    array=(np.asarray(window(past),dtype=np.float32)-mean)/std
                    with torch.no_grad(): action_index=int(model(torch.tensor(array[None],device=device)).argmax(-1))
                    action=module.IDX_TO_ACTION[action_index]
                    if action=='stop':
                        with contextlib.redirect_stdout(io.StringIO()): info=sim.get_info()
                        distance=float(info['geo dis'])
                        stops.append({'step':step,'stage':stage,'geo_m':distance if math.isfinite(distance) else None,'valid':distance<1})
                        stage+=1
                    if approach_manifest:
                        with contextlib.redirect_stdout(io.StringIO()):sim.actor(action)
                        if action=='stop':assert bool(sim.successes[stage-1])==stops[-1]['valid']
                    elif action!='stop':sim.agent.act(action)
                    trace.append({'action':action,'stage':stage-1 if action=='stop' else stage,'position':list(map(float,sim.agent.get_state().position)),
                                  **({'pre_action_geodesic_m':distance_before if math.isfinite(distance_before) else None} if approach_manifest else {})})
                    previous=action_index
                    if stage==2: break
                record={'index':row['index'],'split':row['split'],'variant':variant,'steps':len(trace),'stops':stops,
                    'instruction_source_index':swaps[row['index']]['index'] if paired_controls and variant=='swapped_text' else row['index'] if variant=='normal' else None,
                    'stage0_success':bool(stops and stops[0]['valid']),
                    'full_chain_success':len(stops)==2 and all(s['valid'] for s in stops),'actions':trace}
                rollouts.append(record); save('rollouts.json',rollouts)
                if approach_manifest:
                    record.update({'source_index':row['source_index'],
                                   'minimum_pre_action_geodesic_m':[v if math.isfinite(v) else None for v in minimum]})
                    save('rollouts.json',rollouts)
                print(json.dumps({k:v for k,v in record.items() if k!='actions'}),flush=True)
            finally: sim.close()
    summary={'training_samples':len(ys),'train_episodes':len(rows),'unique_train_tasks':len({r['task_id'] for r in rows}),'dev_episodes':len(dev_rows),'training_last':history[-1],
        'teacher_forced_controls':controls,'closed_loop_train_success':{v:sum(r['full_chain_success'] for r in rollouts if r['variant']==v and r['split']=='train') for v in ('normal','empty_text','swapped_text')},
        'closed_loop_dev_success':{v:sum(r['full_chain_success'] for r in rollouts if r['variant']==v and r['split']=='dev') for v in ('normal','empty_text','swapped_text')},
        'evaluation_counts':{s:{v:sum(r['variant']==v and r['split']==s for r in rollouts) for v in ('normal','empty_text','swapped_text')} for s in ('train','dev')},
        'encoder_mode':'spatial_clip' if clip else 'baseline','parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':review.sha(OUT/'checkpoint.pt'),'scope':'reviewed development pilot; no Table 4 update; no generalization claim'}
    save('summary.json',summary); print(json.dumps(summary),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--bank'); parser.add_argument('--out'); parser.add_argument('--sha'); parser.add_argument('--comparison-cache'); parser.add_argument('--clip',action='store_true'); parser.add_argument('--paired-controls',action='store_true'); parser.add_argument('--recovery-bank',action='store_true')
    parser.add_argument('--new-scene-bank',action='store_true')
    parser.add_argument('--combined-bank',action='store_true')
    cli=parser.parse_args()
    assert all((cli.bank,cli.out,cli.sha)) or not any((cli.bank,cli.out,cli.sha))
    main(cli.bank,cli.out,cli.sha,cli.comparison_cache,cli.clip,cli.paired_controls,cli.recovery_bank,cli.new_scene_bank,cli.combined_bank)
