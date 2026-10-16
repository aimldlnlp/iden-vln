"""Causal masked cache and conflict gate for human-reviewed recovery; no fitting."""
import contextlib,hashlib,importlib.util,io,json,math
from pathlib import Path
import numpy as np
import torch
import train_two_reviewed_il_overfit as il
from spatial_clip_features import SpatialCLIPFeatures
from frozen_approach_resolver import install
from run_single_goal_route_pilot import reset

ROOT=il.ROOT/'runs'
SOURCE=ROOT/'approach_clip8_recovery_collection_20261013_v2'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
OUT=ROOT/'approach_clip8_human_recovery_cache_20261014_v1'
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    assert not OUT.exists(),'refuse overwrite'
    summary=json.loads((SOURCE/'human_review_ingestion_summary.json').read_text())
    manifest=SOURCE/summary['manifest_filename']
    assert il.review.sha(manifest)==summary['manifest_sha256']=='2638e1304efd4721cfe5b2528befc784a3e8197bcfbf55590d606070052a7a51'
    for name,h in summary['exports'].items():assert il.review.sha(SOURCE/'human_review'/name)==h
    bank=list(map(json.loads,manifest.read_text().splitlines()))
    originals=list(map(json.loads,(PARENT/'model_manifest_frozen.jsonl').read_text().splitlines()))
    assert bank[:10]==originals
    recoveries=[r for r in bank if r.get('augmentation')=='clip8_recovery_v2']
    assert len(recoveries)==7 and all(r['human_validated'] and r['split']=='train' for r in recoveries)
    OUT.mkdir()
    paths=[manifest,SOURCE/'human_review_ingestion_summary.json',PARENT/'causal_features.npz',PARENT/'checkpoint.pt',PARENT/'goal_manifest_frozen.json',PARENT/'protocol.json',Path(__file__),Path(il.__file__),Path(il.TRAINER)]
    hashes={str(p):il.review.sha(p) for p in paths}
    save('protocol_frozen.json',{'version':'human_clip8_recovery_cache_v1','source_hashes':hashes,
        'normalization':'archived original bank8 mean/std; same for future BC and BC+recovery arms',
        'label_mask':'originals retained; prefix/perturbation/learner/search excluded from loss but retained in history',
        'conflict_gate':'exact raw causal-window hash with different action labels holds fitting until alternatives audited',
        'dev_used':False,'training_performed':False,'scope':'causal cache and prelaunch gate only; not full human route validation'})
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    spec=importlib.util.spec_from_file_location('human_cache_base',il.TRAINER)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    module=SpatialCLIPFeatures(base,device)
    assert module.protocol(bank)['checkpoint_files_sha256']==json.loads((PARENT/'protocol.json').read_text())['feature_encoder']['checkpoint_files_sha256']
    prior=np.load(PARENT/'causal_features.npz')
    assert prior['x'].shape==(1357,16,6767) and len(prior['y'])==1357
    xs=[];ys=[];tids=[];sample_refs=[];checks=[]
    install(il.evaluator,PARENT/'goal_manifest_frozen.json','visible_approach_framed_v2')
    args=il.review.driver.probe.load_legacy_module(il.review.driver.probe.ROOT).make_args();args.success_dis,args.max_step=1.,400
    for row in recoveries:
        assert il.review.sha(Path(row['source_task_json']))==row['source_sha256']
        task=json.loads(Path(row['source_task_json']).read_text())
        with contextlib.redirect_stdout(io.StringIO()):sim=il.review.driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None;history=[];previous=-1;errors=[];stops=[];labels=0;prefix_errors=[]
        parent=next(r for r in originals if r['index']==row['parent_index'])
        prefix=next(b['collection_prefix_actions'] for b in json.loads((SOURCE/'protocol_frozen.json').read_text())['branches'] if b['index']==row['parent_index'])
        cached=prior['x'][prior['tids']==row['parent_index']]
        try:
            with contextlib.redirect_stdout(io.StringIO()):reset(sim,row['start'])
            start=list(map(float,sim.agent.get_state().position));yaw=il.signed_yaw(sim.agent.get_state().rotation)
            for t,item in enumerate(row['teacher_actions']):
                assert sim.stage==item['stage'] and not sim.done
                value=il.feature(module,sim,start,yaw,previous,il.context(row));history.append(value)
                if t<prefix:
                    assert item['action']==parent['teacher_actions'][t]['action']
                    prefix_errors.append(float(np.max(np.abs(np.asarray(value)-cached[t,-1]))))
                if item['supervise']:
                    assert t>=row['correction_at'] and item['kind'].startswith('recovery_')
                    xs.append(il.window(history));ys.append(module.ACTION_TO_IDX[item['action']]);tids.append(row['index'])
                    sample_refs.append({'index':row['index'],'t':t,'stage':item['stage'],'action':item['action']});labels+=1
                else:assert item['kind'] in ('prefix','perturbation','learner_prefix','teacher_view_search')
                if item['action']=='stop':
                    with contextlib.redirect_stdout(io.StringIO()):geo=float(sim.get_info()['geo dis'])
                    assert geo<1 and item['supervise'];stops.append(geo)
                with contextlib.redirect_stdout(io.StringIO()):sim.actor(item['action'])
                errors.append(math.dist(sim.agent.get_state().position,item['position']))
                previous=module.ACTION_TO_IDX[item['action']]
            assert sim.done and all(sim.successes) and len(stops)==2 and max(errors)<1e-5 and max(prefix_errors)<1e-5
            checks.append({'index':row['index'],'supervised_labels':labels,'max_position_error_m':max(errors),
                          'prefix_feature_error':max(prefix_errors),'STOPs':stops})
        finally:sim.close()
        save('replay_checks.json',checks);print(json.dumps(checks[-1]),flush=True)
    extra=np.asarray(xs,dtype=np.float32);assert extra.shape==(697,16,6767)
    raw=np.concatenate([prior['x'],extra]);y=np.concatenate([prior['y'],np.asarray(ys)]);ids=np.concatenate([prior['tids'],np.asarray(tids)])
    assert raw.shape==(2054,16,6767) and not raw[:,:,47:111].any()
    assert np.array_equal(raw[:1357],prior['x']) and np.array_equal(y[:1357],prior['y'])
    groups={}
    for i,(x,label) in enumerate(zip(raw,y)):
        key=hashlib.sha256(x.tobytes()).hexdigest();groups.setdefault(key,[]).append((i,int(label)))
    conflicts=[{'window_sha256':h,'samples':[i for i,_ in group],'labels':sorted({v for _,v in group})}
               for h,group in groups.items() if len({v for _,v in group})>1]
    np.savez_compressed(OUT/'causal_features.npz',x=raw,y=y,tids=ids,mean=prior['mean'],std=prior['std'])
    save('recovery_sample_refs.json',sample_refs);save('exact_label_conflicts.json',conflicts)
    assert all(il.review.sha(Path(p))==h for p,h in hashes.items())
    save('verification.json',{'passed':True,'training_gate_passed':not conflicts,'exact_conflict_groups':len(conflicts),
        'samples':len(y),'recovery_samples':697,'STOP_labels':int(sum(y==module.ACTION_TO_IDX['stop'])),
        'original_samples_identical':True,'normalization_identical':True,'dev_used':False,'training_performed':False,
        'cache_sha256':il.review.sha(OUT/'causal_features.npz'),'records':checks})
    print(json.dumps({'samples':len(y),'exact_conflicts':len(conflicts),'training_gate_passed':not conflicts}),flush=True)
if __name__=='__main__':main()
