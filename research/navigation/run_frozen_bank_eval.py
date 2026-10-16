"""Frozen release-wide evaluation of historical checkpoints; never trains."""
import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path('/home/aimldl/LH-VLN')
OUT=ROOT/'runs/release2500_eval_20261012_v1'
AUDIT=ROOT/'runs/level2_5_readiness_20261012_v1/audit.json'
RELEASE=ROOT/'nav_gen/runs/ieee_access_release_export_20260708/iden_vln_tasks.jsonl'
COHORT=ROOT/'runs/level2_5_training_cohorts_20261012_v1/training_cohorts.json'
INVENTORY=ROOT/'runs/level2_5_readiness_20261012_v1/checkpoint_inventory_resolved.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def atomic(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def metric(u):
    k=len(u)
    return sum(u)/k,sum(u[j]*(1+(1 if j==0 else u[j-1])) for j in range(k))/(2*k)
def self_check():
    for k in range(2,6):
        for u in itertools.product((0,1),repeat=k):
            a,b=metric(u);assert 0<=a<=1 and 0<=b<=1
            if k==2:assert b==(2*u[0]+u[1]*(1+u[0]))/4
    assert metric([0,0,0])==(0,0) and metric([1,1,1,1,1])==(1,1)

def prepare():
    assert not (OUT/'protocol_frozen.json').exists(),'Frozen protocol already exists'
    OUT.mkdir(exist_ok=True)
    release=[json.loads(l) for l in RELEASE.read_text().splitlines()]
    audit=json.loads(AUDIT.read_text());assert sha(RELEASE)==audit['release_sha256']
    assets={a['task_id']:a for a in audit['assets']}
    inventory=json.loads(INVENTORY.read_text())[:10]
    assert len(release)==2500 and len({r['task_id'] for r in release})==2500
    assert len(inventory)==10
    for p in inventory:
        if p['checkpoint']:assert sha(Path(p['checkpoint']))==p['sha256']
    bank=[]
    for r in sorted(release,key=lambda r:r['task_id']):
        a=assets[r['task_id']];source=Path(a['source']);assert sha(source)==a['source_sha256']
        t=json.loads(source.read_text());k=int(r['level']);first=t['trial']['trial_0']
        assert t['Scene']==r['scene'] and t['Task instruction']==r['task_instruction']
        # Source simulator starts its yaw counter at 180 for identity rotation.
        angle=math.radians(float(first['yaw'][0])-180)
        rotation=[math.cos(angle/2),0.,math.sin(angle/2),0.]
        config={'Task instruction':r['task_instruction'],'Scene':r['scene'],'Robot':r['robot'],'Object':r['object_names'],'Region':r['region_ids'],'Region Name':r['region_names'],'Generated level':k,'gt_step':[len(t['trial'][f'trial_{j}']['action']) for j in range(k)]}
        assert len(config['Object'])==len(config['Region'])==k
        bank.append({'task_id':r['task_id'],'split':r['scene_heldout_split'],'level':k,'scene':r['scene'],'source':str(source),'source_sha256':a['source_sha256'],'config':config,'start_state':{'position':first['pos'][0],'rotation':rotation}})
    atomic(OUT/'bank_frozen.json',bank)
    atomic(OUT/'cohorts_frozen.json',json.loads(COHORT.read_text()))
    smoke=[next(i for i,r in enumerate(bank) if r['task_id']==task_id) for task_id in json.loads((ROOT/'runs/level2_5_readiness_20261012_v1/benchmark_protocol_frozen.json').read_text())['task_ids']]
    atomic(OUT/'protocol_frozen.json',{'version':'release2500_historical_aabb_v1','release_sha256':sha(RELEASE),'bank_sha256':sha(OUT/'bank_frozen.json'),'cohorts_sha256':sha(OUT/'cohorts_frozen.json'),'runner_sha256':sha(Path(__file__)),'policies':inventory,'requested_per_policy':2500,'max_steps':150,'strict_stop_distance_m':1.0,'resolver':'historical index-aware AABB navmesh snap','start':'source trial_0 first position, yaw mapped as radians(source_yaw-180)','stage_transition':'every scored STOP advances, including unsuccessful STOP; initial observation STOP does not score','ISR':'macro mean over episode sum(success)/K; micro also reported','CSR':'local macro mean sum(u_k*(1+u_previous))/(2K), u_0=1','failed_episode':'all K stages zero; failure retained in denominator','invalid_reference':'all K stages zero, counted failure; reference and NE coverage retained','NE':'historical finite observed-stage mean, conditional coverage explicit','policy_inputs':'historical feature contract; target/sequence include release object/room metadata, RGB statistics, pose, previous actions; not a metadata-free visual policy','smoke_indices':smoke,'smoke_gate':'40 runs; every runtime completed, original reset verified, correct K success slots; navigation failure is not gate failure','parallelism':1,'scope':'development audit, not untouched scene generalization; grouped split/level/training exposure','compiled_paper_update':'only after full completion and verification'})
    print('Frozen all2500 bank and ten checkpoints',flush=True)

def summarize(rows):
    n=len(rows)
    def avg(key):
        v=[r[key] for r in rows if r.get(key) is not None];return sum(v)/len(v) if v else None
    return {'N_records':n,'failed':sum(r['status']!='completed' for r in rows),'ISR_macro':avg('ISR'),'ISR_micro':sum(sum(r['successes']) for r in rows)/sum(r['level'] for r in rows) if n else None,'CSR_local_macro':avg('CSR'),'SR':avg('SR'),'OSR':avg('OSR'),'SPL_conditional':avg('SPL'),'SPL_coverage':sum(r.get('SPL') is not None for r in rows),'NE_conditional':avg('NE'),'NE_episode_coverage':sum(r.get('NE') is not None for r in rows)}

def evaluate(phase):
    protocol=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(Path(__file__))==protocol['runner_sha256']
    assert sha(OUT/'bank_frozen.json')==protocol['bank_sha256']
    bank=json.loads((OUT/'bank_frozen.json').read_text());cohorts=json.loads((OUT/'cohorts_frozen.json').read_text())
    if phase=='full':assert json.loads((OUT/'smoke/status.json').read_text())['status']=='passed'
    folder=OUT/phase;folder.mkdir(exist_ok=True)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    import numpy as np
    ordered=load(ROOT/'runs/idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py','release_ordered')
    sweep=load(ROOT/'runs/idenvln_policy_sweep_20260805_014026/run_policy_sweep.py','release_sweep')
    legacy=ordered.load_legacy_module(ROOT);ordered.install_index_aware_target_patch()
    args=legacy.make_args();args.episode_data='source_gt_metadata';legacy.make_args=lambda:args
    def context(c):
        return {'instruction':c['Task instruction'],'objects':legacy.object_labels(c['Object']),'regions':legacy.normalize_regions(c['Region Name']),'robot':c['Robot'].lower(),'level':c['Generated level'],'scene':c['Scene'],'index':0}
    legacy.load_context=context
    dataset=[(r['config'],[]) for r in bank];starts={str(i):r['start_state'] for i,r in enumerate(bank)}
    # The adapter must restore source orientation/position before policy observation.
    reset=ordered.reset_simulator
    def checked_reset(sim,state):
        reset(sim,state);actual=ordered.capture_state(sim)
        assert np.allclose(actual['position'],state['position'],atol=1e-5)
        assert abs(abs(np.dot(actual['rotation'],state['rotation']))-1)<1e-5
    ordered.reset_simulator=checked_reset
    indices=protocol['smoke_indices'] if phase=='smoke' else range(len(bank))
    stop=threading.Event()
    def monitor():
        with (folder/'gpu_samples.jsonl').open('a') as f:
            while not stop.is_set():
                sample={'utc':time.time(),'pid':os.getpid()}
                for key,query in [('device','--query-gpu=name,memory.used,utilization.gpu'),('compute_processes','--query-compute-apps=pid,used_gpu_memory')]:
                    result=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True);sample[key]=result.stdout.strip()
                f.write(json.dumps(sample)+'\n');f.flush();stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    total=0;all_smoke_pass=True
    try:
        for spec in protocol['policies']:
            name=spec['name'];policy_dir=folder/name;policy_dir.mkdir(exist_ok=True)
            if spec['checkpoint']:assert sha(Path(spec['checkpoint']))==spec['sha256']
            state=sweep.load_policy_from_checkpoint(legacy,spec)
            assert state['status']=='loaded',state
            cache={};records=[]
            for i in indices:
                r=bank[i];result_path=policy_dir/f'{i:04d}.json'
                if result_path.exists():
                    row=json.loads(result_path.read_text());assert row['task_id']==r['task_id']
                else:
                    tmp_trace=policy_dir/f'{i:04d}.trace.jsonl';started=time.perf_counter()
                    with open(os.devnull,'w') as null,contextlib.redirect_stdout(null):
                        trace,_=ordered.evaluate_policy(legacy,dataset,name,'planner_prior' if name=='sequence_hybrid' else 'raw',state,[i],starts,tmp_trace,150,1.,61007,policy_dir/'progress.json',cache)
                    t=trace[0];runtime_ok=t['status']=='completed' and len(t.get('successes',[]))==r['level']
                    valid=runtime_ok and t.get('reference_path_length') is not None and all(p is not None for p in t.get('reference_target_points',[]))
                    u=[int(x) for x in t['successes']] if valid else [0]*r['level'];isr,csr=metric(u)
                    family='frame' if name.startswith('frame') else 'target' if name.startswith('target') else 'dagger_final_round2' if name=='dagger_lite' else 'sequence'
                    seen=None if name=='random' else r['scene'] in cohorts['families'][family]['scene_ids']['train']
                    row={'task_id':r['task_id'],'index':i,'policy':name,'level':r['level'],'split':r['split'],'scene':r['scene'],'seen_training_scene':seen,'known_exposure_scene':r['scene'] in cohorts['known_exposure_scene_ids'],'status':'completed' if valid else 'runtime_failure' if not runtime_ok else 'invalid_reference','runtime_completed':runtime_ok,'error':t.get('error'),'actions':len(t.get('actions',[])),'elapsed_s':time.perf_counter()-started,'successes':u,'ISR':isr,'CSR':csr,'SR':int(valid and t.get('full_chain_success',False)),'OSR':int(valid and t.get('oracle_full_chain_success',False)),'SPL':t.get('spl') if valid else None,'NE':t.get('navigation_error') if valid else None}
                    compressed=tmp_trace.with_suffix('.gz')
                    with gzip.open(compressed,'wb') as f:f.write(tmp_trace.read_bytes())
                    tmp_trace.unlink();atomic(result_path,row)
                records.append(row);total+=1;all_smoke_pass=all_smoke_pass and row['runtime_completed']
                if phase=='smoke' or total%25==0:
                    atomic(folder/'status.json',{'status':'running','phase':phase,'records_processed_this_invocation':total,'last_policy':name,'last_index':i,'last_task_id':r['task_id'],'updated_utc':time.time()})
                    print(phase,name,i,row['status'],round(row['elapsed_s'],2),flush=True)
            groups={'all':summarize(records)}
            for key in ('level','split','seen_training_scene','known_exposure_scene'):
                for value in sorted({str(r[key]) for r in records}):groups[f'{key}={value}']=summarize([r for r in records if str(r[key])==value])
            atomic(policy_dir/'summary.json',groups)
        expected=len(protocol['policies'])*len(indices)
        assert total==expected
        atomic(folder/'status.json',{'status':('passed' if all_smoke_pass else 'failed') if phase=='smoke' else 'completed','records':total,'updated_utc':time.time()})
        if phase=='smoke':assert all_smoke_pass,'Smoke runtime compatibility failure; inspect preserved traces'
    finally:
        stop.set();thread.join(timeout=10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','smoke','full','self-check']);parser.add_argument('--output-dir',type=Path,required=True);args_cli=parser.parse_args();OUT=args_cli.output_dir;phase=args_cli.phase
    self_check()
    if phase=='prepare':prepare()
    elif phase!='self-check':evaluate(phase)
