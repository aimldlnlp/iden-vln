"""Matched numerical diagnostic: post-action/unsigned yaw versus causal/signed yaw."""
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/home/aimldl/LH-VLN')
RUNS=ROOT/'runs'
REPAIR=RUNS/'pipeline_repair_20261013_v3'
OUT=RUNS/'balanced100_matched_contract_20261013_v1'
BANK=RUNS/'balanced100_eval_20261013_v1/bank_frozen.json'
SEED=57007
STEPS=4000


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def window(history):return [[0.]*175]*max(0,8-len(history))+history[-8:]


def self_check():
    assert len(window([[1.]*175]))==8
    assert window([[1.]*175])[-1]==[1.]*175
    assert window([[float(i)]*175 for i in range(10)])[0]==[2.]*175


def main():
    import numpy as np
    import torch
    self_check()
    OUT.mkdir(exist_ok=False)
    os.chdir(ROOT)
    sys.path[:0]=[str(ROOT),str(REPAIR)]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sequence=load(REPAIR/'sequence_trainer_original.py','matched_sequence')
    helpers=load(REPAIR/'repair_contract.py','matched_helpers')
    ordered_path=RUNS/'idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py'
    ordered=load(ordered_path,'matched_ordered')
    legacy=ordered.load_legacy_module(ROOT)
    ordered.install_index_aware_target_patch()
    pilot=json.loads((REPAIR/'experiment/protocol_frozen.json').read_text())
    cohort=pilot['cohort']
    assert len(cohort['train'])==len(cohort['dev'])==4
    assert not {r['scene'] for r in cohort['train']} & {r['scene'] for r in cohort['dev']}
    assert json.loads((RUNS/'balanced100_teacher_contract_20261013_v1/independent_verification.json').read_text())['passed']
    bank=json.loads(BANK.read_text())
    assert len(bank)==100 and sha(BANK)=='9f74e219b74adc3b9807bbde230092415b0b2f5c8fe7e6149162874d34dfeec2'
    sources=[Path(__file__),BANK,REPAIR/'sequence_trainer_original.py',REPAIR/'repair_contract.py',REPAIR/'experiment/protocol_frozen.json',ordered_path,RUNS/'idenvln_harmonized_agent_eval_level23_20260707/run_harmonized_agent_eval_fixed.py']
    for rows in cohort.values():
        for row in rows:
            path=Path(row['trial'])/'task.json'
            assert sha(path)==row['task_sha256']
            sources.append(path)
    dependencies={str(p):sha(p) for p in sources}
    save('protocol_frozen.json',{'version':OUT.name,'seed':SEED,'cohort':cohort,'dependencies':dependencies,'started_utc':time.time(),'arms':{'historical_contract':'post-action RGB/pose paired with current action; original unsigned rotation.angle() at runtime','repaired_contract':'pre-action RGB/pose paired with next action; signed yaw at runtime'},'architecture':'same GRU128/window8/input175; global RGB statistics/static metadata hashes/own pose/previous action','updates':STEPS,'batch_size':128,'learning_rate':.0012,'weight_decay':.0001,'weight_power':.45,'STOP_factor':.65,'sampling_and_loss':'identical mild class weights in both arms','checkpoint_selection':'final update4000 for both; no early stopping or dev selection; independent clones common to both','device':'CPU,2threads; deterministic algorithms','horizon':150,'strict_STOP_m':1.,'resolver':'unchanged index-aware snapped AABB','evaluation':'same train4/dev4 plus balanced100; random on same cohorts; paired task reporting','scope':'numerical contract diagnostic, one seed; grounding held not changed; not approved bank expansion or untouched scene generalization','excluded_effects':'does not isolate snapshot bug or reproduce original best-checkpoint selection; changes alignment and runtime yaw as a package','training_gate':'source actions consecutive, exactlyKSTOP, finalSTOP, <=150; replay verification from pilot and feature preflight retained; complete source data/no dropped samples'})
    save('status.json',{'phase':'data_preflight'})
    data={arm:[] for arm in ('historical_contract','repaired_contract')}
    labels=[];data_rows=[];contexts={}
    frames={}
    for rows in cohort.values():
        for row in rows:
            trial=Path(row['trial']);c=sequence.load_context(trial);contexts[row['index']]=c
            dirs=sorted([(sequence.parse_action_dir(p.name)[0],sequence.parse_action_dir(p.name)[1],p) for p in trial.iterdir() if p.is_dir() and sequence.parse_action_dir(p.name)[1] is not None])
            assert [i for i,a,p in dirs]==list(range(1,len(dirs)+1))
            assert [a for i,a,p in dirs]==row['teacher_actions']
            assert len(dirs)<=150 and dirs[-1][1]=='stop' and sum(a=='stop' for i,a,p in dirs)==row['level']
            for path in [p for i,a,p in dirs]+[helpers.before_action(trial,1,c['pos'],c['yaw'])[0]]:
                for camera in ('left','front','right'):
                    image=path/(camera+'.png');frames[str(image)]=sha(image)
            if row not in cohort['train']:continue
            previous=0;history={arm:[] for arm in data}
            for i,action,post_frame in dirs:
                before,pose,yaw=helpers.before_action(trial,i,c['pos'],c['yaw'])
                features={'historical_contract':sequence.frame_feature(post_frame,c['pos'][i],c['yaw'][i],previous,c),'repaired_contract':sequence.frame_feature(before,pose,yaw,previous,c)}
                assert all(f is not None and len(f)==175 and np.isfinite(f).all() for f in features.values())
                label=sequence.ACTION_TO_IDX[action]
                for arm,feature in features.items():
                    history[arm].append(feature);data[arm].append(window(history[arm]))
                labels.append(label);data_rows.append({'tid':row['tid'],'offset':i-1,'label':label})
                previous=label
    assert len(labels)==398 and labels.count(0)==10
    save('source_frame_hashes.json',frames)
    save('training/data_rows.json',data_rows)
    x={arm:torch.tensor(np.asarray(values,dtype=np.float32)) for arm,values in data.items()}
    y=torch.tensor(labels,dtype=torch.long)
    counts=torch.bincount(y,minlength=4).float()
    weights=counts.clamp_min(1).pow(-.45);weights[0]*=.65;weights=weights/weights.sum()*4
    sample_weights=weights[y]
    save('training/data_verification.json',{'passed':True,'samples':len(y),'STOP':int((y==0).sum()),'shapes':{a:list(t.shape) for a,t in x.items()},'label_counts':counts.tolist(),'class_weights':weights.tolist(),'source_frame_count':len(frames),'source_frame_hashes_sha256':sha(OUT/'source_frame_hashes.json'),'matched_labels':True,'features_differ':bool((x['historical_contract']!=x['repaired_contract']).any())})
    states={};training={}
    for arm in data:
        torch.manual_seed(SEED)
        model=sequence.SequenceGRU(175)
        initialization=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.0001)
        loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
        draw_hash=hashlib.sha256();curve=[]
        save('status.json',{'phase':'training','arm':arm,'step':0})
        for step in range(1,STEPS+1):
            model.train()
            ids=torch.multinomial(sample_weights,128,replacement=True)
            draw_hash.update(ids.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x[arm][ids]),y[ids]);loss.backward();optimizer.step()
            if step%500==0:
                model.eval()
                with torch.no_grad():pred=model(x[arm]).argmax(1)
                record={'step':step,'loss':float(loss.detach()),'accuracy':float((pred==y).float().mean()),'STOP_recall':float((pred[y==0]==0).float().mean()),'errors':int((pred!=y).sum())}
                curve.append(record);save('training/'+arm+'_curve.json',curve)
                save('status.json',{'phase':'training','arm':arm,**record})
                print(arm,record,flush=True)
        checkpoint=OUT/'training'/ (arm+'.pt')
        torch.save({'model_state_dict':helpers.best_snapshot(model),'input_dim':175,'window':8,'action_to_idx':sequence.ACTION_TO_IDX,'variant':arm,'seed':SEED,'updates':STEPS},checkpoint)
        model.eval()
        states[arm]={'status':'loaded','model':model,'feature':'seq','window':8}
        training[arm]={'initialization_sha256':initialization,'draws_sha256':draw_hash.hexdigest(),'checkpoint_sha256':sha(checkpoint),'final':curve[-1]}
    assert training['historical_contract']['initialization_sha256']==training['repaired_contract']['initialization_sha256']
    assert training['historical_contract']['draws_sha256']==training['repaired_contract']['draws_sha256']
    save('training/matched_verification.json',{'passed':True,'arms':training})
    args=legacy.make_args();args.episode_data='source_gt_metadata';legacy.make_args=lambda:args
    def context(config):
        return {'instruction':config['Task instruction'],'objects':legacy.object_labels(config['Object']),'regions':legacy.normalize_regions(config['Region Name']),'robot':config['Robot'].lower(),'level':config['Generated level'],'scene':config['Scene']}
    legacy.load_context=context
    source=inspect.getsource(ordered.evaluate_policy)
    old='yaw = float(sim.agent.get_state().rotation.angle())'
    assert source.count(old)==1
    namespace=dict(vars(ordered),signed_yaw=helpers.signed_yaw)
    exec(compile(source.replace(old,'yaw = signed_yaw(sim.agent.get_state().rotation)'),'<matched-signed-evaluator>','exec'),namespace)
    evaluators={'historical_contract':ordered.evaluate_policy,'repaired_contract':namespace['evaluate_policy'],'random':ordered.evaluate_policy}
    null=open(os.devnull,'w')
    summaries={}
    for split in ('train','dev','balanced100'):
        if split=='balanced100':
            dataset=[(r['config'],[]) for r in bank];indices=list(range(100));starts={str(i):r['start_state'] for i,r in enumerate(bank)}
        else:
            dataset={}
            for row in cohort[split]:
                c=contexts[row['index']]
                config=dict(c['task'],Object=c['objects'],**{'Region Name':c['regions'],'Generated level':c['level']})
                dataset[row['index']]=(config,[])
            indices=[r['index'] for r in cohort[split]]
            starts={str(i):{'position':contexts[i]['pos'][0],'rotation':[math.cos(contexts[i]['yaw'][0]/2),0,math.sin(contexts[i]['yaw'][0]/2),0]} for i in indices}
        summaries[split]={}
        for arm,evaluate in evaluators.items():
            folder=OUT/'evaluation'/split/arm;folder.mkdir(parents=True)
            state=states.get(arm,{'status':'loaded'})
            save('status.json',{'phase':'evaluation','split':split,'arm':arm})
            records=[]
            for ordinal,index in enumerate(indices):
                with contextlib.redirect_stdout(null):
                    traces,summary=evaluate(legacy,dataset,arm,'raw',state,[index],starts,folder/(str(index)+'.jsonl'),150,1.,SEED,folder/'progress.json',{})
                assert len(traces)==1
                record=traces[0];records.append(record)
                assert len(record.get('successes',[]))==int(dataset[index][0]['Generated level']),record.get('error')
                assert record['status']=='completed',record.get('error')
                save('status.json',{'phase':'evaluation','split':split,'arm':arm,'completed':ordinal+1,'requested':len(indices),'last_index':index})
                print(split,arm,index,record['successes'],record['executed_steps'] if 'executed_steps' in record else len(record['actions']),flush=True)
            with (folder/'traces.jsonl').open('w') as f:
                for record in records:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            valid=[r for r in records if r.get('reference_path_length') is not None]
            scored=[list(map(int,r['successes'])) if r.get('reference_path_length') is not None else [0]*len(r['successes']) for r in records]
            metrics={'episodes':len(records),'chains':sum(all(u) for u in scored),'stage_success':sum(map(sum,scored)),'stage_count':sum(map(len,scored)),'ISR_macro':sum(sum(u)/len(u) for u in scored)/len(scored),'CSR_local_macro':sum(sum(u[j]*(1+(1 if j==0 else u[j-1])) for j in range(len(u)))/(2*len(u)) for u in scored)/len(scored),'reference_valid_episodes':len(valid),'by_level':{str(k):{'episodes':sum(len(u)==k for u in scored),'chains':sum(all(u) for u in scored if len(u)==k),'stages':sum(sum(u) for u in scored if len(u)==k)} for k in range(2,6)}}
            summaries[split][arm]=metrics;save('evaluation/'+split+'/'+arm+'/metrics.json',metrics)
            print('Metrics',split,arm,metrics,flush=True)
    assert all(sha(Path(p))==h for p,h in dependencies.items())
    assert all(sha(Path(p))==h for p,h in frames.items())
    save('summary.json',summaries)
    save('status.json',{'phase':'completed','dependencies_unchanged':True,'source_frames_unchanged':True,'training_matched':True,'finished_utc':time.time(),'summaries':summaries})
    null.close()


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:self_check();print('self-check passed')
    else:
        assert not OUT.exists(),'Refuse overwrite; use a new version'
        try:main()
        except Exception as exc:
            if OUT.exists():save('failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
            raise
