"""Frozen train-imitation gate for repaired historical Sequence BC, then dev diagnosis."""
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

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/aimldl/LH-VLN')
RUN = HERE/'experiment'
OLD = ROOT/'runs'
SEED = 57007


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    p = RUN/name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, p)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def key(config):
    return (config['Task instruction'], config['Scene'], str(config['Robot']).lower())


def main():
    import numpy as np
    import torch
    from repair_contract import self_check, signed_yaw, best_snapshot
    RUN.mkdir(exist_ok=False)
    save('status.json', {'phase': 'preflight', 'started_utc': time.time()})
    torch.set_num_threads(2)
    checks = self_check()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    sequence = load(HERE/'sequence_trainer_fixed.py', 'repair_sequence')
    causal_path = HERE/'causal_collector.py'
    causal = load(causal_path, 'repair_causal')
    ordered_path = OLD/'idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py'
    harmonized_path = OLD/'idenvln_harmonized_agent_eval_level23_20260707/run_harmonized_agent_eval_fixed.py'
    ordered = load(ordered_path, 'repair_ordered')
    legacy = ordered.load_legacy_module(ROOT)
    ordered.install_index_aware_target_patch()
    from utils.dataset import TaskDataset
    dataset = TaskDataset(legacy.make_args(), 'test')
    trials = sorted(sequence.SOURCE_DATA.glob('task/batch_generated_level23/*/*/success/trial_1'))
    contexts = [sequence.load_context(p) for p in trials]
    index_by_key = {key(dataset[i][0]): i for i in range(len(dataset))}
    assert len(trials) == len(dataset) == len(index_by_key) == 300
    assert sequence.ACTION_TO_IDX == legacy.ACTION_TO_IDX == {'stop':0,'turn_left':1,'move_forward':2,'turn_right':3}
    canonical_path = OLD/'idenvln_policy_sweep_20260805_014026/episode_bank.json'
    canonical = set(json.loads(canonical_path.read_text())['episode_indices'])
    assert canonical == set(range(100))
    retained = [tid for tid,c in enumerate(contexts) if index_by_key[key(c['task'])] not in canonical]
    assert len(retained) == 200
    split = sequence.split_by_traj(retained)
    eligible = {}
    ledger = []
    for tid in retained:
        dirs = sorted([(sequence.parse_action_dir(p.name)[0], sequence.parse_action_dir(p.name)[1]) for p in trials[tid].iterdir() if p.is_dir() and sequence.parse_action_dir(p.name)[1] is not None])
        c = contexts[tid]
        complete = bool(dirs) and dirs[-1][1]=='stop' and sum(a=='stop' for i,a in dirs)==c['level']
        consecutive = [i for i,a in dirs]==list(range(1,len(dirs)+1))
        valid = complete and consecutive and 1<=len(dirs)<=150
        ledger.append({'tid':tid,'level':c['level'],'scene':c['scene'],'actions':len(dirs),'STOP':sum(a=='stop' for i,a in dirs),'last_action':dirs[-1][1] if dirs else None,'complete':complete,'eligible':valid})
        if valid: eligible[tid]=dirs
    save('selection_ledger.json', ledger)
    cohort = {}
    used_scenes = set()
    for name, source_split in [('train','train'),('dev','val')]:
        pool = sorted(split[source_split], key=lambda tid: hashlib.sha256(json.dumps(key(contexts[tid]['task'])).encode()).hexdigest())
        chosen = []
        for level in (2,3):
            for tid in pool:
                c = contexts[tid]
                if c['level'] != level or c['scene'] in used_scenes or tid not in eligible:
                    continue
                dirs = eligible[tid]
                assert [i for i,a in dirs] == list(range(1,len(dirs)+1))
                chosen.append({'tid':tid, 'index':index_by_key[key(c['task'])], 'level':level, 'scene':c['scene'], 'instruction':c['instruction'], 'trial':str(trials[tid]), 'task_sha256':sha(trials[tid]/'task.json'), 'teacher_actions':[a for i,a in dirs]})
                used_scenes.add(c['scene'])
                if sum(r['level']==level for r in chosen)==2:
                    break
            assert sum(r['level']==level for r in chosen)==2
        cohort[name] = chosen
    assert not {r['scene'] for r in cohort['train']} & {r['scene'] for r in cohort['dev']}
    # The cohort is selected from source metadata before inspecting replay or policy scores.
    sources = [ordered_path,harmonized_path,canonical_path,causal_path,HERE/'sequence_trainer_original.py']
    dependencies = {str(p):sha(p) for p in sources}
    save('protocol_frozen.json', {'version':'pipeline_repair_20261013_v3', 'seed':SEED, 'cohort':cohort, 'canonical_excluded':100, 'selection':'SHA256 task key; two tasks per level 2/3 per arm; unique scenes; source action count <=150; consecutive frames; exactly K STOP and final action STOP', 'selection_ledger_sha256':sha(RUN/'selection_ledger.json'), 'parent_failed_gate':'pipeline_repair_20261013_v1 retained; two incomplete source demonstrations; no training', 'dependencies':dependencies, 'architecture':'historical 175-dim GRU128 window8; RGB statistics, static metadata hashes, absolute own pose, previous action', 'scope':'four-task train imitation gate and four-task dev diagnostic; no generalization claim', 'horizon':150, 'strict_STOP_m':1.0, 'resolver':'historical index-aware snapped AABB', 'early_stop':'three consecutive 100-step checks with all 398 train actions correct and STOP recall 100%', 'parent_protocol_sha256':sha(OLD/'pipeline_repair_20261013_v2/experiment/protocol_frozen.json'), 'training_steps_cap':4000, 'batch_size':128, 'learning_rate':0.0012, 'weight_power':0.45, 'stop_penalty':0.65, 'sampling_and_loss':'historical weighting retained; no STOP tuning', 'train_gate':'all four train chains must succeed', 'checkpoint_selection':'train classification only for bounded imitation gate; dev not used to select checkpoint', 'replay_gate':'all eight demonstrations valid STOP and max position error <=0.01m'})
    parent=json.loads((OLD/'pipeline_repair_20261013_v2/experiment/protocol_frozen.json').read_text())
    assert cohort==parent['cohort']
    save('self_check.json', checks)
    print('Frozen cohort', {k:[x['tid'] for x in v] for k,v in cohort.items()}, flush=True)
    source = inspect.getsource(ordered.evaluate_policy)
    old_yaw = 'yaw = float(sim.agent.get_state().rotation.angle())'
    assert source.count(old_yaw)==1
    namespace = dict(vars(ordered), signed_yaw=signed_yaw)
    exec(compile(source.replace(old_yaw,'yaw = signed_yaw(sim.agent.get_state().rotation)'),'<repaired-evaluator>','exec'),namespace)
    evaluate = namespace['evaluate_policy']
    starts = {}
    for rows in cohort.values():
        for row in rows:
            c = contexts[row['tid']]
            yaw = c['yaw'][0]
            starts[str(row['index'])] = {'position':c['pos'][0], 'rotation':[math.cos(yaw/2),0,math.sin(yaw/2),0]}
    save('start_states_frozen.json', starts)
    original_selector = namespace['choose_action']
    actions = {row['index']:row['teacher_actions'] for rows in cohort.values() for row in rows}
    def replay_selector(legacy,name,state,obs,pos,yaw,previous,buffer,context,mode,recent,positions,step):
        assert step < len(actions[context['index']])
        return legacy.ACTION_TO_IDX[actions[context['index']][step]], buffer
    namespace['choose_action'] = replay_selector
    replay_state = {'status':'loaded'}
    replay_rows = []
    null = open(os.devnull, 'w')
    for name, rows in cohort.items():
        with contextlib.redirect_stdout(null):
            records, summary = evaluate(legacy,dataset,'teacher_replay','raw',replay_state,[r['index'] for r in rows],starts,RUN/f'replay/{name}.jsonl',150,1.0,SEED,RUN/f'replay/{name}_progress.json',{})
        for row,t in zip(rows,records):
            c = contexts[row['tid']]
            errors = [float(np.linalg.norm(np.array(s['position_after'])-np.array(c['pos'][s['t']+1]))) for s in t.get('steps',[])]
            valid = t['status']=='completed' and t.get('full_chain_success',False) and len(t.get('actions',[]))==len(row['teacher_actions']) and max(errors,default=1)<=0.01
            replay_rows.append({'arm':name,'tid':row['tid'],'valid':valid,'max_position_error_m':max(errors,default=None),'successes':t.get('successes'),'error':t.get('error')})
        save(f'replay/{name}_summary.json',summary)
    save('replay/verification.json', {'passed':all(r['valid'] for r in replay_rows),'rows':replay_rows})
    print('Replay verification',replay_rows,flush=True)
    if not all(r['valid'] for r in replay_rows):
        save('status.json',{'phase':'replay_gate_failed','training_executed':False,'rows':replay_rows})
        return 2
    namespace['choose_action'] = original_selector
    save('status.json', {'phase':'causal_features'})
    xs,ys,tids,meta = causal.collect_causal_sequences(sequence)
    chosen = {r['tid'] for r in cohort['train']}
    kept = [i for i,tid in enumerate(tids) if tid in chosen]
    x = torch.tensor(np.asarray([xs[i] for i in kept],dtype=np.float32))
    y = torch.tensor([ys[i] for i in kept],dtype=torch.long)
    assert x.shape[1:] == (8,175) and len(set(tids[i] for i in kept))==4
    assert torch.isfinite(x).all() and set(y.tolist())==set(range(4))
    # Assert cached causal features against freshly read corrected frames on the selected tasks.
    # Fresh per-task checks avoid copying or mutating the source data tree.
    from repair_contract import before_action
    errors = []
    for row in cohort['train']:
        tid = row['tid']; c=contexts[tid]; ids=[i for i,t in enumerate(tids) if t==tid]
        previous=0
        for offset,i in enumerate(ids):
            frame,pose,yaw=before_action(trials[tid],offset+1,c['pos'],c['yaw'])
            fresh=sequence.frame_feature(frame,pose,yaw,previous,c)
            errors.append(float(np.max(np.abs(np.array(fresh)-np.array(xs[i][-1])))))
            previous=ys[i]
    assert max(errors)<1e-6
    save('training/data.json',{'source_cache':meta,'train_tids':sorted(chosen),'samples':len(y),'STOP':int((y==0).sum()),'fresh_feature_max_error':max(errors),'dev_in_training':False})
    del xs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(0.15)
    torch.manual_seed(SEED)
    model = sequence.SequenceGRU(175).to(device)
    x=x.to(device);y=y.to(device)
    counts=torch.bincount(y,minlength=4).float()
    weights=counts.clamp_min(1).pow(-0.45);weights[0]*=0.65;weights=weights/weights.sum()*4
    sample_weights=weights[y]
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0012,weight_decay=0.0001)
    consecutive=0
    save('status.json',{'phase':'training','device':str(device)})
    for step in range(1,4001):
        model.train()
        ids=torch.multinomial(sample_weights,min(128,len(y)),replacement=True)
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(x[ids]),y[ids]);loss.backward();optimizer.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():pred=model(x).argmax(1)
            accuracy=float((pred==y).float().mean());stop_recall=float((pred[y==0]==0).float().mean())
            save('training/progress.json',{'step':step,'accuracy':accuracy,'STOP_recall':stop_recall,'loss':float(loss)})
            print('train',step,accuracy,stop_recall,flush=True)
            consecutive=consecutive+1 if accuracy==1.0 and stop_recall==1.0 else 0
            if consecutive>=3:break
    save('training/offline_gate.json', {'steps':step,'accuracy':accuracy,'STOP_recall':stop_recall,'consecutive_perfect_checks':consecutive,'passed':consecutive>=3,'remaining_errors':int((pred!=y).sum())})
    checkpoint=RUN/'training/sequence_repaired_gate.pt'
    torch.save({'model_state_dict':best_snapshot(model),'input_dim':175,'window':8,'action_to_idx':sequence.ACTION_TO_IDX,'variant':'pipeline_repaired_train4_gate'},checkpoint)
    state={'status':'loaded','model':model.cpu().eval(),'feature':'seq','window':8}
    if device.type=='cuda':
        del x,y,optimizer,loss
        torch.cuda.empty_cache()
    metrics={}
    for name,rows in cohort.items():
        save('status.json',{'phase':name+'_rollout'})
        with contextlib.redirect_stdout(null):
            records,summary=evaluate(legacy,dataset,'sequence_repaired','raw',state,[r['index'] for r in rows],starts,RUN/f'rollouts/{name}.jsonl',150,1.0,SEED,RUN/f'rollouts/{name}_progress.json',{})
        assert len(records)==4 and all(t['status']=='completed' for t in records)
        k=sum(len(t['successes']) for t in records)
        metrics[name]={'chains':sum(t['full_chain_success'] for t in records),'episodes':4,'stages':sum(sum(t['successes']) for t in records),'stage_count':k,'ISR_micro':sum(sum(t['successes']) for t in records)/k,'ISR_macro':sum(sum(t['successes'])/len(t['successes']) for t in records)/4,'STOP':sum(t['stop_events'] for t in records),'no_STOP_episodes':sum(t['stop_events']==0 for t in records)}
        save(f'rollouts/{name}_summary.json',{'metrics':metrics[name],'historical_summary':summary})
        print(name,metrics[name],flush=True)
    assert all(sha(Path(p))==h for p,h in dependencies.items())
    save('status.json',{'phase':'completed','offline_gate_passed':consecutive>=3,'train_gate_passed':metrics['train']['chains']==4,'metrics':metrics,'checkpoint_sha256':sha(checkpoint),'dependencies_unchanged':True,'finished_utc':time.time()})
    null.close()


if __name__=='__main__':
    if RUN.exists():
        raise SystemExit('Existing experiment refuses overwrite; create a new version')
    try:
        raise SystemExit(main())
    except Exception as exc:
        if RUN.exists():save('status.json',{'phase':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
