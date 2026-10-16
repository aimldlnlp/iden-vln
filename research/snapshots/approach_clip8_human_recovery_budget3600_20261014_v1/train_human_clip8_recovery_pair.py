"""Matched BC versus BC+recovery fitting after human admission and cache gate."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import train_two_reviewed_il_overfit as il

ROOT=il.ROOT/'runs';CACHE=ROOT/'approach_clip8_human_recovery_cache_20261014_v1'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
OUT=ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def main():
    assert not OUT.exists(),'refuse overwrite'
    verification=json.loads((CACHE/'verification.json').read_text())
    assert verification['passed'] and verification['training_gate_passed'] and verification['exact_conflict_groups']==0
    assert il.review.sha(CACHE/'causal_features.npz')==verification['cache_sha256']
    with np.load(CACHE/'causal_features.npz') as d:
        raw=d['x'].copy();labels=d['y'].astype(np.int64);mean=d['mean'].copy();std=d['std'].copy()
    with np.load(PARENT/'causal_features.npz') as d:
        assert np.array_equal(raw[:1357],d['x']) and np.array_equal(labels[:1357],d['y'])
        assert np.array_equal(mean,d['mean']) and np.array_equal(std,d['std'])
    assert raw.shape==(2054,16,6767) and (std>=1e-3).all()
    OUT.mkdir()
    hashes={str(p):il.review.sha(p) for p in (CACHE/'causal_features.npz',CACHE/'verification.json',PARENT/'model_manifest_frozen.jsonl',PARENT/'goal_manifest_frozen.json',Path(il.TRAINER),Path(__file__))}
    save('protocol_frozen.json',{'version':'human_clip8_budget3600_pair_v1','source_hashes':hashes,
        'arms':{'BC':1357,'BC_recovery':2054},'unique_train_tasks':8,'accepted_dev':2,'full_dev_cohort':5,
        'seed':20260930,'updates':3600,'batch_size':128,'optimizer':'AdamW lr0.0012 weight_decay0.0001; clip5',
        'checkpoint_selection':'final update only; no dev selection','model':'existing GRU128 input6767 window16; frozen CLIP features',
        'normalization':'identical frozen original-bank mean/std','loss_weights':'identical original-bank inverse-action-frequency weights',
        'sampling':'same uniform draws mapped to each arm sample count; original+recovery uniform, no state selection by outcome',
        'evaluation_plan':'same8original starts,8reserved different-prefix/right60 train holdout branches,2dev tasks; scoring approach-v2 unchanged',
        'scope':'one-seed recovery pilot; human review terminal views only; not balanced100 or Table4',
        'budget_difference_from_archived':'new paired minibatch128 experiment; archived CLIP8 used full batches, so archived scores are context only'})
    torch.set_num_threads(2);torch.manual_seed(20260930)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    spec=importlib.util.spec_from_file_location('human_pair_base',il.TRAINER);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    model=base.SequenceGRU(6767).to(device);initial=copy.deepcopy(model.state_dict());initial_hash=state_hash(initial)
    baseline=json.loads((ROOT/"approach_clip8_human_recovery_pair_20261014_v1/training_summary.json").read_text())
    assert all(m["initial_state_sha256"]==initial_hash for m in baseline.values()), "initializer differs from1200"
    rng=np.random.default_rng(20260930);draws=rng.random((3600,128));draw_hash=hashlib.sha256(draws.tobytes()).hexdigest()
    prefix_hash=hashlib.sha256(draws[:1200].tobytes()).hexdigest()
    assert all(m["uniform_draws_sha256"]==prefix_hash for m in baseline.values()), "sampling prefix differs from1200"
    counts=np.bincount(labels[:1357],minlength=4);weights=1357/(4*np.maximum(counts,1));weights=weights/weights.mean()
    weights=torch.tensor(weights,dtype=torch.float32,device=device)
    # shortcut: CPU cache avoids simultaneous full GPU windows; larger cohorts need streaming.
    normalized=torch.from_numpy((raw-mean)/std);target=torch.from_numpy(labels)
    summaries={}
    for arm,n in (('BC',1357),('BC_recovery',2054)):
        model.load_state_dict(initial);assert state_hash(model.state_dict())==initial_hash
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.0001)
        curve=[]
        for step in range(3600):
            ids=torch.from_numpy(np.floor(draws[step]*n).astype(np.int64))
            model.train();optimizer.zero_grad();logits=model(normalized[ids].to(device))
            loss=torch.nn.functional.cross_entropy(logits,target[ids].to(device),weight=weights)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if (step+1)%100==0:
                curve.append({'step':step+1,'batch_loss':float(loss.detach())});save(arm+'_progress.json',curve)
                print(json.dumps({'arm':arm,**curve[-1]}),flush=True)
        model.eval();correct=stop_correct=stop_count=0
        with torch.inference_mode():
            for at in range(0,n,128):
                end=min(at+128,n)
                pred=model(normalized[at:end].to(device)).argmax(-1).cpu();y=target[at:end]
                mask=y==base.ACTION_TO_IDX['stop'];correct+=int((pred==y).sum());stop_correct+=int(((pred==y)&mask).sum());stop_count+=int(mask.sum())
        path=OUT/(arm+'.pt')
        torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'input_dim':6767,'window':16,
            'mean':mean.tolist(),'std':std.tolist(),'protocol_sha256':il.review.sha(OUT/'protocol_frozen.json')},path)
        summaries[arm]={'samples':n,'updates':3600,'offline_accuracy':correct/n,'STOP_recall':stop_correct/stop_count,
            'initial_state_sha256':initial_hash,'uniform_draws_sha256':draw_hash,'checkpoint_sha256':il.review.sha(path)}
        save('training_summary.json',summaries)
    assert all(il.review.sha(Path(p))==h for p,h in hashes.items())
    save('fitting_verification.json',{'passed':True,'arms':summaries,'sources_unchanged':True,'same_initialization':True,
        'same_normalization':True,'same_loss_weights':True,'same_uniform_draws':True,'holdout_or_dev_used_for_training':False,
        'policy_evaluation_completed':False})
    save('status.json',{'phase':'fitting_completed','evaluation_pending':True})
if __name__=='__main__':main()
