"""Matched BC versus BC+recovery fitting after human admission and cache gate."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import train_two_reviewed_il_overfit as il

ROOT=il.ROOT/'runs';CACHE=ROOT/'approach_clip8_human_recovery_cache_20261014_v1'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
OUT=ROOT/'approach_clip8_human_recovery_lr_drop_20261014_v1/experiment'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def fixed_set_diagnostics(model, normalized, target, weights, device):
    # Aggregate weighted numerators and denominators, not minibatch means.
    was_training=model.training;model.eval();result={}
    with torch.inference_mode():
        for name,start,end in (('original',0,1357),('recovery',1357,2054)):
            numerator=denominator=unweighted=0.;confusion=torch.zeros((4,4),dtype=torch.int64)
            for at in range(start,end,128):
                stop=min(at+128,end);y=target[at:stop].to(device)
                logits=model(normalized[at:stop].to(device))
                losses=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                numerator+=float((losses*weights[y]).sum());denominator+=float(weights[y].sum())
                unweighted+=float(losses.sum());pred=logits.argmax(-1).cpu()
                confusion+=torch.bincount(target[at:stop]*4+pred,minlength=16).reshape(4,4)
            result[name]={'weighted_loss':numerator/denominator,'unweighted_loss':unweighted/(end-start),
                'weighted_numerator':numerator,'weight_denominator':denominator,'samples':end-start,
                'accuracy':float(confusion.diag().sum())/(end-start),'confusion_true_rows_pred_columns':confusion.tolist(),
                'STOP_false_positive_teacher_label':int(confusion[1:,0].sum()),
                'STOP_false_negative_teacher_label':int(confusion[0,1:].sum()),'action_order':['stop','turn_left','move_forward','turn_right']}
    model.train(was_training);return result

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
    save('protocol_frozen.json',{'version':'human_clip8_lr_drop_pair_v1','source_hashes':hashes,
        'arms':{'BC':1357,'BC_recovery':2054},'unique_train_tasks':8,'accepted_dev':2,'full_dev_cohort':5,
        'seed':20260930,'updates':3600,'batch_size':128,'optimizer':'AdamW lr0.0012 updates1-1200; lr0.00012 updates1201-3600; weight_decay0.0001; clip5',
        'lr_schedule':{'switch_after_updates':1200,'initial':0.0012,'after':0.00012},
        'diagnostics':'fixed original/recovery sets at1200,2400,3600; gradients and clipping per100 updates; no checkpoint selection',
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
        curve=[];fixed_curve=[];gradient_norms=[];clipped=0
        for step in range(3600):
            if step==1200:
                for group in optimizer.param_groups:group['lr']=.00012
            ids=torch.from_numpy(np.floor(draws[step]*n).astype(np.int64))
            model.train();optimizer.zero_grad();logits=model(normalized[ids].to(device))
            loss=torch.nn.functional.cross_entropy(logits,target[ids].to(device),weight=weights)
            loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            assert torch.isfinite(loss) and torch.isfinite(gradient_norm), 'nonfinite optimization'
            gradient_norms.append(float(gradient_norm));clipped+=int(float(gradient_norm)>5.)
            if (step+1)%100==0:before=[p.detach().clone() for p in model.parameters()]
            optimizer.step()
            if (step+1)%100==0:
                delta_norm=float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(model.parameters(),before))))
                parameter_norm=float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters())))
                curve.append({'step':step+1,'batch_loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],
                    'gradient_norm_mean_last100':float(np.mean(gradient_norms)),'gradient_norm_max_last100':max(gradient_norms),
                    'clipping_fraction_last100':clipped/100,'sampled_update_norm':delta_norm,'parameter_norm':parameter_norm,
                    'batch_label_counts':torch.bincount(target[ids],minlength=4).tolist(),
                    'batch_weight_denominator':float(weights[target[ids].to(device)].sum())});save(arm+'_progress.json',curve)
                gradient_norms=[];clipped=0
                print(json.dumps({'arm':arm,**curve[-1]}),flush=True)
            if (step+1) in (1200,2400,3600):
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
                fixed_curve.append({'step':step+1,'sets':fixed_set_diagnostics(model,normalized,target,weights,device)})
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                save(arm+'_fixed_set_diagnostics.json',fixed_curve)
                if step+1==1200:
                    archived=json.loads((ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'/(arm+'_progress.json')).read_text())
                    assert all(c['batch_loss']==a['batch_loss'] for c,a in zip(curve,archived[:12])), 'training prefix differs'
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
