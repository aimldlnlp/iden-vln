"""Read-only checkpoint/cache diagnosis; no fitting, action intervention or rollout."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import train_two_reviewed_il_overfit as il

ROOT=il.ROOT/'runs';PAIR=ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'
CACHE=ROOT/'approach_clip8_human_recovery_cache_20261014_v1'
PARENT=ROOT/'approach_matched_il_20261011_v1/bank8'
COLLECTION=ROOT/'approach_clip8_recovery_collection_20261013_v2'
OUT=PAIR/'diagnosis'
def main():
    assert not OUT.exists(),'refuse overwrite'
    assert json.loads((PAIR/'final_verification.json').read_text())['passed']
    paths=[PAIR/(a+'.pt') for a in ('BC','BC_recovery')]+[CACHE/'causal_features.npz',PARENT/'model_manifest_frozen.jsonl',PAIR/'evaluation_rollouts.json']
    hashes={str(p):il.review.sha(p) for p in paths}
    with np.load(CACHE/'causal_features.npz') as d:
        raw=d['x'].copy();labels=d['y'].astype(np.int64);tids=d['tids'].copy();mean=d['mean'].copy();std=d['std'].copy()
    rows=list(map(json.loads,(PARENT/'model_manifest_frozen.jsonl').read_text().splitlines()))
    refs=[{'index':r['index'],'t':t,'stage':a['stage'],'action':a['action']} for r in rows if r['split']=='train' for t,a in enumerate(r['teacher_actions'])]
    assert len(refs)==1357
    refs+=json.loads((CACHE/'recovery_sample_refs.json').read_text())
    assert len(refs)==2054 and all(int(tids[i])==r['index'] for i,r in enumerate(refs))
    torch.set_num_threads(2);device=torch.device('cpu')
    spec=importlib.util.spec_from_file_location('regression_base',il.TRAINER);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    x=torch.from_numpy((raw-mean)/std)
    findings={};predictions={};logits={}
    for arm in ('BC','BC_recovery'):
        cp=torch.load(PAIR/(arm+'.pt'),map_location='cpu')
        assert np.array_equal(np.asarray(cp['mean'],dtype=np.float32),mean)
        model=base.SequenceGRU(6767);model.load_state_dict(cp['state_dict']);model.eval()
        with torch.inference_mode():logit=torch.cat([model(x[at:at+128]) for at in range(0,len(x),128)])
        pred=logit.argmax(-1).numpy();prob=logit.softmax(-1).numpy()
        predictions[arm]=pred;logits[arm]=prob
        findings[arm]={}
        for name,start,end in (('original',0,1357),('recovery',1357,2054)):
            y=labels[start:end];p=pred[start:end];stop=base.ACTION_TO_IDX['stop']
            findings[arm][name]={'samples':end-start,'accuracy':float(np.mean(y==p)),'errors':int(sum(y!=p)),
                'confusion_true_by_pred':[[int(sum((y==i)&(p==j))) for j in range(4)] for i in range(4)],
                'STOP_false_positives':int(sum((p==stop)&(y!=stop))),'STOP_false_negatives':int(sum((p!=stop)&(y==stop)))}
    draws=np.random.default_rng(20260930).random((3600,128));exposure={}
    for arm,n in (('BC',1357),('BC_recovery',2054)):
        ids=np.floor(draws*n).astype(np.int64);counts=np.bincount(ids.ravel(),minlength=n)
        exposure[arm]={'sample_draws':int(ids.size),'original_draws':int(sum(ids.ravel()<1357)),
            'recovery_draws':int(sum(ids.ravel()>=1357)),'mean_presentations_per_label':float(counts.mean()),
            'original_mean_presentations':float(counts[:1357].mean()),'min_label_presentations':int(counts.min()),'max_label_presentations':int(counts.max())}
    count_a=np.bincount(labels[:1357],minlength=4);count_b=np.bincount(labels,minlength=4)
    weights=1357/(4*np.maximum(count_a,1));weights/=weights.mean()
    mass_a=count_a*weights;mass_b=count_b*weights
    action_mix={'action_to_index':base.ACTION_TO_IDX,'original_counts':count_a.tolist(),'combined_counts':count_b.tolist(),
        'fixed_weights':weights.tolist(),'BC_weighted_label_mass':(mass_a/mass_a.sum()).tolist(),
        'recovery_weighted_label_mass':(mass_b/mass_b.sum()).tolist(),
        'note':'expected corpus weighted mass; not exact stochastic minibatch loss contribution'}
    rollouts=json.loads((PAIR/'evaluation_rollouts.json').read_text());divergences=[]
    for row in rows:
        if row['split']!='train':continue
        idx=row['index'];teacher=row['teacher_actions'];offset=next(i for i,r in enumerate(refs) if r['index']==idx)
        for arm in ('BC','BC_recovery'):
            trace=next(r['actions'] for r in rollouts if r['cohort']=='train_original' and r['index']==idx and r['arm']==arm)
            mismatch=next((t for t,(a,b) in enumerate(zip(trace,teacher)) if a['action']!=b['action']),None)
            item={'index':idx,'arm':arm,'first_action_difference':mismatch,'lengths':[len(trace),len(teacher)]}
            if mismatch is not None:
                t=mismatch;i=offset+t
                item.update(stage=teacher[t]['stage'],teacher_action=teacher[t]['action'],rollout_action=trace[t]['action'],
                    cached_prediction=base.IDX_TO_ACTION[int(predictions[arm][i])],
                    cached_probability=logits[arm][i].tolist(),
                    prefix_position_error=max((float(np.linalg.norm(np.asarray(trace[j]['position'])-np.asarray(teacher[j]['position']))) for j in range(t)),default=0.))
            divergences.append(item)
    recovery=list(map(json.loads,(COLLECTION/'model_manifest_human_reviewed_frozen.jsonl').read_text().splitlines()))
    stop_history=[]
    for row in recovery:
        if row.get('augmentation')!='clip8_recovery_v2':continue
        for t,a in enumerate(row['teacher_actions']):
            if a['action']=='stop':
                window=row['teacher_actions'][max(0,t-15):t+1]
                stop_history.append({'index':row['index'],'stage':a['stage'],'window_actions':len(window),
                    'masked_search_in_window':sum(v['kind']=='teacher_view_search' for v in window),
                    'history_kinds':[v['kind'] for v in window]})
    assert all(il.review.sha(Path(p))==h for p,h in hashes.items())
    OUT.mkdir()
    result={'source_hashes':hashes,'common_set_prediction_metrics':findings,'sampling_exposure':exposure,
        'archived_full_batch_original_draws':1357*1200,'action_mix':action_mix,'first_divergences':divergences,
        'recovery_STOP_history':stop_history,'no_training_or_rollout_executed':True,'script_sha256':il.review.sha(Path(__file__))}
    (OUT/'diagnostic_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('common_set_prediction_metrics','sampling_exposure','action_mix')}),flush=True)
if __name__=='__main__':main()
