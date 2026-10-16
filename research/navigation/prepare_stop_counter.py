"""Prepare isolated counter-only ablation from the verified LR-drop pipeline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD='approach_clip8_human_recovery_lr_drop_20261014_v1'
NEW='approach_clip8_stop_counter_20261014_v1'
SOURCE=ROOT/OLD;OUT=ROOT/NEW


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def one(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)


def main():
    assert not (OUT/'train_human_clip8_recovery_pair.py').exists() and not (OUT/'contract_frozen.json').exists()
    names=['train_human_clip8_recovery_pair.py','evaluate_human_clip8_recovery_pair.py',
        'verify_human_clip8_recovery_pair.py','audit_recovery_regression_research.py','verify_download.py','launch.sh']
    originals={n:sha(SOURCE/n) for n in names};generated={}
    for name in names:
        s=(SOURCE/name).read_text().replace(OLD,NEW)
        if name.startswith('train_'):
            s=one(s,'import train_two_reviewed_il_overfit as il','import train_two_reviewed_il_overfit as il\nfrom counter_support import build_counters')
            s=one(s,'OUT.mkdir()',"OUT.mkdir()\n    collection=ROOT/'approach_clip8_recovery_collection_20261013_v2'\n    counters,refs=build_counters(PARENT,collection,CACHE)\n    assert counters.shape==(2054,16,1) and set(np.unique(counters))=={0.,1.}\n    with np.load(CACHE/'causal_features.npz') as d:assert all(int(d['tids'][i])==r['index'] for i,r in enumerate(refs))\n    np.savez_compressed(OUT/'counter_windows.npz',counter=counters)\n    save('counter_cache_verification.json',{'passed':True,'samples':2054,'source':'prior executed STOP actions only','no_scorer_input':True,'all_stage_metadata_checks_passed':True})")
            s=one(s,"PARENT/'goal_manifest_frozen.json',Path(il.TRAINER),Path(__file__))", "PARENT/'goal_manifest_frozen.json',collection/'model_manifest_human_reviewed_frozen.jsonl',CACHE/'recovery_sample_refs.json',Path(__file__).with_name('counter_support.py'),Path(il.TRAINER),Path(__file__))")
            s=s.replace('human_clip8_lr_drop_pair_v1','human_clip8_counter_pair_v1')
            s=one(s,"'arms':{'BC':1357,'BC_recovery':2054}","'arms':{'counter_zero':2054,'counter_active':2054},'counter':'prior STOP count on every frame; zero control; no successful-stage feedback'")
            s=s.replace('existing GRU128 input6767 window16','existing GRU128 input6768 window16')
            s=one(s,"'normalization':'identical frozen original-bank mean/std'", "'normalization':'legacy6767 original-bank mean/std unchanged; counter mean0 std1'")
            s=one(s,'model=base.SequenceGRU(6767).to(device);initial=copy.deepcopy(model.state_dict());initial_hash=state_hash(initial)',
                "legacy=base.SequenceGRU(6767).to(device);legacy_state=copy.deepcopy(legacy.state_dict());legacy_hash=state_hash(legacy_state)\n    model=base.SequenceGRU(6768).to(device)\n    initial=copy.deepcopy(legacy_state)\n    initial['gru.weight_ih_l0']=torch.cat([initial['gru.weight_ih_l0'],torch.zeros((384,1),device=device)],dim=1)\n    model.load_state_dict(initial);initial_hash=state_hash(initial)")
            s=one(s,'m["initial_state_sha256"]==initial_hash','m["initial_state_sha256"]==legacy_hash')
            s=one(s,'normalized=torch.from_numpy((raw-mean)/std);target=torch.from_numpy(labels)',
                "legacy_normalized=torch.from_numpy((raw-mean)/std);target=torch.from_numpy(labels)\n    mean=np.concatenate([mean,np.zeros(1,dtype=np.float32)]);std=np.concatenate([std,np.ones(1,dtype=np.float32)])")
            s=one(s,"for arm,n in (('BC',1357),('BC_recovery',2054)):",
                "for arm,n in (('counter_zero',2054),('counter_active',2054)):\n        normalized=torch.cat([legacy_normalized,torch.from_numpy(counters if arm=='counter_active' else np.zeros_like(counters))],dim=-1)")
            start=s.index('                if step+1==1200:')
            end=s.index('        model.eval();correct=',start)
            s=s[:start]+s[end:]
            s=one(s,"'input_dim':6767,'window':16", "'input_dim':6768,'window':16,'counter_active':arm=='counter_active'")
            s=one(s,"'same_uniform_draws':True", "'same_uniform_draws':True,'same_training_samples':True,'counter_input_is_only_arm_difference':True")
        elif name.startswith('evaluate_'):
            s=one(s,'model=module.SequenceGRU(6767).to(device)','model=module.SequenceGRU(6768).to(device)')
            s=one(s,'sim.save_path=None;trace=[];stops=[];history=[];previous=-1;prefix_error=None',
                "sim.save_path=None;trace=[];stops=[];history=[];previous=-1;prefix_error=None;own_stops=0\n            active=arm=='counter_active'")
            s=one(s,'nonlocal previous','nonlocal previous,own_stops')
            s=one(s,"'pre_geo_m':geo if math.isfinite(geo) else None,'kind':kind}",
                "'pre_geo_m':geo if math.isfinite(geo) else None,'kind':kind,'own_STOP_count_before':own_stops,'counter_feature':float(own_stops) if active else 0.}")
            s=one(s,"previous=module.ACTION_TO_IDX[action]", "previous=module.ACTION_TO_IDX[action]\n                    own_stops+=int(action=='stop')")
            s=one(s,'                if case[\'prefix\']:',
                "                def live():\n                    return il.feature(module,sim,start,yaw,previous,il.context(row))+[float(own_stops) if active else 0.]\n                if case['prefix']:")
            s=one(s,"history.append(cached[t,-1].tolist());act(item['action'],'teacher_prefix')", "history.append(cached[t,-1].tolist()+[0.]);act(item['action'],'teacher_prefix')")
            s=s.replace('history.append(il.feature(module,sim,start,yaw,previous,il.context(row)))','history.append(live())')
            s=one(s,"Path(__file__)]+[OUT/(a+'.pt')", "Path(__file__),Path(__file__).with_name('counter_support.py')]+[OUT/(a+'.pt')")
        elif name.startswith('verify_human_'):
            s=s.replace("'BC_recovery'","'counter_active'").replace("'BC'","'counter_zero'")
            s=one(s,"actual=[a for a in r['actions'] if a['action']=='stop']", "count=0\n                for event in r['actions']:\n                    assert event['own_STOP_count_before']==count\n                    assert event['counter_feature']==(float(count) if arm=='counter_active' else 0.)\n                    count+=int(event['action']=='stop')\n                actual=[a for a in r['actions'] if a['action']=='stop']")
        elif name.startswith('audit_'):
            s=one(s,'import train_two_reviewed_il_overfit as il','import train_two_reviewed_il_overfit as il\nfrom counter_support import build_counters')
            s=one(s,'x=torch.from_numpy((raw-mean)/std)',"legacy_x=torch.from_numpy((raw-mean)/std)\n    counters,_=build_counters(PARENT,COLLECTION,CACHE)")
            assert s.count("for arm in ('BC','BC_recovery'):")==2
            s=s.replace("for arm in ('BC','BC_recovery'):","for arm in ('counter_zero','counter_active'):")
            s=one(s,"np.asarray(cp['mean'],dtype=np.float32),mean", "np.asarray(cp['mean'][:-1],dtype=np.float32),mean")
            s=one(s,"assert np.array_equal(np.asarray(cp['mean'][:-1],dtype=np.float32),mean)", "assert np.array_equal(np.asarray(cp['mean'][:-1],dtype=np.float32),mean)\n        assert np.array_equal(np.asarray(cp['std'][:-1],dtype=np.float32),std) and cp['mean'][-1]==0 and cp['std'][-1]==1")
            s=one(s,'model=base.SequenceGRU(6767);',"x=torch.cat([legacy_x,torch.from_numpy(counters if arm=='counter_active' else np.zeros_like(counters))],dim=-1)\n        model=base.SequenceGRU(6768);")
            s=s.replace("('BC',1357),('BC_recovery',2054)","('counter_zero',2054),('counter_active',2054)")
            s=s.replace("('BC','BC_recovery')","('counter_zero','counter_active')")
            s=s.replace("'BC_weighted_label_mass'","'original_corpus_weighted_label_mass'")
        elif name=='verify_download.py':
            start=s.index("    lr=read(HERE/'lr_comparison.json')")
            end=s.index('    result=',start)
            s=s[:start]+"    comparison=read(HERE/'counter_comparison.json');assert comparison['verified']\n    assert comparison['contract_sha256']==sha(HERE/'contract_frozen.json')\n"+s[end:]
        elif name=='launch.sh':
            s=s.replace('budget_contract.py --finish','counter_contract.py --finish')
        if name.endswith('.py'):ast.parse(s)
        generated[name]=s
    for name,s in generated.items():(OUT/name).write_text(s,encoding='utf-8',newline='\n')
    assert all(sha(SOURCE/n)==h for n,h in originals.items())
    (OUT/'preparation.json').write_text(json.dumps({'source_run':OLD,'original_scripts_sha256':originals,
        'generated_scripts_sha256':{n:sha(OUT/n) for n in names},'intervention':'counter zero vs active; both use2054 BC+recovery samples; same6768 architecture and mapped common initializer'},indent=2)+'\n')
    print('prepared',OUT)


if __name__=='__main__':main()
