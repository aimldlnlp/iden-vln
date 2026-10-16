"""Causal STOP count from executed actions; no scorer or stage input."""
import json
import numpy as np


def window_counter(actions, t):
    assert 0 <= t < len(actions)
    before=[];count=0
    for item in actions[:t+1]:
        before.append(count)
        count+=int(item['action']=='stop')
    values=before[max(0,t-15):t+1]
    return [0]*(16-len(values))+values


def build_counters(parent, collection, cache):
    originals=list(map(json.loads,(parent/'model_manifest_frozen.jsonl').read_text().splitlines()))
    recovery=list(map(json.loads,(collection/'model_manifest_human_reviewed_frozen.jsonl').read_text().splitlines()))
    sequences={r['index']:r['teacher_actions'] for r in originals if r['split']=='train'}
    sequences.update({r['index']:r['teacher_actions'] for r in recovery if r.get('augmentation')=='clip8_recovery_v2'})
    refs=[{'index':r['index'],'t':t,'stage':a['stage'],'action':a['action']} for r in originals if r['split']=='train' for t,a in enumerate(r['teacher_actions'])]
    assert len(refs)==1357
    refs+=json.loads((cache/'recovery_sample_refs.json').read_text())
    assert len(refs)==2054
    values=[]
    for ref in refs:
        actions=sequences[ref['index']];t=ref['t']
        assert actions[t]['action']==ref['action']
        value=window_counter(actions,t)
        # This checks metadata; the feature itself uses only prior actions.
        assert value[-1]==ref['stage']
        values.append(value)
    return np.asarray(values,dtype=np.float32)[...,None],refs


def check():
    actions=[{'action':a} for a in ['move_forward','stop','turn_left']+['move_forward']*18+['stop']]
    assert window_counter(actions,1)[-1]==0
    assert window_counter(actions,2)[-1]==1
    assert window_counter(actions,20)==[1]*16
    assert window_counter(actions,0)==[0]*16
    changed=[{**a,'valid':False,'stage':99} for a in actions]
    assert window_counter(changed,20)==window_counter(actions,20)
    changed[21]['action']='turn_right'
    assert window_counter(changed,20)==window_counter(actions,20)
    assert window_counter(actions,21)[-1]==1
    print('counter causal, padding, long-history, and scorer-independence checks passed')


if __name__=='__main__':check()
