"""One-action intervention on the archived v2 policy; diagnostic, not learned score."""
import contextlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_repair_pilot as pilot
from repair_contract import signed_yaw


def main():
    import torch
    torch.set_num_threads(2)
    parent=HERE.parent/'pipeline_repair_20261013_v2/experiment'
    out=HERE/'single_action_intervention'
    out.mkdir(exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    baseline=next(json.loads(l) for l in (parent/'rollouts/train.jsonl').read_text().splitlines() if json.loads(l)['index']==168)
    checkpoint=parent/'training/sequence_repaired_gate.pt'
    starts=json.loads((parent/'start_states_frozen.json').read_text())
    save('protocol_frozen.json',{'checkpoint_sha256':pilot.sha(checkpoint),'baseline_trace_sha256':pilot.sha(parent/'rollouts/train.jsonl'),'index':168,'intervention_zero_based_step':38,'forced_action':'move_forward','all_other_actions':'archived v2 policy','start_state':starts['168'],'horizon':150,'strict_STOP_m':1.0,'scope':'post-hoc diagnostic intervention, excluded from learned performance'})
    sys.path.insert(0,str(pilot.ROOT));os.chdir(pilot.ROOT)
    ordered=pilot.load(pilot.OLD/'idenvln_ordered_eval_20260804_234338/run_ordered_metrics_eval.py','intervention_ordered')
    legacy=ordered.load_legacy_module(pilot.ROOT);ordered.install_index_aware_target_patch()
    from utils.dataset import TaskDataset
    dataset=TaskDataset(legacy.make_args(),'test')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=legacy.SequenceGRU(175);model.load_state_dict(ck['model_state_dict']);model.eval()
    state={'status':'loaded','model':model,'feature':'seq','window':8}
    source=inspect.getsource(ordered.evaluate_policy)
    old='yaw = float(sim.agent.get_state().rotation.angle())'
    assert source.count(old)==1
    namespace=dict(vars(ordered),signed_yaw=signed_yaw)
    exec(compile(source.replace(old,'yaw = signed_yaw(sim.agent.get_state().rotation)'),'<intervention-evaluator>','exec'),namespace)
    original=namespace['choose_action'];interventions=[]
    def choose(*args):
        prediction,buffer=original(*args)
        if args[-1]==38:
            interventions.append({'step':38,'original_action':legacy.IDX_TO_ACTION[prediction],'forced_action':'move_forward'})
            return legacy.ACTION_TO_IDX['move_forward'],buffer
        return prediction,buffer
    namespace['choose_action']=choose
    with open(os.devnull,'w') as null,contextlib.redirect_stdout(null):
        rows,summary=namespace['evaluate_policy'](legacy,dataset,'v2_single_action_intervention','raw',state,[168],starts,out/'trace.jsonl',150,1.,57007,out/'progress.json',{})
    assert len(rows)==1 and rows[0]['status']=='completed' and len(interventions)==1
    t=rows[0]
    assert interventions[0]['original_action']=='turn_left'
    for a,b in zip(baseline['steps'][:38],t['steps'][:38]):
        assert a['action']==b['action'] and math.dist(a['position_before'],b['position_before'])<1e-5
    result={'baseline_successes':baseline['successes'],'intervention_successes':t['successes'],'baseline_chain_success':baseline['full_chain_success'],'intervention_chain_success':t['full_chain_success'],'interventions':interventions,'prefix_verified':True,'checkpoint_unchanged':pilot.sha(checkpoint)==json.loads((out/'protocol_frozen.json').read_text())['checkpoint_sha256'],'STOP_distances':[s['distances_after'][s['active_target']] for s in t['steps'] if s['stop']],'not_a_learned_score':True}
    assert result['checkpoint_unchanged']
    save('result.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
