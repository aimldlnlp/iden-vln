"""Check dataset causal counts and zero-column model mapping before launch."""
import copy
import importlib.util
import json
from pathlib import Path
import torch
from counter_support import build_counters,check
import train_two_reviewed_il_overfit as il

check()
root=il.ROOT/'runs'
counters,refs=build_counters(root/'approach_matched_il_20261011_v1/bank8',
    root/'approach_clip8_recovery_collection_20261013_v2',root/'approach_clip8_human_recovery_cache_20261014_v1')
assert counters.shape==(2054,16,1)
spec=importlib.util.spec_from_file_location('counter_check_base',il.TRAINER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
torch.set_num_threads(2);torch.manual_seed(20260930)
legacy=base.SequenceGRU(6767).eval();state=copy.deepcopy(legacy.state_dict())
model=base.SequenceGRU(6768).eval()
state['gru.weight_ih_l0']=torch.cat([state['gru.weight_ih_l0'],torch.zeros((384,1))],dim=1)
model.load_state_dict(state)
x=torch.randn(3,16,6767);zero=torch.zeros((3,16,1));active=torch.ones_like(zero)
with torch.inference_mode():
    old=legacy(x);control=model(torch.cat([x,zero],dim=-1));treatment=model(torch.cat([x,active],dim=-1))
assert torch.allclose(old,control,atol=1e-6,rtol=1e-6)
assert torch.equal(control,treatment)
model.train();model.zero_grad();model(torch.cat([x,active],dim=-1)).sum().backward()
assert model.gru.weight_ih_l0.grad[:,-1].abs().sum()>0
out=Path(__file__).parent/'counter_self_check.json'
assert not out.exists()
out.write_text(json.dumps({'passed':True,'counter_samples':len(refs),'causal_counter_and_padding_checked':True,
    'initial_model_mapping_checked':True,'active_zero_initial_predictions_equal':True,'counter_channel_can_learn':True},indent=2)+'\n')
print('counter/model checks passed')
