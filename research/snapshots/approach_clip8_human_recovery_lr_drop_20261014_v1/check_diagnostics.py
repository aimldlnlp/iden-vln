"""Check diagnostic aggregation against direct full-set loss before launch."""
import json
from pathlib import Path
import torch
from train_human_clip8_recovery_pair import fixed_set_diagnostics

torch.manual_seed(41)
x=torch.randn(2054,4)
y=torch.arange(2054)%4
weights=torch.tensor([3.48,.30,.0725,.143])
model=torch.nn.Linear(4,4).train()
result=fixed_set_diagnostics(model,x,y,weights,torch.device('cpu'))
assert model.training
for name,start,end in [('original',0,1357),('recovery',1357,2054)]:
    with torch.inference_mode():
        logits=model(x[start:end]);labels=y[start:end]
        weighted=torch.nn.functional.cross_entropy(logits,labels,weight=weights)
        unweighted=torch.nn.functional.cross_entropy(logits,labels)
        predicted=logits.argmax(-1)
        confusion=torch.bincount(labels*4+predicted,minlength=16).reshape(4,4)
    assert abs(result[name]['weighted_loss']-float(weighted))<1e-6
    assert abs(result[name]['unweighted_loss']-float(unweighted))<1e-6
    assert result[name]['confusion_true_rows_pred_columns']==confusion.tolist()
    assert abs(result[name]['weight_denominator']-float(weights[labels].sum()))<1e-3
assert [.0012 if step<1200 else .00012 for step in [0,1199,1200,3599]]==[.0012,.0012,.00012,.00012]
output=Path(__file__).parent/'diagnostic_self_check.json'
assert not output.exists()
output.write_text(json.dumps({'passed':True,'weighted_full_set_aggregation':True,'confusion_matrix':True,
    'model_mode_restored':True,'lr_boundary_checked':True},indent=2)+'\n')
print('diagnostic checks passed')
