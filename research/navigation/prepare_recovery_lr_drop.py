"""Create isolated LR-schedule copies, retaining the archived experiment."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = 'approach_clip8_human_recovery_budget3600_20261014_v1'
NEW = 'approach_clip8_human_recovery_lr_drop_20261014_v1'
SOURCE = ROOT / OLD
OUT = ROOT / NEW


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


DIAGNOSTICS = '''
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

'''


def main():
    assert not OUT.exists(), 'refuse overwrite'
    names = ['train_human_clip8_recovery_pair.py', 'evaluate_human_clip8_recovery_pair.py',
             'verify_human_clip8_recovery_pair.py', 'audit_recovery_regression_research.py',
             'budget_contract.py', 'launch.sh', 'verify_download.py']
    hashes = {name: sha(SOURCE / name) for name in names}
    generated = {}
    for name in names:
        text = (SOURCE / name).read_text(encoding='utf-8').replace(OLD, NEW)
        if name.startswith('train_'):
            text = replace_once(text, 'def main():', DIAGNOSTICS + 'def main():')
            text = text.replace('human_clip8_budget3600_pair_v1', 'human_clip8_lr_drop_pair_v1')
            text = replace_once(text, "'optimizer':'AdamW lr0.0012 weight_decay0.0001; clip5',",
                "'optimizer':'AdamW lr0.0012 updates1-1200; lr0.00012 updates1201-3600; weight_decay0.0001; clip5',\n        'lr_schedule':{'switch_after_updates':1200,'initial':0.0012,'after':0.00012},\n        'diagnostics':'fixed original/recovery sets at1200,2400,3600; gradients and clipping per100 updates; no checkpoint selection',")
            text = replace_once(text, 'curve=[]', 'curve=[];fixed_curve=[];gradient_norms=[];clipped=0')
            text = replace_once(text, 'ids=torch.from_numpy(np.floor(draws[step]*n).astype(np.int64))',
                "if step==1200:\n                for group in optimizer.param_groups:group['lr']=.00012\n            ids=torch.from_numpy(np.floor(draws[step]*n).astype(np.int64))")
            text = replace_once(text, 'loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()',
                "loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)\n            assert torch.isfinite(loss) and torch.isfinite(gradient_norm), 'nonfinite optimization'\n            gradient_norms.append(float(gradient_norm));clipped+=int(float(gradient_norm)>5.)\n            if (step+1)%100==0:before=[p.detach().clone() for p in model.parameters()]\n            optimizer.step()")
            text = replace_once(text, "curve.append({'step':step+1,'batch_loss':float(loss.detach())});save(arm+'_progress.json',curve)",
                "delta_norm=float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(model.parameters(),before))))\n                parameter_norm=float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters())))\n                curve.append({'step':step+1,'batch_loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],\n                    'gradient_norm_mean_last100':float(np.mean(gradient_norms)),'gradient_norm_max_last100':max(gradient_norms),\n                    'clipping_fraction_last100':clipped/100,'sampled_update_norm':delta_norm,'parameter_norm':parameter_norm,\n                    'batch_label_counts':torch.bincount(target[ids],minlength=4).tolist(),\n                    'batch_weight_denominator':float(weights[target[ids].to(device)].sum())});save(arm+'_progress.json',curve)\n                gradient_norms=[];clipped=0")
            text = replace_once(text, "print(json.dumps({'arm':arm,**curve[-1]}),flush=True)",
                "print(json.dumps({'arm':arm,**curve[-1]}),flush=True)\n            if (step+1) in (1200,2400,3600):\n                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None\n                fixed_curve.append({'step':step+1,'sets':fixed_set_diagnostics(model,normalized,target,weights,device)})\n                torch.set_rng_state(cpu_rng)\n                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)\n                save(arm+'_fixed_set_diagnostics.json',fixed_curve)\n                if step+1==1200:\n                    archived=json.loads((ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'/(arm+'_progress.json')).read_text())\n                    assert all(c['batch_loss']==a['batch_loss'] for c,a in zip(curve,archived[:12])), 'training prefix differs'")
        elif name == 'budget_contract.py':
            text = replace_once(text, "'question':'Does increasing optimization budget improve original competence and recovery under the same training package?',",
                "'question':'Does a tenfold LR drop after1200 updates restore fitting and behavior at3600?',")
            text = replace_once(text, "'intervention':'1200 archived vs3600 new updates; fresh common initializer; no warm start',",
                "'intervention':'LR0.0012 through1200; LR0.00012 for1201-3600; fresh common initializer; no warm start',\n        'constant_lr3600_baseline':str(ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'),\n        'lr_schedule':{'switch_after_updates':1200,'initial':0.0012,'after':0.00012},")
            text = text.replace("'normalization','optimizer','goal sets'", "'normalization','optimizer algorithm and weight decay','goal sets'")
            text = replace_once(text, "if key not in ['version','updates','source_hashes']:", "if key not in ['version','updates','source_hashes','optimizer']:")
            text = replace_once(text, "assert new_protocol['updates']==3600", "assert new_protocol['updates']==3600\n    assert new_protocol['lr_schedule']==contract['lr_schedule']")
            text = replace_once(text, "'same_cases_and_training_contract_except_budget':True", "'same_cases_and_training_contract_except_LR_schedule_and_budget_vs1200':True")
            text = replace_once(text, "paths += list(HERE.glob('*.py'))", "constant=ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1'\n    paths += [constant/'experiment'/n for n in ['protocol_frozen.json','training_summary.json','final_verification.json','evaluation_protocol_frozen.json','BC.pt','BC_recovery.pt','BC_progress.json','BC_recovery_progress.json']]\n    paths += list(HERE.glob('*.py'))")
            text = replace_once(text, "print(json.dumps({'verified':True,'metrics3600':new_final['metrics']}),flush=True)",
                "constant=read(ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment/final_verification.json')\n    constant_protocol=read(ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment/protocol_frozen.json')\n    for key in constant_protocol:\n        if key not in ['version','source_hashes','optimizer']:assert constant_protocol[key]==new_protocol[key],key\n    constant_eval=read(ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment/evaluation_protocol_frozen.json')\n    assert constant_eval['cases']==new_eval['cases'] and constant_eval['scoring']==new_eval['scoring']\n    prefixes={}\n    for arm in new_train:\n        new_curve=read(OUT/(arm+'_progress.json'));old_curve=read(ROOT/'approach_clip8_human_recovery_budget3600_20261014_v1/experiment'/(arm+'_progress.json'))\n        assert len(new_curve)==len(old_curve)==36\n        assert all(a['batch_loss']==b['batch_loss'] for a,b in zip(new_curve[:12],old_curve[:12]))\n        assert all(c['lr']==(.0012 if c['step']<=1200 else .00012) for c in new_curve)\n        assert [c['step'] for c in read(OUT/(arm+'_fixed_set_diagnostics.json'))]==[1200,2400,3600]\n        prefixes[arm]={'batch_loss_checks':12,'max_difference':0.0}\n    save(HERE/'lr_comparison.json',{'verified':True,'metrics1200':old_final['metrics'],'metrics_constant3600':constant['metrics'],\n        'metrics_lr_drop3600':new_final['metrics'],'identical_first1200_loss_checks':prefixes,'same_cases_and_scoring':True,\n        'single_seed_development_screening':True,'contract_sha256':sha(HERE/'contract_frozen.json')})\n    print(json.dumps({'verified':True,'metrics3600':new_final['metrics']}),flush=True)")
        elif name == 'verify_download.py':
            text = replace_once(text, "comparison=read(HERE/'budget_comparison.json')", "lr=read(HERE/'lr_comparison.json');assert lr['verified'] and lr['same_cases_and_scoring']\n    assert lr['contract_sha256']==sha(HERE/'contract_frozen.json')\n    comparison=read(HERE/'budget_comparison.json')")
        if name.endswith('.py'):
            ast.parse(text)
        generated[name] = text
    OUT.mkdir()
    for name, text in generated.items():
        (OUT / name).write_text(text, encoding='utf-8', newline='\n')
    assert all(sha(SOURCE / name) == digest for name, digest in hashes.items())
    (OUT / 'preparation.json').write_text(json.dumps({'original_scripts_sha256':hashes,
        'generated_scripts_sha256':{n:sha(OUT/n) for n in names if n.endswith('.py') and n!='budget_contract.py'},
        'schedule':{'initial':0.0012,'switch_after_updates':1200,'after':0.00012},
        'changes':'LR schedule and diagnostic logging; output identity; identical data, model, weights, sampling, scoring'},indent=2)+'\n')
    print('prepared', OUT)


if __name__ == '__main__':
    main()
