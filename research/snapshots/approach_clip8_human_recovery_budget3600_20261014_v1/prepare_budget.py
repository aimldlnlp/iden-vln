"""Prepare isolated budget-only copies; historical scripts remain immutable."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
NAME=HERE.name
OLD='approach_clip8_human_recovery_pair_20261014_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'experiment').exists() and not (HERE/'contract_frozen.json').exists(), 'refuse changing a frozen run'
    inputs={}
    for name in ['train_human_clip8_recovery_pair.py','evaluate_human_clip8_recovery_pair.py',
                 'verify_human_clip8_recovery_pair.py','audit_recovery_regression_research.py']:
        path=SOURCE/name;inputs[name]=sha(path)
        source=path.read_text(encoding='utf-8')
        assert OLD in source
        source=source.replace(OLD,NAME+'/experiment')
        if name.startswith('train_'):
            assert source.count('1200')==4
            source=source.replace('1200','3600')
            source=source.replace('human_clip8_matched_recovery_pair_v1','human_clip8_budget3600_pair_v1')
            line='initial_hash=state_hash(initial)'
            assert source.count(line)==1
            source=source.replace(line,line+'\n    baseline=json.loads((ROOT/"'+OLD+'/training_summary.json").read_text())\n    assert all(m["initial_state_sha256"]==initial_hash for m in baseline.values()), "initializer differs from1200"')
            line='draw_hash=hashlib.sha256(draws.tobytes()).hexdigest()'
            assert source.count(line)==1
            source=source.replace(line,line+'\n    prefix_hash=hashlib.sha256(draws[:1200].tobytes()).hexdigest()\n    assert all(m["uniform_draws_sha256"]==prefix_hash for m in baseline.values()), "sampling prefix differs from1200"')
        elif name.startswith('verify_'):
            assert source.count('meta[\'updates\']==1200')==1
            source=source.replace("meta['updates']==1200","meta['updates']==3600")
            source=source.replace('seven recovery terminal views only','14 terminal STOP views from seven recovery tasks')
        elif name.startswith('audit_'):
            source=source.replace("ROOT/'human_recovery_regression_research_20261014_v1'","PAIR/'diagnosis'")
            assert source.count('random((1200,128))')==1
            source=source.replace('random((1200,128))','random((3600,128))')
        ast.parse(source)
        (HERE/name).write_text(source,encoding='utf-8',newline='\n')
    assert all(sha(SOURCE/name)==h for name,h in inputs.items())
    (HERE/'preparation.json').write_text(json.dumps({'original_scripts_sha256':inputs,
        'generated_scripts_sha256':{n:sha(HERE/n) for n in inputs},'budget':3600,
        'changes':'output/version and budget only; identical initializer and first1200 sampling draws asserted; audit draw budget updated; archived presentation count retained'},indent=2)+'\n')
    print('prepared budget-only copies',NAME)


if __name__=='__main__':main()
