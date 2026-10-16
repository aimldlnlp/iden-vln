"""Verify every full-run row against its preserved trace before paper updates."""
import gzip
import importlib.util
import json
from pathlib import Path
import time
import argparse

ROOT=Path('/home/aimldl/LH-VLN')
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);OUT=parser.parse_args().output_dir
runner=OUT/'run_frozen_bank_eval.py'
spec=importlib.util.spec_from_file_location('release_verify',runner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
protocol=json.loads((OUT/'protocol_frozen.json').read_text())
for dependency in json.loads((OUT/'dependencies_frozen.json').read_text())['files']:
    assert module.sha(Path(dependency['path']))==dependency['sha256']
assert module.sha(runner)==protocol['runner_sha256']
assert module.sha(OUT/'bank_frozen.json')==protocol['bank_sha256']
bank=json.loads((OUT/'bank_frozen.json').read_text())
assert json.loads((OUT/'full/status.json').read_text())['status']=='completed'
count=0;summary={}
for policy in protocol['policies']:
    name=policy['name'];folder=OUT/'full'/name;records=[]
    if policy['checkpoint']:assert module.sha(Path(policy['checkpoint']))==policy['sha256']
    for i,b in enumerate(bank):
        r=json.loads((folder/f'{i:04d}.json').read_text())
        with gzip.open(folder/f'{i:04d}.trace.gz','rt') as f:t=json.load(f)
        assert r['index']==i and r['task_id']==b['task_id'] and r['level']==b['level']
        assert r['scene']==b['scene'] and r['split']==b['split']
        assert r['actions']==len(t.get('actions',[])) and r['actions']<=150
        assert len(r['successes'])==b['level']
        if r['status']=='completed':
            assert t['status']=='completed' and r['successes']==[int(x) for x in t['successes']]
            assert t['start_state']==b['start_state']
            assert r['SR']==int(t['full_chain_success'])
        else:assert r['successes']==[0]*b['level'] and r['SR']==0
        a,c=module.metric(r['successes']);assert (r['ISR'],r['CSR'])==(a,c)
        records.append(r);count+=1
    recorded=json.loads((folder/'summary.json').read_text())
    assert recorded['all']==module.summarize(records)
    assert len(records)==len(bank)
    summary[name]=recorded
assert count==len(bank)*len(protocol['policies'])
module.atomic(OUT/'verified_results.json',{'status':'passed','records':count,'verified_utc':time.time(),'summaries':summary,'protocol_sha256':module.sha(OUT/'protocol_frozen.json')})
print('Verified all frozen bank rows against traces and summaries')
