"""Recompute audit gates and counts from frozen inputs and actual actor traces."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/aimldl/LH-VLN/runs/balanced100_teacher_contract_20261013_v1')
protocol=json.loads((ROOT/'protocol_frozen.json').read_text())
for path,digest in protocol['dependencies'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
features=json.loads((ROOT/'feature_verification.json').read_text())
checks=[c for row in features['rows'] for c in row['checks']]
assert len(checks)==features['samples']==398
assert len(features['rows'])==4
thresholds={'cache_source_error':1e-6,'disk_live_error':1e-6,'evaluator_live_error':1e-6,'position_error':1e-4,'circular_yaw_error':1e-5,'raw_live_cache_error':1e-4}
for key,tolerance in thresholds.items():
    assert max(c[key] for c in checks)==features['maxima'][key]
    assert max(c[key] for c in checks)<=tolerance
assert all(c['cache_pred']==c['live_pred'] for c in checks)
assert features['passed']
outputs={}
for phase in ('smoke','full'):
    rows=[json.loads(line) for line in (ROOT/phase/'traces.jsonl').read_text().splitlines()]
    expected=protocol['smoke_indices'] if phase=='smoke' else list(range(100))
    assert [r['index'] for r in rows]==expected
    assert len({r['task_id'] for r in rows})==len(rows)
    for row in rows:
        assert row['runtime_ok'],(phase,row['index'],row.get('error'))
        assert len(row['successes'])==row['level']
        assert row['full_chain_success']==all(row['successes'])
        assert len(row['steps'])<=150
        stage=0
        for t,step in enumerate(row['steps']):
            assert step['t']==t and step['stage']==stage
            if step['action']=='stop':
                assert step['success_after']==(step['distance_before_m']<1.)
                assert row['successes'][stage]==step['success_after']
                stage+=1
        assert row['full_chain_success']==(row['failure'] is None)
        if row['failure']=='horizon_exhausted':assert len(row['steps'])==150
    aggregate={'episodes':len(rows),'chains':sum(r['full_chain_success'] for r in rows),'stages':sum(sum(r['successes']) for r in rows),'stage_count':sum(r['level'] for r in rows),'failures':{f:sum(r['failure']==f for r in rows) for f in sorted({r['failure'] for r in rows if r['failure']})},'by_level':{str(k):{'episodes':sum(r['level']==k for r in rows),'chains':sum(r['full_chain_success'] for r in rows if r['level']==k)} for k in range(2,6)}}
    saved=json.loads((ROOT/phase/'verification.json').read_text())
    assert saved['passed'] and saved['dependencies_unchanged']
    assert all(saved[k]==v for k,v in aggregate.items())
    outputs[phase]=aggregate
result={'passed':True,'feature_samples':398,'phases':outputs,'scope':'trace arithmetic, actual STOP and frozen dependencies; not human grounding validation'}
(ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
print(json.dumps(result,indent=2))
