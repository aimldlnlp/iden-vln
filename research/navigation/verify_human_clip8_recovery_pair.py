"""Recompute final paired outcomes and check source/checkpoint identities, read-only."""
import json,math
from pathlib import Path
from build_route_instruction_pilot import sha

OUT=Path('/home/aimldl/LH-VLN/runs/approach_clip8_human_recovery_pair_20261014_v1')
def main():
    status=json.loads((OUT/'status.json').read_text());assert status['phase']=='completed'
    protocol=json.loads((OUT/'evaluation_protocol_frozen.json').read_text())
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    fitting=json.loads((OUT/'fitting_verification.json').read_text())
    assert fitting['passed'] and fitting['same_initialization'] and fitting['same_normalization'] and fitting['same_loss_weights']
    for arm,meta in fitting['arms'].items():assert sha(OUT/(arm+'.pt'))==meta['checkpoint_sha256'] and meta['updates']==1200
    records=json.loads((OUT/'evaluation_rollouts.json').read_text())
    replay=json.loads((OUT/'independent_evaluation_verification.json').read_text())
    assert replay['passed'] and replay['replayed_rollouts']==len(records)==36 and replay['max_position_error_m']<1e-5
    expected={(case['cohort'],case['index'],arm) for case in protocol['cases'] for arm in protocol['arms']}
    assert {(r['cohort'],r['index'],r['arm']) for r in records}==expected
    metrics={};pairs={}
    for cohort,n in (('train_original',8),('train_holdout',8),('dev_original',2)):
        metrics[cohort]={};pairs[cohort]=[]
        for arm in protocol['arms']:
            rs=[r for r in records if r['cohort']==cohort and r['arm']==arm];assert len(rs)==n
            for r in rs:
                assert len(r['actions'])==r['steps']<=400
                actual=[a for a in r['actions'] if a['action']=='stop']
                assert len(actual)==len(r['stops'])<=2
                for a,s in zip(actual,r['stops']):
                    assert a['kind']=='policy' and a['stage']==s['stage']
                    assert s['valid']==(s['geo_m'] is not None and s['geo_m']<1)
                    assert (a['pre_geo_m'] is None and s['geo_m'] is None) or abs(a['pre_geo_m']-s['geo_m'])<1e-5
                assert [s['stage'] for s in r['stops']]==list(range(len(actual)))
                assert r['success']==(len(actual)==2 and all(s['valid'] for s in r['stops']))
                assert r['stage_success']==sum(s['valid'] for s in r['stops'])
            metrics[cohort][arm]={'episodes':n,'chain_success':sum(r['success'] for r in rs),'stage_success':sum(r['stage_success'] for r in rs),
                'SR':sum(r['success'] for r in rs)/n,'ISR':sum(r['stage_success'] for r in rs)/(2*n)}
        indices=sorted({r['index'] for r in records if r['cohort']==cohort})
        for i in indices:
            a=next(r for r in records if r['cohort']==cohort and r['index']==i and r['arm']=='BC')
            b=next(r for r in records if r['cohort']==cohort and r['index']==i and r['arm']=='BC_recovery')
            assert math.dist(a['branch_position'],b['branch_position'])<1e-5
            pairs[cohort].append({'index':i,'BC':a['success'],'BC_recovery':b['success'],'stage_gain':b['stage_success']-a['stage_success']})
    assert metrics==json.loads((OUT/'evaluation_summary.json').read_text())
    result={'passed':True,'rollouts_verified':36,'metrics':metrics,'paired':pairs,'source_hashes_unchanged':True,
        'single_seed':True,'human_review_scope':'seven recovery terminal views only; original/dev not newly human-validated',
        'script_sha256':sha(Path(__file__))}
    path=OUT/'final_verification.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'metrics':metrics}),flush=True)
if __name__=='__main__':main()
