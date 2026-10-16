"""Check downloaded checkpoints, case coverage and scores against frozen artifacts."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiment'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (HERE/'exit').read_text().strip()=='0'
    contract=read(HERE/'contract_frozen.json')
    assert contract['updates']==3600
    for raw,expected in contract['dependencies'].items():
        if '/'+HERE.name+'/' in raw:
            local=HERE/raw.split('/'+HERE.name+'/',1)[1]
            assert sha(local)==expected,raw
    final=read(OUT/'final_verification.json');records=read(OUT/'evaluation_rollouts.json')
    protocol=read(OUT/'evaluation_protocol_frozen.json');training=read(OUT/'training_summary.json')
    assert final['passed'] and final['rollouts_verified']==len(records)==36
    expected={(c['cohort'],c['index'],a) for c in protocol['cases'] for a in protocol['arms']}
    assert len(expected)==36 and {(r['cohort'],r['index'],r['arm']) for r in records}==expected
    for arm,meta in training.items():
        assert meta['updates']==3600 and sha(OUT/(arm+'.pt'))==meta['checkpoint_sha256']
    for cohort,arms in final['metrics'].items():
        for arm,metric in arms.items():
            group=[r for r in records if r['cohort']==cohort and r['arm']==arm]
            assert len(group)==metric['episodes']
            for row in group:
                assert row['steps']==len(row['actions'])<=400
                assert [s['stage'] for s in row['stops']]==list(range(len(row['stops'])))
                assert all(s['valid']==(s['geo_m'] is not None and s['geo_m']<1.) for s in row['stops'])
                assert row['stage_success']==sum(s['valid'] for s in row['stops'])
                assert row['success']==(len(row['stops'])==2 and all(s['valid'] for s in row['stops']))
            assert metric['chain_success']==sum(r['success'] for r in group)
            assert metric['stage_success']==sum(r['stage_success'] for r in group)
            assert metric['SR']==metric['chain_success']/len(group)
            assert metric['ISR']==metric['stage_success']/(2*len(group))
    comparison=read(HERE/'budget_comparison.json')
    assert comparison['verified'] and comparison['baseline_and_sources_unchanged']
    assert comparison['contract_sha256']==sha(HERE/'contract_frozen.json')
    result={'passed':True,'rollouts':36,'metrics_recomputed':True,'checkpoint_hashes_match':True,
        'frozen_code_hashes_match':True,'archive_sha256':sha(HERE/'snapshot.tgz'),
        'remote_actor_replay_verified':read(OUT/'independent_evaluation_verification.json')['passed']}
    (HERE/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
