"""Recompute pilot metrics and check unchanged historical dependencies/checkpoints."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE/'experiment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN/'verification.json').exists(), 'Refuse overwrite'
    status = json.loads((RUN/'status.json').read_text())
    assert status['phase']=='completed' and (HERE/'exit').read_text().strip()=='0'
    protocol = json.loads((RUN/'protocol_frozen.json').read_text())
    parent=json.loads((HERE.parent/'pipeline_repair_20261013_v2/experiment/protocol_frozen.json').read_text())
    assert protocol['cohort']==parent['cohort']
    offline=json.loads((RUN/'training/offline_gate.json').read_text())
    assert offline['passed']==status['offline_gate_passed'] and offline['steps']<=4000
    if offline['passed']:
        assert offline['accuracy']==offline['STOP_recall']==1.0
        assert offline['remaining_errors']==0 and offline['consecutive_perfect_checks']>=3
        logged=[line.split() for line in (HERE/'run.log').read_text().splitlines() if line.startswith('train ') and not line.startswith('train {')]
        assert all(float(row[2])==float(row[3])==1.0 for row in logged[-3:])
    assert sha(RUN/'selection_ledger.json')==protocol['selection_ledger_sha256']
    assert all(sha(Path(p))==h for p,h in protocol['dependencies'].items())
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    for family,v in manifest.items():
        assert sha(HERE/(family+'_original.py'))==v['original_sha256']
        assert sha(HERE/(family+'_fixed.py'))==v['fixed_sha256']
    history = HERE.parent/'balanced100_eval_20261013_v1/protocol_frozen.json'
    historical = json.loads(history.read_text())
    assert all(not r['checkpoint'] or sha(Path(r['checkpoint']))==r['sha256'] for r in historical['policies'])
    checkpoint = RUN/'training/sequence_repaired_gate.pt'
    assert sha(checkpoint)==status['checkpoint_sha256']
    starts = json.loads((RUN/'start_states_frozen.json').read_text())
    assert len(starts)==8
    assert not {r['scene'] for r in protocol['cohort']['train']} & {r['scene'] for r in protocol['cohort']['dev']}
    replay = json.loads((RUN/'replay/verification.json').read_text())
    assert replay['passed'] and len(replay['rows'])==8 and all(r['valid'] and r['max_position_error_m']<=0.01 for r in replay['rows'])
    metrics = {}
    for arm,cohort in protocol['cohort'].items():
        rows = [json.loads(l) for l in (RUN/f'rollouts/{arm}.jsonl').read_text().splitlines()]
        teacher = [json.loads(l) for l in (RUN/f'replay/{arm}.jsonl').read_text().splitlines()]
        assert len(rows)==len(teacher)==len(cohort)==4
        for selected,t,r in zip(cohort,teacher,rows):
            assert t['index']==r['index']==selected['index']
            assert t['actions']==selected['teacher_actions']
            assert sha(Path(selected['trial'])/'task.json')==selected['task_sha256']
            assert r['status']=='completed' and len(r['successes'])==selected['level']
            assert r['start_state']==starts[str(selected['index'])]
            assert t['start_state']==r['start_state']
            stops=[s for s in r['steps'] if s['stop']]
            assert len(stops)==r['stop_events'] and len(r['actions'])<=150
            for s in stops:
                k=s['active_target'];d=s['distances_after'][k]
                assert bool(r['successes'][k])==(d is not None and d<1.0)
        k=sum(len(r['successes']) for r in rows)
        metrics[arm]={'chains':sum(r['full_chain_success'] for r in rows),'episodes':4,'stages':sum(sum(r['successes']) for r in rows),'stage_count':k,'ISR_micro':sum(sum(r['successes']) for r in rows)/k,'ISR_macro':sum(sum(r['successes'])/len(r['successes']) for r in rows)/4,'STOP':sum(r['stop_events'] for r in rows),'no_STOP_episodes':sum(r['stop_events']==0 for r in rows)}
        assert metrics[arm]==status['metrics'][arm]
    data=json.loads((RUN/'training/data.json').read_text())
    assert data['dev_in_training'] is False and data['fresh_feature_max_error']<1e-6
    assert set(data['train_tids'])=={r['tid'] for r in protocol['cohort']['train']}
    assert all(json.loads((RUN/'self_check.json').read_text()).values())
    intervention=json.loads((HERE/'single_action_intervention/result.json').read_text())
    intervention_protocol=json.loads((HERE/'single_action_intervention/protocol_frozen.json').read_text())
    old_checkpoint=HERE.parent/'pipeline_repair_20261013_v2/experiment/training/sequence_repaired_gate.pt'
    assert sha(old_checkpoint)==intervention_protocol['checkpoint_sha256']
    assert intervention['prefix_verified'] and intervention['checkpoint_unchanged'] and intervention['not_a_learned_score']
    assert intervention['interventions']==[{'step':38,'original_action':'turn_left','forced_action':'move_forward'}]
    result={'status':'passed','metrics':metrics,'train_gate_passed':status['train_gate_passed'],'historical_checkpoints_unchanged':True,'source_hashes_unchanged':True,'task_hashes_unchanged':True,'protocol_sha256':sha(RUN/'protocol_frozen.json'),'runner_sha256':sha(HERE/'run_repair_pilot.py'),'checkpoint_sha256':sha(checkpoint),'verifier_sha256':sha(Path(__file__))}
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
