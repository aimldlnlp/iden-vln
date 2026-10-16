"""Read-only reconstruction of original boundary failures from verified artifacts."""
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parent
RUN=BASE/'approach_clip8_stop_counter_20261014_v1'
OUT=BASE/'counter_boundary_review_20261014_v1'
MANIFEST=BASE/'approach_matched_il_20261011_v1/bank8/model_manifest_frozen.jsonl'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(),'Refuse overwrite; use a new version'
    assert read(RUN/'local_verification.json')['passed']
    paths=[RUN/'local_verification.json',RUN/'counter_comparison.json',RUN/'experiment/evaluation_rollouts.json',RUN/'experiment/diagnosis/diagnostic_results.json',MANIFEST]
    hashes={str(p):sha(p) for p in paths}
    traces=read(paths[2]);diagnosis=read(paths[3])
    teachers=[r for r in map(json.loads,MANIFEST.read_text(encoding='utf-8').splitlines()) if r['split']=='train']
    assert len(teachers)==8
    findings=[]
    for teacher in teachers:
        actions=teacher['teacher_actions']
        boundary=next(t for t,a in enumerate(actions) if a['stage']==1)
        assert boundary>0 and actions[boundary-1]['action']=='stop'
        for arm in ('counter_zero','counter_active'):
            trace=next(r for r in traces if r['cohort']=='train_original' and r['index']==teacher['index'] and r['arm']==arm)
            first=next((t for t,(actual,reference) in enumerate(zip(trace['actions'],actions)) if actual['action']!=reference['action']),None)
            old=next(d for d in diagnosis['first_divergences'] if d['index']==teacher['index'] and d['arm']==arm)
            assert first==old['first_action_difference']
            prefix_error=max((math.dist(trace['actions'][t]['position'],actions[t]['position']) for t in range(first or 0)),default=0.)
            assert abs(prefix_error-old['prefix_position_error'])<1e-8
            item={'index':teacher['index'],'arm':arm,'teacher_boundary_t':boundary,'first_divergence_t':first,'prefix_position_error_m':prefix_error,'stage0_valid':any(s['stage']==0 and s['valid'] for s in trace['stops']),'stage1_entered':any(a['stage']==1 for a in trace['actions']),'first_divergence_at_teacher_boundary':first==boundary}
            if first is not None:
                event=trace['actions'][first]
                assert old['cached_prediction']==event['action']
                item.update(teacher_action=actions[first]['action'],actual_action=event['action'],cached_prediction=old['cached_prediction'],cached_STOP_probability=old['cached_probability'][0])
            if first==boundary:
                assert event['stage']==1 and event['action']=='stop'
                assert actions[first]['action']!='stop' and event['pre_geo_m']>=1.
                assert event['own_STOP_count_before']==1
                assert event['counter_feature']==(1. if arm=='counter_active' else 0.)
                item.update(classification='invalid_STOP_at_teacher_boundary_also_predicted_on_cache',distance_m=event['pre_geo_m'],own_STOP_count_before=1,counter_feature=event['counter_feature'])
            else:item['classification']='earlier_divergence_before_teacher_boundary' if first is not None and first<boundary else 'other'
            findings.append(item)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    summary={arm:{'episodes':8,'boundary_first_divergences':sum(r['first_divergence_at_teacher_boundary'] for r in findings if r['arm']==arm),'earlier_divergences':sum(r['classification']=='earlier_divergence_before_teacher_boundary' for r in findings if r['arm']==arm)} for arm in ('counter_zero','counter_active')}
    result={'passed':True,'source_hashes':hashes,'script_sha256':sha(Path(__file__)),'selection':'all eight original train tasks and both arms; no success-based selection','summary':summary,'rows':findings,'no_training_or_rollout':True,'limits':'cache and live top1 agree with matching action/position prefix; numerical equality of every live feature is not tested here; teacher-state fitting errors do not exclude later covariate shift'}
    OUT.mkdir()
    (OUT/'boundary_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'summary':summary},indent=2))


if __name__=='__main__':main()
