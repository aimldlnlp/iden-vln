"""Freeze only the RGB-reviewed navigation projection addition; retain held pair members."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

PARENT_SHA='cf6122e038639a1f6f6f52dd4f5c8175f8b710decaad037a1b46fb20308fa70d'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(root, verify_sources=False):
    root=Path(root); out=root/'paired_original_navigation_bank_20261001'
    parent=root/'original_projection_data_20261001/manifest_original_projection.jsonl'
    assert sha(parent)==PARENT_SHA
    selection={r['index']:r for r in map(json.loads,(out/'selection_manifest.jsonl').read_text().splitlines())}
    eligible={r['index']:r for r in json.loads((out/'eligibility.json').read_text())}
    rows={r['index']:r for r in json.loads((out/'candidate_rows.json').read_text())}
    evidence={r['index']:r for r in json.loads((out/'review_evidence.json').read_text())}
    assert len(selection)==26 and not eligible[0]['holds']
    decisions=[{'source_index':0,'accepted':True,'reason':'Armchair and door identifiable in RGB/panorama; desk/workspace compatible with source Office label; route contact sheet and replay reviewed.', 'reviewer':'Codex; not human annotation'},
        {'source_index':18,'accepted':False,'reason':'toilet_paper_target_not_identifiable_in_RGB_mesh; semantic pixel gate alone insufficient', 'reviewer':'Codex; not human annotation'}]
    assert rows[0]['teacher_replay_pass'] and evidence[0]['max_position_error_m']<1e-5
    assert all(s['geo_m']<1 and 25<=s['front_target_pixels']<=.2*512*512 for s in evidence[0]['stops'])
    assert all(stage['matching_regions']==[stage['region']] for stage in eligible[0]['stages'])
    combined=[json.loads(l) for l in parent.read_text().splitlines()]
    for row in combined:
        row['bank_partition']='legacy_projection_anchor'
        row['original_language_uniqueness_review']='Not re-audited here; legacy dev has known Office+box ambiguity.'
    original=selection[0]; replay=rows[0]
    addition={'index':len(combined),'source_candidate_index':0,'source_cohort':out.name,'task_id':replay['task_id'],
        'split':replay['split'],'scene':replay['scene'],'pair_id':original['pair_id'],
        'source_task_json':replay['source_task_json'],'source_sha256':replay['source_sha256'],
        'instruction_original':original['instruction_original'],'instruction_revised':evidence[0]['instruction_revised_draft'],
        'instruction_version':'original_verbatim_navigation_projection_v1','instruction_provenance':'verbatim original; navigation-only scoring',
        'policy_context':{'instruction':original['instruction_original'],'robot':original['robot']},
        'teacher_actions':replay['teacher_actions'],'start':original['start'],'start_rotation_simulator_argument':original['start_rotation_simulator_argument'],
        'navigation_projection_stages':2,'navigation_projection_ready':True,'pair_ready':False,'literal_instruction_pair_ready':False,
        'human_validated':False,'benchmark_training_ready':False,'bank_partition':'new_original_unique_reviewed',
        'protocol_version':json.loads((out/'protocol.json').read_text())['version'],'protocol_sha256':sha(out/'protocol.json'),
        'goal_resolver':'obb_v1','original_language_uniqueness_review':'both stage room-object regions unique with tie labels treated as matches',
        'scope':'navigation projection; no pick/place success claim; incomplete counterfactual pair'}
    combined.append(addition)
    assert len({r['task_id'] for r in combined})==len(combined)
    train=[r for r in combined if r['split']=='train'];dev=[r for r in combined if r['split']=='dev']
    assert len(train)==5 and len(dev)==1 and {r['scene'] for r in train}.isdisjoint(r['scene'] for r in dev)
    for r in combined:
        assert r['policy_context']['instruction']==r['instruction_original'] and r['navigation_projection_stages']==2
        assert sum(a['action']=='stop' for a in r['teacher_actions'])==2
        if verify_sources:assert sha(Path(r['source_task_json']))==r['source_sha256']
    manifest=out/'model_manifest_original_projection_frozen.jsonl'
    manifest.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined).encode('utf-8'))
    (out/'new_original_unique_manifest.jsonl').write_text(json.dumps(addition,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'visual_review_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    ledger=[]
    for i,r in selection.items():
        holds=list(eligible[i]['holds'])
        if i in rows and not rows[i]['teacher_replay_pass']:holds.append('teacher_replay_failed')
        if i==18:holds.append('toilet_paper_target_not_identifiable_in_RGB_mesh')
        ledger.append({'index':i,'task_id':r['task_id'],'pair_id':r['pair_id'],'split':r['split'],'accepted':i==0,'holds':holds})
    (out/'final_candidate_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    summary={'combined_train_tasks':5,'combined_dev_tasks':1,'new_train_tasks':1,'new_dev_tasks':0,'complete_reviewed_same_start_pairs':0,
        'train_action_samples':sum(len(r['teacher_actions']) for r in train),'new_train_action_samples':len(addition['teacher_actions']),
        'manifest_sha256':sha(manifest),'parent_manifest_sha256':PARENT_SHA,'sources_verified':verify_sources,'new_model_trained':False,
        'scope':'legacy anchors + one new train; no new dev or complete pair obtained; original projection only; not benchmark ready'}
    (out/'frozen_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runs-root',default=str(Path(__file__).parent));parser.add_argument('--verify-sources',action='store_true')
    args=parser.parse_args();main(args.runs_root,args.verify_sources)
