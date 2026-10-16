#!/usr/bin/env python3
"""Triage original level-2 IL candidates; targets and semantics stay scorer-only."""
import contextlib
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis
import probe_single_goal_final_visual_train as probe
from audit_original_instruction_contract import clauses, label_in
from audit_release_simulator_assets import RELEASE, rows
from pilot_level_balanced_route import geodesic

BASE=Path('/home/aimldl/LH-VLN/runs/route_instruction_pilot_v1_20260930')
OUT=Path('/home/aimldl/LH-VLN/runs/level2_il_candidate_validation_20260930')


def sampled_indices(length, count=6):
    assert length>0
    return sorted({round(k*(length-1)/(count-1)) for k in range(count)})


def finite(value):
    return float(value) if math.isfinite(value) else None


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert sampled_indices(1)==[0] and sampled_indices(2)==[0,1]
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    bank=rows(BASE/'manifest_replayed.jsonl')
    candidates=[r for r in bank if r['level']==2]
    assert len(candidates)==16 and Counter(r['split'] for r in candidates)=={'train':8,'val':8}
    labels=defaultdict(set)
    for r in rows(RELEASE):
        for obj in r['objects']:
            labels[(r['scene'],str(obj['region_id']))].add(obj['region_name'].lower().strip())
    probe.install_index_aware_target_patch()
    probe.prior.add_side_semantics()
    args=probe.load_legacy_module(probe.ROOT).make_args()
    results=[]
    for row in candidates+[r for r in bank if r['index']==8]:
        path=Path(row['source_task_json'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_task_sha256']
        task=json.loads(path.read_text())
        pieces=clauses(row['instruction_original'])
        auxiliary=row['level']!=2
        with contextlib.redirect_stdout(io.StringIO()):
            sim=probe.SceneSimulator(args=args,config=task)
        stage_results=[]
        try:
            sim.save_path=None
            for stage,target in enumerate(row['ordered_targets']):
                source=task['trial'][f'trial_{stage}']
                endpoint=source['pos'][-1]
                sim.stage=stage
                ids,other_ids=probe.prior.instances(sim,stage)
                coord=sim.get_coord(str(sim.target[stage]))
                assert len(coord)==len(ids)
                raw_distances=[geodesic(sim.pathfinder,endpoint,c) for c in coord]
                snapped=[sim.pathfinder.snap_point(np.asarray(c,dtype=np.float32)) for c in coord]
                snap_distances=[geodesic(sim.pathfinder,endpoint,c) for c in snapped if np.isfinite(c).all()]
                other_regions={str(region.id)[1:] for region in sim.sim.semantic_scene.regions
                               for obj in region.objects if int(obj.id.rsplit('_',1)[1]) in other_ids}
                same_room_regions=[r for r in sorted(other_regions) if target['room'] in labels[(row['scene'],r)]]
                unknown_regions=[r for r in sorted(other_regions) if not labels[(row['scene'],r)]]
                sites=[]
                best_pixels=-1
                best_image=None
                best_boxes=[]
                if not auxiliary:
                    for step in sampled_indices(len(source['pos'])):
                        for reverse in ((False,True) if step==len(source['pos'])-1 else (False,)):
                            yaw=math.radians(float(source['yaw'][step])-180)+(math.pi if reverse else 0)
                            sim.set_state(source['pos'][step],quat_from_angle_axis(yaw,np.array([0.,1.,0.])))
                            obs=sim.sim.get_sensor_observations()
                            for code in ('l','f','r'):
                                boxes=probe.prior.boxes(obs['semantic_sensor' if code=='f' else 'semantic_sensor_'+code],ids)
                                pixels=max((b['pixels'] for b in boxes),default=0)
                                sites.append({'step':step,'camera':code,'reverse':reverse,'instance_pixels':pixels})
                                if not reverse and pixels>best_pixels:
                                    best_pixels=pixels
                                    best_image=Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3])
                                    best_boxes=boxes
                    assert best_image is not None
                    draw=ImageDraw.Draw(best_image)
                    for box in best_boxes:
                        draw.rectangle(box['box'],outline='red',width=2)
                    best_image.save(OUT/'images'/f'{row["index"]:03d}_{stage}.png')
                clause=pieces[stage] if stage<len(pieces) else ''
                stage_results.append({'stage':stage,'target':target,'region_id':str(sim.region[stage]),
                    'matching_instances':len(ids),'other_same_name_room_regions':same_room_regions,
                    'unknown_other_region_names':unknown_regions,'region_names_in_release':sorted(labels[(row['scene'],str(sim.region[stage]))]),
                    'raw_center_geodesic_m':finite(min(raw_distances,default=math.inf)),
                    'snapped_center_geodesic_m':finite(min(snap_distances,default=math.inf)),
                    'object_and_room_in_clause':label_in(clause,target['object']) and label_in(clause,target['room']),
                    'normal_route_visible25':any(s['instance_pixels']>=25 and not s['reverse'] for s in sites),
                    'endpoint_six_view_visible25':any(s['instance_pixels']>=25 and s['step']==len(source['pos'])-1 for s in sites),
                    'sampled_views':sites,'best_normal_image':None if auxiliary else f'images/{row["index"]:03d}_{stage}.png'})
        finally:
            sim.close()
        reasons=[]
        for s in stage_results:
            if not s['object_and_room_in_clause']: reasons.append('incomplete_original_clause')
            if s['snapped_center_geodesic_m'] is None or s['snapped_center_geodesic_m']>=1: reasons.append('endpoint_fails_active_target_radius1')
            if s['other_same_name_room_regions']: reasons.append('object_room_pair_not_unique_across_regions')
            if s['unknown_other_region_names']: reasons.append('other_region_room_names_not_verified')
            if not auxiliary and not s['normal_route_visible25']: reasons.append('target_not_visible_in_sampled_original_views')
        results.append({'index':row['index'],'task_id':row['task_id'],'split':row['split'],'scene':row['scene'],
            'auxiliary_repeated_category_case':auxiliary,'instruction_original':row['instruction_original'],
            'source_task_sha256':row['source_task_sha256'],'stages':stage_results,
            'automatic_gate_passed':not reasons,'hold_reasons':sorted(set(reasons)),
            'semantic_language_review':'pending','training_ready':False})
        save(OUT/'candidate_rows.json',results)
        print('validated',len(results),'of 17',flush=True)
    ordinary=[r for r in results if not r['auxiliary_repeated_category_case']]
    stages=[s for r in ordinary for s in r['stages']]
    summary={'candidate_tasks':16,'candidate_stages':32,'source_verified_tasks':17,
        'automatic_gate_pass_tasks':sum(r['automatic_gate_passed'] for r in ordinary),
        'hold_reason_task_counts':dict(Counter(reason for r in ordinary for reason in r['hold_reasons'])),
        'endpoint_raw_geodesic_under1_stages':sum(s['raw_center_geodesic_m'] is not None and s['raw_center_geodesic_m']<1 for s in stages),
        'endpoint_runtime_snapped_geodesic_under1_stages':sum(s['snapped_center_geodesic_m'] is not None and s['snapped_center_geodesic_m']<1 for s in stages),
        'normal_sampled_route_visible_stages':sum(s['normal_route_visible25'] for s in stages),
        'endpoint_six_view_visible_stages':sum(s['endpoint_six_view_visible25'] for s in stages),
        'sampled_views':sum(len(s['sampled_views']) for s in stages),
        'repeated_category_auxiliary':next(r for r in results if r['auxiliary_repeated_category_case']),
        'training_ready_tasks':0,'status':'automatic_triage_complete_manual_semantic_review_pending'}
    save(OUT/'summary.json',summary)
    review=['# Level-2 original-instruction IL candidates','',
        'Automatic checks only. Navigation projection excludes Grab/Release; semantic language review pending.','']
    for r in ordinary:
        review += [f'## Index {r["index"]}: {r["task_id"]} ({r["split"]})','',r['instruction_original'],'',
                   'Automatic hold reasons: '+(', '.join(r['hold_reasons']) or 'none'),'']
        for s in r['stages']:
            review += [f'- Stage {s["stage"]+1}: {s["target"]}; geodesic {s["raw_center_geodesic_m"]}; route visible {s["normal_route_visible25"]}.']
        review += ['', 'Semantic/language review: pending.','']
    (OUT/'review.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='repeated_category_auxiliary'},indent=2),flush=True)


if __name__=='__main__':
    main()
