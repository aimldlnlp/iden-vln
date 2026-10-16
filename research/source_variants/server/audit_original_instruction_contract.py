#!/usr/bin/env python3
"""Audit original text and source targets; render a fixed 16-task subset."""
import contextlib
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis
import probe_single_goal_final_visual_train as probe
from build_route_instruction_pilot import tokens, ordered_mentions
from pilot_level_balanced_route import geodesic

BASE = Path('/home/aimldl/LH-VLN/runs/route_instruction_pilot_v1_20260930')
OUT = Path('/home/aimldl/LH-VLN/runs/original_instruction_contract_20260930')


def clauses(text):
    # ponytail: source-template clause split, not a general semantic parser; review new templates manually.
    return re.split(r',\s*(?:kemudian|lalu)\s+', text.lower())


def label_in(text, label):
    return ordered_mentions(text, [label])


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert clauses('Pergi ke Bedroom lalu ambil bed, kemudian move ke Hall dan letakkan on table, lalu move ke Kitchen cek target') == ['pergi ke bedroom lalu ambil bed', 'move ke hall dan letakkan on table', 'move ke kitchen cek target']
    assert not label_in('cek target', 'tv')
    assert not label_in('door knob', 'doorway')
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    manifest = [json.loads(line) for line in (BASE/'manifest_replayed.jsonl').read_text().splitlines()]
    assert len(manifest) == 64
    selected = {min(r['index'] for r in manifest if r['split']==split and r['level']==level)+offset
                for split in ('train','val') for level in range(2,6) for offset in (0,1)}
    assert len(selected)==16
    records, tasks = [], {}
    for row in manifest:
        path = Path(row['source_task_json'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_task_sha256']
        task = json.loads(path.read_text())
        tasks[row['index']] = task
        assert task['Task instruction']==row['instruction_original']
        assert [x.lower().strip() for x in task['Object']]==[x['object'] for x in row['ordered_targets']]
        assert [x.lower().strip() for x in task['Region Name']]==[x['room'] for x in row['ordered_targets']]
        pieces = clauses(row['instruction_original'])
        stages = []
        for stage,target in enumerate(row['ordered_targets']):
            text = pieces[stage] if stage < len(pieces) else ''
            stages.append({'stage':stage,'object':target['object'],'room':target['room'],'clause':text,
                'object_explicit_in_own_clause':label_in(text,target['object']),
                'room_explicit_in_own_clause':label_in(text,target['room']),
                'generic_target': 'cek target' in text,
                'endpoint':row['stage_endpoints'][stage]})
        records.append({'index':row['index'],'task_id':row['task_id'],'split':row['split'],'level':row['level'],
            'scene':row['scene'],'instruction_original':row['instruction_original'],'stages':stages,
            'source_hash_verified':True,'source_keys':list(task),'source_region_ids':task.get('Region'),
            'length':len(row['instruction_original']),'ends_end':row['instruction_original'].endswith(' end'),
            'has_direction_words':bool(set(tokens(row['instruction_original'])) & {'left','right','kiri','kanan','straight','lurus'}),
            'all_targets_explicit_in_own_clauses':all(s['object_explicit_in_own_clause'] and s['room_explicit_in_own_clause'] for s in stages),
            'geometry_replay_passed':row['simulator_replay_status']=='passed','visual_subset':row['index'] in selected})
    save(OUT/'text_rows.json',records)
    probe.install_index_aware_target_patch()
    probe.prior.add_side_semantics()
    args = probe.load_legacy_module(probe.ROOT).make_args()
    visual=[]
    for row in manifest:
        if row['index'] not in selected:
            continue
        task=tasks[row['index']]
        with contextlib.redirect_stdout(io.StringIO()):
            sim=probe.SceneSimulator(args=args,config=task)
        try:
            for stage in range(row['level']):
                source=task['trial'][f'trial_{stage}']
                position=source['pos'][-1]
                yaw=math.radians(float(source['yaw'][-1])-180)
                ids,distractors=probe.prior.instances(sim,stage)
                objects=[(region,obj) for region in sim.sim.semantic_scene.regions for obj in region.objects
                         if int(obj.id.rsplit('_',1)[1]) in ids]
                centers=[]
                for region,obj in objects:
                    center=obj.aabb.center
                    center=center() if callable(center) else center
                    center=np.asarray(center,dtype=np.float32)
                    snapped=sim.pathfinder.snap_point(center)
                    centers.append({'instance_id':int(obj.id.rsplit('_',1)[1]),'region_id':str(region.id),
                        'category':obj.category.name(),'center':center.tolist(),
                        'center_euclidean_m':math.dist(position,center),
                        'snap_displacement_m':math.dist(center,snapped) if np.isfinite(snapped).all() else None,
                        'endpoint_to_snapped_center_geodesic_m':geodesic(sim.pathfinder,position,snapped) if np.isfinite(snapped).all() else None})
                views=[]
                for reverse in (False,True):
                    sim.set_state(position,quat_from_angle_axis(yaw+(math.pi if reverse else 0),np.array([0.,1.,0.])))
                    obs=sim.sim.get_sensor_observations()
                    for code in ('l','f','r'):
                        mask=obs['semantic_sensor' if code=='f' else 'semantic_sensor_'+code]
                        boxes=probe.prior.boxes(mask,ids)
                        views.append({'camera':code,'reverse':reverse,'target_pixels':sum(b['pixels'] for b in boxes),
                                      'max_instance_pixels':max((b['pixels'] for b in boxes),default=0),
                                      'distractor_pixels':int(np.isin(mask,distractors).sum())})
                        if code=='f' and not reverse:
                            filename=f'{row["index"]:03d}_stage{stage}.png'
                            image=Image.fromarray(np.asarray(obs['color_sensor_f'])[...,:3])
                            draw=ImageDraw.Draw(image)
                            for box in boxes:
                                draw.rectangle(box['box'],outline='red',width=2)
                            image.save(OUT/'images'/filename)
                visual.append({'index':row['index'],'task_id':row['task_id'],'level':row['level'],'stage':stage,
                    'target':row['ordered_targets'][stage],'sim_target':str(sim.target[stage]),'sim_region':str(sim.region[stage]),
                    'matched_instances':len(ids),'same_category_other_region_instances':len(distractors),'instances':centers,
                    'views':views,'any_visible_instance25':any(v['max_instance_pixels']>=25 for v in views),
                    'normal_visible_instance25':any(v['max_instance_pixels']>=25 and not v['reverse'] for v in views),
                    'front_image':f'images/{row["index"]:03d}_stage{stage}.png'})
        finally:
            sim.close()
        save(OUT/'visual_rows.json',visual)
        print('audited',len({v['index'] for v in visual}),'of 16',flush=True)
    assert len(visual)==56
    by_level={}
    for level in range(2,6):
        rr=[r for r in records if r['level']==level]
        ss=[s for r in rr for s in r['stages']]
        by_level[str(level)]={'tasks':len(rr),'stages':len(ss),
            'complete_text_targets':sum(r['all_targets_explicit_in_own_clauses'] for r in rr),
            'missing_object_stages':sum(not s['object_explicit_in_own_clause'] for s in ss),
            'missing_room_stages':sum(not s['room_explicit_in_own_clause'] for s in ss),
            'length180':sum(r['length']==180 for r in rr),'ends_end':sum(r['ends_end'] for r in rr)}
    summary={'scope':'64 preselected geometry-valid tasks; fixed first two per split/level for 16-task semantic audit',
        'manifest_sha256':hashlib.sha256((BASE/'manifest_replayed.jsonl').read_bytes()).hexdigest(),
        'text_tasks':len(records),'source_hashes_verified':len(records),'text_by_level':by_level,
        'direction_word_tasks':sum(r['has_direction_words'] for r in records),
        'visual_indices':sorted(selected),'visual_tasks':16,'visual_stages':len(visual),
        'stages_no_matching_instance':sum(v['matched_instances']==0 for v in visual),
        'stages_multiple_matching_instances':sum(v['matched_instances']>1 for v in visual),
        'normal_visible_stages':sum(v['normal_visible_instance25'] for v in visual),
        'six_view_visible_stages':sum(v['any_visible_instance25'] for v in visual),
        'min_center_euclidean_over3_stages':sum(bool(v['instances']) and min(i['center_euclidean_m'] for i in v['instances'])>3 for v in visual),
        'limitations':['Literal clause screening, not synonym-aware semantic evaluation',
            'Semantic category/region matches do not validate human room naming',
            'Endpoint poses are teleported only for audit observations, not policy rollouts',
            'Six endpoint views are not exhaustive route visibility',
            'No learned-policy score; no causal attribution of Table 4 failure']}
    save(OUT/'summary.json',summary)
    chosen=[v for v in visual if v['index'] in {0,8,16,24,32,40,48,56} and v['stage']==v['level']-1]
    sheet=Image.new('RGB',(4*360,2*300),'white')
    draw=ImageDraw.Draw(sheet)
    for n,v in enumerate(chosen):
        image=Image.open(OUT/v['front_image']).convert('RGB')
        image.thumbnail((350,250))
        x,y=(n%4)*360,(n//4)*300
        sheet.paste(image,(x,y+40))
        draw.text((x+4,y+3),f'index {v["index"]} stage {v["stage"]+1}: {v["target"]["object"]}',fill='black')
        draw.text((x+4,y+18),f'{v["target"]["room"]}; matches {v["matched_instances"]}',fill='black')
    sheet.save(OUT/'endpoint_contact_sheet.jpg')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
