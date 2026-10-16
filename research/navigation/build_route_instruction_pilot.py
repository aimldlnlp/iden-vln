#!/usr/bin/env python3
"""Audit original instructions; build a versioned, reviewable route pilot."""
import hashlib
import json
import math
import re
from itertools import groupby
from collections import Counter, defaultdict
from pathlib import Path

import habitat_sim
import numpy as np
from audit_release_simulator_assets import HM3D, OLD, NEW, RELEASE, rows
from pilot_level_balanced_route import geodesic, source_path
from run_single_goal_route_pilot import interpolate
from single_goal_input_contract import ROOMS

ROOT = Path('/home/aimldl/LH-VLN/runs')
OUT = ROOT / 'route_instruction_pilot_v1_20260930'
BASE = ROOT / 'single_goal_sequence_v2_train_20260930/protocol.json'
BANK = ROOT / 'single_goal_heldout_bank_v2_100_20260929.jsonl'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return re.findall(r'[a-z0-9_]+', text.lower())


def ordered_mentions(text, labels):
    words, cursor = tokens(text), 0
    for label in labels:
        phrase = tokens(label)
        match = next((i for i in range(cursor, len(words)-len(phrase)+1)
                      if phrase and words[i:i+len(phrase)] == phrase), None)
        if match is None:
            return False
        cursor = match + len(phrase)
    return True


def navigation_instruction(objects, rooms, stages=None):
    if stages is None:
        parts = [f'Dari posisi awal, navigate menuju {objects[0]} di {rooms[0]}.']
        parts += [f'Kemudian lanjut ke {obj} di {room}.' for obj,room in zip(objects[1:],rooms[1:])]
    else:
        parts = ['Dari posisi awal, ikuti rute ini.']
        previous_pos, previous_yaw = stages[0]['pos'][0], float(stages[0]['yaw'][0])
        for stage,obj,room in zip(stages,objects,rooms):
            steps=[]
            start_index = 1 if stage is stages[0] else 0
            for action,position,yaw in zip(stage['action'][start_index:],stage['pos'][start_index:],stage['yaw'][start_index:]):
                if action=='move_forward':
                    amount=math.hypot(position[0]-previous_pos[0],position[2]-previous_pos[2])
                elif action in ('turn_left','turn_right'):
                    amount=abs((float(yaw)-previous_yaw+180)%360-180)
                else:
                    amount=0
                if action!='stop':
                    steps.append((action,amount))
                previous_pos,previous_yaw=position,float(yaw)
            commands=[]
            for action,group in groupby(steps,key=lambda x:x[0]):
                amount=sum(item[1] for item in group)
                if amount<.01:
                    continue
                commands.append(f'maju sekitar {amount:.2f} m' if action=='move_forward'
                                else f'turn {"left" if action=="turn_left" else "right"} {amount:.0f} derajat')
            parts.append((', lalu '.join(commands) if commands else 'Tetap di posisi ini')+f', sampai di {obj} di {room}.')
    parts.append('Ikuti urutan tujuan tersebut; stop hanya setelah mencapai tujuan terakhir.')
    return ' '.join(parts)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert ordered_mentions('go to chair then table', ['chair','table'])
    assert not ordered_mentions('go to table then chair', ['chair','table'])
    assert not ordered_mentions('go to chair', ['chair','chair'])
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(BANK) == '66668f486851aa74e2bb4bbae9bc176d570f1a8fa9568728af843f10612cf795'
    groups = json.loads(BASE.read_text())['scene_groups']
    reserved = {r['scene'] for r in rows(BANK)}
    assert set(groups['train']).isdisjoint(groups['val'])
    assert not (reserved & (set(groups['train']) | set(groups['val'])))
    old, new = rows(OLD), rows(NEW)
    candidates, audit, counts = [], [], defaultdict(Counter)
    for record in rows(RELEASE):
        if record['scene_heldout_split'] != 'train' or record['scene'] in reserved:
            continue
        split = next((s for s in ('train','val') if record['scene'] in groups[s]), None)
        if split is None:
            continue
        path = source_path(record, old, new)
        task = json.loads(path.read_text())
        objects = [str(x).strip().lower() for x in task['Object']]
        rooms = [str(x).strip().lower() for x in task['Region Name']]
        text = str(task.get('Task instruction') or '')
        level = int(record['level'])
        assert len(objects) == len(rooms) == len(task['trial']) == level
        flags = {'object_mentions_in_order':ordered_mentions(text, objects),
                 'room_mentions_in_order':ordered_mentions(text, rooms),
                 'length_180_suspected_truncation':len(text)==180,
                 'generic_cek_target':'cek target' in text.lower(),
                 'contains_manipulation_verbs':bool(set(tokens(text)) & {'ambil','letakkan','pick','place','put','grab'}),
                 'valid_room_labels':all(room in ROOMS for room in rooms)}
        literal = flags['object_mentions_in_order'] and flags['room_mentions_in_order']
        counts[str(level)]['tasks'] += 1
        counts[str(level)]['all_labels_in_order'] += literal
        counts[str(level)]['literal_navigation_candidate'] += literal and not any(flags[k] for k in
                    ('length_180_suspected_truncation','generic_cek_target','contains_manipulation_verbs'))
        for key, value in flags.items():
            counts[str(level)][key] += value
        audit.append({'task_id':record['task_id'], 'split':split, 'level':level, 'scene':record['scene'],
                      'source_task_json':str(path), 'instruction_original':text, **flags})
        if flags['valid_room_labels'] and all(objects):
            candidates.append((record,path,task,objects,rooms,flags,split))
    with (OUT/'original_instruction_audit.jsonl').open('w') as handle:
        for row in audit:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    meshes, used, manifest, rejected = {}, set(), [], Counter()
    for split in ('train','val'):
        for level in range(2,6):
            selected = 0
            pool = sorted((r for r in candidates if r[6]==split and int(r[0]['level'])==level),
                          key=lambda r:hashlib.sha256(('route_pilot_v1:'+r[0]['task_id']).encode()).digest())
            for record,path,task,objects,rooms,flags,_ in pool:
                stages = [task['trial'][f'trial_{i}'] for i in range(level)]
                start, final = stages[0]['pos'][0], stages[-1]['pos'][-1]
                key = (record['scene'],tuple(start),tuple(zip(objects,rooms)))
                if key in used:
                    rejected['duplicate_scene_start_targets'] += 1
                    continue
                scene = record['scene']
                mesh = next(p for s in ('train','val') if (p:=HM3D/s/scene/(scene.split('-')[-1]+'.basis.navmesh')).exists())
                if str(mesh) not in meshes:
                    pf = habitat_sim.PathFinder()
                    assert pf.load_nav_mesh(str(mesh))
                    meshes[str(mesh)] = pf
                pf = meshes[str(mesh)]
                initial = geodesic(pf,start,final)
                if not math.isfinite(initial) or initial <= 5:
                    rejected['initial_distance_not_gt5'] += 1
                    continue
                assert all(len(s['pos'])==len(s['action'])==len(s['yaw']) and s['action'][-1]=='stop' for s in stages)
                gaps = [math.dist(a['pos'][-1],b['pos'][0]) for a,b in zip(stages,stages[1:])]
                gap_geo = [geodesic(pf,a['pos'][-1],b['pos'][0]) for a,b in zip(stages,stages[1:])]
                if any(not math.isfinite(g) or g > .35 for g in gap_geo):
                    rejected['boundary_geodesic_gt_0p35_or_unreachable'] += 1
                    continue
                if any(d>.1 and b['action'][0]!='move_forward' for d,b in zip(gaps,stages[1:])):
                    rejected['unexplained_boundary_action'] += 1
                    continue
                paths = [s['pos'] for s in stages]
                # Validate every recorded segment, including stage boundaries.
                points = [p for path_points in paths for p in path_points]
                distances = [geodesic(pf,a,b) for a,b in zip(points,points[1:])]
                if any(not math.isfinite(g) or g>.35 or g>math.dist(a,b)+.05
                       for a,b,g in zip(points,points[1:],distances)):
                    rejected['invalid_recorded_segment'] += 1
                    continue
                reference = interpolate(points)
                actions = [a for i,s in enumerate(stages) for a in (s['action'][1:] if i==0 else s['action']) if a!='stop'] + ['stop']
                instruction = navigation_instruction(objects,rooms,stages)
                assert ordered_mentions(instruction,objects) and ordered_mentions(instruction,rooms)
                assert actions.count('stop')==1 and actions[-1]=='stop'
                manifest.append({'index':len(manifest),'task_id':record['task_id'],'split':split,'level':level,
                    'scene':scene,'robot':str(task.get('Robot','')).lower(),'source_task_json':str(path),
                    'source_task_sha256':sha(path),'navmesh':str(mesh),'navmesh_sha256':sha(mesh),
                    'instruction_original':str(task.get('Task instruction') or ''),'original_flags':flags,
                    'instruction':instruction,'instruction_version':'route_actions_navigation_v1',
                    'instruction_target_order_only':navigation_instruction(objects,rooms),
                    'instruction_provenance':'synthetic route rewrite from original recorded actions, yaw deltas, horizontal displacement, and ordered object/room metadata',
                    'semantic_review_status':'pending','simulator_replay_status':'pending','training_ready':False,
                    'start':start,'final':final,'stage_endpoints':[s['pos'][-1] for s in stages],
                    'ordered_targets':[{'object':o,'room':r} for o,r in zip(objects,rooms)],
                    'reference':reference,'reference_origin':'original stage positions; no shortest-route replacement',
                    'boundary_euclidean_m':gaps,'boundary_geodesic_m':gap_geo,
                    'reference_geodesic_length_m':sum(distances),'initial_geodesic_m':initial,
                    'candidate_actions':actions,'candidate_actions_count':len(actions),
                    'action_alignment':'initial recorded action omitted; intermediate STOPs removed; simulator replay still required'})
                used.add(key)
                selected += 1
                if selected==8:
                    break
            assert selected==8,(split,level,selected,dict(rejected))
    assert len(manifest)==64
    assert len({r['task_id'] for r in manifest})==64
    assert {r['scene'] for r in manifest if r['split']=='train'}.isdisjoint(r['scene'] for r in manifest if r['split']=='val')
    with (OUT/'manifest.jsonl').open('w') as handle:
        for row in manifest:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary = {'tasks_audited':len(audit),'original_audit_by_level':dict(counts),
        'pilot_tasks':len(manifest),'pilot_split_counts':dict(Counter(r['split'] for r in manifest)),
        'pilot_scene_counts':{s:len({r['scene'] for r in manifest if r['split']==s}) for s in ('train','val')},
        'rejected_geometry_or_duplicates':dict(rejected),'max_boundary_gap_m':max(max(r['boundary_euclidean_m']) for r in manifest),
        'max_boundary_geodesic_m':max(max(r['boundary_geodesic_m']) for r in manifest),
        'manifest_sha256':sha(OUT/'manifest.jsonl'),'release_sha256':sha(RELEASE),'scene_split_protocol_sha256':sha(BASE),
        'horizon_status':'not frozen until candidate action sequences pass simulator replay',
        'training_ready_tasks':0,'next':'replay candidate actions; review language/object-room alignment before training'}
    write_json(OUT/'summary.json',summary)
    review = ['# Route pilot instruction review','', 'Synthetic navigation instructions from recorded routes and metadata. Human semantic review pending; see replay_summary.json for simulator validation.','']
    for r in manifest:
        review += [f'## {r["index"]}: {r["task_id"]} ({r["split"]}, level {r["level"]})','',
                   '**Original:** '+r['instruction_original'],'','**Navigation v1:** '+r['instruction'],'',
                   'Scene: '+r['scene']+'. Target order: '+json.dumps(r['ordered_targets'],ensure_ascii=False),
                   '', 'Review: pending.','']
    (OUT/'instruction_review.md').write_text('\n'.join(review)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
