#!/usr/bin/env python3
"""Train-only final-goal RGB grounding probe; semantics are scorer-only."""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image, ImageEnhance

ROOT=Path('/home/aimldl/LH-VLN')
RUN=ROOT/'runs/single_goal_final_visual_train_20260929'
os.environ.setdefault('IDENVLN_RUN_DIR',str(RUN))
sys.path.insert(0,str(ROOT))
from habitat_base.simulation import SceneSimulator
from audit_release_simulator_assets import OLD,NEW,RELEASE,rows
from pilot_level_balanced_route import source_path
from run_ordered_metrics_eval import install_index_aware_target_patch,load_legacy_module
from single_goal_input_contract import final_goal
import run_rollout_target_proposals_train as prior


def select(per_level):
    reserved={r['scene'] for r in rows(ROOT/'runs/single_goal_heldout_bank_v2_100_20260929.jsonl')}
    old,new=rows(OLD),rows(NEW)
    chosen=[]
    for level in range(2,6):
        pool=sorted((r for r in rows(RELEASE) if r['scene_heldout_split']=='train' and int(r['level'])==level and r['scene'] not in reserved),
                    key=lambda r:hashlib.sha256(('final_visual_v1:'+r['task_id']).encode()).hexdigest())
        for r in pool:
            task_path=source_path(r,old,new)
            task=json.loads(task_path.read_text())
            try: goal=final_goal(task)
            except ValueError: continue
            chosen.append((r,task_path,task,goal))
            if sum(x[0]['level']==r['level'] for x in chosen)==per_level: break
    assert len(chosen)==4*per_level
    return chosen


def capture(per_level):
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'images').mkdir(exist_ok=True)
    install_index_aware_target_patch()
    prior.add_side_semantics()
    args=load_legacy_module(ROOT).make_args()
    output=[]
    for ordinal,(record,path,task,goal) in enumerate(select(per_level),1):
        stage=int(record['level'])-1
        trajectory=task['trial'][f'trial_{stage}']
        position=trajectory['pos'][-1]
        angle=math.radians(float(trajectory['yaw'][-1])-180.0)
        frames=sorted((p for p in path.parent.iterdir() if p.is_dir() and p.name.split('_',1)[0].isdigit()),key=lambda p:int(p.name.split('_',1)[0]))
        saved=np.asarray(Image.open(frames[-1]/'front.png').convert('RGBA'),dtype=np.int16)
        with contextlib.redirect_stdout(io.StringIO()):
            sim=SceneSimulator(args=args,config=task)
        try:
            sim.set_state(position,quat_from_angle_axis(angle,np.array([0.,1.,0.])))
            obs=sim.sim.get_sensor_observations()
            rendered=ImageEnhance.Contrast(Image.fromarray(np.asarray(obs['color_sensor_f']),mode='RGBA')).enhance(1.5).resize((366,366))
            error=float(np.abs(saved-np.asarray(rendered,dtype=np.int16)).mean())
            assert error<0.02,(record['task_id'],error)
            target_ids,distractor_ids=prior.instances(sim,stage)
            for camera,code in (('left','l'),('front','f'),('right','r')):
                semantic='semantic_sensor' if code=='f' else 'semantic_sensor_'+code
                mask=np.asarray(obs[semantic])
                name=f'{ordinal:03d}_{camera}.png'
                Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3]).save(RUN/'images'/name)
                output.append({'task_id':record['task_id'],'level':int(record['level']),
                               'scene':record['scene'],'camera':camera,'image':'images/'+name,
                               'goal_object':goal['goal_object'],'goal_room':goal['goal_room'],
                               'target_boxes':prior.boxes(mask,target_ids),
                               'distractor_boxes':prior.boxes(mask,distractor_ids),
                               'rgb_replay_mae':round(error,6)})
        finally: sim.close()
        print('captured',ordinal,'of',4*per_level,flush=True)
    (RUN/'frames.json').write_text(json.dumps(output,indent=2)+'\n')
    print('capture_complete',len(output),'views',flush=True)


def score():
    import torch
    from transformers import OwlViTForObjectDetection,OwlViTProcessor
    frames=json.loads((RUN/'frames.json').read_text())
    model_name='google/owlvit-base-patch32'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    processor=OwlViTProcessor.from_pretrained(model_name)
    model=OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()
    scored=[]
    with torch.inference_mode():
        for start in range(0,len(frames),4):
            batch=frames[start:start+4]
            images=[Image.open(RUN/r['image']).convert('RGB') for r in batch]
            prompts=[[f'a photo of a {r["goal_object"]}'] for r in batch]
            inputs=processor(text=prompts,images=images,return_tensors='pt')
            predictions=processor.post_process_object_detection(
                model(**{k:v.to(device) for k,v in inputs.items()}),threshold=0.0,
                target_sizes=[image.size[::-1] for image in images])
            for row,pred in zip(batch,predictions):
                gold=[b['box'] for b in row['target_boxes'] if b['pixels']>=25]
                distractors=[b['box'] for b in row['distractor_boxes'] if b['pixels']>=25]
                rank=pred['scores'].argsort(descending=True)[:20]
                proposals=[[float(z) for z in pred['boxes'][i]] for i in rank]
                hit_rank=next((j for j,p in enumerate(proposals,1) if any(prior.iou(p,g)>=.5 for g in gold)),None)
                scored.append({**row,'target_visible':bool(gold),'distractor_visible':bool(distractors),
                               'top1_hit':bool(gold and any(prior.iou(proposals[0],g)>=.5 for g in gold)),
                               'top20_hit':hit_rank is not None,'first_hit_rank':hit_rank,
                               'top1_distractor':bool(distractors and any(prior.iou(proposals[0],d)>=.5 for d in distractors)),
                               'top1_score':float(pred['scores'][rank[0]])})
            print('scored',min(start+4,len(frames)),'of',len(frames),flush=True)
    visible=[r for r in scored if r['target_visible']]
    result={'tasks':len({r['task_id'] for r in scored}),'views':len(scored),
            'target_visible_tasks':len({r['task_id'] for r in visible}),
            'target_visible_views':len(visible),'top1_hits_visible':sum(r['top1_hit'] for r in visible),
            'top20_hits_visible':sum(r['top20_hit'] for r in visible),
            'distractor_views':sum(r['distractor_visible'] for r in scored),
            'top1_distractor_views':sum(r['top1_distractor'] for r in scored),
            'max_rgb_replay_mae':max(r['rgb_replay_mae'] for r in scored),
            'prompt':'gold final object text from v2 task input, RGB only; semantic labels scorer-only',
            'scope':'deterministic train-only expert final poses; no heldout scoring'}
    (RUN/'scored.json').write_text(json.dumps(scored,indent=2)+'\n')
    (RUN/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('capture','score'))
    p.add_argument('--per-level',type=int,default=8)
    a=p.parse_args()
    if a.mode=='capture':capture(a.per_level)
    else:score()
