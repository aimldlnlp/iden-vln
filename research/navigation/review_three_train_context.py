"""Rebuild and replay three train candidates with context views, then export RGB review evidence."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import audit_level2_train_cohort as driver
import goal_resolver as resolver
import run_ordered_metrics_eval as evaluator
from build_route_instruction_pilot import sha
from run_single_goal_route_pilot import reset

ROOT = Path('/home/aimldl/LH-VLN')
OLD = ROOT / 'runs/level2_train_cohort_obb_v1_1_20260930'
OUT = ROOT / 'runs/three_train_context_review_20260930'

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    OUT.mkdir(exist_ok=True)
    rows = json.loads((OLD/'candidate_rows.json').read_text())
    boxes = json.loads((OLD/'box_audit.json').read_text())
    chosen = [r for r in rows if r['index'] in (1,4,5)]
    assert len(chosen)==3 and all(r['split']=='train' and r['teacher_replay_pass'] for r in chosen)
    assert all(b['obb_mesh_max_surface_distance_m']<=.1 for b in boxes if b['index'] in (1,4,5))
    save('protocol.json', {'version':'level2_context_view_obb_v1','parent_version':'level2_navigation_obb_v1_1',
        'goal_resolver':'obb_v1','radius_strict_m':1.,'horizon':400,'stop_rule':'one STOP per ordered stage',
        'pose_selection':'fixed seed; source endpoint and 24 nearby samples per goal; <0.8 m goal distance; front target 25..20% image pixels',
        'stop_view':'full scan then recorded shortest yaw turn to best front view within 25..20% pixel gate',
        'teacher_inputs':'privileged semantic mask/navmesh for demonstrations only',
        'policy_inputs':['instruction','robot','RGB_l_f_r','relative_pose_yaw_history','own_action_history'],
        'indices':[1,4,5],'parent_candidates_sha256':sha(OLD/'candidate_rows.json'),
        'resolver_sha256':sha(Path(resolver.__file__)),'driver_sha256':sha(Path(driver.__file__)),
        'script_sha256':sha(Path(__file__)),'scope':'train-only engineering review; no learned-policy score'})
    driver.OUT = OUT
    driver.probe.install_index_aware_target_patch = lambda: resolver.install(evaluator,'obb_v1')
    driver.main(review_subset=True)
    candidates = json.loads((OUT/'candidate_rows.json').read_text())
    args = driver.probe.load_legacy_module(driver.probe.ROOT).make_args()
    args.success_dis,args.max_step=1.,400
    with (ROOT/'nav_gen/scene/Per_Scene_Region_Weighted_Votes.csv').open() as f:
        rooms={(r['Scene Name'],r['Region #']):r['Weighted Room Proposal'].strip().lower() for r in csv.DictReader(f)}
    reviews=[]
    for row in candidates:
        source=Path(row['source_task_json'])
        assert sha(source)==row['source_sha256']
        task=json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()): sim=driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None
        try:
            objects=[o for o in sim.sim.semantic_scene.objects if o is not None and o.region is not None and o.category is not None]
            room_objects={}
            for o in objects:
                region=o.region.id.split('_')[-1]
                room_objects.setdefault(region,set()).add(o.category.name())
            inventory=[{'region':k,'room':rooms.get((row['scene'],k)),'objects':sorted(v)} for k,v in room_objects.items()]
            with contextlib.redirect_stdout(io.StringIO()): reset(sim,task['trial']['trial_0']['pos'][0])
            sampled,stops,errors=[],[],[]
            for step,item in enumerate(row['teacher_actions']):
                sim.stage=item['stage']
                if step%8==0 or item['action']=='stop':
                    obs=sim.sim.get_sensor_observations()
                    sampled.append((step,item['stage'],item['action'],Image.fromarray(np.asarray(obs['color_sensor_f'])[...,:3])))
                if item['action']=='stop':
                    with contextlib.redirect_stdout(io.StringIO()): info=sim.get_info()
                    assert info['geo dis']<1 and abs(info['geo dis']-item['runtime_geo_m'])<1e-5
                    ids,_=driver.probe.prior.instances(sim,sim.stage)
                    pixels=max((b['pixels'] for b in driver.probe.prior.boxes(obs['semantic_sensor'],ids)),default=0)
                    assert 25<=pixels<=.2*obs['semantic_sensor'].size
                    assert pixels==row['stages'][sim.stage]['stop_front_visible_pixels']
                    sheet=Image.new('RGB',(3*384,408),'white')
                    for col,code in enumerate(('l','f','r')):
                        frame=Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3])
                        frame.save(OUT/f'{row["index"]:03d}_{sim.stage}_stop_rgb_{code}.png')
                        sheet.paste(frame.resize((384,384)),(col*384,0))
                        ImageDraw.Draw(sheet).text((col*384+4,388),f'{code}: task {row["index"]}, stage {sim.stage}',fill='black')
                    sheet.save(OUT/f'{row["index"]:03d}_{sim.stage}_stop_context.png')
                    stops.append({'stage':sim.stage,'step':step,'geo_m':float(info['geo dis']),'front_target_pixels':pixels})
                else: sim.agent.act(item['action'])
                errors.append(math.dist(sim.agent.get_state().position,item['position']))
            if row['teacher_replay_pass']: assert len(stops)==2 and max(errors)<1e-5
            for start in range(0,len(sampled),20):
                group=sampled[start:start+20]
                sheet=Image.new('RGB',(4*256,math.ceil(len(group)/4)*280),'white')
                for j,(step,stage,action,frame) in enumerate(group):
                    x,y=(j%4)*256,(j//4)*280
                    sheet.paste(frame.resize((256,256)),(x,y))
                    ImageDraw.Draw(sheet).text((x+3,y+260),f'{step}: stage {stage} {action}',fill='black')
                sheet.save(OUT/f'{row["index"]:03d}_route_{start//20:02d}.png')
            reviews.append({'index':row['index'],'scene':row['scene'],'region_inventory':inventory,
                'stops':stops,'route_sample_interval':8,'sampled_frames':len(sampled),
                'max_position_error_m':max(errors) if errors else None,'complete_replay':row['teacher_replay_pass']})
        finally: sim.close()
        save('review_evidence.json',reviews)
        print(json.dumps({'index':row['index'],'review_replay':row['teacher_replay_pass'],'stops':stops}),flush=True)
    print('review export complete',flush=True)

if __name__=='__main__': main()
