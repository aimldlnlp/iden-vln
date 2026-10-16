"""Explain held train targets without changing eligibility or evaluator geometry."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path

import magnum as mn
import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis
import audit_level2_train_cohort as driver
import goal_resolver as resolver
import run_ordered_metrics_eval as evaluator
from build_route_instruction_pilot import sha
from pilot_level_balanced_route import geodesic
from run_cohort_obb_v1 import mesh_bounds

ROOT = Path('/home/aimldl/LH-VLN/runs')
SOURCE = ROOT/'scene_disjoint_original_train_bank_20261011'
OUT = ROOT/'held_train_target_diagnostic_20261011'


def finite(value):
    return float(value) if math.isfinite(value) else None


def main():
    if OUT.exists():
        assert not (OUT/'diagnostic.json').exists(), 'refuse to overwrite diagnostic'
        (OUT/'protocol.json').rename(OUT/'failed_startup_protocol.json')
    else:
        OUT.mkdir()
    selected = {r['index']:r for r in map(json.loads,(SOURCE/'selection_manifest.jsonl').read_text().splitlines())}
    checks = json.loads((SOURCE/'eligibility.json').read_text())
    ids = sorted({r['index'] for r in checks if r['holds']} | {1,8,11})
    assert ids == [1,2,3,4,5,6,7,8,9,11]
    assert all(selected[i]['split']=='train' for i in ids)
    protocol = {'version':'held_train_target_diagnostic_v1','selected_indices':ids,
                'selection_sha256':sha(SOURCE/'selection_manifest.jsonl'),
                'eligibility_sha256':sha(SOURCE/'eligibility.json'),
                'geometry':'same raw mesh transform; vertex-count distributions plus native OBB distance checks',
                'pose_search':'reproduce original seed, sites and yaw order; count each filter separately; additionally record source endpoint distance under AABB and OBB',
                'eligibility_changed':False,'training_performed':False,'script_sha256':sha(Path(__file__))}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    resolver.install(evaluator,'obb_v1')
    args = driver.probe.load_legacy_module(driver.probe.ROOT).make_args()
    args.success_dis,args.max_step = 1.,400
    results = []
    for index in ids:
        row = selected[index]
        path = Path(row['source_task_json'])
        assert sha(path)==row['source_task_sha256']
        task = json.loads(path.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            sim = driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path = None
        record = {'index':index,'task_id':row['task_id'],'scene':row['scene'],
                  'instruction_original':row['instruction_original'],'boxes':[],'pose_stages':[]}
        try:
            objects = [resolver.objects_for_stage(sim,k) for k in range(2)]
            directory = Path(sim.sim_settings['scene'])
            if not directory.is_dir():directory=directory.parent
            stem = row['scene'].split('-')[-1]
            mesh = directory/(stem+'.semantic.glb')
            with (directory/(stem+'.semantic.txt')).open() as stream:
                annotations = {int(r[0]):r for r in csv.reader(stream) if r and r[0].strip().isdigit()}
            colors = {annotations[int(o.id.split('_')[-1])][1].strip().lstrip('#') for group in objects for o in group}
            bounds, vertices = mesh_bounds(mesh,colors)
            for stage,group in enumerate(objects):
                for obj in group:
                    ident = int(obj.id.split('_')[-1])
                    color = annotations[ident][1].strip().lstrip('#')
                    points = vertices[color]
                    low, high = bounds[color]
                    matrix = np.asarray(obj.obb.world_to_local,dtype=float)
                    oc, ac = resolver.center(obj,'obb_v1'), resolver.center(obj,'aabb_v0')
                    assert np.allclose((matrix@np.r_[oc,1])[:3],0,atol=1e-4)
                    local = (np.c_[points,np.ones(len(points))]@matrix.T)[:,:3]
                    distances = np.linalg.norm((local-np.clip(local,-1,1))*np.asarray(obj.obb.half_extents),axis=1)
                    for n in np.unique(np.linspace(0,len(points)-1,7,dtype=int).tolist()+[int(distances.argmax())]):
                        assert abs(float(obj.obb.distance(mn.Vector3(*points[n])))-float(distances[n]))<1e-4
                    snap = np.asarray(sim.pathfinder.snap_point(oc),dtype=float)
                    record['boxes'].append({'stage':stage,'instance_id':ident,'object':obj.category.name(),
                        'vertices':len(points),'outside10cm_vertices':int((distances>.1).sum()),
                        'outside10cm_fraction':float((distances>.1).mean()),
                        'outside_distance_quantiles_m':dict(zip(('p50','p90','p95','p99','maximum'),map(float,np.quantile(distances,[.5,.9,.95,.99,1])))),
                        'mesh_bounds':[low.tolist(),high.tolist()],'obb_center':oc.tolist(),'aabb_center':ac.tolist(),
                        'obb_mesh_center_error_m':float(np.linalg.norm(oc-(low+high)/2)),
                        'aabb_mesh_center_error_m':float(np.linalg.norm(ac-(low+high)/2)),
                        'obb_snap':snap.tolist() if np.isfinite(snap).all() else None,
                        'snap_minus_center_y_m':float(snap[1]-oc[1]) if np.isfinite(snap).all() else None,
                        'mesh_sha256':sha(mesh)})
            if index in (1,8,11):
                sim.pathfinder.seed(57031+index)
                for stage in range(2):
                    sim.stage=stage
                    goals=resolver.goals_for_stage(sim,stage,'obb_v1')
                    endpoint=np.asarray(task['trial'][f'trial_{stage}']['pos'][-1],dtype=np.float32)
                    sites=[endpoint]
                    for goal in goals:
                        sites.append(goal)
                        sites.extend(sim.pathfinder.get_random_navigable_point_near(goal,.9,100) for _ in range(24))
                    stats={'stage':stage,'target':str(sim.target[stage]),'sites':len(sites),'nonfinite_sites':0,
                           'outside_radius_sites':0,'within_radius_sites':0,'views':0,'below25_views':0,
                           'above20percent_views':0,'original_gate_views':0,'minimum25_only_views':0,
                           'source_endpoint':endpoint.tolist(),
                           'source_endpoint_obb_geo_m':finite(min(geodesic(sim.pathfinder,endpoint,g) for g in goals)),
                           'goal_from_start_geo_m':[finite(geodesic(sim.pathfinder,row['start'],g)) for g in goals]}
                    old=resolver.goals_for_stage(sim,stage,'aabb_v0')
                    stats['source_endpoint_aabb_geo_m']=finite(min(geodesic(sim.pathfinder,endpoint,g) for g in old if np.isfinite(g).all()))
                    target_ids,_=driver.probe.prior.instances(sim,stage)
                    best=None
                    for pos in sites:
                        if not np.isfinite(pos).all():stats['nonfinite_sites']+=1;continue
                        distance=min(geodesic(sim.pathfinder,pos,g) for g in goals)
                        if distance>=.8:stats['outside_radius_sites']+=1;continue
                        stats['within_radius_sites']+=1
                        for turn in range(12):
                            sim.set_state(pos,quat_from_angle_axis(turn*math.pi/6,np.array([0.,1.,0.])))
                            obs=sim.sim.get_sensor_observations()
                            boxes=driver.probe.prior.boxes(obs['semantic_sensor'],target_ids)
                            pixels=max((b['pixels'] for b in boxes),default=0)
                            stats['views']+=1
                            stats['below25_views']+=pixels<25
                            stats['above20percent_views']+=pixels>.2*obs['semantic_sensor'].size
                            stats['original_gate_views']+=25<=pixels<=.2*obs['semantic_sensor'].size
                            stats['minimum25_only_views']+=pixels>=25
                            if best is None or pixels>best['pixels']:
                                best={'pixels':pixels,'position':list(map(float,pos)),'yaw_degrees':turn*30,'geo_m':float(distance)}
                                frame=Image.fromarray(np.asarray(obs['color_sensor_f'])[...,:3])
                                for box in boxes:ImageDraw.Draw(frame).rectangle(box['box'],outline='red',width=2)
                                frame.save(OUT/f'{index:03d}_{stage}_max_visibility.png')
                    assert stats['views']==12*stats['within_radius_sites']
                    assert stats['views']==stats['below25_views']+stats['above20percent_views']+stats['original_gate_views']
                    stats['best']=best
                    record['pose_stages'].append(stats)
                    # Original teacher fails at its first stage without a feasible view, so do not claim later seeds reproduced.
                    if stats['original_gate_views']==0:break
        finally:
            sim.close()
        results.append(record)
        (OUT/'diagnostic.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'index':index,'failing_boxes':sum(b['outside10cm_vertices']>0 for b in record['boxes']),
                          'pose_stages':record['pose_stages']}),flush=True)
    assert len(results)==10
    (OUT/'complete.json').write_text(json.dumps({'status':'complete','tasks':10,'training_performed':False,
        'eligibility_changed':False,'diagnostic_sha256':sha(OUT/'diagnostic.json')},indent=2)+'\n')


if __name__=='__main__':main()
