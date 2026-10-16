"""Freeze bounded train approach goals before producing any new demonstrations."""
import contextlib
import io
import json
import math
import sys
from pathlib import Path
import magnum as mn
import numpy as np
from PIL import Image,ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis
import diagnose_held_train_targets as audit
from frozen_approach_resolver import VERSION,fingerprint

OUT=audit.ROOT/'visible_approach_pilot_20261011'


def framed_box(box,width,height,margin=8):
    x0,y0,x1,y1=box
    return x0>=margin and y0>=margin and x1<=width-margin and y1<=height-margin


def main(selection_manifest=None, output=None):
    global OUT
    framed='--framed-v2' in sys.argv
    dev='--dev' in sys.argv
    expand='--expand-train' in sys.argv
    assert sys.argv[1:] in ([],['--framed-v2'],['--framed-v2','--dev'],['--framed-v2','--expand-train'])
    version='visible_approach_framed_v2' if framed else VERSION
    if framed:OUT=audit.ROOT/'visible_approach_framed_v2_20261011'
    if dev:OUT=audit.ROOT/'visible_approach_framed_v2_dev_20261011'
    if expand:OUT=audit.ROOT/'visible_approach_framed_v2_train_expand_20261011'
    if selection_manifest:
        assert framed and not dev and not expand and output
        OUT=Path(output)
    assert not OUT.exists(),'refuse to overwrite approach pilot'
    OUT.mkdir()
    selected={r['index']:r for r in map(json.loads,(audit.SOURCE/'selection_manifest.jsonl').read_text().splitlines())}
    indices=[1,8,11]
    if framed:
        parent=audit.SOURCE/'model_manifest_frozen.jsonl'
        assert audit.sha(parent)=='d52833eafdb15ce8bd2a12dfcd5e17d12db6e9fcf670576cf02c2bbdf6c37a7b'
        bank={r['index']:r for r in map(json.loads,parent.read_text().splitlines())}
        indices=[1,11]
        for i in (0,1,2,4,8,9):
            row=bank[i];task=json.loads(Path(row['source_task_json']).read_text())
            assert row['split']=='train'
            selected[100+i]={**row,'source_task_sha256':row['source_sha256'],'split':'train',
                              'start':task['trial']['trial_0']['pos'][0],'parent_bank_index':i}
            indices.append(100+i)
        assert len({selected[i]['scene'] for i in indices})==6
        if dev:
            indices=[]
            train_scenes={r['scene'] for r in bank.values() if r['split']=='train'}
            for i,row in sorted(bank.items()):
                if row['split']!='dev':continue
                task=json.loads(Path(row['source_task_json']).read_text())
                assert row['scene'] not in train_scenes
                selected[200+i]={**row,'source_task_sha256':row['source_sha256'],
                    'start':task['trial']['trial_0']['pos'][0],'parent_bank_index':i}
                indices.append(200+i)
            assert indices==[203,207]
        # The known Kitchen/bathroom mismatch remains held; framing cannot repair its instruction.
        if expand:
            indices=[0,2,3,4]
            forbidden={r['scene'] for r in bank.values() if r['split']=='dev'}
            canonical=json.loads((audit.ROOT/'spatial_clip_v3_canonical100_aabb_20261011'/'episode_bank.json').read_text())
            forbidden.update(r['scene'] for r in canonical)
            old=json.loads((audit.ROOT/'visible_approach_framed_v2_20261011'/'construction_ledger.json').read_text())
            assert {selected[i]['task_id'] for i in indices}.isdisjoint(r['task_id'] for r in old)
            assert all(selected[i]['split']=='train' and selected[i]['scene'] not in forbidden for i in indices)
    if selection_manifest:
        selection_rows=[json.loads(l) for l in Path(selection_manifest).read_text().splitlines()]
        selected={r['index']:r for r in selection_rows}
        indices=list(selected)
        assert len(selected)==len(selection_rows) and all(r['split'] in ('train','dev') for r in selection_rows)
    protocol={'version':version,'source_indices':indices,'scope':'train-only goal construction; Codex RGB review pending; no policy scores',
        'selection_sha256':audit.sha(audit.SOURCE/'selection_manifest.jsonl'),
        'query_height_offsets_m':[-3.,-2.,-1.5,-1.,-.5,0.],
        'query_horizontal_offsets_m':[[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],
        'candidate_rule':'finite navigable snapped query; reachable from original start; foot point within 2 m of instance OBB surface',
        'view_rule':'12 yaw headings; individual instance >=25 and <=20% image pixels',
        'ranking':'one position/heading per instance; closest target occupancy to 5%, then horizontal center distance, position and yaw',
        'stage_goal_set':'all qualifying target instances in the original active region; fail task if either stage has zero qualifying instances',
        'STOP_rule':'strict geodesic <1 m to any frozen stage approach position; heading not scored',
        'horizon':400,'stops_per_episode':2,'goal_construction_before_rollout':True,
        'policy_inputs':['original instruction','robot','RGB','relative pose/yaw','own action history'],
        'forbidden_policy_inputs':['goal positions','target regions','semantic masks','NavMesh','success flags'],
        'historical_resolvers_modified':False,'training_performed':False,'script_sha256':audit.sha(Path(__file__))}
    if framed:
        protocol.update({'parent_manifest_sha256':audit.sha(parent),'border_margin_pixels':8,'maximum_surface_distance_m':3.,
            'query_horizontal_offsets_m':[[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[2.,0.],[-2.,0.],[0.,2.],[0.,-2.]],
            'candidate_rule':'same finite/reachable checks; foot point within 3 m of instance OBB surface',
            'view_rule':'same 25..20% occupancy plus all bounding-box edges >=8 pixels from image edges',
            'ranking':'same 5% occupancy preference, among framed views only',
            'selection_rule':'held painting task, accepted flowerpot task, and first original parent task from each of six safe train scenes; fixed before v2 results',
            'excluded_known_language_hold':[8]})
    if expand:
        protocol.update({'selection_rule':'first four previously held selection indices not used in v2: 0,2,3,4; fixed before construction',
                         'split':'train','excluded_known_language_hold':[]})
    if dev or expand:
        if dev:protocol.update({'scope':'fixed dev-only goal audit; never training supervision; no policy scores',
            'split':'dev','selection_rule':'all two original dev tasks from frozen parent bank',
            'excluded_known_language_hold':[]})
        train_protocol=json.loads((audit.ROOT/'visible_approach_framed_v2_20261011'/'protocol.json').read_text())
        for key in ('version','query_height_offsets_m','query_horizontal_offsets_m','candidate_rule','view_rule',
                    'ranking','stage_goal_set','STOP_rule','horizon','stops_per_episode','border_margin_pixels',
                    'maximum_surface_distance_m','policy_inputs','forbidden_policy_inputs'):
            assert protocol[key]==train_protocol[key],key
        protocol['train_protocol_sha256']=audit.sha(audit.ROOT/'visible_approach_framed_v2_20261011'/'protocol.json')
    if selection_manifest:
        protocol.update({'selection_sha256':audit.sha(Path(selection_manifest)),
                         'selection_rule':'fixed external selection; no model-score filtering',
                         'scope':'train/dev goal audit; dev never training; no policy scores',
                         'excluded_known_language_hold':[]})
    assert framed_box([8,8,504,504],512,512) and not framed_box([7,8,504,504],512,512)
    assert not framed_box([8,8,505,504],512,512)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    args=audit.driver.probe.load_legacy_module(audit.driver.probe.ROOT).make_args()
    args.success_dis,args.max_step=1.,400
    audit.resolver.install(audit.evaluator,'obb_v1')
    records=[]
    for index in protocol['source_indices']:
        row=selected[index];source=Path(row['source_task_json'])
        assert audit.sha(source)==row['source_task_sha256']
        assert row['split'] in ('train','dev') if selection_manifest else row['split']==('dev' if dev else 'train')
        task=json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()):sim=audit.driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None
        record={'source_index':index,'task_id':row['task_id'],'scene':row['scene'],'source_task_json':str(source),
                'source_sha256':audit.sha(source),'config_fingerprint':fingerprint(task),'goal_resolver':version,
                'instruction_original':row['instruction_original'],'start':row['start'],'split':row['split'],
                'stages':[],'RGB_review':'pending'}
        try:
            for stage in range(2):
                sim.stage=stage;goals=[];instances=[]
                objects=sorted(audit.resolver.objects_for_stage(sim,stage),key=lambda o:int(o.id.split('_')[-1]))
                assert 0<len(objects)<=50
                for obj in objects:
                    ident=int(obj.id.split('_')[-1]);center=audit.resolver.center(obj,'obb_v1');positions={}
                    for dy in protocol['query_height_offsets_m']:
                        for dx,dz in protocol['query_horizontal_offsets_m']:
                            pos=np.asarray(sim.pathfinder.snap_point(center+np.array([dx,dy,dz])),dtype=np.float32)
                            if not np.isfinite(pos).all() or not sim.pathfinder.is_navigable(pos):continue
                            if not math.isfinite(audit.geodesic(sim.pathfinder,row['start'],pos)):continue
                            if float(obj.obb.distance(mn.Vector3(*pos)))>protocol.get('maximum_surface_distance_m',2.):continue
                            positions.setdefault(tuple(np.round(pos,4)),pos)
                    choices=[];border_rejected=0
                    for pos in positions.values():
                        for turn in range(12):
                            sim.set_state(pos,quat_from_angle_axis(turn*math.pi/6,np.array([0.,1.,0.])))
                            obs=sim.sim.get_sensor_observations()
                            boxes=audit.driver.probe.prior.boxes(obs['semantic_sensor'],[ident])
                            pixels=max((b['pixels'] for b in boxes),default=0)
                            if not 25<=pixels<=.2*obs['semantic_sensor'].size:continue
                            if framed and not framed_box(boxes[0]['box'],obs['semantic_sensor'].shape[1],obs['semantic_sensor'].shape[0]):
                                border_rejected+=1;continue
                            choices.append({'position':list(map(float,pos)),'yaw_degrees':turn*30,'pixels':pixels,'instance_id':ident,
                                'surface_distance_m':float(obj.obb.distance(mn.Vector3(*pos))),
                                **({'target_box':boxes[0]['box']} if framed else {})})
                    instances.append({'instance_id':ident,'poses':len(positions),'qualifying_views':len(choices),
                                      **({'border_rejected_views':border_rejected} if framed else {})})
                    if choices:
                        choice=min(choices,key=lambda c:(abs(c['pixels']-.05*512*512),
                            math.hypot(c['position'][0]-center[0],c['position'][2]-center[2]),tuple(c['position']),c['yaw_degrees']))
                        goals.append(choice)
                        sim.set_state(choice['position'],quat_from_angle_axis(choice['yaw_degrees']*math.pi/180,np.array([0.,1.,0.])))
                        frame=Image.fromarray(np.asarray(sim.sim.get_sensor_observations()['color_sensor_f'])[...,:3])
                        frame.save(OUT/f'{index:03d}_{stage}_{ident}_approach.png')
                record['stages'].append({'stage':stage,'target':str(sim.target[stage]),'region':str(sim.region[stage]),'goals':goals,'instances':instances})
            record['both_stages_have_goals']=all(s['goals'] for s in record['stages'])
        finally:sim.close()
        records.append(record)
        (OUT/'construction_ledger.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({'index':index,'goals_per_stage':[len(s['goals']) for s in record['stages']]}),flush=True)
    manifest=OUT/'candidate_goal_manifest.json'
    manifest.write_text(json.dumps([r for r in records if r['both_stages_have_goals']],indent=2)+'\n')
    (OUT/'construction_summary.json').write_text(json.dumps({'tasks':len(indices),'complete_goal_tasks':sum(r['both_stages_have_goals'] for r in records),
        'manifest_sha256':audit.sha(manifest),'RGB_review_pending':True,'training_performed':False},indent=2)+'\n')


if __name__=='__main__':main()
