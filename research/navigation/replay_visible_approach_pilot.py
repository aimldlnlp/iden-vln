"""Freeze diagnostic goals, demonstrate ordered navigation and independently replay actions."""
import contextlib
import io
import json
import math
import shutil
import sys
from pathlib import Path
import habitat_sim
import numpy as np
from PIL import Image,ImageDraw
import diagnose_held_train_targets as audit
from frozen_approach_resolver import install,VERSION
from run_single_goal_route_pilot import reset
from build_visible_approach_pilot import framed_box

OUT=audit.ROOT/'visible_approach_pilot_20261011'


def main(output=None):
    global OUT
    framed='--framed-v2' in sys.argv
    assert sys.argv[1:] in ([],['--framed-v2'],['--framed-v2','--dev'],['--framed-v2','--expand-train'])
    if framed:OUT=audit.ROOT/'visible_approach_framed_v2_20261011'
    if '--dev' in sys.argv:OUT=audit.ROOT/'visible_approach_framed_v2_dev_20261011'
    if '--expand-train' in sys.argv:OUT=audit.ROOT/'visible_approach_framed_v2_train_expand_20261011'
    if output:
        assert framed
        OUT=Path(output)
    version=json.loads((OUT/'protocol.json').read_text())['version']
    manifest=OUT/'frozen_diagnostic_goal_manifest.json'
    assert not manifest.exists(),'refuse to replace frozen goals'
    shutil.copyfile(OUT/'candidate_goal_manifest.json',manifest)
    goal_hash=audit.sha(manifest)
    records=json.loads(manifest.read_text())
    assert records
    if not framed:assert len(records)==3
    goals=install(audit.evaluator,manifest,version)
    args=audit.driver.probe.load_legacy_module(audit.driver.probe.ROOT).make_args()
    args.success_dis,args.max_step=1.,400
    results=[]
    def visible_pixels(obs,ids):
        boxes=audit.driver.probe.prior.boxes(obs['semantic_sensor'],ids)
        if not framed:return max((b['pixels'] for b in boxes),default=0)
        valid=[b for b in boxes if 25<=b['pixels']<=.2*obs['semantic_sensor'].size and
               framed_box(b['box'],obs['semantic_sensor'].shape[1],obs['semantic_sensor'].shape[0])]
        return min(valid,key=lambda b:abs(b['pixels']-.05*512*512))['pixels'] if valid else 0
    for row in records:
        path=Path(row['source_task_json']);assert audit.sha(path)==row['source_sha256']
        task=json.loads(path.read_text())
        with contextlib.redirect_stdout(io.StringIO()):sim=audit.driver.probe.SceneSimulator(args=args,config=task)
        sim.save_path=None
        trace=[];stops=[];failure=None
        try:
            with contextlib.redirect_stdout(io.StringIO()):reset(sim,row['start'])
            def act(action,stage):
                assert len(trace)<400,'teacher_horizon'
                sim.agent.act(action)
                trace.append({'action':action,'stage':stage,'position':list(map(float,sim.agent.get_state().position))})
            follower=habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder,sim.agent,goal_radius=.15,
                stop_key='stop',forward_key='move_forward',left_key='turn_left',right_key='turn_right')
            for stage in range(2):
                sim.stage=stage
                target=min(goals(sim,stage),key=lambda g:audit.geodesic(sim.pathfinder,sim.agent.get_state().position,g))
                while True:
                    action=follower.next_action_along(target)
                    if action=='stop':break
                    act(action,stage)
                ids,_=audit.driver.probe.prior.instances(sim,stage)
                choices=[]
                for turn in range(12):
                    obs=sim.sim.get_sensor_observations()
                    pixels=visible_pixels(obs,ids)
                    if 25<=pixels<=.2*obs['semantic_sensor'].size:choices.append((abs(pixels-.05*512*512),turn))
                    act('turn_right',stage)
                assert choices,'no_visible_STOP_view_after_follower'
                turn=min(choices)[1]
                for _ in range(min(turn,12-turn)):act('turn_right' if turn<=6 else 'turn_left',stage)
                with contextlib.redirect_stdout(io.StringIO()):distance=float(sim.get_info()['geo dis'])
                pos=sim.agent.get_state().position
                expected=min(audit.geodesic(sim.pathfinder,pos,g) for g in goals(sim,stage))
                assert distance<1 and abs(distance-expected)<1e-5
                obs=sim.sim.get_sensor_observations()
                pixels=visible_pixels(obs,ids)
                assert 25<=pixels<=52428.8
                for code in ('l','f','r'):
                    Image.fromarray(np.asarray(obs['color_sensor_'+code])[...,:3]).save(OUT/f'{row["source_index"]:03d}_{stage}_STOP_{code}.png')
                previous={v:min(audit.geodesic(sim.pathfinder,pos,g) for g in audit.resolver.goals_for_stage(sim,stage,v)) for v in ('aabb_v0','obb_v1')}
                stops.append({'stage':stage,'approach_geo_m':distance,'front_target_pixels':pixels,
                              'historical_distances_m':{v:audit.finite(d) for v,d in previous.items()},
                              **({'framing_gate_pass':True} if framed else {})})
                assert len(trace)<400
                trace.append({'action':'stop','stage':stage,'position':list(map(float,pos))})
            assert len(trace)<=400 and [a['stage'] for a in trace if a['action']=='stop']==[0,1]
        except (AssertionError,RuntimeError) as error:failure=type(error).__name__+': '+str(error)
        finally:sim.close()
        if failure is None:
            with contextlib.redirect_stdout(io.StringIO()):
                sim=audit.driver.probe.SceneSimulator(args=args,config=task)
                reset(sim,row['start'])
            errors=[];replayed_stops=[]
            try:
                for item in trace:
                    sim.stage=item['stage']
                    if item['action']=='stop':
                        with contextlib.redirect_stdout(io.StringIO()):distance=float(sim.get_info()['geo dis'])
                        expected=min(audit.geodesic(sim.pathfinder,sim.agent.get_state().position,g) for g in goals(sim,sim.stage))
                        assert distance<1 and abs(distance-expected)<1e-5
                        replayed_stops.append(distance)
                    else:sim.agent.act(item['action'])
                    error=math.dist(sim.agent.get_state().position,item['position']);assert error<1e-5;errors.append(error)
                assert len(replayed_stops)==2
            finally:sim.close()
        result={**row,'teacher_actions':trace,'teacher_failure':failure,'independent_replay_pass':failure is None,
                'max_replay_error_m':max(errors) if failure is None else None,'stops':stops,
                'goal_manifest_sha256':goal_hash,'training_approved':False,'human_validated':False}
        results.append(result)
        (OUT/'demonstration_candidates.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({'source_index':row['source_index'],'actions':len(trace),'failure':failure,'stops':stops}),flush=True)
    assert audit.sha(manifest)==goal_hash
    (OUT/'replay_summary.json').write_text(json.dumps({'tasks':len(records),'independent_replay_pass':sum(r['independent_replay_pass'] for r in results),
        'goal_resolver':version,'frozen_goal_sha256':goal_hash,'goal_manifest_unchanged_during_replay':True,
        'training_performed':False,'RGB_and_language_review_pending':True,'script_sha256':audit.sha(Path(__file__))},indent=2)+'\n')


if __name__=='__main__':main()
