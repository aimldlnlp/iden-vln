"""Bounded teacher/visibility repair and full room-map audit of eight train tasks."""
import contextlib
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis
import probe_single_goal_final_visual_train as probe
from audit_original_instruction_contract import clauses, label_in
from build_route_instruction_pilot import sha
from pilot_level_balanced_route import geodesic
from run_single_goal_route_pilot import reset

ROOT = Path('/home/aimldl/LH-VLN')
OUT = ROOT / 'runs/level2_train_cohort_audit_20260930'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def geometry_followup():
    probe.install_index_aware_target_patch()
    args = probe.load_legacy_module(probe.ROOT).make_args()
    checks = []
    for row in json.loads((OUT / 'candidate_rows.json').read_text()):
        source = Path(row['source_task_json'])
        assert sha(source) == row['source_sha256']
        task = json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            sim = probe.SceneSimulator(args=args, config=task)
        sim.save_path = None
        try:
            for s in row['stages']:
                sim.stage = s['stage']
                pairs = []
                for center in sim.get_coord(sim.target[sim.stage]):
                    snap = sim.pathfinder.snap_point(np.asarray(center, dtype=np.float32))
                    if np.isfinite(snap).all():
                        pairs.append({'center': list(map(float, center)), 'snapped': list(map(float, snap)),
                                      'snap_minus_center_y_m': float(snap[1] - center[1])})
                checks.append({'index': row['index'], 'stage': sim.stage, 'object': s['target']['object'],
                    'source_endpoint': task['trial'][f'trial_{sim.stage}']['pos'][-1],
                    'selected_audit_pose': s['audit_best']['position'], 'center_snap_pairs': pairs})
        finally:
            sim.close()
    assert len(checks) == 16
    save('geometry_followup.json', checks)
    print(json.dumps({'stages_checked': len(checks), 'stages_with_any_snap_above_center_gt1_5m':
        sum(any(p['snap_minus_center_y_m'] > 1.5 for p in c['center_snap_pairs']) for c in checks)}), flush=True)


def main(review_subset=False, manifest_path=None):
    OUT.mkdir(exist_ok=True)
    votes_path = ROOT / 'nav_gen/scene/Per_Scene_Region_Weighted_Votes.csv'
    with votes_path.open() as f:
        rooms = {(r['Scene Name'], r['Region #']): r['Weighted Room Proposal'].strip().lower()
                 for r in csv.DictReader(f)}
    manifest = Path(manifest_path) if manifest_path is not None else ROOT / 'runs/route_instruction_pilot_v1_20260930/manifest_replayed.jsonl'
    bank = [json.loads(s) for s in manifest.read_text().splitlines()]
    selected = [r for r in bank if r['level'] == 2 and (manifest_path is not None or r['split'] == 'train')]
    if review_subset and manifest_path is None:
        selected = [r for r in selected if r['index'] in (1, 4, 5)]
    if manifest_path is None:
        assert [r['index'] for r in selected] == ([1, 4, 5] if review_subset else list(range(8)))
    else:
        assert selected and len({r['index'] for r in selected}) == len(selected)
    probe.install_index_aware_target_patch()
    probe.prior.add_side_semantics()
    args = probe.load_legacy_module(probe.ROOT).make_args()
    args.success_dis, args.max_step = 1., 400
    results = []
    overview = Image.new('RGB', (4 * 320, (math.ceil(2*len(selected)/4) if manifest_path is not None else 4) * 280), 'white')
    for row_number, row in enumerate(selected):
        source = Path(row['source_task_json'])
        assert sha(source) == row['source_task_sha256']
        task = json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            sim = probe.SceneSimulator(args=args, config=task)
        sim.save_path = None
        stages, trace, waypoints, failure = [], [], [], None
        try:
            objects = [{'name': o.category.name(), 'region': o.region.id.split('_')[-1]}
                       for o in sim.sim.semantic_scene.objects if o is not None and o.region is not None]
            pieces = clauses(row['instruction_original'])
            sim.pathfinder.seed(57031 + row['index'])
            for stage, target in enumerate(row['ordered_targets']):
                sim.stage = stage
                region = str(sim.region[stage])
                possible = sorted({o['region'] for o in objects if o['name'] == target['object']
                                   and rooms.get((row['scene'], o['region'])) == target['room']})
                unknown = sorted({o['region'] for o in objects if o['name'] == target['object']
                                  and (row['scene'], o['region']) not in rooms})
                ids, _ = probe.prior.instances(sim, stage)
                goals = [sim.pathfinder.snap_point(np.asarray(c, dtype=np.float32))
                         for c in sim.get_coord(sim.target[stage])]
                goals = [g for g in goals if np.isfinite(g).all()]
                assert ids and goals
                sites = [np.asarray(task['trial'][f'trial_{stage}']['pos'][-1], dtype=np.float32)]
                for goal in goals:
                    sites.append(goal)
                    sites.extend(sim.pathfinder.get_random_navigable_point_near(goal, .9, 100) for _ in range(24 if review_subset else 6))
                best, frame, count = None, None, 0
                for pos in sites:
                    if not np.isfinite(pos).all() or min(geodesic(sim.pathfinder, pos, g) for g in goals) >= .8:
                        continue
                    for turn in range(12):
                        sim.set_state(pos, quat_from_angle_axis(turn * math.pi / 6, np.array([0., 1., 0.])))
                        obs = sim.sim.get_sensor_observations()
                        boxes = probe.prior.boxes(obs['semantic_sensor'], ids)
                        pixels = max((b['pixels'] for b in boxes), default=0)
                        count += 1
                        if review_subset and (pixels < 25 or pixels > .2 * obs['semantic_sensor'].size):
                            continue
                        if best is None or pixels > best['pixels']:
                            best = {'position': list(map(float, pos)), 'pixels': pixels, 'turn': turn}
                            frame = Image.fromarray(np.asarray(obs['color_sensor_f'])[..., :3])
                            for b in boxes:
                                ImageDraw.Draw(frame).rectangle(b['box'], outline='red', width=2)
                assert best is not None, 'no finite audit sites within radius'
                name = f'{row["index"]:03d}_{stage}_best.png'
                frame.save(OUT / name)
                ordinal = 2 * (row_number if manifest_path is not None else row['index']) + stage
                x, y = (ordinal % 4) * 320, (ordinal // 4) * 280
                overview.paste(frame.resize((320, 256)), (x, y))
                ImageDraw.Draw(overview).text((x + 2, y + 258), f'{row["index"]}:{stage} {target["object"]} px={best["pixels"]}', fill='black')
                clause = pieces[stage] if stage < len(pieces) else ''
                stages.append({'stage': stage, 'target': target, 'gold_region': region,
                    'upstream_room': rooms.get((row['scene'], region)), 'matching_object_room_regions': possible,
                    'unknown_room_regions': unknown, 'original_clause_complete': label_in(clause, target['object']) and label_in(clause, target['room']),
                    'region_objects': sorted({o['name'] for o in objects if o['region'] == region}),
                    'audit_search_views': count, 'audit_best': best, 'best_image': name})
                waypoints.append(best['position'])
            with contextlib.redirect_stdout(io.StringIO()):
                reset(sim, row['start'])
            follower = habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder, sim.agent, goal_radius=.15,
                stop_key='stop', forward_key='move_forward', left_key='turn_left', right_key='turn_right')
            for stage, waypoint in enumerate(waypoints):
                sim.stage = stage
                while len(trace) < (362 if review_subset else 374):  # Reserve scans, STOPs, and review heading alignment within 400.
                    action = follower.next_action_along(np.asarray(waypoint, dtype=np.float32))
                    if action == 'stop':
                        break
                    sim.agent.act(action)
                    trace.append({'action': action, 'stage': stage, 'position': list(map(float, sim.agent.get_state().position))})
                else:
                    raise RuntimeError('teacher_horizon')
                ids, _ = probe.prior.instances(sim, stage)
                visible, best_turn = 0, 0
                for scan_turn in range(12):
                    obs = sim.sim.get_sensor_observations()
                    pixels = max((b['pixels'] for b in probe.prior.boxes(obs['semantic_sensor'], ids)), default=0)
                    if pixels > visible and (not review_subset or pixels <= .2 * obs['semantic_sensor'].size):
                        visible, best_turn = pixels, scan_turn
                    sim.agent.act('turn_right')
                    trace.append({'action': 'turn_right', 'stage': stage, 'position': list(map(float, sim.agent.get_state().position))})
                if review_subset:
                    # Align the real trace with the best context view; audit rotations alone are not teacher actions.
                    turn_action = 'turn_right' if best_turn <= 6 else 'turn_left'
                    for _ in range(min(best_turn, 12 - best_turn)):
                        sim.agent.act(turn_action)
                        trace.append({'action': turn_action, 'stage': stage, 'position': list(map(float, sim.agent.get_state().position))})
                    obs = sim.sim.get_sensor_observations()
                    stop_pixels = max((b['pixels'] for b in probe.prior.boxes(obs['semantic_sensor'], ids)), default=0)
                    assert 25 <= stop_pixels <= .2 * obs['semantic_sensor'].size, ('stop_context_view', stage, stop_pixels)
                    stages[stage]['stop_front_visible_pixels'] = stop_pixels
                with contextlib.redirect_stdout(io.StringIO()):
                    info = sim.get_info()
                assert info['geo dis'] < 1., ('teacher_stop_radius', stage, info['geo dis'])
                trace.append({'action': 'stop', 'stage': stage, 'position': list(map(float, sim.agent.get_state().position)),
                              'runtime_geo_m': float(info['geo dis']), 'scan_visible_pixels': visible})
                stages[stage]['teacher_scan_visible_pixels'] = visible
            assert len(trace) <= 400
            with contextlib.redirect_stdout(io.StringIO()):
                reset(sim, row['start'])
            errors = []
            for item in trace:
                sim.stage = item['stage']
                if item['action'] == 'stop':
                    with contextlib.redirect_stdout(io.StringIO()):
                        info = sim.get_info()
                    assert info['geo dis'] < 1.
                else:
                    sim.agent.act(item['action'])
                errors.append(math.dist(sim.agent.get_state().position, item['position']))
            assert max(errors) < 1e-5
        except (RuntimeError, AssertionError) as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            sim.close()
        holds = []
        if failure: holds.append('teacher_repair_or_replay_failed')
        for s in stages:
            if not s['original_clause_complete']: holds.append('incomplete_original_clause')
            if s['matching_object_room_regions'] != [s['gold_region']]: holds.append('ambiguous_or_mismatched_original_object_room')
            if s['unknown_room_regions']: holds.append('room_mapping_incomplete')
            if s.get('teacher_scan_visible_pixels', 0) < 25: holds.append('target_visibility_below25_in_repaired_scan')
        result = {'index': row['index'], 'task_id': row['task_id'], 'split': row['split'], 'scene': row['scene'],
            'instruction_original': row['instruction_original'], 'source_task_json': str(source),
            'source_sha256': sha(source), 'stages': stages, 'teacher_actions': trace,
            'teacher_replay_pass': failure is None, 'teacher_failure': failure,
            'replay_max_position_error_m': max(errors) if failure is None else None,
            'hold_reasons': sorted(set(holds)), 'automatic_candidate_pass': not holds,
            'human_validated': False, 'training_ready': False}
        results.append(result)
        save('candidate_rows.json', results)
        overview.save(OUT / 'overview.png')
        print(json.dumps({'index': row['index'], 'teacher_actions': len(trace), 'replay_pass': failure is None,
                          'holds': result['hold_reasons']}, ensure_ascii=False), flush=True)
    save('summary.json', {'tasks': len(results), 'stages': sum(len(r['stages']) for r in results),
        'room_csv_sha256': sha(votes_path), 'manifest_sha256': sha(manifest),
        'replay_pass_tasks': sum(r['teacher_replay_pass'] for r in results),
        'automatic_candidate_pass_tasks': sum(r['automatic_candidate_pass'] for r in results),
        'hold_task_counts': dict(Counter(h for r in results for h in r['hold_reasons'])),
        'training_ready_tasks': 0, 'scope': 'Engineering audit, bounded pose search, privileged teacher; visual/language review required'})


if __name__ == '__main__':
    assert sys.argv[1:] in ([], ['--geometry-only'])
    geometry_followup() if sys.argv[1:] else main()
