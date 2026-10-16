"""Audit cohort boxes, rescore old demonstrations, then rebuild/replay under OBB v1."""
import contextlib
import io
import json
import math
from pathlib import Path

import magnum as mn
import numpy as np
import trimesh
import audit_level2_train_cohort as driver
import goal_resolver as resolver
import run_ordered_metrics_eval as evaluator
from build_route_instruction_pilot import sha
from run_single_goal_route_pilot import reset
from pilot_level_balanced_route import geodesic

ROOT = Path('/home/aimldl/LH-VLN')
OLD = ROOT / 'runs/level2_train_cohort_audit_20260930'
OUT = ROOT / 'runs/level2_train_cohort_obb_v1_1_20260930'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def mesh_bounds(mesh_path, colors):
    scene = trimesh.load(mesh_path, force='scene', process=False)
    bounds, vertices = {}, {}
    for node in scene.graph.nodes_geometry:
        matrix, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        vc = mesh.visual.to_color().vertex_colors if mesh.visual.kind == 'texture' else getattr(mesh.visual, 'vertex_colors', None)
        if vc is None or len(vc) != len(mesh.vertices): continue
        for color in colors:
            selected = np.all(np.asarray(vc)[:, :3] == np.array(list(bytes.fromhex(color))), axis=1)
            if not selected.any(): continue
            raw = trimesh.transform_points(np.asarray(mesh.vertices)[selected], matrix)
            points = raw[:, [0, 2, 1]] * [1, 1, -1]
            low, high = points.min(axis=0), points.max(axis=0)
            if color in bounds:
                low, high = np.minimum(low, bounds[color][0]), np.maximum(high, bounds[color][1])
            bounds[color] = (low, high)
            vertices.setdefault(color, []).append(points)
    return bounds, {c: np.concatenate(v) for c, v in vertices.items()}


def main():
    import csv
    OUT.mkdir(exist_ok=True)
    old_rows = json.loads((OLD / 'candidate_rows.json').read_text())
    assert len(old_rows) == 8 and all(r['split'] == 'train' for r in old_rows)
    protocol = {'version': 'level2_navigation_obb_v1_1', 'goal_resolver': 'obb_v1',
        'legacy_resolver': 'aabb_v0', 'mesh_check': 'OBB surface distance of raw target vertices; >0.1 m holds task; center difference diagnostic only', 'radius_strict_m': 1., 'horizon': 400,
        'target_rule': 'minimum geodesic to snapped OBB center of category in active target region',
        'stop_rule': 'each STOP closes a stage, final STOP closes episode',
        'policy_inputs': ['instruction', 'robot', 'RGB_l_f_r', 'relative_pose_yaw_history', 'own_action_history'],
        'forbidden_policy_inputs': ['target_category_list', 'target_regions', 'target_coordinates', 'semantic_masks', 'navmesh', 'success_flags'],
        'scope': 'train-only privileged geometry/replay diagnostic, language review still required',
        'resolver_sha256': sha(Path(resolver.__file__)), 'orchestration_sha256': sha(Path(__file__)), 'driver_sha256': sha(Path(driver.__file__)),
        'old_candidate_sha256': sha(OLD / 'candidate_rows.json'), 'training_ready': False}
    save('protocol.json', protocol)
    boxes, replays, cache = [], [], {}
    args = driver.probe.load_legacy_module(driver.probe.ROOT).make_args()
    args.success_dis, args.max_step = 1., 400
    resolver.install(evaluator, 'obb_v1')
    for row in old_rows:
        source = Path(row['source_task_json'])
        assert sha(source) == row['source_sha256']
        task = json.loads(source.read_text())
        with contextlib.redirect_stdout(io.StringIO()): sim = driver.probe.SceneSimulator(args=args, config=task)
        sim.save_path = None
        try:
            directory = Path(sim.sim_settings['scene'])
            if not directory.is_dir(): directory = directory.parent
            stem = row['scene'].split('-')[-1]
            mesh = directory / (stem + '.semantic.glb')
            if row['scene'] not in cache:
                with (directory / (stem + '.semantic.txt')).open() as f:
                    annotations = {int(r[0]): r for r in csv.reader(f) if r and r[0].strip().isdigit()}
                # Gather colors for this scene's task targets before loading the mesh once.
                targets = {(s['target']['object'], s['gold_region']) for rr in old_rows if rr['scene'] == row['scene'] for s in rr['stages']}
                ids = {int(o.id.split('_')[-1]) for o in sim.sim.semantic_scene.objects
                       if o is not None and o.region is not None and o.category is not None
                       and (o.category.name(), o.region.id.split('_')[-1]) in targets}
                colors = {annotations[i][1].strip().lstrip('#') for i in ids}
                bb, vv = mesh_bounds(mesh, colors)
                cache[row['scene']] = (annotations, bb, vv, sha(mesh))
            annotations, bounds, vertices, mesh_sha = cache[row['scene']]
            for stage in range(2):
                sim.stage = stage
                for obj in resolver.objects_for_stage(sim, stage):
                    ident = int(obj.id.split('_')[-1])
                    color = annotations[ident][1].strip().lstrip('#')
                    assert color in bounds, f'missing raw mesh color {ident}'
                    low, high = bounds[color]
                    mc = (low + high) / 2
                    ac, oc = resolver.center(obj, 'aabb_v0'), resolver.center(obj, 'obb_v1')
                    matrix = np.asarray(obj.obb.world_to_local, dtype=float)
                    assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
                    # Installed Habitat maps the OBB center to zero despite [0,1] in its docstring.
                    assert np.allclose((matrix @ np.r_[oc, 1])[:3], [0, 0, 0], atol=1e-4)
                    points = vertices[color]
                    local = (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]
                    excess = (local - np.clip(local, -1, 1)) * np.asarray(obj.obb.half_extents)
                    distances = np.linalg.norm(excess, axis=1)
                    worst = int(np.argmax(distances))
                    native = float(obj.obb.distance(mn.Vector3(*points[worst])))
                    assert abs(native - distances[worst]) < 1e-4, (ident, native, distances[worst])
                    boxes.append({'index': row['index'], 'stage': stage, 'instance_id': ident,
                        'object': sim.target[stage], 'region': str(sim.region[stage]), 'source_sha256': sha(source),
                        'mesh_sha256': mesh_sha, 'mesh_center': mc.tolist(), 'mesh_bounds': [low.tolist(), high.tolist()],
                        'aabb_center': ac.tolist(), 'obb_center': oc.tolist(),
                        'aabb_mesh_center_error_m': float(np.linalg.norm(ac - mc)),
                        'obb_mesh_center_error_m': float(np.linalg.norm(oc - mc)),
                        'mesh_vertices': len(points), 'obb_mesh_max_surface_distance_m': float(distances.max()),
                        'obb_mesh_vertices_outside10cm_fraction': float(np.mean(distances > .1))})
                obbgoals = resolver.goals_for_stage(sim, stage, 'obb_v1')
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.set_state(task['trial'][f'trial_{stage}']['pos'][-1], [1., 0., 0., 0.])
                    info = sim.get_info()
                expected = min(geodesic(sim.pathfinder, sim.agent.get_state().position, g) for g in obbgoals)
                assert (math.isinf(info['geo dis']) and math.isinf(expected)) or abs(info['geo dis'] - expected) < 1e-5, (row['index'], stage, info['geo dis'], expected)
                assert np.allclose(evaluator.target_coords_for_index(sim, stage), obbgoals)
            if row['teacher_replay_pass']:
                with contextlib.redirect_stdout(io.StringIO()): reset(sim, task['trial']['trial_0']['pos'][0])
                stops, errors = [], []
                for item in row['teacher_actions']:
                    sim.stage = item['stage']
                    if item['action'] == 'stop':
                        old = resolver.goals_for_stage(sim, sim.stage, 'aabb_v0')
                        new = resolver.goals_for_stage(sim, sim.stage, 'obb_v1')
                        pos = sim.agent.get_state().position
                        a = min(geodesic(sim.pathfinder, pos, g) for g in old)
                        b = min(geodesic(sim.pathfinder, pos, g) for g in new)
                        assert abs(a - item['runtime_geo_m']) < 1e-5
                        stops.append({'stage': sim.stage, 'aabb_geo_m': a, 'obb_geo_m': b if math.isfinite(b) else None, 'obb_reachable': math.isfinite(b), 'aabb_valid': a < 1, 'obb_valid': b < 1})
                    else: sim.agent.act(item['action'])
                    errors.append(math.dist(sim.agent.get_state().position, item['position']))
                assert max(errors) < 1e-5 and len(stops) == 2
                replays.append({'index': row['index'], 'max_position_error_m': max(errors), 'stops': stops})
        finally: sim.close()
        save('box_audit.json', boxes)
        save('old_teacher_rescore.json', replays)
        print('box/replay audit', row['index'], flush=True)
    assert len(boxes) == 29 and len(replays) == 7
    # v1.1: center mismatch is diagnostic; actual surface mismatch holds training eligibility.
    mesh_holds = {b['index'] for b in boxes if b['obb_mesh_max_surface_distance_m'] > .1}
    driver.OUT = OUT
    driver.probe.install_index_aware_target_patch = lambda: resolver.install(evaluator, 'obb_v1')
    driver.main()
    candidates = json.loads((OUT / 'candidate_rows.json').read_text())
    for row in candidates:
        row['protocol_version'] = protocol['version']
        row['goal_resolver'] = 'obb_v1'
        if row['index'] in mesh_holds:
            row['hold_reasons'] = sorted(set(row['hold_reasons'] + ['mesh_outside_obb_gt10cm']))
            row['automatic_candidate_pass'] = False
    save('candidate_rows.json', candidates)
    summary = json.loads((OUT / 'summary.json').read_text())
    summary.update({'protocol_version': protocol['version'], 'instances_audited': len(boxes), 'mesh_surface_hold_tasks': sorted(mesh_holds),
        'obb_mesh_surface_agreement_under10cm_instances': sum(b['obb_mesh_max_surface_distance_m'] <= .1 for b in boxes),
        'obb_mesh_agreement_under10cm_instances': sum(b['obb_mesh_center_error_m'] < .1 for b in boxes),
        'aabb_mesh_error_over10cm_instances': sum(b['aabb_mesh_center_error_m'] > .1 for b in boxes),
        'old_teacher_stops_valid_obb': sum(s['obb_valid'] for r in replays for s in r['stops'])})
    from collections import Counter
    summary['hold_task_counts'] = dict(Counter(h for r in candidates for h in r['hold_reasons']))
    summary['automatic_candidate_pass_tasks'] = sum(r['automatic_candidate_pass'] for r in candidates)
    save('summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
