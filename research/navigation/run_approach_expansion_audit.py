"""Construct frozen goals, replay teachers and verify actual STOPs for a fixed batch."""
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import build_visible_approach_pilot as builder
import replay_visible_approach_pilot as replay
from frozen_approach_resolver import install


def main():
    out = builder.audit.ROOT / 'approach_expansion_audit_20261011_v1'
    selection = builder.audit.ROOT / 'approach_expansion_selection_20261011_v1'
    protocol = json.loads((selection / 'protocol.json').read_text())
    manifest = selection / 'selection_manifest.jsonl'
    assert builder.audit.sha(manifest) == protocol['selection_sha256']
    sys.argv = [sys.argv[0], '--framed-v2']
    builder.main(selection_manifest=manifest, output=out)
    replay.main(output=out)
    frozen = out / 'frozen_diagnostic_goal_manifest.json'
    goal_hash = builder.audit.sha(frozen)
    goals = install(builder.audit.evaluator, frozen, 'visible_approach_framed_v2')
    args = builder.audit.driver.probe.load_legacy_module(builder.audit.driver.probe.ROOT).make_args()
    args.success_dis, args.max_step = 1., 400
    records = []
    for row in json.loads((out / 'demonstration_candidates.json').read_text()):
        if not row['independent_replay_pass']:
            continue
        source = Path(row['source_task_json'])
        assert builder.audit.sha(source) == row['source_sha256']
        with contextlib.redirect_stdout(io.StringIO()):
            sim = builder.audit.driver.probe.SceneSimulator(args=args, config=json.loads(source.read_text()))
            sim.save_path = None
            replay.reset(sim, row['start'])
        stops = []
        max_error = 0.
        try:
            for item in row['teacher_actions']:
                assert sim.stage == item['stage']
                stage = sim.stage
                if item['action'] == 'stop':
                    expected = min(builder.audit.geodesic(sim.pathfinder, sim.agent.get_state().position, g)
                                   for g in goals(sim, stage))
                    assert expected < 1.
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.actor(item['action'])
                error = math.dist(sim.agent.get_state().position, item['position'])
                assert error < 1e-5
                max_error = max(max_error, error)
                if item['action'] == 'stop':
                    assert sim.successes[stage]
                    assert (sim.stage == 1 and not sim.done) if stage == 0 else sim.done
                    stops.append({'stage': stage, 'distance_m': expected, 'runtime_success': True})
            assert sim.done and all(sim.successes) and len(stops) == 2
            records.append({'source_index': row['source_index'], 'split': row['split'],
                            'stops': stops, 'max_position_error_m': max_error})
        finally:
            sim.close()
    assert builder.audit.sha(frozen) == goal_hash
    result = {'status': 'passed', 'tasks': len(records), 'runtime_STOPs_verified': 2*len(records),
              'records': records, 'manifest_sha256': goal_hash, 'manifest_unchanged': True,
              'RGB_review_pending': True, 'training_performed': False, 'script_sha256': builder.audit.sha(Path(__file__))}
    (out / 'runtime_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
