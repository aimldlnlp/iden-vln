"""Opt-in scorer for immutable, pre-rollout approach goals; historical resolvers stay separate."""
import hashlib
import json
import numpy as np

VERSION = 'visible_approach_v1'


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def install(evaluator, manifest, version=VERSION):
    from habitat_base.simulation import SceneSimulator
    records = json.loads(manifest.read_text())
    lookup = {r['config_fingerprint']:r for r in records}
    assert len(lookup)==len(records) and records
    for row in records:
        assert row['goal_resolver']==version
        for stage in row['stages']:
            assert stage['goals']
            assert np.isfinite(np.asarray([g['position'] for g in stage['goals']],dtype=float)).all()

    def goals(sim,index):
        record=lookup[fingerprint(sim.config)]  # Unknown tasks fail; no fallback to a different protocol.
        stage=record['stages'][index]
        assert stage['stage']==index and stage['target']==str(sim.target[index])
        assert stage['region']==str(sim.region[index])
        values=[np.asarray(g['position'],dtype=np.float32) for g in stage['goals']]
        assert all(sim.pathfinder.is_navigable(v) for v in values)
        assert all(np.allclose(sim.pathfinder.snap_point(v),v,atol=1e-5) for v in values)
        return values

    def get_coord(sim,target):
        index=min(max(int(sim.stage),0),len(sim.target)-1)
        assert str(sim.target[index])==str(target)
        return goals(sim,index)

    SceneSimulator.get_coord=get_coord
    evaluator.target_coords_for_index=goals
    return goals
