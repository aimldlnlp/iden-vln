#!/usr/bin/env python3
"""Exercise ordered endpoint STOP against the real SceneSimulator on four tasks."""

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/home/aimldl/LH-VLN")
sys.path.insert(0, "/home/aimldl/LH-VLN/runs/idenvln_ordered_eval_20260804_234338")

from habitat_base.simulation import SceneSimulator
from run_ordered_metrics_eval import install_index_aware_target_patch

from audit_release_simulator_assets import NEW, OLD, rows
from pilot_level_balanced_route import BANK, ROOT, geodesic, source_path, stop_transition


def reset(sim, start):
    sim.set_state(start, [1.0, 0.0, 0.0, 0.0])  # Recorded initial yaw is 180°: generator's identity rotation.
    sim.observations = sim.sim.get_sensor_observations()
    sim.step = -1
    sim.stage = 0
    sim.successes = [False] * len(sim.target)
    sim.oracle_successes = [False] * len(sim.target)
    sim.nav_steps = []
    sim.nav_errors = []
    sim.info = sim.get_info()
    sim.gt_path = [sim.info["geo dis"]]
    sim.done = False
    sim.episode_over = False
    obs, _, _ = sim.actor("stop")  # Initialization at step -1; does not count as a decision.
    assert sim.step == 0 and sim.stage == 0
    return obs


def route_stop(sim, endpoint):
    before = sim.stage
    distance = geodesic(sim.pathfinder, sim.agent.get_state().position, endpoint)
    after, outcome = stop_transition(before, len(sim.target), distance)
    if outcome == "failed_stop":
        sim.episode_over = True
    else:
        sim.actor("stop")
    assert sim.stage == after, (before, after, sim.stage, outcome)
    assert sim.done == (outcome == "completed")
    return round(distance, 4), outcome


def main():
    install_index_aware_target_patch()
    bank, old, new = rows(BANK), rows(OLD), rows(NEW)
    args = SimpleNamespace(scene=str(ROOT / "data/hm3d") + "/",
                           scene_dataset=str(ROOT / "data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"),
                           max_step=750, success_dis=1.0)
    selected = bank if "--stationary" in sys.argv else [next(x for x in bank if x["level"] == level) for level in range(2, 6)]
    for record in selected:
        level = record["level"]
        task = json.loads(source_path(record, old, new).read_text())
        paths = [task["trial"][f"trial_{i}"]["pos"] for i in range(level)]
        assert task["trial"]["trial_0"]["yaw"][0] == 180
        with contextlib.redirect_stdout(io.StringIO()):
            sim = SceneSimulator(args=args, config=task)
            try:
                reset(sim, paths[0][0])
                if "--stationary" in sys.argv:
                    events = []
                    for positions in paths:
                        events.append(route_stop(sim, positions[-1]))
                        if sim.episode_over:
                            break
                    result = {"task_id": record["task_id"], "level": level,
                              "stationary_complete": sim.done, "completed_stages": sim.stage,
                              "events": events}
                else:
                    successes = []
                    for positions in paths:
                        sim.set_state(positions[-1], [1.0, 0.0, 0.0, 0.0])
                        sim.observations = sim.sim.get_sensor_observations()
                        successes.append(route_stop(sim, positions[-1]))
                    assert sim.done and sim.stage == level
                    reset(sim, paths[0][0])
                    initial = route_stop(sim, paths[0][-1])
                    if initial[0] > 3.0:
                        assert initial[1] == "failed_stop" and sim.stage == 0 and sim.episode_over
                    result = {"task_id": record["task_id"], "level": level,
                              "success_stops": successes, "initial_stop": initial,
                              "start_position": paths[0][0]}
            finally:
                sim.close()
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
