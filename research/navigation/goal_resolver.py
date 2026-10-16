#!/usr/bin/env python3
"""Versioned object-goal geometry for ordered IDEN-VLN evaluation."""
import numpy as np

VERSIONS = ('aabb_v0', 'obb_v1')


def center(obj, version):
    if version == 'aabb_v0':
        value = obj.aabb.center
    elif version == 'obb_v1':
        value = obj.obb.center
    else:
        raise ValueError(f'unknown goal version: {version}')
    result = np.asarray(value() if callable(value) else value, dtype=np.float32)
    if version == 'obb_v1':
        sizes = np.asarray(obj.obb.sizes, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all() or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError(f'invalid OBB geometry: {obj.id}')
    return result


def goals_for_stage(sim, index, version):
    objects = objects_for_stage(sim, index)
    if version == 'obb_v1' and not objects:
        raise ValueError(f'missing target instances at stage {index}')
    goals = [np.asarray(sim.pathfinder.snap_point(center(obj, version)), dtype=np.float32) for obj in objects]
    if version == 'obb_v1' and any(not np.isfinite(g).all() for g in goals):
        raise ValueError(f'nonfinite OBB snap at stage {index}')
    return goals


def objects_for_stage(sim, index):
    label = sim.target[index]
    region_id = str(sim.region[index])
    return [obj for region in sim.sim.semantic_scene.regions
            if str(region.id)[1:] == region_id
            for obj in region.objects
            if obj.category is not None and obj.category.name() == label]


def install(evaluator, version):
    """Patch simulator STOP and evaluator distances in the current process."""
    if version not in VERSIONS:
        raise ValueError(f'unknown goal version: {version}')
    from habitat_base.simulation import SceneSimulator

    def get_coord(self, obj_target):
        if not self.target:
            return []
        index = min(max(int(self.stage), 0), len(self.target) - 1)
        if self.target[index] != obj_target:
            matches = [i for i, label in enumerate(self.target) if label == obj_target]
            index = matches[0] if matches else index
        if version == 'obb_v1':
            goals_for_stage(self, index, version)
        return [center(obj, version) for obj in objects_for_stage(self, index)]

    def target_coords_for_index(sim, index):
        return goals_for_stage(sim, index, version)

    SceneSimulator.get_coord = get_coord
    evaluator.target_coords_for_index = target_coords_for_index
