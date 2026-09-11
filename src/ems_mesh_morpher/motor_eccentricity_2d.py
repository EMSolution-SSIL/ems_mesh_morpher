from __future__ import annotations

import meshio

from .config import MorphConfig
from .mesh_motion import MeshMotionResult, apply_mesh_motion


MorphResult = MeshMotionResult


def apply_motor_eccentricity_2d(mesh: meshio.Mesh, config: MorphConfig) -> MorphResult:
    if config.mode != "motor_eccentricity_2d":
        raise ValueError("apply_motor_eccentricity_2d requires motor_eccentricity_2d mode")
    return apply_mesh_motion(mesh, config)
