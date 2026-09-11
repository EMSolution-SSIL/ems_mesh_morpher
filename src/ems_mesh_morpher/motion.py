from __future__ import annotations

import math

import numpy as np

from .config import MotionConfig


def _fit_vector(
    value: tuple[float, ...], dimensions: int, key: str, fill: float = 0.0
) -> np.ndarray:
    if not value or len(value) > dimensions:
        raise ValueError(f"{key} must contain between one and {dimensions} values")
    result = np.full(dimensions, fill, dtype=float)
    result[: len(value)] = value
    return result


def translation_vector(displacement: tuple[float, ...], dimensions: int = 3) -> np.ndarray:
    return _fit_vector(displacement, dimensions, "displacement")


def translation_displacement_field(points: np.ndarray, displacement: tuple[float, ...]) -> np.ndarray:
    delta = translation_vector(displacement, points.shape[1])
    return np.broadcast_to(delta, points.shape).copy()


def rotation_displacement_field(
    points: np.ndarray,
    center: tuple[float, ...],
    angle_degrees: float,
    axis: tuple[float, ...] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    source = np.asarray(points, dtype=float)
    dimensions = source.shape[1]
    origin = _fit_vector(center, dimensions, "center")
    relative = source - origin
    angle = math.radians(float(angle_degrees))

    if dimensions == 2:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
        rotated = relative @ rotation.T
    elif dimensions == 3:
        axis_vector = _fit_vector(axis, 3, "axis")
        norm = float(np.linalg.norm(axis_vector))
        if norm == 0.0:
            raise ValueError("rotation axis must be non-zero")
        unit_axis = axis_vector / norm
        cosine = math.cos(angle)
        sine = math.sin(angle)
        cross = np.cross(np.broadcast_to(unit_axis, relative.shape), relative)
        dot = relative @ unit_axis
        rotated = (
            relative * cosine
            + cross * sine
            + dot[:, None] * unit_axis[None, :] * (1.0 - cosine)
        )
    else:
        raise ValueError("rotation supports only 2D or 3D point arrays")
    return rotated - relative


def scaling_displacement_field(
    points: np.ndarray, center: tuple[float, ...], scale: tuple[float, ...]
) -> np.ndarray:
    source = np.asarray(points, dtype=float)
    dimensions = source.shape[1]
    origin = _fit_vector(center, dimensions, "center")
    if len(scale) == 1:
        factors = np.full(dimensions, float(scale[0]), dtype=float)
    else:
        factors = _fit_vector(scale, dimensions, "scale", fill=1.0)
    relative = source - origin
    return relative * factors - relative


def prescribed_displacement_field(points: np.ndarray, motion: MotionConfig) -> tuple[np.ndarray, np.ndarray]:
    field = np.zeros_like(points, dtype=float)
    assigned = np.zeros(len(points), dtype=bool)
    for item in motion.point_displacements:
        index = int(item.point_index)
        if index < 0 or index >= len(points):
            raise ValueError(f"prescribed point index is outside the mesh: {index}")
        if assigned[index]:
            raise ValueError(f"duplicate prescribed displacement for point index {index}")
        field[index] = _fit_vector(item.displacement, points.shape[1], "prescribed displacement")
        assigned[index] = True
    if not np.any(assigned):
        raise ValueError("prescribed motion requires point_displacements")
    return field, assigned


def motion_displacement_field(points: np.ndarray, motion: MotionConfig) -> tuple[np.ndarray, np.ndarray | None]:
    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] not in {2, 3}:
        raise ValueError("mesh points must be an N x 2 or N x 3 array")
    if motion.type == "translation":
        return translation_displacement_field(source, motion.displacement), None
    if motion.type == "rotation":
        return rotation_displacement_field(
            source,
            center=motion.center,
            angle_degrees=motion.angle_degrees,
            axis=motion.axis,
        ), None
    if motion.type == "scaling":
        return scaling_displacement_field(source, center=motion.center, scale=motion.scale), None
    if motion.type == "prescribed":
        return prescribed_displacement_field(source, motion)
    raise ValueError(f"unsupported motion type: {motion.type}")


def apply_displacement(points: np.ndarray, node_mask: np.ndarray, displacement: np.ndarray) -> None:
    if displacement.shape != points.shape:
        raise ValueError("displacement field shape must match mesh points")
    points[node_mask] += displacement[node_mask]


def translate_points(points: np.ndarray, node_mask: np.ndarray, displacement: tuple[float, ...]) -> None:
    points[node_mask] += translation_vector(displacement, points.shape[1])
