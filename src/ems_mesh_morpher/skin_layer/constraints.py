from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlaneConstraint:
    normal: tuple[float, float, float]
    offset: float
    tolerance: float = 1.0e-9
    name: str = ""

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("plane normal must contain three finite values")
        if float(np.linalg.norm(normal)) <= np.finfo(float).eps:
            raise ValueError("plane normal must be non-zero")
        if self.tolerance < 0.0:
            raise ValueError("plane tolerance must be non-negative")

    @property
    def unit_normal(self) -> np.ndarray:
        normal = np.asarray(self.normal, dtype=float)
        return normal / np.linalg.norm(normal)

    @property
    def unit_offset(self) -> float:
        return float(self.offset) / float(np.linalg.norm(self.normal))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.unit_normal - self.unit_offset

    def matches(self, points: np.ndarray) -> bool:
        return bool(np.all(np.abs(self.signed_distance(points)) <= self.tolerance))


def project_vector_to_planes(
    vector: np.ndarray,
    planes: Sequence[PlaneConstraint],
    *,
    rank_tolerance: float = 1.0e-12,
) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,):
        raise ValueError("vector must contain three values")
    if not planes:
        return value.copy()
    normals = np.vstack([plane.unit_normal for plane in planes])
    _, singular_values, vh = np.linalg.svd(normals, full_matrices=True)
    if singular_values.size == 0:
        return value.copy()
    threshold = rank_tolerance * max(normals.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    row_basis = vh[:rank]
    return value - row_basis.T @ (row_basis @ value)


def apply_directional_constraints(
    displacements: np.ndarray,
    plane_constraints: Mapping[int, Sequence[PlaneConstraint]],
    *,
    fixed_nodes: Sequence[int] = (),
) -> np.ndarray:
    result = np.asarray(displacements, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("displacements must be an N x 3 array")
    for node_index, planes in plane_constraints.items():
        result[int(node_index)] = project_vector_to_planes(result[int(node_index)], planes)
    if fixed_nodes:
        result[np.asarray(tuple(fixed_nodes), dtype=int)] = 0.0
    return result
