from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import meshio
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class IDWConfig:
    power: float = 2.0
    neighbors: int | None = 32
    radius: float | None = None
    smoothing: float = 0.0
    outside_policy: str = "nearest"
    query_chunk_size: int = 8192
    exact_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.power <= 0.0 or not np.isfinite(self.power):
            raise ValueError("IDW power must be a positive finite value")
        if self.neighbors is not None and (
            isinstance(self.neighbors, bool) or self.neighbors <= 0
        ):
            raise ValueError("IDW neighbors must be a positive integer or None")
        if self.radius is not None and (
            self.radius <= 0.0 or not np.isfinite(self.radius)
        ):
            raise ValueError("IDW radius must be a positive finite value")
        if self.smoothing < 0.0 or not np.isfinite(self.smoothing):
            raise ValueError("IDW smoothing must be a non-negative finite value")
        if self.outside_policy not in {"nearest", "zero", "error"}:
            raise ValueError("IDW outside_policy must be 'nearest', 'zero', or 'error'")
        if isinstance(self.query_chunk_size, bool) or self.query_chunk_size <= 0:
            raise ValueError("IDW query_chunk_size must be a positive integer")
        if self.exact_tolerance < 0.0 or not np.isfinite(self.exact_tolerance):
            raise ValueError("IDW exact_tolerance must be non-negative and finite")


@dataclass(frozen=True)
class IDWInterpolationResult:
    source_points: np.ndarray
    displacement: np.ndarray
    control_indices: np.ndarray
    fixed_indices: np.ndarray
    query_indices: np.ndarray
    config: IDWConfig

    @property
    def target_points(self) -> np.ndarray:
        return self.source_points + self.displacement

    def apply_to_mesh(self, mesh: meshio.Mesh) -> meshio.Mesh:
        points = np.asarray(mesh.points, dtype=float)
        if points.shape != self.source_points.shape or not np.allclose(
            points, self.source_points, rtol=0.0, atol=0.0
        ):
            raise ValueError("mesh points do not match the IDW source points")
        result = copy.deepcopy(mesh)
        result.points = self.target_points.copy()
        return result


def _point_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] not in {2, 3}:
        raise ValueError(f"{name} must be an N x 2 or N x 3 array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _index_array(
    values: Iterable[int],
    point_count: int,
    name: str,
    *,
    allow_empty: bool,
) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=int)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(result < 0) or np.any(result >= point_count):
        raise ValueError(f"{name} contains an index outside the point array")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return result


def idw_interpolate(
    query_points: np.ndarray,
    control_points: np.ndarray,
    control_values: np.ndarray,
    config: IDWConfig | None = None,
) -> np.ndarray:
    resolved = config or IDWConfig()
    query = _point_array(query_points, "query_points")
    controls = _point_array(control_points, "control_points")
    values = np.asarray(control_values, dtype=float)
    if query.shape[1] != controls.shape[1]:
        raise ValueError("query_points and control_points dimensions must match")
    if values.ndim != 2 or values.shape[0] != len(controls):
        raise ValueError("control_values must have one row per control point")
    if not np.all(np.isfinite(values)):
        raise ValueError("control_values must contain only finite values")
    if len(controls) == 0:
        raise ValueError("IDW interpolation requires at least one control point")
    if len(query) == 0:
        return np.empty((0, values.shape[1]), dtype=float)

    tree = cKDTree(controls)
    neighbor_count = min(
        len(controls) if resolved.neighbors is None else resolved.neighbors,
        len(controls),
    )
    upper_bound = np.inf if resolved.radius is None else resolved.radius
    scale = max(float(np.ptp(controls, axis=0).max()), 1.0)
    exact_tolerance = max(
        resolved.exact_tolerance,
        np.finfo(float).eps * scale * 16.0,
    )
    result = np.zeros((len(query), values.shape[1]), dtype=float)

    for start in range(0, len(query), resolved.query_chunk_size):
        stop = min(start + resolved.query_chunk_size, len(query))
        query_chunk = query[start:stop]
        distances, indices = tree.query(
            query_chunk,
            k=neighbor_count,
            distance_upper_bound=upper_bound,
            workers=1,
        )
        distances = np.asarray(distances, dtype=float)
        indices = np.asarray(indices, dtype=int)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        chunk_result = np.zeros((len(query_chunk), values.shape[1]), dtype=float)
        missing_rows: list[int] = []
        for row in range(len(query_chunk)):
            valid = np.isfinite(distances[row]) & (indices[row] < len(controls))
            if not np.any(valid):
                missing_rows.append(row)
                continue
            row_distances = distances[row, valid]
            row_indices = indices[row, valid]
            exact = row_distances <= exact_tolerance
            if np.any(exact):
                chunk_result[row] = np.mean(values[row_indices[exact]], axis=0)
                continue
            denominator = np.power(
                row_distances * row_distances + resolved.smoothing**2,
                0.5 * resolved.power,
            )
            weights = 1.0 / denominator
            chunk_result[row] = np.sum(
                values[row_indices] * weights[:, None], axis=0
            ) / float(np.sum(weights))

        if missing_rows:
            if resolved.outside_policy == "error":
                raise ValueError(
                    f"{len(missing_rows)} query points have no IDW controls within radius"
                )
            if resolved.outside_policy == "nearest":
                _, nearest = tree.query(query_chunk[missing_rows], k=1, workers=1)
                chunk_result[missing_rows] = values[np.asarray(nearest, dtype=int)]
        result[start:stop] = chunk_result
    return result


def build_idw_displacement_field(
    points: np.ndarray,
    control_indices: Iterable[int],
    *,
    control_displacements: np.ndarray | None = None,
    control_target_points: np.ndarray | None = None,
    fixed_indices: Iterable[int] = (),
    query_indices: Iterable[int] | None = None,
    config: IDWConfig | None = None,
) -> IDWInterpolationResult:
    source = _point_array(points, "points")
    controls = _index_array(
        control_indices, len(source), "control_indices", allow_empty=False
    )
    fixed = _index_array(fixed_indices, len(source), "fixed_indices", allow_empty=True)
    if np.intersect1d(controls, fixed).size:
        raise ValueError("control_indices and fixed_indices must not overlap")
    if (control_displacements is None) == (control_target_points is None):
        raise ValueError(
            "provide exactly one of control_displacements or control_target_points"
        )

    if control_target_points is not None:
        targets = np.asarray(control_target_points, dtype=float)
        if targets.shape != (len(controls), source.shape[1]):
            raise ValueError(
                "control_target_points must have one row per control index"
            )
        values = targets - source[controls]
    else:
        values = np.asarray(control_displacements, dtype=float)
        if values.shape != (len(controls), source.shape[1]):
            raise ValueError(
                "control_displacements must have one row per control index"
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("control displacement data must contain only finite values")

    known = np.concatenate((controls, fixed))
    known_values = np.vstack(
        (values, np.zeros((len(fixed), source.shape[1]), dtype=float))
    )
    if query_indices is None:
        query_mask = np.ones(len(source), dtype=bool)
        query_mask[known] = False
        queries = np.flatnonzero(query_mask)
    else:
        queries = _index_array(
            query_indices, len(source), "query_indices", allow_empty=True
        )
        if np.intersect1d(queries, known).size:
            raise ValueError("query_indices must not overlap control or fixed indices")

    displacement = np.zeros_like(source, dtype=float)
    displacement[controls] = values
    if len(queries):
        displacement[queries] = idw_interpolate(
            source[queries],
            source[known],
            known_values,
            config,
        )
    return IDWInterpolationResult(
        source_points=source.copy(),
        displacement=displacement,
        control_indices=controls,
        fixed_indices=fixed,
        query_indices=queries,
        config=config or IDWConfig(),
    )
