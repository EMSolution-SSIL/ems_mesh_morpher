from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import meshio
import numpy as np
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RBFConfig:
    kernel: str = "wendland_c2"
    radius: float | None = None
    radius_scale: float = 1.25
    neighbors: int | None = 32
    regularization: float = 1.0e-12
    polynomial_degree: int = 1
    outside_policy: str = "nearest"
    query_chunk_size: int = 4096
    exact_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.kernel != "wendland_c2":
            raise ValueError("RBF kernel must be 'wendland_c2'")
        if self.radius is not None and (
            self.radius <= 0.0 or not np.isfinite(self.radius)
        ):
            raise ValueError("RBF radius must be a positive finite value")
        if self.radius_scale <= 1.0 or not np.isfinite(self.radius_scale):
            raise ValueError("RBF radius_scale must be finite and greater than 1")
        if self.neighbors is not None and (
            isinstance(self.neighbors, bool) or self.neighbors <= 0
        ):
            raise ValueError("RBF neighbors must be a positive integer or None")
        if self.regularization < 0.0 or not np.isfinite(self.regularization):
            raise ValueError("RBF regularization must be non-negative and finite")
        if self.polynomial_degree not in {-1, 0, 1}:
            raise ValueError("RBF polynomial_degree must be -1, 0, or 1")
        if self.outside_policy not in {"nearest", "zero", "error"}:
            raise ValueError("RBF outside_policy must be 'nearest', 'zero', or 'error'")
        if isinstance(self.query_chunk_size, bool) or self.query_chunk_size <= 0:
            raise ValueError("RBF query_chunk_size must be a positive integer")
        if self.exact_tolerance < 0.0 or not np.isfinite(self.exact_tolerance):
            raise ValueError("RBF exact_tolerance must be non-negative and finite")


@dataclass(frozen=True)
class RBFInterpolationResult:
    source_points: np.ndarray
    displacement: np.ndarray
    control_indices: np.ndarray
    fixed_indices: np.ndarray
    query_indices: np.ndarray
    config: RBFConfig
    resolved_radius: float

    @property
    def target_points(self) -> np.ndarray:
        return self.source_points + self.displacement

    def apply_to_mesh(self, mesh: meshio.Mesh) -> meshio.Mesh:
        points = np.asarray(mesh.points, dtype=float)
        if points.shape != self.source_points.shape or not np.allclose(
            points, self.source_points, rtol=0.0, atol=0.0
        ):
            raise ValueError("mesh points do not match the RBF source points")
        result = copy.deepcopy(mesh)
        result.points = self.target_points.copy()
        return result


@dataclass(frozen=True)
class _PolynomialTransform:
    center: np.ndarray
    axes: np.ndarray
    scales: np.ndarray
    degree: int

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        if self.degree < 0:
            return np.empty((len(points), 0), dtype=float)
        columns = [np.ones(len(points), dtype=float)]
        if self.degree == 1 and len(self.axes):
            coordinates = (points - self.center) @ self.axes.T
            columns.extend((coordinates / self.scales).T)
        return np.column_stack(columns)


@dataclass(frozen=True)
class _RBFModel:
    control_points: np.ndarray
    control_values: np.ndarray
    tree: cKDTree
    radius: float
    weights: np.ndarray
    polynomial_weights: np.ndarray
    polynomial: _PolynomialTransform
    config: RBFConfig


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
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


def _wendland_c2(normalized_distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(normalized_distance, dtype=float)
    remainder = np.maximum(1.0 - distance, 0.0)
    return remainder**4 * (4.0 * distance + 1.0)


def _polynomial_transform(points: np.ndarray, degree: int) -> _PolynomialTransform:
    center = np.mean(points, axis=0)
    if degree < 1 or len(points) == 1:
        return _PolynomialTransform(
            center=center,
            axes=np.empty((0, points.shape[1]), dtype=float),
            scales=np.empty(0, dtype=float),
            degree=degree,
        )

    centered = points - center
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        rank = 0
    else:
        tolerance = (
            max(centered.shape) * np.finfo(float).eps * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
    selected_axes = axes[:rank]
    coordinates = centered @ selected_axes.T
    scales = np.max(np.abs(coordinates), axis=0) if rank else np.empty(0)
    return _PolynomialTransform(center, selected_axes, scales, degree)


def _automatic_radius(
    tree: cKDTree,
    control_points: np.ndarray,
    query_points: np.ndarray,
    config: RBFConfig,
) -> float:
    candidates: list[float] = []
    if len(control_points) > 1:
        neighbor_count = (
            len(control_points)
            if config.neighbors is None
            else min(len(control_points), config.neighbors + 1)
        )
        distances, _ = tree.query(control_points, k=neighbor_count)
        if distances.ndim == 1:
            distances = distances[:, None]
        candidates.append(float(np.max(distances[:, -1])))
    if len(query_points):
        nearest, _ = tree.query(query_points, k=1)
        candidates.append(float(np.max(nearest)))

    base_radius = max(candidates, default=0.0)
    if base_radius <= 0.0:
        extent = float(np.linalg.norm(np.ptp(control_points, axis=0)))
        base_radius = extent if extent > 0.0 else 1.0
    return base_radius * config.radius_scale


def _kernel_matrix(
    first_tree: cKDTree,
    second_tree: cKDTree,
    radius: float,
) -> csr_matrix:
    distances = first_tree.sparse_distance_matrix(
        second_tree, radius, output_type="coo_matrix"
    )
    values = _wendland_c2(distances.data / radius)
    return csr_matrix(
        (values, (distances.row, distances.col)), shape=distances.shape
    )


def _fit_rbf(
    control_points: np.ndarray,
    control_values: np.ndarray,
    query_points: np.ndarray,
    config: RBFConfig,
) -> _RBFModel:
    if len(control_points) == 0:
        raise ValueError("RBF interpolation requires at least one control point")
    if len(np.unique(control_points, axis=0)) != len(control_points):
        raise ValueError("RBF control_points must not contain duplicate coordinates")

    tree = cKDTree(control_points)
    radius = (
        config.radius
        if config.radius is not None
        else _automatic_radius(tree, control_points, query_points, config)
    )
    kernel = _kernel_matrix(tree, tree, radius)
    if config.regularization:
        kernel = kernel + config.regularization * eye(len(control_points), format="csr")

    polynomial = _polynomial_transform(control_points, config.polynomial_degree)
    polynomial_matrix = polynomial.evaluate(control_points)
    polynomial_size = polynomial_matrix.shape[1]
    if polynomial_size:
        polynomial_sparse = csr_matrix(polynomial_matrix)
        system = bmat(
            [
                [kernel, polynomial_sparse],
                [polynomial_sparse.T, None],
            ],
            format="csc",
        )
        right_hand_side = np.vstack(
            (
                control_values,
                np.zeros((polynomial_size, control_values.shape[1]), dtype=float),
            )
        )
    else:
        system = kernel.tocsc()
        right_hand_side = control_values

    try:
        solution = splu(system).solve(right_hand_side)
    except RuntimeError as exc:
        raise ValueError(
            "RBF system is singular; increase radius or regularization"
        ) from exc
    if not np.all(np.isfinite(solution)):
        raise ValueError("RBF solve produced non-finite coefficients")
    return _RBFModel(
        control_points=control_points,
        control_values=control_values,
        tree=tree,
        radius=float(radius),
        weights=solution[: len(control_points)],
        polynomial_weights=solution[len(control_points) :],
        polynomial=polynomial,
        config=config,
    )


def _evaluate_rbf(model: _RBFModel, query_points: np.ndarray) -> np.ndarray:
    result = np.zeros(
        (len(query_points), model.control_values.shape[1]), dtype=float
    )
    for start in range(0, len(query_points), model.config.query_chunk_size):
        stop = min(start + model.config.query_chunk_size, len(query_points))
        chunk = query_points[start:stop]
        query_tree = cKDTree(chunk)
        kernel = _kernel_matrix(query_tree, model.tree, model.radius)
        chunk_result = kernel @ model.weights
        if len(model.polynomial_weights):
            chunk_result += (
                model.polynomial.evaluate(chunk) @ model.polynomial_weights
            )

        supported = np.diff(kernel.indptr) > 0
        nearest_distance, nearest_index = model.tree.query(chunk, k=1)
        exact = nearest_distance <= model.config.exact_tolerance
        chunk_result[exact] = model.control_values[nearest_index[exact]]
        missing = ~supported & ~exact
        if np.any(missing):
            if model.config.outside_policy == "nearest":
                chunk_result[missing] = model.control_values[nearest_index[missing]]
            elif model.config.outside_policy == "zero":
                chunk_result[missing] = 0.0
            else:
                count = int(np.count_nonzero(missing))
                raise ValueError(
                    f"{count} query points have no RBF controls within radius"
                )
        result[start:stop] = chunk_result
    return result


def rbf_interpolate(
    query_points: np.ndarray,
    control_points: np.ndarray,
    control_values: np.ndarray,
    config: RBFConfig | None = None,
) -> np.ndarray:
    resolved = config or RBFConfig()
    queries = _point_array(query_points, "query_points")
    controls = _point_array(control_points, "control_points")
    if queries.shape[1] != controls.shape[1]:
        raise ValueError("query_points and control_points dimensions must match")
    values = np.asarray(control_values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(controls):
        raise ValueError("control_values must have one row per control point")
    if not np.all(np.isfinite(values)):
        raise ValueError("control_values must contain only finite values")
    model = _fit_rbf(controls, values, queries, resolved)
    return _evaluate_rbf(model, queries)


def build_rbf_displacement_field(
    points: np.ndarray,
    control_indices: Iterable[int],
    *,
    control_displacements: np.ndarray | None = None,
    control_target_points: np.ndarray | None = None,
    fixed_indices: Iterable[int] = (),
    query_indices: Iterable[int] | None = None,
    config: RBFConfig | None = None,
) -> RBFInterpolationResult:
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

    resolved = config or RBFConfig()
    model = _fit_rbf(source[known], known_values, source[queries], resolved)
    displacement = np.zeros_like(source, dtype=float)
    displacement[controls] = values
    if len(queries):
        displacement[queries] = _evaluate_rbf(model, source[queries])
    return RBFInterpolationResult(
        source_points=source.copy(),
        displacement=displacement,
        control_indices=controls,
        fixed_indices=fixed,
        query_indices=queries,
        config=resolved,
        resolved_radius=model.radius,
    )
