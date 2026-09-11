from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping

import meshio
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


_CELL_EDGES = {
    "line": ((0, 1),),
    "triangle": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "tetra": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    "hexahedron": (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ),
    "wedge": (
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5),
    ),
    "pyramid": (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 4), (2, 4), (3, 4),
    ),
}

@dataclass(frozen=True)
class LaplaceConfig:
    weighting: str = "inverse_distance"
    distance_power: float = 1.0
    minimum_distance: float = 1.0e-12
    element_size_power: float = 1.0
    element_aspect_power: float = 1.0

    def __post_init__(self) -> None:
        if self.weighting not in {
            "uniform",
            "inverse_distance",
            "inverse_distance_element",
        }:
            raise ValueError(
                "Laplace weighting must be 'uniform', 'inverse_distance', or "
                "'inverse_distance_element'"
            )
        if self.distance_power < 0.0 or not np.isfinite(self.distance_power):
            raise ValueError("Laplace distance_power must be non-negative and finite")
        if self.minimum_distance <= 0.0 or not np.isfinite(self.minimum_distance):
            raise ValueError("Laplace minimum_distance must be positive and finite")
        if self.element_size_power < 0.0 or not np.isfinite(self.element_size_power):
            raise ValueError("Laplace element_size_power must be non-negative and finite")
        if self.element_aspect_power < 0.0 or not np.isfinite(
            self.element_aspect_power
        ):
            raise ValueError(
                "Laplace element_aspect_power must be non-negative and finite"
            )


@dataclass(frozen=True)
class LaplaceInterpolationResult:
    source_points: np.ndarray
    displacement: np.ndarray
    control_indices: np.ndarray
    fixed_indices: np.ndarray
    query_indices: np.ndarray
    config: LaplaceConfig

    @property
    def target_points(self) -> np.ndarray:
        return self.source_points + self.displacement

    def apply_to_mesh(self, mesh: meshio.Mesh) -> meshio.Mesh:
        points = np.asarray(mesh.points, dtype=float)
        if points.shape != self.source_points.shape or not np.allclose(
            points, self.source_points, rtol=0.0, atol=0.0
        ):
            raise ValueError("mesh points do not match the Laplace source points")
        result = copy.deepcopy(mesh)
        result.points = self.target_points.copy()
        return result


@dataclass(frozen=True)
class PreparedLaplaceInterpolator:
    source_points: np.ndarray
    query_indices: np.ndarray
    control_indices: np.ndarray
    config: LaplaceConfig
    _factorization: object
    _boundary_matrix: csr_matrix

    def solve(self, control_values: np.ndarray) -> np.ndarray:
        values = np.asarray(control_values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or len(values) != len(self.control_indices):
            raise ValueError("control_values must have one row per control index")
        if not np.all(np.isfinite(values)):
            raise ValueError("control_values must contain only finite values")
        if not len(self.query_indices):
            return np.empty((0, values.shape[1]), dtype=float)
        result = self._factorization.solve(self._boundary_matrix @ values)
        if not np.all(np.isfinite(result)):
            raise ValueError("Laplace solve produced non-finite displacement")
        return np.asarray(result)


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


def _cell_mask(
    block_index: int,
    cell_count: int,
    cell_masks: Mapping[int, np.ndarray] | None,
) -> np.ndarray | None:
    if cell_masks is None:
        return np.ones(cell_count, dtype=bool)
    if block_index not in cell_masks:
        return None
    mask = np.asarray(cell_masks[block_index], dtype=bool)
    if mask.shape != (cell_count,):
        raise ValueError(f"cell mask {block_index} does not match its cell block")
    return mask


def _active_edges(
    mesh: meshio.Mesh,
    active: np.ndarray,
    cell_masks: Mapping[int, np.ndarray] | None,
) -> np.ndarray:
    candidates: list[tuple[int, np.ndarray]] = []
    for block_index, block in enumerate(mesh.cells):
        if block.type not in _CELL_EDGES:
            continue
        mask = _cell_mask(block_index, len(block.data), cell_masks)
        if mask is None or not np.any(mask):
            continue
        candidates.append((block_index, mask))
    if not candidates:
        raise ValueError("Laplace morphing requires supported mesh cells")

    edge_arrays: list[np.ndarray] = []
    for block_index, mask in candidates:
        block = mesh.cells[block_index]
        cells = np.asarray(block.data, dtype=int)[mask]
        for first, second in _CELL_EDGES[block.type]:
            edges = np.sort(cells[:, [first, second]], axis=1)
            edges = edges[active[edges[:, 0]] & active[edges[:, 1]]]
            if len(edges):
                edge_arrays.append(edges)
    if not edge_arrays:
        raise ValueError("Laplace morphing found no edges between active nodes")
    return np.unique(np.vstack(edge_arrays), axis=0)


def _edge_weights(
    mesh: meshio.Mesh,
    points: np.ndarray,
    edges: np.ndarray,
    config: LaplaceConfig,
    cell_masks: Mapping[int, np.ndarray] | None,
) -> np.ndarray:
    if config.weighting == "uniform" or config.distance_power == 0.0:
        weights = np.ones(len(edges), dtype=float)
    else:
        lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        if np.any(lengths < config.minimum_distance):
            raise ValueError("Laplace graph contains a zero-length or too-short edge")
        weights = 1.0 / np.maximum(lengths, config.minimum_distance) ** config.distance_power
    if config.weighting != "inverse_distance_element":
        return weights

    edge_rows = {tuple(edge): index for index, edge in enumerate(edges.tolist())}
    stiffness = np.zeros(len(edges), dtype=float)
    for block_index, block in enumerate(mesh.cells):
        if block.type not in _CELL_EDGES:
            continue
        mask = _cell_mask(block_index, len(block.data), cell_masks)
        if mask is None:
            continue
        for cell in np.asarray(block.data, dtype=int)[mask]:
            local_edges = np.array(
                [np.sort(cell[[first, second]]) for first, second in _CELL_EDGES[block.type]],
                dtype=int,
            )
            lengths = np.linalg.norm(
                points[local_edges[:, 0]] - points[local_edges[:, 1]], axis=1
            )
            minimum = max(float(np.min(lengths)), config.minimum_distance)
            mean = max(float(np.mean(lengths)), config.minimum_distance)
            aspect = float(np.max(lengths)) / minimum
            cell_stiffness = (
                mean ** (-config.element_size_power)
                * aspect ** config.element_aspect_power
            )
            for edge in local_edges:
                row = edge_rows.get(tuple(edge.tolist()))
                if row is not None:
                    stiffness[row] = max(stiffness[row], cell_stiffness)
    stiffness[stiffness == 0.0] = 1.0
    return weights * stiffness


def _validate_anchored_components(
    active_indices: np.ndarray,
    edges: np.ndarray,
    controls: np.ndarray,
    queries: np.ndarray,
) -> None:
    local = np.full(int(np.max(active_indices)) + 1, -1, dtype=int)
    local[active_indices] = np.arange(len(active_indices))
    local_edges = local[edges]
    graph = coo_matrix(
        (
            np.ones(2 * len(edges), dtype=float),
            (
                np.concatenate((local_edges[:, 0], local_edges[:, 1])),
                np.concatenate((local_edges[:, 1], local_edges[:, 0])),
            ),
        ),
        shape=(len(active_indices), len(active_indices)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    control_components = set(labels[local[controls]])
    query_components = set(labels[local[queries]])
    missing = query_components - control_components
    if missing:
        count = int(sum(np.count_nonzero(labels[local[queries]] == value) for value in missing))
        raise ValueError(f"{count} Laplace query nodes belong to an unanchored component")


def prepare_laplace_interpolator(
    mesh: meshio.Mesh,
    query_indices: Iterable[int],
    control_indices: Iterable[int],
    config: LaplaceConfig | None = None,
    *,
    cell_masks: Mapping[int, np.ndarray] | None = None,
) -> PreparedLaplaceInterpolator:
    points = np.asarray(mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("mesh points must be an N x 2 or N x 3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("mesh points must contain only finite values")
    queries = _index_array(
        query_indices, len(points), "query_indices", allow_empty=True
    )
    controls = _index_array(
        control_indices, len(points), "control_indices", allow_empty=False
    )
    if np.intersect1d(queries, controls).size:
        raise ValueError("query_indices and control_indices must not overlap")
    if not len(queries):
        return PreparedLaplaceInterpolator(
            source_points=points.copy(),
            query_indices=queries,
            control_indices=controls,
            config=config or LaplaceConfig(),
            _factorization=None,
            _boundary_matrix=csr_matrix((0, len(controls))),
        )

    resolved = config or LaplaceConfig()
    active = np.zeros(len(points), dtype=bool)
    active[queries] = True
    active[controls] = True
    edges = _active_edges(mesh, active, cell_masks)
    weights = _edge_weights(mesh, points, edges, resolved, cell_masks)
    active_indices = np.flatnonzero(active)
    _validate_anchored_components(active_indices, edges, controls, queries)

    query_row = np.full(len(points), -1, dtype=int)
    query_row[queries] = np.arange(len(queries))
    control_row = np.full(len(points), -1, dtype=int)
    control_row[controls] = np.arange(len(controls))
    diagonal = np.zeros(len(queries), dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    boundary_rows: list[int] = []
    boundary_columns: list[int] = []
    boundary_values: list[float] = []

    for (first, second), weight in zip(edges, weights, strict=True):
        first_query = query_row[first]
        second_query = query_row[second]
        if first_query >= 0:
            diagonal[first_query] += weight
            if second_query >= 0:
                rows.append(first_query)
                columns.append(second_query)
                coefficients.append(-weight)
            else:
                boundary_rows.append(first_query)
                boundary_columns.append(control_row[second])
                boundary_values.append(weight)
        if second_query >= 0:
            diagonal[second_query] += weight
            if first_query >= 0:
                rows.append(second_query)
                columns.append(first_query)
                coefficients.append(-weight)
            else:
                boundary_rows.append(second_query)
                boundary_columns.append(control_row[first])
                boundary_values.append(weight)

    if np.any(diagonal <= 0.0):
        node = int(queries[np.flatnonzero(diagonal <= 0.0)[0]])
        raise ValueError(f"isolated Laplace query node {node}")
    row_indices = np.concatenate((np.asarray(rows, dtype=int), np.arange(len(queries))))
    column_indices = np.concatenate(
        (np.asarray(columns, dtype=int), np.arange(len(queries)))
    )
    matrix_values = np.concatenate((np.asarray(coefficients), diagonal))
    matrix = csr_matrix(
        (matrix_values, (row_indices, column_indices)),
        shape=(len(queries), len(queries)),
    )
    boundary_matrix = csr_matrix(
        (boundary_values, (boundary_rows, boundary_columns)),
        shape=(len(queries), len(controls)),
    )
    try:
        factorization = splu(matrix.tocsc())
    except RuntimeError as exc:
        raise ValueError("Laplace system is singular") from exc
    return PreparedLaplaceInterpolator(
        source_points=points.copy(),
        query_indices=queries,
        control_indices=controls,
        config=resolved,
        _factorization=factorization,
        _boundary_matrix=boundary_matrix,
    )


def laplace_interpolate(
    mesh: meshio.Mesh,
    query_indices: Iterable[int],
    control_indices: Iterable[int],
    control_values: np.ndarray,
    config: LaplaceConfig | None = None,
    *,
    cell_masks: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    prepared = prepare_laplace_interpolator(
        mesh,
        query_indices,
        control_indices,
        config,
        cell_masks=cell_masks,
    )
    return prepared.solve(control_values)


def build_laplace_displacement_field(
    mesh: meshio.Mesh,
    control_indices: Iterable[int],
    *,
    control_displacements: np.ndarray | None = None,
    control_target_points: np.ndarray | None = None,
    fixed_indices: Iterable[int] = (),
    query_indices: Iterable[int] | None = None,
    config: LaplaceConfig | None = None,
    cell_masks: Mapping[int, np.ndarray] | None = None,
) -> LaplaceInterpolationResult:
    points = np.asarray(mesh.points, dtype=float)
    controls = _index_array(
        control_indices, len(points), "control_indices", allow_empty=False
    )
    fixed = _index_array(fixed_indices, len(points), "fixed_indices", allow_empty=True)
    if np.intersect1d(controls, fixed).size:
        raise ValueError("control_indices and fixed_indices must not overlap")
    if (control_displacements is None) == (control_target_points is None):
        raise ValueError(
            "provide exactly one of control_displacements or control_target_points"
        )
    if control_target_points is not None:
        targets = np.asarray(control_target_points, dtype=float)
        if targets.shape != (len(controls), points.shape[1]):
            raise ValueError("control_target_points must have one row per control index")
        moving_values = targets - points[controls]
    else:
        moving_values = np.asarray(control_displacements, dtype=float)
        if moving_values.shape != (len(controls), points.shape[1]):
            raise ValueError("control_displacements must have one row per control index")
    if not np.all(np.isfinite(moving_values)):
        raise ValueError("control displacement data must contain only finite values")

    known = np.concatenate((controls, fixed))
    known_values = np.vstack(
        (moving_values, np.zeros((len(fixed), points.shape[1]), dtype=float))
    )
    if query_indices is None:
        query_mask = np.ones(len(points), dtype=bool)
        query_mask[known] = False
        queries = np.flatnonzero(query_mask)
    else:
        queries = _index_array(
            query_indices, len(points), "query_indices", allow_empty=True
        )
        if np.intersect1d(queries, known).size:
            raise ValueError("query_indices must not overlap control or fixed indices")

    resolved = config or LaplaceConfig()
    displacement = np.zeros_like(points, dtype=float)
    displacement[controls] = moving_values
    if len(queries):
        displacement[queries] = laplace_interpolate(
            mesh,
            queries,
            known,
            known_values,
            resolved,
            cell_masks=cell_masks,
        )
    return LaplaceInterpolationResult(
        source_points=points.copy(),
        displacement=displacement,
        control_indices=controls,
        fixed_indices=fixed,
        query_indices=queries,
        config=resolved,
    )
