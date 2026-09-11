from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import meshio
import numpy as np

from ..morph.blend import shape_blend_weights
from ..morph.distance import nearest_reference
from ..morph.idw import IDWConfig, idw_interpolate
from ..morph.laplace import LaplaceConfig, laplace_interpolate
from ..morph.rbf import RBFConfig, rbf_interpolate
from .offset import OffsetResult
from .surface import SurfaceTopology, _points3d, _property_key


_CELL_EDGES = {
    "tetra": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    "hexahedron": (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
}


@dataclass(frozen=True)
class CoreMorphingConfig:
    method: str = "auto"
    zone_depth: float | None = None
    blend: str = "smoothstep"
    distance_chunk_size: int = 2048
    idw: IDWConfig = field(default_factory=IDWConfig)
    rbf: RBFConfig = field(default_factory=RBFConfig)
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)

    def __post_init__(self) -> None:
        if self.method not in {
            "auto",
            "distance_blend",
            "idw",
            "rbf",
            "laplace",
            "none",
        }:
            raise ValueError(
                "core morphing method must be auto, distance_blend, idw, rbf, "
                "laplace, or none"
            )
        if self.zone_depth is not None and self.zone_depth <= 0.0:
            raise ValueError("zone_depth must be positive")
        if self.distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be positive")


@dataclass(frozen=True)
class CoreMorphingResult:
    displacements: np.ndarray
    movable_nodes: np.ndarray
    selected_nodes: np.ndarray
    method: str


def selected_volume_masks(
    mesh: meshio.Mesh,
    volume_property_ids: Iterable[int] | None,
) -> dict[int, np.ndarray]:
    requested = None if volume_property_ids is None else {int(value) for value in volume_property_ids}
    property_key = _property_key(mesh)
    if requested is not None and property_key is None:
        raise ValueError("volume property selection requires 'property_id' or 'gmsh:physical'")
    masks: dict[int, np.ndarray] = {}
    for block_index, block in enumerate(mesh.cells):
        if block.type not in _CELL_EDGES:
            continue
        if requested is None:
            mask = np.ones(len(block.data), dtype=bool)
        else:
            properties = np.asarray(mesh.cell_data[property_key][block_index], dtype=int).reshape(-1)
            mask = np.isin(properties, list(requested))
        if np.any(mask):
            masks[block_index] = mask
    if not masks:
        raise ValueError("no tetrahedral or hexahedral volume elements were selected")
    return masks


def _node_masks(
    mesh: meshio.Mesh,
    selections: dict[int, np.ndarray],
    topology: SurfaceTopology,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.zeros(len(mesh.points), dtype=bool)
    nonselected = np.zeros(len(mesh.points), dtype=bool)
    for block_index, block in enumerate(mesh.cells):
        if block.type not in _CELL_EDGES:
            continue
        selection = selections.get(block_index, np.zeros(len(block.data), dtype=bool))
        if np.any(selection):
            selected[np.asarray(block.data)[selection].ravel()] = True
        if np.any(~selection):
            nonselected[np.asarray(block.data)[~selection].ravel()] = True
    boundary = np.zeros(len(mesh.points), dtype=bool)
    boundary[np.fromiter(topology.node_to_faces, dtype=int)] = True
    movable = selected & ~boundary & ~nonselected
    return selected, movable


def _distance_displacement(
    points: np.ndarray,
    movable: np.ndarray,
    boundary_displacement: np.ndarray,
    active_boundary: np.ndarray,
    config: CoreMorphingConfig,
) -> np.ndarray:
    result = boundary_displacement.copy()
    if not np.any(movable):
        return result
    if not np.any(active_boundary):
        raise ValueError("distance core morphing requires at least one skin boundary node")
    distances, nearest = nearest_reference(
        points[movable], points[active_boundary], config.distance_chunk_size
    )
    active_indices = np.flatnonzero(active_boundary)
    depth = config.zone_depth
    if depth is None:
        depth = max(float(np.max(distances)), np.finfo(float).eps)
    raw = np.clip(1.0 - distances / depth, 0.0, 1.0)
    weights = shape_blend_weights(raw, config.blend)
    result[movable] = boundary_displacement[active_indices[nearest]] * weights[:, None]
    return result


def _laplace_displacement(
    mesh: meshio.Mesh,
    selections: dict[int, np.ndarray],
    movable: np.ndarray,
    boundary_displacement: np.ndarray,
    config: LaplaceConfig,
) -> np.ndarray:
    result = boundary_displacement.copy()
    free_nodes = np.flatnonzero(movable)
    if len(free_nodes) == 0:
        return result
    selected_nodes = np.zeros(len(mesh.points), dtype=bool)
    for block_index, mask in selections.items():
        selected_nodes[np.asarray(mesh.cells[block_index].data)[mask].ravel()] = True
    controls = np.flatnonzero(selected_nodes & ~movable)
    if not len(controls):
        raise ValueError("Laplace core morphing requires boundary control nodes")
    result[free_nodes] = laplace_interpolate(
        mesh,
        free_nodes,
        controls,
        boundary_displacement[controls],
        config,
        cell_masks=selections,
    )
    return result


def _idw_displacement(
    points: np.ndarray,
    movable: np.ndarray,
    boundary: np.ndarray,
    boundary_displacement: np.ndarray,
    config: IDWConfig,
) -> np.ndarray:
    result = boundary_displacement.copy()
    if not np.any(movable):
        return result
    if not np.any(boundary):
        raise ValueError("IDW core morphing requires at least one boundary node")
    result[movable] = idw_interpolate(
        points[movable],
        points[boundary],
        boundary_displacement[boundary],
        config,
    )
    return result


def _rbf_displacement(
    points: np.ndarray,
    movable: np.ndarray,
    boundary: np.ndarray,
    boundary_displacement: np.ndarray,
    config: RBFConfig,
) -> np.ndarray:
    result = boundary_displacement.copy()
    if not np.any(movable):
        return result
    if not np.any(boundary):
        raise ValueError("RBF core morphing requires at least one boundary node")
    result[movable] = rbf_interpolate(
        points[movable],
        points[boundary],
        boundary_displacement[boundary],
        config,
    )
    return result


def compute_core_displacement(
    mesh: meshio.Mesh,
    topology: SurfaceTopology,
    offset: OffsetResult,
    volume_property_ids: Iterable[int] | None,
    config: CoreMorphingConfig,
    *,
    method: str | None = None,
) -> CoreMorphingResult:
    resolved_method = config.method if method is None else method
    if resolved_method == "auto":
        raise ValueError("auto core morphing must be resolved by the skin-layer generator")
    selections = selected_volume_masks(mesh, volume_property_ids)
    selected, movable = _node_masks(mesh, selections, topology)
    boundary = np.zeros(len(mesh.points), dtype=bool)
    boundary[np.fromiter(topology.node_to_faces, dtype=int)] = True
    boundary_displacement = np.zeros((len(mesh.points), 3), dtype=float)
    boundary_displacement[boundary] = offset.displacements[boundary]
    active_boundary = boundary & offset.active_nodes

    if resolved_method == "distance_blend":
        displacement = _distance_displacement(
            _points3d(mesh.points),
            movable,
            boundary_displacement,
            active_boundary,
            config,
        )
    elif resolved_method == "idw":
        displacement = _idw_displacement(
            _points3d(mesh.points),
            movable,
            boundary,
            boundary_displacement,
            config.idw,
        )
    elif resolved_method == "rbf":
        displacement = _rbf_displacement(
            _points3d(mesh.points),
            movable,
            boundary,
            boundary_displacement,
            config.rbf,
        )
    elif resolved_method == "laplace":
        displacement = _laplace_displacement(
            mesh, selections, movable, boundary_displacement, config.laplace
        )
    elif resolved_method == "none":
        displacement = boundary_displacement
    else:
        raise ValueError(f"unsupported resolved core morphing method: {resolved_method}")
    return CoreMorphingResult(displacement, movable, selected, resolved_method)
