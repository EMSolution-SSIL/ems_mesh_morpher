from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import meshio
import numpy as np

from ems_file_format_converter import write_mesh

from ..morph import LaplaceConfig, build_laplace_displacement_field
from .boundary_roles import PlaneSelector
from .constraints import PlaneConstraint, apply_directional_constraints
from .generator import compute_layer_thicknesses
from .topology import (
    _NewBlock,
    _add_field_data,
    _new_cell_values,
    _next_positive_id,
    _normalised_cell_data,
)


_CELL_EDGES = {
    "triangle": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
}

_VOLUME_CELL_PREFIXES = (
    "tetra",
    "hexahedron",
    "wedge",
    "pyramid",
    "voxel",
    "polyhedron",
)


@dataclass(frozen=True)
class BoundaryEdge2D:
    node_indices: tuple[int, int]
    owner_block_index: int
    owner_element_index: int


@dataclass(frozen=True)
class BoundaryTopology2D:
    edges: tuple[BoundaryEdge2D, ...]
    boundary_nodes: tuple[int, ...]
    selected_element_count: int


@dataclass(frozen=True)
class _BoundarySelection2D:
    skin_edges: tuple[BoundaryEdge2D, ...]
    excluded_edges: tuple[BoundaryEdge2D, ...]
    node_plane_constraints: dict[int, tuple[PlaneConstraint, ...]]
    fixed_nodes: frozenset[int]


@dataclass(frozen=True)
class SkinLayer2DQualityConfig:
    minimum_core_scaled_jacobian: float = 1.0e-8
    minimum_layer_scaled_jacobian: float = 1.0e-8
    minimum_core_measure_ratio: float = 0.1
    fail_on_inverted: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_core_scaled_jacobian <= 1.0:
            raise ValueError("minimum_core_scaled_jacobian must be between 0 and 1")
        if not 0.0 <= self.minimum_layer_scaled_jacobian <= 1.0:
            raise ValueError("minimum_layer_scaled_jacobian must be between 0 and 1")
        if self.minimum_core_measure_ratio < 0.0:
            raise ValueError("minimum_core_measure_ratio must be non-negative")


@dataclass(frozen=True)
class SkinLayer2DConfig:
    thickness: float
    region_property_ids: tuple[int, ...] | None = None
    layer_count: int = 1
    growth_ratio: float = 1.0
    core_morphing: str = "weighted_laplace"
    laplace: LaplaceConfig = field(
        default_factory=lambda: LaplaceConfig(weighting="inverse_distance_element")
    )
    quality: SkinLayer2DQualityConfig = field(default_factory=SkinLayer2DQualityConfig)
    skin_layer_property_id: int | None = None
    skin_layer_name: str = "skin_layer_2d"
    excluded_planes: tuple[PlaneSelector, ...] = ()
    maximum_miter_ratio: float = 4.0
    planarity_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.thickness <= 0.0 or not np.isfinite(self.thickness):
            raise ValueError("thickness must be a positive finite value")
        compute_layer_thicknesses(self.thickness, self.layer_count, self.growth_ratio)
        if self.core_morphing not in {"weighted_laplace", "laplace", "none"}:
            raise ValueError(
                "core_morphing must be 'weighted_laplace', 'laplace', or 'none'"
            )
        if self.maximum_miter_ratio < 1.0 or not np.isfinite(self.maximum_miter_ratio):
            raise ValueError("maximum_miter_ratio must be finite and at least 1")
        if self.planarity_tolerance < 0.0:
            raise ValueError("planarity_tolerance must be non-negative")

    @property
    def layer_thicknesses(self) -> tuple[float, ...]:
        return compute_layer_thicknesses(
            self.thickness, self.layer_count, self.growth_ratio
        )

    @property
    def layer_fractions(self) -> tuple[float, ...]:
        values = np.cumsum(self.layer_thicknesses) / self.thickness
        values[-1] = 1.0
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class SkinLayer2DReport:
    selected_element_count: int
    boundary_edge_count: int
    skin_edge_count: int
    excluded_edge_count: int
    boundary_node_count: int
    constrained_node_count: int
    new_node_count: int
    layer_element_count: int
    requested_thickness: float
    layer_count: int
    growth_ratio: float
    layer_thicknesses: tuple[float, ...]
    core_morphing_method: str
    minimum_core_measure_ratio: float
    minimum_core_scaled_jacobian: float
    minimum_layer_scaled_jacobian: float
    core_orientation_flip_count: int
    core_degenerate_count: int
    layer_inverted_count: int
    layer_degenerate_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SkinLayer2DResult:
    mesh: meshio.Mesh
    config: SkinLayer2DConfig
    boundary: BoundaryTopology2D
    report: SkinLayer2DReport
    layer_node_maps: tuple[dict[int, int], ...]
    core_node_map: dict[int, int]
    layer_block_index: int

    def write(self, path: str | Path, *, file_format: str | None = None) -> None:
        write_mesh(path, self.mesh, file_format=file_format)


class SkinLayer2DQualityError(ValueError):
    pass


def _points3d(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {2, 3}:
        raise ValueError("mesh points must have two or three coordinates")
    if values.shape[1] == 3:
        return values
    return np.column_stack((values, np.zeros(len(values), dtype=float)))


def _signed_area(points: np.ndarray) -> float:
    xy = np.asarray(points, dtype=float)[:, :2]
    return 0.5 * float(
        np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
        - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
    )


def _scaled_jacobian(points: np.ndarray) -> float:
    xy = np.asarray(points, dtype=float)[:, :2]
    values = []
    for index in range(len(xy)):
        outgoing = xy[(index + 1) % len(xy)] - xy[index]
        incoming = xy[(index - 1) % len(xy)] - xy[index]
        denominator = float(np.linalg.norm(outgoing) * np.linalg.norm(incoming))
        if denominator <= np.finfo(float).eps:
            values.append(0.0)
        else:
            determinant = float(np.linalg.det(np.vstack((outgoing, incoming))))
            values.append(abs(determinant) / denominator)
    return min(values)


def _property_key(mesh: meshio.Mesh) -> str | None:
    if "property_id" in mesh.cell_data:
        return "property_id"
    if "gmsh:physical" in mesh.cell_data:
        return "gmsh:physical"
    return None


def _require_2d_only_mesh(mesh: meshio.Mesh) -> None:
    for block in mesh.cells:
        if len(block.data) and block.type.startswith(_VOLUME_CELL_PREFIXES):
            raise ValueError(
                "generate_skin_layer_2d requires a 2D-only mesh; "
                f"found volume cell block '{block.type}'"
            )


def _selected_masks(
    mesh: meshio.Mesh, property_ids: Iterable[int] | None
) -> dict[int, np.ndarray]:
    requested = None if property_ids is None else {int(value) for value in property_ids}
    key = _property_key(mesh)
    if requested is not None and key is None:
        raise ValueError("2D region selection requires property_id or gmsh:physical")
    selections: dict[int, np.ndarray] = {}
    for block_index, block in enumerate(mesh.cells):
        if block.type not in _CELL_EDGES:
            continue
        if requested is None:
            mask = np.ones(len(block.data), dtype=bool)
        else:
            properties = np.asarray(mesh.cell_data[key][block_index], dtype=int)
            mask = np.isin(properties, tuple(requested))
        if np.any(mask):
            selections[block_index] = mask
    if not selections:
        raise ValueError("no triangle or quadrilateral elements were selected")
    return selections


def extract_boundary_edges_2d(
    mesh: meshio.Mesh,
    region_property_ids: Iterable[int] | None = None,
    *,
    planarity_tolerance: float = 1.0e-10,
) -> tuple[BoundaryTopology2D, dict[int, np.ndarray]]:
    _require_2d_only_mesh(mesh)
    selections = _selected_masks(mesh, region_property_ids)
    points = _points3d(mesh.points)
    selected_nodes = np.unique(
        np.concatenate(
            [
                np.asarray(mesh.cells[index].data, dtype=int)[mask].reshape(-1)
                for index, mask in selections.items()
            ]
        )
    )
    if float(np.ptp(points[selected_nodes, 2])) > planarity_tolerance:
        raise ValueError("2D skin-layer input must lie in a plane parallel to XY")

    occurrences: dict[tuple[int, int], list[BoundaryEdge2D]] = {}
    selected_count = 0
    for block_index, mask in selections.items():
        block = mesh.cells[block_index]
        for element_index in np.flatnonzero(mask):
            nodes = np.asarray(block.data[element_index], dtype=int)
            area = _signed_area(points[nodes])
            if abs(area) <= np.finfo(float).eps:
                raise ValueError(
                    f"selected 2D element ({block_index}, {element_index}) is degenerate"
                )
            ordered = nodes if area > 0.0 else nodes[::-1]
            selected_count += 1
            for first_index, second_index in _CELL_EDGES[block.type]:
                first = int(ordered[first_index])
                second = int(ordered[second_index])
                key = (min(first, second), max(first, second))
                occurrences.setdefault(key, []).append(
                    BoundaryEdge2D(
                        node_indices=(first, second),
                        owner_block_index=block_index,
                        owner_element_index=int(element_index),
                    )
                )
    nonmanifold = [key for key, edges in occurrences.items() if len(edges) > 2]
    if nonmanifold:
        raise ValueError(f"non-manifold 2D edges found: {len(nonmanifold)}")
    boundary = tuple(edges[0] for edges in occurrences.values() if len(edges) == 1)
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    for edge in boundary:
        first, second = edge.node_indices
        if first in outgoing or second in incoming:
            raise ValueError("2D boundary is branched or inconsistently oriented")
        outgoing[first] = second
        incoming[second] = first
    nodes = tuple(sorted({node for edge in boundary for node in edge.node_indices}))
    if not nodes or any(node not in incoming or node not in outgoing for node in nodes):
        raise ValueError("2D boundary contains an open chain")
    return BoundaryTopology2D(boundary, nodes, selected_count), selections


def _classify_boundary_edges_2d(
    points: np.ndarray,
    boundary: BoundaryTopology2D,
    excluded_planes: tuple[PlaneSelector, ...],
) -> _BoundarySelection2D:
    skin_edges = []
    excluded_edges = []
    node_planes: dict[int, list[PlaneConstraint]] = {}
    fixed_nodes: set[int] = set()
    for edge in boundary.edges:
        coordinates = points[list(edge.node_indices)]
        matches = [
            selector
            for selector in excluded_planes
            if selector.plane.matches(coordinates)
        ]
        if not matches:
            skin_edges.append(edge)
            continue
        excluded_edges.append(edge)
        for selector in matches:
            if selector.node_constraint == "fixed":
                fixed_nodes.update(edge.node_indices)
            elif selector.node_constraint == "slip_plane":
                for node in edge.node_indices:
                    planes = node_planes.setdefault(node, [])
                    if selector.plane not in planes:
                        planes.append(selector.plane)
    if not skin_edges:
        raise ValueError("all 2D boundary edges are excluded from skin-layer generation")
    return _BoundarySelection2D(
        skin_edges=tuple(skin_edges),
        excluded_edges=tuple(excluded_edges),
        node_plane_constraints={
            node: tuple(planes) for node, planes in node_planes.items()
        },
        fixed_nodes=frozenset(fixed_nodes),
    )


def _boundary_displacements(
    points: np.ndarray,
    boundary: BoundaryTopology2D,
    selection: _BoundarySelection2D,
    thickness: float,
    maximum_miter_ratio: float,
) -> np.ndarray:
    incident: dict[int, list[BoundaryEdge2D]] = {}
    for edge in boundary.edges:
        for node in edge.node_indices:
            incident.setdefault(node, []).append(edge)
    skin_edges = set(selection.skin_edges)
    skin_nodes = {
        node for edge in selection.skin_edges for node in edge.node_indices
    }
    displacement = np.zeros_like(points, dtype=float)
    for node in skin_nodes:
        rows = []
        right_hand_side = []
        for edge in incident[node]:
            if edge not in skin_edges:
                continue
            first, second = edge.node_indices
            tangent = points[second, :2] - points[first, :2]
            length = float(np.linalg.norm(tangent))
            if length <= np.finfo(float).eps:
                raise ValueError("2D boundary contains a zero-length edge")
            tangent /= length
            rows.append(np.array((-tangent[1], tangent[0]), dtype=float))
            right_hand_side.append(thickness)
        for plane in selection.node_plane_constraints.get(node, ()):
            rows.append(plane.unit_normal[:2])
            right_hand_side.append(0.0)
        matrix = np.vstack(rows)
        value, _, _, _ = np.linalg.lstsq(
            matrix, np.asarray(right_hand_side, dtype=float), rcond=None
        )
        residual = matrix @ value - right_hand_side
        if float(np.linalg.norm(residual, ord=np.inf)) > max(thickness, 1.0) * 1.0e-10:
            raise ValueError(
                f"2D boundary constraints are inconsistent at node {node}"
            )
        ratio = float(np.linalg.norm(value) / thickness)
        if ratio > maximum_miter_ratio:
            raise ValueError(
                f"2D corner miter ratio {ratio:.6g} exceeds "
                f"maximum_miter_ratio at node {node}"
            )
        displacement[node, :2] = value
    if selection.fixed_nodes:
        displacement[np.asarray(tuple(selection.fixed_nodes), dtype=int)] = 0.0
    return displacement


def _point_data_with_copies(
    mesh: meshio.Mesh,
    source_nodes: np.ndarray,
    *,
    boundary_copy_count: int,
    boundary_node_count: int,
    skin_layer_property_id: int,
    core_property_id: int,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(values).copy() for key, values in mesh.point_data.items()}
    if "id" not in result:
        if "gmsh:node_tags" in result:
            result["id"] = np.asarray(result["gmsh:node_tags"], dtype=int).copy()
        else:
            result["id"] = np.arange(1, len(mesh.points) + 1, dtype=int)
    id_sources = [np.asarray(result["id"], dtype=int)]
    if "gmsh:node_tags" in result:
        id_sources.append(np.asarray(result["gmsh:node_tags"], dtype=int))
    next_id = _next_positive_id(id_sources)
    new_ids = np.arange(next_id, next_id + len(source_nodes), dtype=int)
    for key, values in tuple(result.items()):
        source = np.asarray(values)
        if len(source) != len(mesh.points):
            raise ValueError(f"point_data['{key}'] length does not match mesh points")
        if key in {"id", "gmsh:node_tags"}:
            appended = new_ids.astype(source.dtype, copy=False)
        elif key == "gmsh:dim_tags":
            appended = np.empty((len(source_nodes), 2), dtype=source.dtype)
            split = boundary_copy_count * boundary_node_count
            appended[:split] = (2, skin_layer_property_id)
            appended[split:] = (2, core_property_id)
        else:
            appended = source[source_nodes].copy()
        result[key] = np.concatenate((source, appended), axis=0)
    return result


def _material_id(mesh: meshio.Mesh, edge: BoundaryEdge2D) -> int:
    if "material_id" not in mesh.cell_data:
        return 1
    values = np.asarray(mesh.cell_data["material_id"][edge.owner_block_index])
    return int(values[edge.owner_element_index])


def _resolve_skin_property(mesh: meshio.Mesh, requested: int | None) -> int:
    key = _property_key(mesh)
    used = set()
    if key is not None:
        used = {
            int(value)
            for values in mesh.cell_data[key]
            for value in np.asarray(values).reshape(-1)
        }
    if requested is not None:
        if requested in used:
            raise ValueError(f"skin_layer_property_id {requested} is already in use")
        return int(requested)
    return max(used, default=0) + 1


def _quality_summary(
    source: meshio.Mesh,
    points: np.ndarray,
    cells: list[np.ndarray],
    selections: Mapping[int, np.ndarray],
    layer_connectivity: np.ndarray,
    *,
    zero_tolerance: float = 1.0e-15,
) -> dict[str, float | int]:
    source_points = _points3d(source.points)
    ratios = []
    core_jacobians = []
    flips = 0
    core_degenerate = 0
    for block_index, mask in selections.items():
        before_cells = np.asarray(source.cells[block_index].data, dtype=int)
        after_cells = cells[block_index]
        for element_index in np.flatnonzero(mask):
            before = _signed_area(source_points[before_cells[element_index]])
            after = _signed_area(points[after_cells[element_index]])
            if abs(before) <= zero_tolerance:
                raise ValueError("selected reference 2D element is degenerate")
            ratios.append(abs(after) / abs(before))
            core_jacobians.append(_scaled_jacobian(points[after_cells[element_index]]))
            flips += int(before * after < 0.0)
            core_degenerate += int(abs(after) <= zero_tolerance)
    layer_areas = np.array(
        [_signed_area(points[connectivity]) for connectivity in layer_connectivity]
    )
    layer_jacobians = np.array(
        [_scaled_jacobian(points[connectivity]) for connectivity in layer_connectivity]
    )
    return {
        "minimum_core_measure_ratio": float(min(ratios, default=1.0)),
        "minimum_core_scaled_jacobian": float(min(core_jacobians, default=1.0)),
        "minimum_layer_scaled_jacobian": float(np.min(layer_jacobians)),
        "core_orientation_flip_count": flips,
        "core_degenerate_count": core_degenerate,
        "layer_inverted_count": int(np.count_nonzero(layer_areas < -zero_tolerance)),
        "layer_degenerate_count": int(
            np.count_nonzero(np.abs(layer_areas) <= zero_tolerance)
        ),
    }


def generate_skin_layer_2d(
    mesh: meshio.Mesh,
    config: SkinLayer2DConfig,
) -> SkinLayer2DResult:
    points = _points3d(mesh.points)
    boundary, selections = extract_boundary_edges_2d(
        mesh,
        config.region_property_ids,
        planarity_tolerance=config.planarity_tolerance,
    )
    boundary_selection = _classify_boundary_edges_2d(
        points, boundary, config.excluded_planes
    )
    offset = _boundary_displacements(
        points,
        boundary,
        boundary_selection,
        config.thickness,
        config.maximum_miter_ratio,
    )
    boundary_nodes = np.asarray(boundary.boundary_nodes, dtype=int)
    skin_nodes = np.asarray(
        sorted(
            {
                node
                for edge in boundary_selection.skin_edges
                for node in edge.node_indices
            }
        ),
        dtype=int,
    )
    selected_nodes = np.unique(
        np.concatenate(
            [
                np.asarray(mesh.cells[index].data, dtype=int)[mask].reshape(-1)
                for index, mask in selections.items()
            ]
        )
    )
    fixed_nodes = np.asarray(sorted(boundary_selection.fixed_nodes), dtype=int)
    control_nodes = np.union1d(skin_nodes, fixed_nodes)
    deformable_nodes = np.setdiff1d(selected_nodes, control_nodes)
    core_displacement = np.zeros_like(points)
    core_displacement[control_nodes] = offset[control_nodes]
    if len(deformable_nodes) and config.core_morphing != "none":
        interpolation = build_laplace_displacement_field(
            mesh,
            control_nodes,
            control_displacements=offset[control_nodes],
            query_indices=deformable_nodes,
            config=config.laplace,
            cell_masks=selections,
        )
        core_displacement[deformable_nodes] = interpolation.displacement[
            deformable_nodes
        ]
    core_displacement = apply_directional_constraints(
        core_displacement,
        boundary_selection.node_plane_constraints,
        fixed_nodes=tuple(int(node) for node in fixed_nodes),
    )

    layer_node_maps = []
    layer_points = []
    next_node = len(points)
    for fraction in config.layer_fractions:
        node_map = {
            int(node): next_node + local
            for local, node in enumerate(skin_nodes)
        }
        layer_node_maps.append(node_map)
        layer_points.append(points[skin_nodes] + fraction * offset[skin_nodes])
        next_node += len(skin_nodes)
    remaining_nodes = np.setdiff1d(selected_nodes, skin_nodes)
    remaining_node_map = {
        int(node): next_node + local for local, node in enumerate(remaining_nodes)
    }
    remaining_points = (
        points[remaining_nodes] + core_displacement[remaining_nodes]
    )
    output_points = np.vstack((*([points] + layer_points), remaining_points))
    core_node_map = dict(layer_node_maps[-1])
    core_node_map.update(remaining_node_map)

    cells = [np.asarray(block.data, dtype=int).copy() for block in mesh.cells]
    for block_index, mask in selections.items():
        for element_index in np.flatnonzero(mask):
            cells[block_index][element_index] = np.asarray(
                [core_node_map[int(node)] for node in cells[block_index][element_index]],
                dtype=int,
            )

    layer_quads = []
    layer_indices = []
    layer_materials = []
    for layer_offset, current_map in enumerate(layer_node_maps):
        previous_map = None if layer_offset == 0 else layer_node_maps[layer_offset - 1]
        for edge in boundary_selection.skin_edges:
            first, second = edge.node_indices
            outer_first = first if previous_map is None else previous_map[first]
            outer_second = second if previous_map is None else previous_map[second]
            candidate = (
                outer_first,
                outer_second,
                current_map[second],
                current_map[first],
            )
            if _signed_area(output_points[list(candidate)]) < 0.0:
                candidate = tuple(reversed(candidate))
            layer_quads.append(candidate)
            layer_indices.append(layer_offset + 1)
            layer_materials.append(_material_id(mesh, edge))
    layer_connectivity = np.asarray(layer_quads, dtype=int)

    skin_property = _resolve_skin_property(mesh, config.skin_layer_property_id)
    new_block = _NewBlock(
        cell_type="quad",
        connectivity=layer_connectivity,
        property_ids=np.full(len(layer_connectivity), skin_property, dtype=int),
        material_ids=np.asarray(layer_materials, dtype=int),
        category="layer_2d",
        layer_indices=np.asarray(layer_indices, dtype=int),
    )
    cell_data = _normalised_cell_data(mesh)
    id_sources = [np.asarray(values) for values in cell_data["element_id"]]
    if "gmsh:element_tags" in cell_data:
        id_sources.extend(np.asarray(values) for values in cell_data["gmsh:element_tags"])
    next_element_id = _next_positive_id(id_sources)
    element_ids = np.arange(
        next_element_id, next_element_id + len(layer_connectivity), dtype=int
    )
    for key, arrays in cell_data.items():
        arrays.append(_new_cell_values(key, arrays, new_block, element_ids))

    requested_properties = config.region_property_ids or (1,)
    source_nodes = np.concatenate(
        [np.tile(skin_nodes, len(layer_node_maps)), remaining_nodes]
    )
    point_data = _point_data_with_copies(
        mesh,
        source_nodes,
        boundary_copy_count=len(layer_node_maps),
        boundary_node_count=len(skin_nodes),
        skin_layer_property_id=skin_property,
        core_property_id=int(requested_properties[0]),
    )
    field_data = {
        key: np.asarray(value).copy() for key, value in (mesh.field_data or {}).items()
    }
    _add_field_data(field_data, config.skin_layer_name, skin_property, 2)

    point_sets = {
        name: np.asarray(indices, dtype=int).copy()
        for name, indices in (mesh.point_sets or {}).items()
    }
    for layer_index, node_map in enumerate(layer_node_maps, start=1):
        point_sets[f"{config.skin_layer_name}_{layer_index}_nodes"] = np.fromiter(
            node_map.values(), dtype=int
        )
    cell_sets = {
        name: [np.asarray(indices, dtype=int).copy() for indices in arrays]
        for name, arrays in (mesh.cell_sets or {}).items()
    }
    for arrays in cell_sets.values():
        arrays.append(np.empty(0, dtype=int))
    cell_sets[config.skin_layer_name] = [
        *[np.empty(0, dtype=int) for _ in mesh.cells],
        np.arange(len(layer_connectivity), dtype=int),
    ]

    output = meshio.Mesh(
        points=output_points,
        cells=[
            *((mesh.cells[index].type, data) for index, data in enumerate(cells)),
            ("quad", layer_connectivity),
        ],
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
        gmsh_periodic=mesh.gmsh_periodic,
        info=dict(mesh.info or {}),
    )

    quality = _quality_summary(
        mesh, output_points, cells, selections, layer_connectivity
    )
    failures = []
    if config.quality.fail_on_inverted:
        if quality["core_orientation_flip_count"] or quality["core_degenerate_count"]:
            failures.append("2D core contains flipped or degenerate elements")
        if quality["layer_inverted_count"] or quality["layer_degenerate_count"]:
            failures.append("2D skin layer contains inverted or degenerate elements")
    if quality["minimum_core_measure_ratio"] < config.quality.minimum_core_measure_ratio:
        failures.append("2D core measure ratio is below the configured minimum")
    if (
        quality["minimum_core_scaled_jacobian"]
        < config.quality.minimum_core_scaled_jacobian
    ):
        failures.append("2D core scaled Jacobian is below the configured minimum")
    if (
        quality["minimum_layer_scaled_jacobian"]
        < config.quality.minimum_layer_scaled_jacobian
    ):
        failures.append("2D skin-layer scaled Jacobian is below the configured minimum")
    if failures:
        raise SkinLayer2DQualityError("; ".join(failures))

    report = SkinLayer2DReport(
        selected_element_count=boundary.selected_element_count,
        boundary_edge_count=len(boundary.edges),
        skin_edge_count=len(boundary_selection.skin_edges),
        excluded_edge_count=len(boundary_selection.excluded_edges),
        boundary_node_count=len(boundary.boundary_nodes),
        constrained_node_count=len(boundary_selection.node_plane_constraints),
        new_node_count=len(output_points) - len(points),
        layer_element_count=len(layer_connectivity),
        requested_thickness=config.thickness,
        layer_count=config.layer_count,
        growth_ratio=config.growth_ratio,
        layer_thicknesses=config.layer_thicknesses,
        core_morphing_method=config.core_morphing,
        **quality,
    )
    return SkinLayer2DResult(
        mesh=output,
        config=config,
        boundary=boundary,
        report=report,
        layer_node_maps=tuple(layer_node_maps),
        core_node_map=core_node_map,
        layer_block_index=len(mesh.cells),
    )
