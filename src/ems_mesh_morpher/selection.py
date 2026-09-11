from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np

from .config import GeometricSelectionConfig, RegionConfig, SelectionConfig


@dataclass(frozen=True)
class NodeClassification:
    moving: np.ndarray
    fixed: np.ndarray
    deformable: np.ndarray
    protected: np.ndarray | None = None
    morphing_zone: np.ndarray | None = None

    @property
    def counts(self) -> dict[str, int]:
        # Keep the Phase 1 summary shape backward-compatible.
        return {
            "moving": int(np.count_nonzero(self.moving)),
            "fixed": int(np.count_nonzero(self.fixed)),
            "deformable": int(np.count_nonzero(self.deformable)),
        }


def property_data_key(mesh: meshio.Mesh) -> str:
    if "property_id" in mesh.cell_data:
        return "property_id"
    if "gmsh:physical" in mesh.cell_data:
        return "gmsh:physical"
    raise ValueError("mesh.cell_data must contain 'property_id' or 'gmsh:physical'")


def available_property_ids(mesh: meshio.Mesh) -> set[int]:
    key = property_data_key(mesh)
    ids: set[int] = set()
    for values in mesh.cell_data[key]:
        ids.update(int(v) for v in np.asarray(values).ravel())
    return ids


def nodes_for_property_ids(mesh: meshio.Mesh, property_ids: set[int] | tuple[int, ...]) -> np.ndarray:
    ids = {int(v) for v in property_ids}
    mask = np.zeros(len(mesh.points), dtype=bool)
    if not ids:
        return mask

    key = property_data_key(mesh)
    for block, values in zip(mesh.cells, mesh.cell_data[key], strict=True):
        values_array = np.asarray(values)
        selected = np.isin(values_array, list(ids))
        if np.any(selected):
            mask[np.asarray(block.data)[selected].ravel()] = True
    return mask


def validate_property_ids(mesh: meshio.Mesh, property_ids: set[int] | tuple[int, ...]) -> None:
    requested = {int(v) for v in property_ids}
    if not requested:
        return
    missing = sorted(requested - available_property_ids(mesh))
    if missing:
        raise ValueError(f"property IDs not found in mesh: {missing}")


def _point_array(points: np.ndarray, dimensions: int = 3) -> np.ndarray:
    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] not in {2, 3}:
        raise ValueError("mesh points must be an N x 2 or N x 3 array")
    if source.shape[1] >= dimensions:
        return source[:, :dimensions]
    return np.pad(source, ((0, 0), (0, dimensions - source.shape[1])))


def _vector(value: tuple[float, ...], dimensions: int, key: str) -> np.ndarray:
    if len(value) not in {2, 3}:
        raise ValueError(f"{key} must contain two or three values")
    result = np.zeros(dimensions, dtype=float)
    count = min(len(value), dimensions)
    result[:count] = value[:count]
    return result


def geometric_node_mask(points: np.ndarray, geometry: GeometricSelectionConfig) -> np.ndarray:
    tolerance = float(geometry.tolerance)
    if tolerance < 0.0:
        raise ValueError("selection tolerance must be non-negative")
    kind = geometry.type

    if kind == "all":
        return np.ones(len(points), dtype=bool)

    if kind == "box":
        if not geometry.minimum or not geometry.maximum:
            raise ValueError("box selection requires minimum and maximum")
        if len(geometry.minimum) != len(geometry.maximum):
            raise ValueError("box minimum and maximum dimensions must match")
        dimensions = len(geometry.minimum)
        coords = _point_array(points, dimensions)
        minimum = np.asarray(geometry.minimum, dtype=float)
        maximum = np.asarray(geometry.maximum, dtype=float)
        if np.any(maximum < minimum):
            raise ValueError("box maximum must not be less than minimum")
        return np.all((coords >= minimum - tolerance) & (coords <= maximum + tolerance), axis=1)

    if kind in {"circle", "annulus"}:
        if not geometry.center:
            raise ValueError(f"{kind} selection requires center")
        center = _vector(geometry.center, 2, "selection.center")
        radius = np.linalg.norm(_point_array(points, 2) - center, axis=1)
        if kind == "circle":
            if geometry.radius is None or geometry.radius < 0.0:
                raise ValueError("circle selection requires a non-negative radius")
            return radius <= geometry.radius + tolerance
        if geometry.inner_radius is None or geometry.outer_radius is None:
            raise ValueError("annulus selection requires inner_radius and outer_radius")
        if geometry.inner_radius < 0.0 or geometry.outer_radius < geometry.inner_radius:
            raise ValueError("annulus radii must satisfy 0 <= inner_radius <= outer_radius")
        return (radius >= geometry.inner_radius - tolerance) & (
            radius <= geometry.outer_radius + tolerance
        )

    if kind == "sphere":
        if not geometry.center or geometry.radius is None or geometry.radius < 0.0:
            raise ValueError("sphere selection requires center and a non-negative radius")
        center = _vector(geometry.center, 3, "selection.center")
        radius = np.linalg.norm(_point_array(points, 3) - center, axis=1)
        return radius <= geometry.radius + tolerance

    if kind == "cylinder":
        if not geometry.axis_start or not geometry.axis_end:
            raise ValueError("cylinder selection requires axis_start and axis_end")
        if geometry.radius is None or geometry.radius < 0.0:
            raise ValueError("cylinder selection requires a non-negative radius")
        coords = _point_array(points, 3)
        start = _vector(geometry.axis_start, 3, "selection.axis_start")
        end = _vector(geometry.axis_end, 3, "selection.axis_end")
        axis = end - start
        length_squared = float(np.dot(axis, axis))
        if length_squared == 0.0:
            raise ValueError("cylinder axis_start and axis_end must differ")
        projection = ((coords - start) @ axis) / length_squared
        closest = start + projection[:, None] * axis
        radial_distance = np.linalg.norm(coords - closest, axis=1)
        axial_tolerance = tolerance / np.sqrt(length_squared)
        return (
            (projection >= -axial_tolerance)
            & (projection <= 1.0 + axial_tolerance)
            & (radial_distance <= geometry.radius + tolerance)
        )

    raise ValueError(f"unsupported geometric selection type: {kind}")


def select_nodes(mesh: meshio.Mesh, selection: SelectionConfig) -> np.ndarray:
    criteria: list[np.ndarray] = []
    if selection.property_ids:
        validate_property_ids(mesh, selection.property_ids)
        criteria.append(nodes_for_property_ids(mesh, selection.property_ids))
    if selection.point_indices:
        point_indices = np.asarray(selection.point_indices, dtype=int)
        if np.any(point_indices < 0) or np.any(point_indices >= len(mesh.points)):
            raise ValueError("selection point index is outside the mesh")
        point_mask = np.zeros(len(mesh.points), dtype=bool)
        point_mask[point_indices] = True
        criteria.append(point_mask)
    criteria.extend(geometric_node_mask(mesh.points, geometry) for geometry in selection.geometries)

    if not criteria:
        result = np.zeros(len(mesh.points), dtype=bool)
    elif selection.combine == "union":
        result = np.logical_or.reduce(criteria)
    elif selection.combine == "intersection":
        result = np.logical_and.reduce(criteria)
    else:
        raise ValueError(f"unsupported selection combination: {selection.combine}")
    return ~result if selection.invert else result


def _classification_from_raw_masks(
    moving: np.ndarray,
    deformable: np.ndarray,
    fixed: np.ndarray,
    protected: np.ndarray | None = None,
    morphing_zone: np.ndarray | None = None,
    conflict_policy: str = "priority",
) -> NodeClassification:
    protected_mask = np.zeros_like(moving) if protected is None else protected.copy()
    overlap = (moving & (fixed | protected_mask)) | (deformable & (fixed | protected_mask))
    if conflict_policy == "error" and np.any(overlap):
        raise ValueError(f"conflicting region assignments for {np.count_nonzero(overlap)} nodes")
    if morphing_zone is not None:
        deformable = deformable & morphing_zone

    # Protected nodes never move. Remaining Phase 1 priority is moving > fixed > deformable.
    moving = moving & ~protected_mask
    fixed = (fixed | protected_mask) & ~moving
    deformable = deformable & ~moving & ~fixed
    return NodeClassification(
        moving=moving,
        fixed=fixed,
        deformable=deformable,
        protected=protected_mask,
        morphing_zone=morphing_zone,
    )


def classify_nodes_from_regions(mesh: meshio.Mesh, regions: RegionConfig) -> NodeClassification:
    legacy_ids = (
        *regions.moving_property_ids,
        *regions.deformable_property_ids,
        *regions.fixed_property_ids,
    )
    validate_property_ids(mesh, legacy_ids)
    moving = select_nodes(mesh, regions.moving) | nodes_for_property_ids(
        mesh, regions.moving_property_ids
    )
    deformable = select_nodes(mesh, regions.deformable) | nodes_for_property_ids(
        mesh, regions.deformable_property_ids
    )
    fixed = select_nodes(mesh, regions.fixed) | nodes_for_property_ids(
        mesh, regions.fixed_property_ids
    )
    protected = select_nodes(mesh, regions.protected)
    zone = None if regions.morphing_zone is None else select_nodes(mesh, regions.morphing_zone)
    if zone is not None and not np.any(deformable):
        deformable = zone.copy()
    return _classification_from_raw_masks(
        moving,
        deformable,
        fixed,
        protected=protected,
        morphing_zone=zone,
        conflict_policy=regions.conflict_policy,
    )


def classify_nodes(
    mesh: meshio.Mesh,
    moving_property_ids: tuple[int, ...],
    deformable_property_ids: tuple[int, ...],
    fixed_property_ids: tuple[int, ...],
) -> NodeClassification:
    validate_property_ids(
        mesh,
        tuple(moving_property_ids) + tuple(deformable_property_ids) + tuple(fixed_property_ids),
    )
    return _classification_from_raw_masks(
        nodes_for_property_ids(mesh, moving_property_ids),
        nodes_for_property_ids(mesh, deformable_property_ids),
        nodes_for_property_ids(mesh, fixed_property_ids),
    )
