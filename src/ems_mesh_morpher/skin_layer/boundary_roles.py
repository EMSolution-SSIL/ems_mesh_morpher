from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import meshio
import numpy as np

from .constraints import PlaneConstraint
from .surface import SurfaceTopology, _points3d


class BoundaryRole(str, Enum):
    SKIN = "skin"
    EXCLUDED = "excluded"
    PROTECTED = "protected"


@dataclass(frozen=True)
class PlaneSelector:
    plane: PlaneConstraint
    node_constraint: str = "slip_plane"

    def __post_init__(self) -> None:
        if self.node_constraint not in {"slip_plane", "fixed", "none"}:
            raise ValueError("node_constraint must be 'slip_plane', 'fixed', or 'none'")


@dataclass(frozen=True)
class BoxSelector:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("box minimum and maximum must contain three values")
        if np.any(maximum < minimum):
            raise ValueError("box maximum must not be less than minimum")
        if self.tolerance < 0.0:
            raise ValueError("box tolerance must be non-negative")

    def matches(self, points: np.ndarray) -> bool:
        minimum = np.asarray(self.minimum, dtype=float) - self.tolerance
        maximum = np.asarray(self.maximum, dtype=float) + self.tolerance
        coordinates = np.asarray(points, dtype=float)
        return bool(np.all((coordinates >= minimum) & (coordinates <= maximum)))


@dataclass(frozen=True)
class BoundaryRoleConfig:
    excluded_surface_property_ids: tuple[int, ...] = ()
    protected_surface_property_ids: tuple[int, ...] = ()
    excluded_planes: tuple[PlaneSelector, ...] = ()
    protected_planes: tuple[PlaneSelector, ...] = ()
    excluded_boxes: tuple[BoxSelector, ...] = ()
    protected_boxes: tuple[BoxSelector, ...] = ()
    default_role: BoundaryRole = BoundaryRole.SKIN

    def __post_init__(self) -> None:
        if self.default_role not in {BoundaryRole.SKIN, BoundaryRole.EXCLUDED}:
            raise ValueError("default_role must be skin or excluded")


@dataclass(frozen=True)
class BoundaryClassification:
    roles: tuple[BoundaryRole, ...]
    node_plane_constraints: dict[int, tuple[PlaneConstraint, ...]]
    fixed_nodes: frozenset[int]
    conflicts: tuple[str, ...] = ()

    @property
    def skin_face_indices(self) -> tuple[int, ...]:
        return tuple(index for index, role in enumerate(self.roles) if role == BoundaryRole.SKIN)

    @property
    def excluded_face_indices(self) -> tuple[int, ...]:
        return tuple(index for index, role in enumerate(self.roles) if role == BoundaryRole.EXCLUDED)

    @property
    def protected_face_indices(self) -> tuple[int, ...]:
        return tuple(index for index, role in enumerate(self.roles) if role == BoundaryRole.PROTECTED)


def _matching_planes(
    points,
    face_nodes: Iterable[int],
    selectors: tuple[PlaneSelector, ...],
) -> list[PlaneSelector]:
    coordinates = points[list(face_nodes)]
    return [selector for selector in selectors if selector.plane.matches(coordinates)]


def _constraint_key(plane: PlaneConstraint) -> tuple[float, ...]:
    normal = plane.unit_normal
    return tuple(round(float(value), 14) for value in (*normal, plane.unit_offset))


def classify_boundary_faces(
    mesh: meshio.Mesh,
    topology: SurfaceTopology,
    config: BoundaryRoleConfig | None = None,
) -> BoundaryClassification:
    config = config or BoundaryRoleConfig()
    points = _points3d(mesh.points)
    excluded_ids = {int(value) for value in config.excluded_surface_property_ids}
    protected_ids = {int(value) for value in config.protected_surface_property_ids}
    roles: list[BoundaryRole] = []
    constraints: dict[int, dict[tuple[float, ...], PlaneConstraint]] = {}
    fixed_nodes: set[int] = set()
    conflicts: list[str] = []

    for face_index, face in enumerate(topology.faces):
        property_id = face.surface_property_id
        excluded_by_property = property_id is not None and property_id in excluded_ids
        protected_by_property = property_id is not None and property_id in protected_ids
        excluded_planes = _matching_planes(points, face.node_indices, config.excluded_planes)
        protected_planes = _matching_planes(points, face.node_indices, config.protected_planes)
        coordinates = points[list(face.node_indices)]
        excluded_by_box = any(selector.matches(coordinates) for selector in config.excluded_boxes)
        protected_by_box = any(selector.matches(coordinates) for selector in config.protected_boxes)

        if protected_by_property or protected_planes or protected_by_box:
            role = BoundaryRole.PROTECTED
        elif excluded_by_property or excluded_planes or excluded_by_box:
            role = BoundaryRole.EXCLUDED
        else:
            role = config.default_role
        roles.append(role)

        if (protected_by_property or protected_planes or protected_by_box) and (
            excluded_by_property or excluded_planes or excluded_by_box
        ):
            conflicts.append(f"face {face_index}: protected overrides excluded")

        if protected_by_property or protected_by_box:
            fixed_nodes.update(face.node_indices)
        for selector in (*excluded_planes, *protected_planes):
            if selector.node_constraint == "fixed":
                fixed_nodes.update(face.node_indices)
            elif selector.node_constraint == "slip_plane":
                for node in face.node_indices:
                    constraints.setdefault(node, {})[_constraint_key(selector.plane)] = selector.plane

    frozen_constraints = {
        node: tuple(values.values()) for node, values in constraints.items()
    }
    return BoundaryClassification(
        roles=tuple(roles),
        node_plane_constraints=frozen_constraints,
        fixed_nodes=frozenset(fixed_nodes),
        conflicts=tuple(conflicts),
    )
