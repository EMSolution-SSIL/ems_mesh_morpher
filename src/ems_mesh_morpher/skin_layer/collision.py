from __future__ import annotations

from dataclasses import dataclass
import heapq

import meshio
import numpy as np

from .boundary_roles import BoundaryClassification, BoundaryRole
from .offset import OffsetResult
from .surface import SurfaceTopology, _points3d


@dataclass(frozen=True)
class SkinLayerCollisionConfig:
    policy: str = "fail"
    maximum_thickness_to_edge_ratio: float | None = 1.0
    opposing_surface_safety_factor: float | None = 0.4
    minimum_thickness_ratio: float = 0.05
    maximum_thickness_gradient: float | None = 0.5

    def __post_init__(self) -> None:
        if self.policy not in {"fail", "reduce", "allow"}:
            raise ValueError("collision policy must be 'fail', 'reduce', or 'allow'")
        if (
            self.maximum_thickness_to_edge_ratio is not None
            and self.maximum_thickness_to_edge_ratio <= 0.0
        ):
            raise ValueError("maximum_thickness_to_edge_ratio must be positive")
        if self.opposing_surface_safety_factor is not None and not (
            0.0 < self.opposing_surface_safety_factor < 0.5
        ):
            raise ValueError(
                "opposing_surface_safety_factor must be between 0 and 0.5"
            )
        if not 0.0 < self.minimum_thickness_ratio <= 1.0:
            raise ValueError("minimum_thickness_ratio must be in (0, 1]")
        if (
            self.maximum_thickness_gradient is not None
            and self.maximum_thickness_gradient <= 0.0
        ):
            raise ValueError("maximum_thickness_gradient must be positive")


@dataclass(frozen=True)
class LocalThicknessResult:
    requested_thickness: float
    requested_thicknesses: np.ndarray
    applied_thicknesses: np.ndarray
    maximum_thicknesses: np.ndarray
    active_nodes: np.ndarray
    risk_nodes: np.ndarray
    reduced_nodes: np.ndarray
    limiting_reasons: dict[int, tuple[str, ...]]
    opposing_clearances: dict[int, float]

    @property
    def risk_node_count(self) -> int:
        return int(np.count_nonzero(self.risk_nodes))

    @property
    def reduced_node_count(self) -> int:
        return int(np.count_nonzero(self.reduced_nodes))

    @property
    def opposing_surface_risk_node_count(self) -> int:
        return int(
            sum(
                bool(self.risk_nodes[node])
                and "opposing_surface" in self.limiting_reasons.get(int(node), ())
                for node in np.flatnonzero(self.active_nodes)
            )
        )

    @property
    def minimum_applied_thickness(self) -> float:
        return float(np.min(self.applied_thicknesses[self.active_nodes]))

    @property
    def maximum_applied_thickness(self) -> float:
        return float(np.max(self.applied_thicknesses[self.active_nodes]))

    @property
    def mean_applied_thickness(self) -> float:
        return float(np.mean(self.applied_thicknesses[self.active_nodes]))

    @property
    def minimum_opposing_clearance(self) -> float | None:
        if not self.opposing_clearances:
            return None
        return float(min(self.opposing_clearances.values()))


class SkinLayerCollisionError(ValueError):
    pass


def _skin_edges(
    topology: SurfaceTopology,
    classification: BoundaryClassification,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face_index in classification.skin_face_indices:
        nodes = topology.faces[face_index].node_indices
        for index, node in enumerate(nodes):
            other = nodes[(index + 1) % len(nodes)]
            edges.add((node, other) if node < other else (other, node))
    return edges


def _surface_triangles(
    topology: SurfaceTopology,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    face_indices: list[int] = []
    triangles: list[np.ndarray] = []
    for face_index, face in enumerate(topology.faces):
        nodes = face.node_indices
        for index in range(1, len(nodes) - 1):
            face_indices.append(face_index)
            triangles.append(points[[nodes[0], nodes[index], nodes[index + 1]]])
    return np.asarray(face_indices, dtype=int), np.asarray(triangles, dtype=float)


def _ray_triangle_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
    maximum_distance: float,
    tolerance: float,
) -> float | None:
    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    cross = np.cross(direction, second_edge)
    determinant = float(np.dot(first_edge, cross))
    if abs(determinant) <= tolerance:
        return None
    inverse = 1.0 / determinant
    relative = origin - triangle[0]
    first_coordinate = float(np.dot(relative, cross) * inverse)
    if first_coordinate < -tolerance or first_coordinate > 1.0 + tolerance:
        return None
    second_cross = np.cross(relative, first_edge)
    second_coordinate = float(np.dot(direction, second_cross) * inverse)
    if (
        second_coordinate < -tolerance
        or first_coordinate + second_coordinate > 1.0 + tolerance
    ):
        return None
    distance = float(np.dot(second_edge, second_cross) * inverse)
    if distance <= tolerance or distance > maximum_distance + tolerance:
        return None
    return distance


def _smooth_limits(
    limits: np.ndarray,
    active: np.ndarray,
    edges: set[tuple[int, int]],
    points: np.ndarray,
    maximum_gradient: float | None,
) -> np.ndarray:
    if maximum_gradient is None:
        return limits
    adjacency: dict[int, list[tuple[int, float]]] = {
        int(node): [] for node in np.flatnonzero(active)
    }
    for first, second in edges:
        if not active[first] or not active[second]:
            continue
        length = float(np.linalg.norm(points[first] - points[second]))
        adjacency[first].append((second, length))
        adjacency[second].append((first, length))

    smoothed = limits.copy()
    queue = [
        (float(smoothed[node]), int(node))
        for node in np.flatnonzero(active)
        if np.isfinite(smoothed[node])
    ]
    heapq.heapify(queue)
    while queue:
        value, node = heapq.heappop(queue)
        if value > smoothed[node]:
            continue
        for neighbour, distance in adjacency.get(node, ()):
            candidate = value + maximum_gradient * distance
            if candidate < smoothed[neighbour]:
                smoothed[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return smoothed


def compute_local_thickness_limits(
    mesh: meshio.Mesh,
    topology: SurfaceTopology,
    classification: BoundaryClassification,
    offset: OffsetResult,
    requested_thickness: float,
    config: SkinLayerCollisionConfig,
) -> LocalThicknessResult:
    points = _points3d(mesh.points)
    active = offset.active_nodes.copy()
    requested = np.zeros(len(points), dtype=float)
    requested[active] = requested_thickness
    limits = np.full(len(points), np.inf, dtype=float)
    reasons: dict[int, set[str]] = {}
    clearances: dict[int, float] = {}
    edges = _skin_edges(topology, classification)

    def apply_limit(node: int, value: float, reason: str) -> None:
        if value < requested_thickness:
            reasons.setdefault(node, set()).add(reason)
        if value < limits[node]:
            limits[node] = value
        elif np.isclose(value, limits[node], rtol=1.0e-10, atol=0.0):
            reasons.setdefault(node, set()).add(reason)

    if config.maximum_thickness_to_edge_ratio is not None:
        for first, second in edges:
            length = float(np.linalg.norm(points[first] - points[second]))
            for node in (first, second):
                if not active[node]:
                    continue
                displacement_length = float(np.linalg.norm(offset.displacements[node]))
                miter_ratio = displacement_length / requested_thickness
                apply_limit(
                    node,
                    config.maximum_thickness_to_edge_ratio * length / miter_ratio,
                    "local_edge_length",
                )

    if config.opposing_surface_safety_factor is not None:
        triangle_faces, triangles = _surface_triangles(topology, points)
        triangle_minimum = np.min(triangles, axis=1)
        triangle_maximum = np.max(triangles, axis=1)
        scale = max(float(np.ptp(points, axis=0).max()), 1.0)
        tolerance = scale * 1.0e-10
        for node in np.flatnonzero(active):
            displacement = offset.displacements[node]
            displacement_length = float(np.linalg.norm(displacement))
            if displacement_length <= tolerance:
                continue
            miter_ratio = displacement_length / requested_thickness
            direction = displacement / displacement_length
            maximum_distance = (
                requested_thickness
                * miter_ratio
                / config.opposing_surface_safety_factor
            )
            incident = set(topology.node_to_faces[int(node)])
            nearest = np.inf
            segment_end = points[node] + direction * maximum_distance
            segment_minimum = np.minimum(points[node], segment_end) - tolerance
            segment_maximum = np.maximum(points[node], segment_end) + tolerance
            candidates = np.flatnonzero(
                np.all(triangle_maximum >= segment_minimum, axis=1)
                & np.all(triangle_minimum <= segment_maximum, axis=1)
            )
            for triangle_index in candidates:
                face_index = int(triangle_faces[triangle_index])
                if face_index in incident:
                    continue
                distance = _ray_triangle_distance(
                    points[node],
                    direction,
                    triangles[triangle_index],
                    maximum_distance,
                    tolerance,
                )
                if distance is not None:
                    nearest = min(nearest, distance)
            if np.isfinite(nearest):
                clearances[int(node)] = float(nearest)
                apply_limit(
                    int(node),
                    config.opposing_surface_safety_factor * nearest / miter_ratio,
                    "opposing_surface",
                )

    raw_limits = limits.copy()
    limits = _smooth_limits(
        limits,
        active,
        edges,
        points,
        config.maximum_thickness_gradient,
    )
    for node in np.flatnonzero(active):
        if limits[node] < raw_limits[node]:
            reasons.setdefault(int(node), set()).add("thickness_gradient")

    comparison_tolerance = max(requested_thickness * 1.0e-10, 1.0e-14)
    risk = active & (limits < requested_thickness - comparison_tolerance)
    applied = requested.copy()
    reduced = np.zeros(len(points), dtype=bool)
    if config.policy == "reduce":
        applied[risk] = limits[risk]
        reduced = risk.copy()
        minimum_allowed = requested_thickness * config.minimum_thickness_ratio
        if np.any(applied[active] < minimum_allowed):
            minimum = float(np.min(applied[active]))
            raise SkinLayerCollisionError(
                "local thickness reduction would fall below minimum_thickness_ratio: "
                f"{minimum:.6g} < {minimum_allowed:.6g}"
            )
    elif config.policy == "fail" and np.any(risk):
        risk_indices = np.flatnonzero(risk)
        minimum = float(np.min(limits[risk]))
        categories = sorted(
            {reason for node in risk_indices for reason in reasons.get(int(node), ())}
        )
        raise SkinLayerCollisionError(
            f"requested thickness {requested_thickness:.6g} exceeds local limits at "
            f"{len(risk_indices)} nodes (minimum {minimum:.6g}; "
            f"reasons: {', '.join(categories)}); use collision policy 'reduce' "
            "to apply explicit local reduction"
        )

    return LocalThicknessResult(
        requested_thickness=requested_thickness,
        requested_thicknesses=requested,
        applied_thicknesses=applied,
        maximum_thicknesses=limits,
        active_nodes=active,
        risk_nodes=risk,
        reduced_nodes=reduced,
        limiting_reasons={
            node: tuple(sorted(values)) for node, values in reasons.items()
        },
        opposing_clearances=clearances,
    )
