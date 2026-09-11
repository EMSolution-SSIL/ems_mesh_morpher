from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import meshio
import numpy as np


_VOLUME_FACES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tetra": (
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
    ),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}


@dataclass(frozen=True)
class BoundaryFace:
    node_indices: tuple[int, ...]
    face_type: str
    owner_cell_type: str
    owner_block_index: int
    owner_element_index: int
    owner_local_face_index: int
    owner_property_id: int | None
    normal: np.ndarray
    area: float
    centroid: np.ndarray
    owner_centroid: np.ndarray
    planarity_error: float = 0.0
    surface_property_id: int | None = None
    surface_block_index: int | None = None
    surface_element_index: int | None = None

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_indices))


@dataclass(frozen=True)
class SurfaceTopology:
    faces: tuple[BoundaryFace, ...]
    edge_to_faces: dict[tuple[int, int], tuple[int, ...]]
    node_to_faces: dict[int, tuple[int, ...]]
    boundary_edges: tuple[tuple[int, int], ...]
    nonmanifold_edges: tuple[tuple[int, int], ...]
    selected_volume_elements: int

    @property
    def triangle_count(self) -> int:
        return sum(face.face_type == "triangle" for face in self.faces)

    @property
    def quad_count(self) -> int:
        return sum(face.face_type == "quad" for face in self.faces)

    @property
    def max_quad_planarity_error(self) -> float:
        return max(
            (face.planarity_error for face in self.faces if face.face_type == "quad"),
            default=0.0,
        )


def _points3d(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {2, 3}:
        raise ValueError("mesh points must be an N x 2 or N x 3 array")
    if values.shape[1] == 2:
        return np.pad(values, ((0, 0), (0, 1)))
    return values


def _property_key(mesh: meshio.Mesh) -> str | None:
    if "property_id" in mesh.cell_data:
        return "property_id"
    if "gmsh:physical" in mesh.cell_data:
        return "gmsh:physical"
    return None


def _face_geometry(coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    if len(coordinates) == 3:
        area_vector = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    else:
        area_vector = np.zeros(3, dtype=float)
        for index in range(len(coordinates)):
            area_vector += np.cross(coordinates[index], coordinates[(index + 1) % len(coordinates)])
    magnitude = float(np.linalg.norm(area_vector))
    if magnitude <= np.finfo(float).eps:
        raise ValueError("degenerate boundary face with zero area")
    return area_vector / magnitude, 0.5 * magnitude


def _quad_planarity_error(coordinates: np.ndarray) -> float:
    centered = coordinates - np.mean(coordinates, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return float(np.max(np.abs(centered @ vh[-1])))


def _face_edges(nodes: tuple[int, ...]) -> Iterable[tuple[int, int]]:
    for index, node in enumerate(nodes):
        other = nodes[(index + 1) % len(nodes)]
        yield (node, other) if node < other else (other, node)


def _build_topology(
    faces: tuple[BoundaryFace, ...], selected_volume_elements: int
) -> SurfaceTopology:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    node_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for edge in _face_edges(face.node_indices):
            edge_faces.setdefault(edge, []).append(face_index)
        for node in face.node_indices:
            node_faces.setdefault(node, []).append(face_index)
    frozen_edges = {edge: tuple(indices) for edge, indices in edge_faces.items()}
    return SurfaceTopology(
        faces=faces,
        edge_to_faces=frozen_edges,
        node_to_faces={node: tuple(indices) for node, indices in node_faces.items()},
        boundary_edges=tuple(sorted(edge for edge, indices in frozen_edges.items() if len(indices) == 1)),
        nonmanifold_edges=tuple(sorted(edge for edge, indices in frozen_edges.items() if len(indices) > 2)),
        selected_volume_elements=selected_volume_elements,
    )


def extract_boundary_surface(
    mesh: meshio.Mesh,
    volume_property_ids: Iterable[int] | None = None,
    *,
    match_existing: bool = True,
) -> SurfaceTopology:
    points = _points3d(mesh.points)
    requested = None if volume_property_ids is None else {int(value) for value in volume_property_ids}
    property_key = _property_key(mesh)
    if requested is not None and property_key is None:
        raise ValueError("volume property selection requires 'property_id' or 'gmsh:physical'")

    occurrences: dict[tuple[int, ...], list[BoundaryFace]] = {}
    selected_count = 0
    for block_index, block in enumerate(mesh.cells):
        templates = _VOLUME_FACES.get(block.type)
        if templates is None:
            continue
        connectivity = np.asarray(block.data, dtype=int)
        properties = None
        if property_key is not None:
            properties = np.asarray(mesh.cell_data[property_key][block_index], dtype=int).reshape(-1)
            if len(properties) != len(connectivity):
                raise ValueError(f"cell_data['{property_key}'] length does not match cell block")

        for element_index, element_nodes in enumerate(connectivity):
            property_id = None if properties is None else int(properties[element_index])
            if requested is not None and property_id not in requested:
                continue
            selected_count += 1
            owner_centroid = np.mean(points[element_nodes], axis=0)
            for local_face_index, template in enumerate(templates):
                nodes = tuple(int(element_nodes[index]) for index in template)
                coordinates = points[list(nodes)]
                normal, area = _face_geometry(coordinates)
                centroid = np.mean(coordinates, axis=0)
                if float(np.dot(normal, centroid - owner_centroid)) < 0.0:
                    nodes = tuple(reversed(nodes))
                    coordinates = points[list(nodes)]
                    normal, area = _face_geometry(coordinates)
                face_type = "triangle" if len(nodes) == 3 else "quad"
                planarity = _quad_planarity_error(coordinates) if len(nodes) == 4 else 0.0
                face = BoundaryFace(
                    node_indices=nodes,
                    face_type=face_type,
                    owner_cell_type=block.type,
                    owner_block_index=block_index,
                    owner_element_index=element_index,
                    owner_local_face_index=local_face_index,
                    owner_property_id=property_id,
                    normal=normal,
                    area=area,
                    centroid=centroid,
                    owner_centroid=owner_centroid,
                    planarity_error=planarity,
                )
                occurrences.setdefault(face.key, []).append(face)

    if selected_count == 0:
        raise ValueError("no tetrahedral or hexahedral volume elements were selected")
    duplicate = [key for key, faces in occurrences.items() if len(faces) > 2]
    if duplicate:
        raise ValueError(f"non-manifold volume faces found: {len(duplicate)}")

    boundary_faces = tuple(faces[0] for faces in occurrences.values() if len(faces) == 1)
    topology = _build_topology(boundary_faces, selected_count)
    if match_existing:
        topology = match_existing_surface_elements(mesh, topology)
    return topology


def match_existing_surface_elements(
    mesh: meshio.Mesh, topology: SurfaceTopology
) -> SurfaceTopology:
    property_key = _property_key(mesh)
    if property_key is None:
        return topology

    matches: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for block_index, block in enumerate(mesh.cells):
        if block.type not in {"triangle", "quad"}:
            continue
        properties = np.asarray(mesh.cell_data[property_key][block_index], dtype=int).reshape(-1)
        for element_index, nodes in enumerate(np.asarray(block.data, dtype=int)):
            key = tuple(sorted(int(node) for node in nodes))
            if key in matches:
                raise ValueError(f"duplicate existing surface elements for face {key}")
            matches[key] = (block_index, element_index, int(properties[element_index]))

    updated = []
    for face in topology.faces:
        match = matches.get(face.key)
        if match is None:
            updated.append(face)
            continue
        block_index, element_index, property_id = match
        updated.append(
            replace(
                face,
                surface_property_id=property_id,
                surface_block_index=block_index,
                surface_element_index=element_index,
            )
        )
    return _build_topology(tuple(updated), topology.selected_volume_elements)


def validate_surface_topology(
    topology: SurfaceTopology,
    *,
    require_closed: bool = True,
    quad_planarity_tolerance: float | None = None,
) -> None:
    if topology.nonmanifold_edges:
        raise ValueError(f"surface contains {len(topology.nonmanifold_edges)} non-manifold edges")
    if require_closed and topology.boundary_edges:
        raise ValueError(f"surface is open at {len(topology.boundary_edges)} edges")
    if quad_planarity_tolerance is not None:
        if quad_planarity_tolerance < 0.0:
            raise ValueError("quad_planarity_tolerance must be non-negative")
        if topology.max_quad_planarity_error > quad_planarity_tolerance:
            raise ValueError(
                "quad planarity error exceeds tolerance: "
                f"{topology.max_quad_planarity_error:.6g} > {quad_planarity_tolerance:.6g}"
            )
