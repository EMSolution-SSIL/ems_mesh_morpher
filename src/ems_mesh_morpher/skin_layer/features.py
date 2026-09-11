from __future__ import annotations

from dataclasses import dataclass
from math import degrees
from typing import Iterable

import meshio
import numpy as np

from .surface import SurfaceTopology, _points3d


@dataclass(frozen=True)
class EdgeFeature:
    edge: tuple[int, int]
    face_indices: tuple[int, ...]
    angle_degrees: float
    kind: str
    concave: bool = False


@dataclass(frozen=True)
class VertexFeature:
    node_index: int
    face_indices: tuple[int, ...]
    normal_clusters: tuple[np.ndarray, ...]
    kind: str


@dataclass(frozen=True)
class FeatureTopology:
    edges: dict[tuple[int, int], EdgeFeature]
    vertices: dict[int, VertexFeature]
    feature_angle_degrees: float

    @property
    def sharp_edges(self) -> tuple[EdgeFeature, ...]:
        return tuple(edge for edge in self.edges.values() if edge.kind == "sharp")

    @property
    def concave_edges(self) -> tuple[EdgeFeature, ...]:
        return tuple(edge for edge in self.edges.values() if edge.concave)

    @property
    def corner_vertices(self) -> tuple[VertexFeature, ...]:
        return tuple(vertex for vertex in self.vertices.values() if vertex.kind == "corner")


def cluster_normals(
    normals: Iterable[np.ndarray],
    weights: Iterable[float],
    feature_angle_degrees: float,
) -> tuple[np.ndarray, ...]:
    cosine_limit = float(np.cos(np.deg2rad(feature_angle_degrees)))
    clusters: list[tuple[np.ndarray, float]] = []
    for normal, weight in zip(normals, weights, strict=True):
        unit = np.asarray(normal, dtype=float)
        unit /= np.linalg.norm(unit)
        selected = None
        best_dot = -1.0
        for index, (weighted_sum, _) in enumerate(clusters):
            center = weighted_sum / np.linalg.norm(weighted_sum)
            similarity = float(np.dot(unit, center))
            if similarity >= cosine_limit and similarity > best_dot:
                selected = index
                best_dot = similarity
        if selected is None:
            clusters.append((unit * float(weight), float(weight)))
        else:
            weighted_sum, total_weight = clusters[selected]
            clusters[selected] = (
                weighted_sum + unit * float(weight),
                total_weight + float(weight),
            )
    return tuple(weighted_sum / np.linalg.norm(weighted_sum) for weighted_sum, _ in clusters)


def _edge_is_concave(
    points: np.ndarray,
    topology: SurfaceTopology,
    edge: tuple[int, int],
    face_indices: tuple[int, int],
    tolerance: float,
) -> bool:
    midpoint = np.mean(points[list(edge)], axis=0)
    first = topology.faces[face_indices[0]]
    second = topology.faces[face_indices[1]]
    first_side = float(np.dot(second.centroid - midpoint, first.normal))
    second_side = float(np.dot(first.centroid - midpoint, second.normal))
    return first_side > tolerance or second_side > tolerance


def detect_surface_features(
    mesh: meshio.Mesh,
    topology: SurfaceTopology,
    *,
    feature_angle_degrees: float = 30.0,
    concavity_tolerance: float = 1.0e-10,
) -> FeatureTopology:
    if not 0.0 < feature_angle_degrees < 180.0:
        raise ValueError("feature_angle_degrees must be between 0 and 180")
    points = _points3d(mesh.points)
    edges: dict[tuple[int, int], EdgeFeature] = {}
    for edge, face_indices in topology.edge_to_faces.items():
        if len(face_indices) != 2:
            edges[edge] = EdgeFeature(edge, face_indices, 180.0, "boundary", False)
            continue
        first, second = (topology.faces[index] for index in face_indices)
        cosine = float(np.clip(np.dot(first.normal, second.normal), -1.0, 1.0))
        angle = degrees(float(np.arccos(cosine)))
        kind = "sharp" if angle >= feature_angle_degrees else "smooth"
        concave = kind == "sharp" and _edge_is_concave(
            points,
            topology,
            edge,
            (face_indices[0], face_indices[1]),
            concavity_tolerance,
        )
        edges[edge] = EdgeFeature(edge, face_indices, angle, kind, concave)

    vertices: dict[int, VertexFeature] = {}
    for node, face_indices in topology.node_to_faces.items():
        faces = [topology.faces[index] for index in face_indices]
        clusters = cluster_normals(
            (face.normal for face in faces),
            (face.area for face in faces),
            feature_angle_degrees,
        )
        kind = "corner" if len(clusters) >= 3 else "sharp_edge" if len(clusters) == 2 else "smooth"
        vertices[node] = VertexFeature(node, face_indices, clusters, kind)
    return FeatureTopology(edges, vertices, feature_angle_degrees)
