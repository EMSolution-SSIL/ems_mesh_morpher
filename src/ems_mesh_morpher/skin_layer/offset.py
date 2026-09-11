from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np

from .boundary_roles import (
    BoundaryClassification,
    BoundaryRole,
    classify_boundary_faces,
)
from .features import FeatureTopology, cluster_normals, detect_surface_features
from .surface import SurfaceTopology, _points3d


@dataclass(frozen=True)
class OffsetResult:
    target_points: np.ndarray
    displacements: np.ndarray
    active_nodes: np.ndarray
    normal_thicknesses: np.ndarray
    node_methods: dict[int, str]
    miter_ratios: dict[int, float]
    equation_residuals: dict[int, float]


def _has_active_concave_edge(
    node: int,
    features: FeatureTopology,
    classification: BoundaryClassification,
) -> bool:
    for edge, feature in features.edges.items():
        if node not in edge or not feature.concave:
            continue
        if all(classification.roles[index] == BoundaryRole.SKIN for index in feature.face_indices):
            return True
    return False


def compute_inward_offset(
    mesh: meshio.Mesh,
    topology: SurfaceTopology,
    thickness: float,
    *,
    classification: BoundaryClassification | None = None,
    features: FeatureTopology | None = None,
    feature_angle_degrees: float = 30.0,
    maximum_miter_ratio: float = 4.0,
    equation_tolerance: float | None = None,
    concave_corner_policy: str = "fail",
    node_thicknesses: np.ndarray | None = None,
) -> OffsetResult:
    if thickness <= 0.0 or not np.isfinite(thickness):
        raise ValueError("thickness must be a positive finite value")
    if maximum_miter_ratio < 1.0:
        raise ValueError("maximum_miter_ratio must be at least 1")
    if concave_corner_policy not in {"fail", "allow"}:
        raise ValueError("concave_corner_policy must be 'fail' or 'allow'")

    points = _points3d(mesh.points)
    local_thicknesses = np.zeros(len(points), dtype=float)
    if node_thicknesses is not None:
        supplied = np.asarray(node_thicknesses, dtype=float)
        if supplied.shape != (len(points),):
            raise ValueError("node_thicknesses must contain one value per mesh node")
        if not np.all(np.isfinite(supplied)) or np.any(supplied < 0.0):
            raise ValueError("node_thicknesses must be finite and non-negative")
        local_thicknesses[:] = supplied
    classification = classification or classify_boundary_faces(mesh, topology)
    features = features or detect_surface_features(
        mesh, topology, feature_angle_degrees=feature_angle_degrees
    )
    if len(classification.roles) != len(topology.faces):
        raise ValueError("boundary classification does not match surface topology")

    tolerance = equation_tolerance
    if tolerance is None:
        tolerance = max(1.0e-12, abs(thickness) * 1.0e-8)
    displacements = np.zeros((len(points), 3), dtype=float)
    active = np.zeros(len(points), dtype=bool)
    methods: dict[int, str] = {}
    miter_ratios: dict[int, float] = {}
    residuals: dict[int, float] = {}

    for node, incident_indices in topology.node_to_faces.items():
        skin_indices = tuple(
            index for index in incident_indices if classification.roles[index] == BoundaryRole.SKIN
        )
        if not skin_indices:
            continue
        local_thickness = (
            thickness if node_thicknesses is None else local_thicknesses[node]
        )
        if local_thickness <= 0.0:
            raise ValueError(f"skin node {node} has non-positive local thickness")
        if node in classification.fixed_nodes:
            raise ValueError(f"skin node {node} is fixed by a protected boundary")
        if concave_corner_policy == "fail" and _has_active_concave_edge(
            node, features, classification
        ):
            raise ValueError(f"concave sharp feature reaches skin node {node}")

        skin_faces = [topology.faces[index] for index in skin_indices]
        clusters = cluster_normals(
            (face.normal for face in skin_faces),
            (face.area for face in skin_faces),
            features.feature_angle_degrees,
        )
        rows = list(clusters)
        values = [-float(local_thickness)] * len(rows)
        for plane in classification.node_plane_constraints.get(node, ()):
            rows.append(plane.unit_normal)
            values.append(0.0)
        matrix = np.vstack(rows)
        right_hand_side = np.asarray(values, dtype=float)
        displacement, _, _, _ = np.linalg.lstsq(matrix, right_hand_side, rcond=None)
        residual = float(np.max(np.abs(matrix @ displacement - right_hand_side)))
        if residual > tolerance:
            raise ValueError(
                f"offset constraints are inconsistent at node {node}: residual {residual:.6g}"
            )
        miter_ratio = float(np.linalg.norm(displacement) / local_thickness)
        if miter_ratio > maximum_miter_ratio:
            raise ValueError(
                f"miter ratio exceeds limit at node {node}: "
                f"{miter_ratio:.6g} > {maximum_miter_ratio:.6g}"
            )

        displacements[node] = displacement
        active[node] = True
        local_thicknesses[node] = local_thickness
        methods[node] = (
            "smooth_normal"
            if len(clusters) == 1
            else "offset_plane_edge"
            if len(clusters) == 2
            else "offset_plane_corner"
        )
        miter_ratios[node] = miter_ratio
        residuals[node] = residual

    return OffsetResult(
        target_points=points + displacements,
        displacements=displacements,
        active_nodes=active,
        normal_thicknesses=local_thicknesses,
        node_methods=methods,
        miter_ratios=miter_ratios,
        equation_residuals=residuals,
    )
