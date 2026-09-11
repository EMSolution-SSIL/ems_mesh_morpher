from __future__ import annotations

import copy
from dataclasses import dataclass

import meshio
import numpy as np

from .config import MorphConfig
from .morph import (
    IDWConfig,
    LaplaceConfig,
    PreparedLaplaceInterpolator,
    RBFConfig,
    distance_blend_weights,
    idw_interpolate,
    laplace_interpolate,
    nearest_moving_displacements,
    radial_blend_weights,
    rbf_interpolate,
)
from .motion import motion_displacement_field
from .quality import (
    MeshQualityReport,
    QualityGate,
    QualityReport,
    evaluate_2d_quality,
    evaluate_mesh_quality,
    limit_displacement_by_quality,
)
from .selection import NodeClassification, classify_nodes_from_regions


@dataclass(frozen=True)
class MeshMotionResult:
    mesh: meshio.Mesh
    classification: NodeClassification
    displacement: np.ndarray
    quality_before: QualityReport | None
    quality_after: QualityReport | None
    orientation_flip_count: int
    relative_quality: MeshQualityReport | None = None
    applied_displacement_scale: float = 1.0

    def summary(self) -> dict[str, object]:
        magnitudes = np.linalg.norm(self.displacement, axis=1)
        protected_count = (
            0
            if self.classification.protected is None
            else int(np.count_nonzero(self.classification.protected))
        )
        zone_count = (
            0
            if self.classification.morphing_zone is None
            else int(np.count_nonzero(self.classification.morphing_zone))
        )
        return {
            "nodes": self.classification.counts,
            "protected_nodes": protected_count,
            "morphing_zone_nodes": zone_count,
            "max_displacement": float(np.max(magnitudes)) if len(magnitudes) else 0.0,
            "orientation_flip_count": self.orientation_flip_count,
            "applied_displacement_scale": self.applied_displacement_scale,
            "relative_quality": (
                None if self.relative_quality is None else self.relative_quality.to_dict()
            ),
            "quality_before": None if self.quality_before is None else self.quality_before.to_dict(),
            "quality_after": None if self.quality_after is None else self.quality_after.to_dict(),
        }


def _with_inferred_prescribed_nodes(
    classification: NodeClassification,
    assigned: np.ndarray,
    require_selected_assignments: bool = True,
) -> NodeClassification:
    protected = (
        np.zeros_like(assigned)
        if classification.protected is None
        else classification.protected
    )
    explicitly_moving = classification.moving
    missing = explicitly_moving & ~assigned
    if require_selected_assignments and np.any(missing):
        raise ValueError(
            "prescribed motion has no displacement for "
            f"{np.count_nonzero(missing)} selected moving nodes"
        )
    moving = (explicitly_moving | assigned) & ~protected
    fixed = classification.fixed & ~moving
    deformable = classification.deformable & ~moving & ~fixed
    return NodeClassification(
        moving=moving,
        fixed=fixed,
        deformable=deformable,
        protected=protected,
        morphing_zone=classification.morphing_zone,
    )


def _validated_external_displacement(mesh: meshio.Mesh, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(displacement, dtype=float)
    if field.shape != mesh.points.shape:
        raise ValueError("external displacement field shape must match mesh.points")
    assigned = np.linalg.norm(field, axis=1) > 0.0
    return field.copy(), assigned


def apply_mesh_motion(
    mesh: meshio.Mesh,
    config: MorphConfig,
    prescribed_displacement: np.ndarray | None = None,
    *,
    prepared_laplace: PreparedLaplaceInterpolator | None = None,
) -> MeshMotionResult:
    classification = classify_nodes_from_regions(mesh, config.regions)
    external_displacement = prescribed_displacement is not None
    if prescribed_displacement is None:
        ideal_displacement, assigned = motion_displacement_field(mesh.points, config.motion)
    else:
        ideal_displacement, assigned = _validated_external_displacement(mesh, prescribed_displacement)

    if assigned is not None:
        classification = _with_inferred_prescribed_nodes(
            classification,
            assigned,
            require_selected_assignments=not external_displacement,
        )
    if not np.any(classification.moving):
        raise ValueError("mesh motion requires at least one moving node")

    quality_before = evaluate_2d_quality(mesh) if config.quality.enabled else None
    total_displacement = np.zeros_like(mesh.points, dtype=float)
    total_displacement[classification.moving] = ideal_displacement[classification.moving]

    deformable = classification.deformable
    if np.any(deformable) and config.morphing.type != "none":
        if config.morphing.type in {"idw", "rbf", "laplace", "weighted_laplace"}:
            controls = classification.moving | classification.fixed
            control_displacement = np.zeros(
                (int(np.count_nonzero(controls)), mesh.points.shape[1]),
                dtype=float,
            )
            control_displacement[classification.moving[controls]] = ideal_displacement[
                classification.moving
            ]
            if config.morphing.type == "idw":
                total_displacement[deformable] = idw_interpolate(
                    mesh.points[deformable],
                    mesh.points[controls],
                    control_displacement,
                    IDWConfig(
                        power=config.morphing.power,
                        neighbors=config.morphing.neighbors,
                        radius=config.morphing.radius,
                        smoothing=config.morphing.smoothing,
                        outside_policy=config.morphing.outside_policy,
                        query_chunk_size=config.morphing.query_chunk_size,
                    ),
                )
            elif config.morphing.type == "rbf":
                total_displacement[deformable] = rbf_interpolate(
                    mesh.points[deformable],
                    mesh.points[controls],
                    control_displacement,
                    RBFConfig(
                        kernel=config.morphing.kernel,
                        radius=config.morphing.radius,
                        radius_scale=config.morphing.radius_scale,
                        neighbors=config.morphing.neighbors,
                        regularization=config.morphing.regularization,
                        polynomial_degree=config.morphing.polynomial_degree,
                        outside_policy=config.morphing.outside_policy,
                        query_chunk_size=config.morphing.query_chunk_size,
                    ),
                )
            else:
                query_indices = np.flatnonzero(deformable)
                control_indices = np.flatnonzero(controls)
                laplace_config = LaplaceConfig(
                    weighting=config.morphing.weighting,
                    distance_power=config.morphing.distance_power,
                    minimum_distance=config.morphing.minimum_distance,
                    element_size_power=config.morphing.element_size_power,
                    element_aspect_power=config.morphing.element_aspect_power,
                )
                if prepared_laplace is None:
                    total_displacement[deformable] = laplace_interpolate(
                        mesh,
                        query_indices,
                        control_indices,
                        control_displacement,
                        laplace_config,
                    )
                else:
                    if not np.array_equal(prepared_laplace.query_indices, query_indices):
                        raise ValueError(
                            "prepared Laplace query indices do not match the motion regions"
                        )
                    if not np.array_equal(
                        prepared_laplace.control_indices, control_indices
                    ):
                        raise ValueError(
                            "prepared Laplace control indices do not match the motion regions"
                        )
                    total_displacement[deformable] = prepared_laplace.solve(
                        control_displacement
                    )
        else:
            if assigned is None:
                deformable_target = ideal_displacement[deformable]
            else:
                deformable_target = nearest_moving_displacements(
                    mesh.points,
                    deformable,
                    classification.moving,
                    ideal_displacement,
                    chunk_size=config.morphing.distance_chunk_size,
                )

        if config.morphing.type == "radial_blend":
            weights = radial_blend_weights(
                mesh.points,
                center=config.geometry.center,
                moving_radius=config.geometry.moving_radius,
                fixed_radius=config.geometry.fixed_radius,
                blend=config.morphing.blend,
            )
        elif config.morphing.type == "distance_blend":
            weights = distance_blend_weights(
                mesh.points,
                deformable,
                classification.moving,
                classification.fixed,
                blend=config.morphing.blend,
                chunk_size=config.morphing.distance_chunk_size,
            )
        elif config.morphing.type not in {
            "idw",
            "rbf",
            "laplace",
            "weighted_laplace",
        }:
            raise ValueError(f"unsupported morphing type: {config.morphing.type}")
        if config.morphing.type not in {
            "idw",
            "rbf",
            "laplace",
            "weighted_laplace",
        }:
            total_displacement[deformable] = (
                weights[deformable, None] * deformable_target
            )

    morphed = copy.deepcopy(mesh)
    morphed.points = np.asarray(mesh.points, dtype=float).copy() + total_displacement

    relative_quality = None
    applied_scale = 1.0
    if config.quality.enabled:
        gate = QualityGate(
            reject_orientation_flips=config.quality.fail_on_inverted,
            reject_degenerate_elements=config.quality.reject_degenerate_elements,
            minimum_measure_ratio=config.quality.minimum_measure_ratio,
            minimum_scaled_jacobian=config.quality.minimum_scaled_jacobian,
        )
        relative_quality = evaluate_mesh_quality(mesh, morphed)
        reason = gate.rejection_reason(relative_quality)
        if reason is not None and config.quality.failure_policy == "scale":
            limited = limit_displacement_by_quality(
                mesh,
                total_displacement,
                gate,
                minimum_scale=config.quality.minimum_scale,
                max_iterations=config.quality.max_scale_iterations,
                scale_tolerance=config.quality.scale_tolerance,
            )
            total_displacement = limited.displacement
            applied_scale = limited.scale
            relative_quality = limited.report
            morphed.points = np.asarray(mesh.points, dtype=float) + total_displacement
            reason = None
        if reason is not None:
            raise ValueError(f"mesh morphing quality rejected: {reason}")
    quality_after = evaluate_2d_quality(morphed) if config.quality.enabled else None
    orientation_flip_count = (
        0 if relative_quality is None else relative_quality.orientation_flip_count
    )

    return MeshMotionResult(
        mesh=morphed,
        classification=classification,
        displacement=total_displacement,
        quality_before=quality_before,
        quality_after=quality_after,
        orientation_flip_count=orientation_flip_count,
        relative_quality=relative_quality,
        applied_displacement_scale=applied_scale,
    )
