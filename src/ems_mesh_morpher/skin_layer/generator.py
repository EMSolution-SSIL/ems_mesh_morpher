from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import meshio
import numpy as np

from ems_file_format_converter import write_mesh

from .boundary_roles import (
    BoundaryClassification,
    BoundaryRoleConfig,
    classify_boundary_faces,
)
from .collision import (
    LocalThicknessResult,
    SkinLayerCollisionConfig,
    compute_local_thickness_limits,
)
from .core_morph import (
    CoreMorphingConfig,
    compute_core_displacement,
    selected_volume_masks,
)
from .features import FeatureTopology, detect_surface_features
from .offset import OffsetResult, compute_inward_offset
from .quality import (
    VolumeQualityReport,
    evaluate_3d_quality,
    minimum_measure_ratio,
)
from .surface import SurfaceTopology, extract_boundary_surface, validate_surface_topology
from .topology import TopologyBuildResult, build_layered_mesh


def compute_layer_thicknesses(
    total_thickness: float,
    layer_count: int,
    growth_ratio: float = 1.0,
) -> tuple[float, ...]:
    """Split total thickness geometrically from the outer to inner surface."""
    if total_thickness <= 0.0 or not np.isfinite(total_thickness):
        raise ValueError("total_thickness must be a positive finite value")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1:
        raise ValueError("layer_count must be a positive integer")
    if growth_ratio <= 0.0 or not np.isfinite(growth_ratio):
        raise ValueError("growth_ratio must be a positive finite value")
    logarithms = np.arange(layer_count, dtype=float) * np.log(growth_ratio)
    logarithms -= float(np.max(logarithms))
    weights = np.exp(logarithms)
    values = total_thickness * weights / float(np.sum(weights))
    values[-1] += total_thickness - float(np.sum(values))
    if np.any(values <= 0.0):
        raise ValueError("growth_ratio produces a layer too thin for floating point")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class SkinLayerQualityConfig:
    minimum_core_scaled_jacobian: float = 1.0e-8
    minimum_layer_scaled_jacobian: float = 1.0e-8
    minimum_core_measure_ratio: float = 0.1
    fail_on_inverted: bool = True

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_core_scaled_jacobian <= 1.0:
            raise ValueError("minimum_core_scaled_jacobian must be between -1 and 1")
        if not -1.0 <= self.minimum_layer_scaled_jacobian <= 1.0:
            raise ValueError("minimum_layer_scaled_jacobian must be between -1 and 1")
        if self.minimum_core_measure_ratio < 0.0:
            raise ValueError("minimum_core_measure_ratio must be non-negative")


@dataclass(frozen=True)
class SkinLayerConfig:
    thickness: float
    volume_property_ids: tuple[int, ...] | None = None
    layer_count: int = 1
    growth_ratio: float = 1.0
    boundary_roles: BoundaryRoleConfig = field(default_factory=BoundaryRoleConfig)
    core_morphing: CoreMorphingConfig = field(default_factory=CoreMorphingConfig)
    quality: SkinLayerQualityConfig = field(default_factory=SkinLayerQualityConfig)
    collision: SkinLayerCollisionConfig = field(
        default_factory=SkinLayerCollisionConfig
    )
    skin_layer_property_id: int | None = None
    inner_surface_property_id: int | None = None
    side_surface_property_id: int | None = None
    skin_layer_name: str = "skin_layer"
    inner_surface_name: str = "skin_inner_surface"
    feature_angle_degrees: float = 30.0
    maximum_miter_ratio: float = 4.0
    concave_corner_policy: str = "fail"
    quad_planarity_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0.0 or not np.isfinite(self.thickness):
            raise ValueError("thickness must be a positive finite value")
        if not self.skin_layer_name:
            raise ValueError("skin_layer_name must not be empty")
        if not self.inner_surface_name:
            raise ValueError("inner_surface_name must not be empty")
        compute_layer_thicknesses(
            self.thickness,
            self.layer_count,
            self.growth_ratio,
        )

    @property
    def layer_thicknesses(self) -> tuple[float, ...]:
        return compute_layer_thicknesses(
            self.thickness,
            self.layer_count,
            self.growth_ratio,
        )

    @property
    def layer_fractions(self) -> tuple[float, ...]:
        fractions = np.cumsum(self.layer_thicknesses) / self.thickness
        fractions[-1] = 1.0
        return tuple(float(value) for value in fractions)


@dataclass(frozen=True)
class SkinLayerReport:
    core_morphing_method: str
    selected_volume_element_count: int
    skin_face_count: int
    excluded_face_count: int
    protected_face_count: int
    new_node_count: int
    layer_element_count: int
    inner_surface_element_count: int
    side_surface_element_count: int
    requested_thickness: float
    minimum_applied_thickness: float
    maximum_applied_thickness: float
    mean_applied_thickness: float
    thickness_limit_node_count: int
    thickness_reduced_node_count: int
    opposing_surface_risk_node_count: int
    minimum_opposing_clearance: float | None
    layer_count: int
    growth_ratio: float
    layer_thicknesses: tuple[float, ...]
    actual_layer_thickness_min: tuple[float, ...]
    actual_layer_thickness_max: tuple[float, ...]
    actual_layer_thickness_mean: tuple[float, ...]
    actual_normal_thickness_min: float
    actual_normal_thickness_max: float
    actual_normal_thickness_mean: float
    minimum_core_measure_ratio: float
    quality_before: VolumeQualityReport
    quality_after_core: VolumeQualityReport
    quality_layer: VolumeQualityReport
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SkinLayerResult:
    mesh: meshio.Mesh
    config: SkinLayerConfig
    surface: SurfaceTopology
    classification: BoundaryClassification
    features: FeatureTopology
    offset: OffsetResult
    local_thickness: LocalThicknessResult
    topology: TopologyBuildResult
    report: SkinLayerReport

    @property
    def inner_node_map(self) -> dict[int, int]:
        return self.topology.inner_node_map

    @property
    def layer_node_maps(self) -> tuple[dict[int, int], ...]:
        return self.topology.layer_node_maps

    def write(
        self,
        path: str | Path,
        *,
        file_format: str | None = None,
    ) -> None:
        write_mesh(path, self.mesh, file_format=file_format)


class SkinLayerQualityError(ValueError):
    pass


def _property_ids(mesh: meshio.Mesh) -> set[int]:
    key = (
        "property_id"
        if "property_id" in mesh.cell_data
        else "gmsh:physical"
        if "gmsh:physical" in mesh.cell_data
        else None
    )
    if key is None:
        return set()
    return {
        int(value)
        for values in mesh.cell_data[key]
        for value in np.asarray(values).reshape(-1)
    }


def _resolve_new_properties(
    mesh: meshio.Mesh,
    layer_property_id: int | None,
    inner_property_id: int | None,
) -> tuple[int, int]:
    used = _property_ids(mesh)
    # The metadata normalizer assigns property 1 to legacy meshes that have
    # no property array, so keep it available for their original elements.
    if not used and mesh.cells:
        used.add(1)
    next_id = max(used, default=0) + 1
    layer = next_id if layer_property_id is None else int(layer_property_id)
    if layer in used:
        raise ValueError(f"skin layer property ID already exists: {layer}")
    used.add(layer)
    next_id = max(max(used), next_id) + 1
    inner = next_id if inner_property_id is None else int(inner_property_id)
    if inner in used:
        raise ValueError(f"inner surface property ID already exists: {inner}")
    return layer, inner


def _layer_selections(topology: TopologyBuildResult) -> dict[int, np.ndarray]:
    return {
        block_index: np.ones(len(topology.mesh.cells[block_index].data), dtype=bool)
        for block_index in topology.layer_block_indices
    }


def _quality_failures(
    config: SkinLayerQualityConfig,
    core: VolumeQualityReport,
    layer: VolumeQualityReport,
    measure_ratio: float,
) -> list[str]:
    failures = []
    if config.fail_on_inverted:
        if core.inverted_count or core.degenerate_count:
            failures.append(
                f"core has {core.inverted_count} inverted and "
                f"{core.degenerate_count} degenerate elements"
            )
        if layer.inverted_count or layer.degenerate_count:
            failures.append(
                f"layer has {layer.inverted_count} inverted and "
                f"{layer.degenerate_count} degenerate elements"
            )
    if (
        core.minimum_scaled_jacobian is not None
        and core.minimum_scaled_jacobian < config.minimum_core_scaled_jacobian
    ):
        failures.append(
            "core minimum scaled Jacobian "
            f"{core.minimum_scaled_jacobian:.6g} is below "
            f"{config.minimum_core_scaled_jacobian:.6g}"
        )
    if (
        layer.minimum_scaled_jacobian is not None
        and layer.minimum_scaled_jacobian < config.minimum_layer_scaled_jacobian
    ):
        failures.append(
            "layer minimum scaled Jacobian "
            f"{layer.minimum_scaled_jacobian:.6g} is below "
            f"{config.minimum_layer_scaled_jacobian:.6g}"
        )
    if measure_ratio < config.minimum_core_measure_ratio:
        failures.append(
            f"minimum core measure ratio {measure_ratio:.6g} is below "
            f"{config.minimum_core_measure_ratio:.6g}"
        )
    return failures


def _normal_thickness_samples(
    surface: SurfaceTopology,
    classification: BoundaryClassification,
    offset: OffsetResult,
) -> np.ndarray:
    values = []
    for face_index in classification.skin_face_indices:
        face = surface.faces[face_index]
        for node in face.node_indices:
            values.append(-float(np.dot(offset.displacements[node], face.normal)))
    return np.asarray(values, dtype=float)


def _attempt_generation(
    mesh: meshio.Mesh,
    config: SkinLayerConfig,
    surface: SurfaceTopology,
    classification: BoundaryClassification,
    offset: OffsetResult,
    selections: dict[int, np.ndarray],
    layer_property_id: int,
    inner_property_id: int,
    method: str,
) -> tuple[TopologyBuildResult, VolumeQualityReport, VolumeQualityReport, float]:
    core_morph = compute_core_displacement(
        mesh,
        surface,
        offset,
        config.volume_property_ids,
        config.core_morphing,
        method=method,
    )
    topology = build_layered_mesh(
        mesh,
        surface,
        classification,
        offset,
        core_morph,
        selections,
        skin_layer_property_id=layer_property_id,
        inner_surface_property_id=inner_property_id,
        side_surface_property_id=config.side_surface_property_id,
        skin_layer_name=config.skin_layer_name,
        inner_surface_name=config.inner_surface_name,
        layer_fractions=config.layer_fractions,
    )
    core_quality = evaluate_3d_quality(topology.mesh, selections)
    layer_quality = evaluate_3d_quality(topology.mesh, _layer_selections(topology))
    measure_ratio = minimum_measure_ratio(mesh, topology.mesh, selections)
    failures = _quality_failures(config.quality, core_quality, layer_quality, measure_ratio)
    if failures:
        raise SkinLayerQualityError("; ".join(failures))
    return topology, core_quality, layer_quality, measure_ratio


def generate_skin_layer(
    mesh: meshio.Mesh,
    config: SkinLayerConfig,
) -> SkinLayerResult:
    surface = extract_boundary_surface(mesh, config.volume_property_ids)
    validate_surface_topology(
        surface,
        require_closed=True,
        quad_planarity_tolerance=config.quad_planarity_tolerance,
    )
    classification = classify_boundary_faces(mesh, surface, config.boundary_roles)
    if not classification.skin_face_indices:
        raise ValueError("boundary role selection produced no skin faces")
    features = detect_surface_features(
        mesh,
        surface,
        feature_angle_degrees=config.feature_angle_degrees,
    )
    requested_offset = compute_inward_offset(
        mesh,
        surface,
        config.thickness,
        classification=classification,
        features=features,
        maximum_miter_ratio=config.maximum_miter_ratio,
        concave_corner_policy=config.concave_corner_policy,
    )
    local_thickness = compute_local_thickness_limits(
        mesh,
        surface,
        classification,
        requested_offset,
        config.thickness,
        config.collision,
    )
    offset = requested_offset
    collision_warnings: list[str] = []
    if local_thickness.reduced_node_count:
        offset = compute_inward_offset(
            mesh,
            surface,
            config.thickness,
            classification=classification,
            features=features,
            maximum_miter_ratio=config.maximum_miter_ratio,
            concave_corner_policy=config.concave_corner_policy,
            node_thicknesses=local_thickness.applied_thicknesses,
        )
        collision_warnings.append(
            "local thickness reduced at "
            f"{local_thickness.reduced_node_count} nodes; minimum applied thickness "
            f"is {local_thickness.minimum_applied_thickness:.6g}"
        )
    elif config.collision.policy == "allow" and local_thickness.risk_node_count:
        collision_warnings.append(
            f"{local_thickness.risk_node_count} local thickness risks were allowed"
        )
    selections = selected_volume_masks(mesh, config.volume_property_ids)
    quality_before = evaluate_3d_quality(mesh, selections)
    if config.quality.fail_on_inverted and (
        quality_before.inverted_count or quality_before.degenerate_count
    ):
        raise SkinLayerQualityError(
            "selected input core contains inverted or degenerate elements"
        )
    layer_property_id, inner_property_id = _resolve_new_properties(
        mesh,
        config.skin_layer_property_id,
        config.inner_surface_property_id,
    )

    methods: Iterable[str]
    if config.core_morphing.method == "auto":
        methods = ("distance_blend", "laplace")
    else:
        methods = (config.core_morphing.method,)
    errors: list[str] = []
    for method in methods:
        try:
            topology, core_quality, layer_quality, measure_ratio = _attempt_generation(
                mesh,
                config,
                surface,
                classification,
                offset,
                selections,
                layer_property_id,
                inner_property_id,
                method,
            )
            break
        except (ValueError, RuntimeError) as exc:
            errors.append(f"{method}: {exc}")
    else:
        raise SkinLayerQualityError(
            "all core morphing methods failed: " + " | ".join(errors)
        )

    thickness = _normal_thickness_samples(surface, classification, offset)
    fraction_steps = np.diff((0.0, *config.layer_fractions))
    actual_layer_samples = [thickness * step for step in fraction_steps]
    layer_element_count = sum(
        len(topology.mesh.cells[index].data) for index in topology.layer_block_indices
    )
    inner_element_count = sum(
        len(topology.mesh.cells[index].data)
        for index in topology.inner_surface_block_indices
    )
    side_element_count = sum(
        len(topology.mesh.cells[index].data)
        for index in topology.side_surface_block_indices
    )
    report = SkinLayerReport(
        core_morphing_method=method,
        selected_volume_element_count=sum(
            int(np.count_nonzero(mask)) for mask in selections.values()
        ),
        skin_face_count=len(classification.skin_face_indices),
        excluded_face_count=len(classification.excluded_face_indices),
        protected_face_count=len(classification.protected_face_indices),
        new_node_count=sum(len(node_map) for node_map in topology.layer_node_maps),
        layer_element_count=layer_element_count,
        inner_surface_element_count=inner_element_count,
        side_surface_element_count=side_element_count,
        requested_thickness=config.thickness,
        minimum_applied_thickness=local_thickness.minimum_applied_thickness,
        maximum_applied_thickness=local_thickness.maximum_applied_thickness,
        mean_applied_thickness=local_thickness.mean_applied_thickness,
        thickness_limit_node_count=local_thickness.risk_node_count,
        thickness_reduced_node_count=local_thickness.reduced_node_count,
        opposing_surface_risk_node_count=(
            local_thickness.opposing_surface_risk_node_count
        ),
        minimum_opposing_clearance=local_thickness.minimum_opposing_clearance,
        layer_count=config.layer_count,
        growth_ratio=config.growth_ratio,
        layer_thicknesses=config.layer_thicknesses,
        actual_layer_thickness_min=tuple(
            float(np.min(values)) for values in actual_layer_samples
        ),
        actual_layer_thickness_max=tuple(
            float(np.max(values)) for values in actual_layer_samples
        ),
        actual_layer_thickness_mean=tuple(
            float(np.mean(values)) for values in actual_layer_samples
        ),
        actual_normal_thickness_min=float(np.min(thickness)),
        actual_normal_thickness_max=float(np.max(thickness)),
        actual_normal_thickness_mean=float(np.mean(thickness)),
        minimum_core_measure_ratio=measure_ratio,
        quality_before=quality_before,
        quality_after_core=core_quality,
        quality_layer=layer_quality,
        warnings=tuple((*collision_warnings, *errors)),
    )
    return SkinLayerResult(
        mesh=topology.mesh,
        config=config,
        surface=surface,
        classification=classification,
        features=features,
        offset=offset,
        local_thickness=local_thickness,
        topology=topology,
        report=report,
    )
