from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    center: tuple[float, float] = (0.0, 0.0)
    moving_radius: float = 0.0
    fixed_radius: float = 0.0


@dataclass(frozen=True)
class GeometricSelectionConfig:
    type: str
    center: tuple[float, ...] = ()
    minimum: tuple[float, ...] = ()
    maximum: tuple[float, ...] = ()
    radius: float | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None
    axis_start: tuple[float, ...] = ()
    axis_end: tuple[float, ...] = ()
    tolerance: float = 0.0


@dataclass(frozen=True)
class SelectionConfig:
    property_ids: tuple[int, ...] = ()
    point_indices: tuple[int, ...] = ()
    geometries: tuple[GeometricSelectionConfig, ...] = ()
    combine: str = "union"
    invert: bool = False


@dataclass(frozen=True)
class RegionConfig:
    # The property-ID fields retain the Phase 1 Python API.
    moving_property_ids: tuple[int, ...] = ()
    deformable_property_ids: tuple[int, ...] = ()
    fixed_property_ids: tuple[int, ...] = ()
    moving: SelectionConfig = field(default_factory=SelectionConfig)
    deformable: SelectionConfig = field(default_factory=SelectionConfig)
    fixed: SelectionConfig = field(default_factory=SelectionConfig)
    protected: SelectionConfig = field(default_factory=SelectionConfig)
    morphing_zone: SelectionConfig | None = None
    conflict_policy: str = "priority"


@dataclass(frozen=True)
class PointDisplacementConfig:
    point_index: int
    displacement: tuple[float, ...]


@dataclass(frozen=True)
class MotionConfig:
    type: str = "translation"
    displacement: tuple[float, ...] = (0.0, 0.0)
    center: tuple[float, ...] = (0.0, 0.0, 0.0)
    axis: tuple[float, ...] = (0.0, 0.0, 1.0)
    angle_degrees: float = 0.0
    scale: tuple[float, ...] = (1.0,)
    point_displacements: tuple[PointDisplacementConfig, ...] = ()


@dataclass(frozen=True)
class MorphingConfig:
    type: str = "radial_blend"
    blend: str = "smoothstep"
    distance_chunk_size: int = 2048
    kernel: str = "wendland_c2"
    power: float = 2.0
    neighbors: int | None = 32
    radius: float | None = None
    radius_scale: float = 1.25
    smoothing: float = 0.0
    regularization: float = 1.0e-12
    polynomial_degree: int = 1
    weighting: str = "inverse_distance"
    distance_power: float = 1.0
    minimum_distance: float = 1.0e-12
    element_size_power: float = 1.0
    element_aspect_power: float = 1.0
    outside_policy: str = "nearest"
    query_chunk_size: int = 8192

    def __post_init__(self) -> None:
        if self.type not in {
            "none",
            "radial_blend",
            "distance_blend",
            "idw",
            "rbf",
            "laplace",
            "weighted_laplace",
        }:
            raise ValueError(f"unsupported morphing.type: {self.type}")
        if self.kernel != "wendland_c2":
            raise ValueError("morphing.kernel must be 'wendland_c2'")
        if self.power <= 0.0:
            raise ValueError("morphing.power must be positive")
        if self.neighbors is not None and self.neighbors <= 0:
            raise ValueError("morphing.neighbors must be positive or null")
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError("morphing.radius must be positive or null")
        if self.radius_scale <= 1.0:
            raise ValueError("morphing.radius_scale must be greater than 1")
        if self.smoothing < 0.0:
            raise ValueError("morphing.smoothing must be non-negative")
        if self.regularization < 0.0:
            raise ValueError("morphing.regularization must be non-negative")
        if self.polynomial_degree not in {-1, 0, 1}:
            raise ValueError("morphing.polynomial_degree must be -1, 0, or 1")
        if self.weighting not in {
            "uniform",
            "inverse_distance",
            "inverse_distance_element",
        }:
            raise ValueError(
                "morphing.weighting must be 'uniform', 'inverse_distance', or "
                "'inverse_distance_element'"
            )
        if self.distance_power < 0.0:
            raise ValueError("morphing.distance_power must be non-negative")
        if self.minimum_distance <= 0.0:
            raise ValueError("morphing.minimum_distance must be positive")
        if self.element_size_power < 0.0:
            raise ValueError("morphing.element_size_power must be non-negative")
        if self.element_aspect_power < 0.0:
            raise ValueError("morphing.element_aspect_power must be non-negative")
        if self.outside_policy not in {"nearest", "zero", "error"}:
            raise ValueError(
                "morphing.outside_policy must be 'nearest', 'zero', or 'error'"
            )
        if self.query_chunk_size <= 0:
            raise ValueError("morphing.query_chunk_size must be positive")


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool = True
    fail_on_inverted: bool = True
    reject_degenerate_elements: bool = True
    minimum_measure_ratio: float = 0.0
    minimum_scaled_jacobian: float | None = None
    failure_policy: str = "error"
    minimum_scale: float = 0.0
    max_scale_iterations: int = 32
    scale_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.minimum_measure_ratio < 0.0:
            raise ValueError("quality.minimum_measure_ratio must be non-negative")
        if self.minimum_scaled_jacobian is not None and not np.isfinite(
            self.minimum_scaled_jacobian
        ):
            raise ValueError("quality.minimum_scaled_jacobian must be finite or null")
        if self.failure_policy not in {"error", "scale"}:
            raise ValueError("quality.failure_policy must be 'error' or 'scale'")
        if not 0.0 <= self.minimum_scale <= 1.0:
            raise ValueError("quality.minimum_scale must be between zero and one")
        if self.max_scale_iterations <= 0 or self.scale_tolerance <= 0.0:
            raise ValueError(
                "quality.max_scale_iterations and scale_tolerance must be positive"
            )


@dataclass(frozen=True)
class OutputConfig:
    file_format: str | None = None


@dataclass(frozen=True)
class MorphConfig:
    mode: str = "motor_eccentricity_2d"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    morphing: MorphingConfig = field(default_factory=MorphingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _tuple_floats(value: Any, key: str, lengths: tuple[int, ...] | None = None) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    if lengths is not None and len(value) not in lengths:
        allowed = " or ".join(str(length) for length in lengths)
        raise ValueError(f"{key} must contain {allowed} items")
    return tuple(float(item) for item in value)


def _tuple_float2(value: Any, key: str) -> tuple[float, float]:
    result = _tuple_floats(value, key, (2,))
    return (result[0], result[1])


def _tuple_ints(value: Any, key: str = "integer list") -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    ids: list[int] = []
    for item in value:
        if isinstance(item, dict):
            start = int(item["start"])
            stop = int(item["stop"])
            step = int(item.get("step", 1))
            if step == 0:
                raise ValueError(f"{key} range step must not be zero")
            if (stop - start) * step < 0:
                raise ValueError(f"{key} range step direction does not reach stop")
            ids.extend(range(start, stop + (1 if step > 0 else -1), step))
        else:
            ids.append(int(item))
    return tuple(ids)


def _optional_vector(data: dict[str, Any], key: str) -> tuple[float, ...]:
    value = data.get(key)
    return () if value is None else _tuple_floats(value, key, (2, 3))


def _geometric_selection_from_dict(data: dict[str, Any]) -> GeometricSelectionConfig:
    selection_type = str(data.get("type", "")).lower()
    supported = {"all", "box", "circle", "annulus", "sphere", "cylinder"}
    if selection_type not in supported:
        raise ValueError(f"unsupported geometric selection type: {selection_type}")

    minimum_value = data.get("minimum", data.get("min"))
    maximum_value = data.get("maximum", data.get("max"))
    minimum = () if minimum_value is None else _tuple_floats(minimum_value, "selection.minimum", (2, 3))
    maximum = () if maximum_value is None else _tuple_floats(maximum_value, "selection.maximum", (2, 3))

    return GeometricSelectionConfig(
        type=selection_type,
        center=_optional_vector(data, "center"),
        minimum=minimum,
        maximum=maximum,
        radius=None if data.get("radius") is None else float(data["radius"]),
        inner_radius=None if data.get("inner_radius") is None else float(data["inner_radius"]),
        outer_radius=None if data.get("outer_radius") is None else float(data["outer_radius"]),
        axis_start=_optional_vector(data, "axis_start"),
        axis_end=_optional_vector(data, "axis_end"),
        tolerance=float(data.get("tolerance", 0.0)),
    )


def _selection_from_dict(data: Any, fallback_property_ids: tuple[int, ...] = ()) -> SelectionConfig:
    if data is None:
        return SelectionConfig(property_ids=fallback_property_ids)
    if not isinstance(data, dict):
        raise ValueError("region selection must be an object")

    geometry_data = data.get("geometries")
    if geometry_data is None:
        single_geometry = data.get("geometry")
        geometry_data = [] if single_geometry is None else [single_geometry]
    if not isinstance(geometry_data, (list, tuple)):
        raise ValueError("selection.geometries must be an array")
    geometries = tuple(_geometric_selection_from_dict(item) for item in geometry_data)

    combine = str(data.get("combine", "union")).lower()
    if combine not in {"union", "intersection"}:
        raise ValueError("selection.combine must be 'union' or 'intersection'")

    property_ids = _tuple_ints(data.get("property_ids"), "selection.property_ids")
    if fallback_property_ids:
        property_ids = tuple(dict.fromkeys((*fallback_property_ids, *property_ids)))

    return SelectionConfig(
        property_ids=property_ids,
        point_indices=_tuple_ints(data.get("point_indices"), "selection.point_indices"),
        geometries=geometries,
        combine=combine,
        invert=bool(data.get("invert", False)),
    )


def _geometry_from_dict(data: dict[str, Any], required: bool) -> GeometryConfig:
    center = _tuple_float2(data.get("center", [0.0, 0.0]), "geometry.center")

    if "moving_radius" in data or "fixed_radius" in data:
        if "moving_radius" not in data or "fixed_radius" not in data:
            raise ValueError("geometry requires both moving_radius and fixed_radius")
        moving_radius = float(data["moving_radius"])
        fixed_radius = float(data["fixed_radius"])
    elif "airgap_inner_radius" in data or "airgap_outer_radius" in data:
        if "airgap_inner_radius" not in data or "airgap_outer_radius" not in data:
            raise ValueError("geometry requires both airgap radii")
        inner_radius = float(data["airgap_inner_radius"])
        outer_radius = float(data["airgap_outer_radius"])
        moving_boundary = data.get("moving_boundary", "inner")
        if moving_boundary == "inner":
            moving_radius, fixed_radius = inner_radius, outer_radius
        elif moving_boundary == "outer":
            moving_radius, fixed_radius = outer_radius, inner_radius
        else:
            raise ValueError("geometry.moving_boundary must be 'inner' or 'outer'")
    elif required:
        raise ValueError("radial_blend requires moving_radius and fixed_radius")
    else:
        moving_radius = fixed_radius = 0.0

    if required and moving_radius == fixed_radius:
        raise ValueError("moving_radius and fixed_radius must differ")
    return GeometryConfig(center=center, moving_radius=moving_radius, fixed_radius=fixed_radius)


def _point_displacements_from_dict(value: Any) -> tuple[PointDisplacementConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("motion.point_displacements must be an array")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each point displacement must be an object")
        result.append(
            PointDisplacementConfig(
                point_index=int(item["point_index"]),
                displacement=_tuple_floats(
                    item["displacement"], "motion.point_displacements.displacement", (2, 3)
                ),
            )
        )
    return tuple(result)


def config_from_dict(data: dict[str, Any]) -> MorphConfig:
    mode = str(data.get("mode", "motor_eccentricity_2d"))
    if mode not in {"motor_eccentricity_2d", "mesh_motion"}:
        raise ValueError(f"unsupported mode: {mode}")

    regions_data = data.get("regions", {})
    motion_data = data.get("motion", {})
    morphing_data = data.get("morphing", {})
    quality_data = data.get("quality", {})
    output_data = data.get("output", {})

    motion_type = str(motion_data.get("type", "translation")).lower()
    if motion_type not in {"translation", "rotation", "scaling", "prescribed"}:
        raise ValueError(f"unsupported motion.type: {motion_type}")

    default_morphing_type = "radial_blend" if mode == "motor_eccentricity_2d" else "none"
    morphing_type = str(morphing_data.get("type", default_morphing_type)).lower()
    if morphing_type not in {
        "none",
        "radial_blend",
        "distance_blend",
        "idw",
        "rbf",
        "laplace",
        "weighted_laplace",
    }:
        raise ValueError(f"unsupported morphing.type: {morphing_type}")
    blend = str(morphing_data.get("blend", "smoothstep")).lower()
    if blend not in {"linear", "smoothstep"}:
        raise ValueError(f"unsupported morphing.blend: {blend}")

    if mode == "motor_eccentricity_2d" and motion_type != "translation":
        raise ValueError("motor_eccentricity_2d requires translation motion")
    if mode == "motor_eccentricity_2d" and morphing_type != "radial_blend":
        raise ValueError("motor_eccentricity_2d requires radial_blend morphing")

    moving_ids = _tuple_ints(regions_data.get("moving_property_ids"), "moving_property_ids")
    deformable_ids = _tuple_ints(regions_data.get("deformable_property_ids"), "deformable_property_ids")
    fixed_ids = _tuple_ints(regions_data.get("fixed_property_ids"), "fixed_property_ids")
    conflict_policy = str(regions_data.get("conflict_policy", "priority")).lower()
    if conflict_policy not in {"priority", "error"}:
        raise ValueError("regions.conflict_policy must be 'priority' or 'error'")

    scale_value = motion_data.get("scale", 1.0)
    if isinstance(scale_value, (int, float)):
        scale = (float(scale_value),)
    else:
        scale = _tuple_floats(scale_value, "motion.scale", (2, 3))

    neighbors_value = morphing_data.get("neighbors", 32)
    radius_value = morphing_data.get("radius")

    return MorphConfig(
        mode=mode,
        geometry=_geometry_from_dict(data.get("geometry", {}), required=morphing_type == "radial_blend"),
        regions=RegionConfig(
            moving_property_ids=moving_ids,
            deformable_property_ids=deformable_ids,
            fixed_property_ids=fixed_ids,
            moving=_selection_from_dict(regions_data.get("moving"), moving_ids),
            deformable=_selection_from_dict(regions_data.get("deformable"), deformable_ids),
            fixed=_selection_from_dict(regions_data.get("fixed"), fixed_ids),
            protected=_selection_from_dict(regions_data.get("protected")),
            morphing_zone=(
                None
                if regions_data.get("morphing_zone") is None
                else _selection_from_dict(regions_data["morphing_zone"])
            ),
            conflict_policy=conflict_policy,
        ),
        motion=MotionConfig(
            type=motion_type,
            displacement=_tuple_floats(
                motion_data.get("displacement", [0.0, 0.0]), "motion.displacement", (2, 3)
            ),
            center=_tuple_floats(motion_data.get("center", [0.0, 0.0, 0.0]), "motion.center", (2, 3)),
            axis=_tuple_floats(motion_data.get("axis", [0.0, 0.0, 1.0]), "motion.axis", (3,)),
            angle_degrees=float(motion_data.get("angle_degrees", motion_data.get("angle_deg", 0.0))),
            scale=scale,
            point_displacements=_point_displacements_from_dict(motion_data.get("point_displacements")),
        ),
        morphing=MorphingConfig(
            type=morphing_type,
            blend=blend,
            distance_chunk_size=int(morphing_data.get("distance_chunk_size", 2048)),
            kernel=str(morphing_data.get("kernel", "wendland_c2")).lower(),
            power=float(morphing_data.get("power", 2.0)),
            neighbors=None if neighbors_value is None else int(neighbors_value),
            radius=None if radius_value is None else float(radius_value),
            radius_scale=float(morphing_data.get("radius_scale", 1.25)),
            smoothing=float(morphing_data.get("smoothing", 0.0)),
            regularization=float(morphing_data.get("regularization", 1.0e-12)),
            polynomial_degree=int(morphing_data.get("polynomial_degree", 1)),
            weighting=str(
                morphing_data.get("weighting", "inverse_distance")
            ).lower(),
            distance_power=float(morphing_data.get("distance_power", 1.0)),
            minimum_distance=float(morphing_data.get("minimum_distance", 1.0e-12)),
            element_size_power=float(morphing_data.get("element_size_power", 1.0)),
            element_aspect_power=float(
                morphing_data.get("element_aspect_power", 1.0)
            ),
            outside_policy=str(
                morphing_data.get("outside_policy", "nearest")
            ).lower(),
            query_chunk_size=int(morphing_data.get("query_chunk_size", 8192)),
        ),
        quality=QualityConfig(
            enabled=bool(quality_data.get("enabled", True)),
            fail_on_inverted=bool(quality_data.get("fail_on_inverted", True)),
            reject_degenerate_elements=bool(
                quality_data.get("reject_degenerate_elements", True)
            ),
            minimum_measure_ratio=float(
                quality_data.get("minimum_measure_ratio", 0.0)
            ),
            minimum_scaled_jacobian=(
                None
                if quality_data.get("minimum_scaled_jacobian") is None
                else float(quality_data["minimum_scaled_jacobian"])
            ),
            failure_policy=str(quality_data.get("failure_policy", "error")).lower(),
            minimum_scale=float(quality_data.get("minimum_scale", 0.0)),
            max_scale_iterations=int(quality_data.get("max_scale_iterations", 32)),
            scale_tolerance=float(quality_data.get("scale_tolerance", 1.0e-6)),
        ),
        output=OutputConfig(file_format=output_data.get("file_format")),
    )


def load_config(path: str | Path) -> MorphConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))
