from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_mesh_morpher import (
    CoreMorphingConfig,
    SkinLayerCollisionConfig,
    SkinLayerCollisionError,
    SkinLayerConfig,
    classify_boundary_faces,
    compute_inward_offset,
    compute_local_thickness_limits,
    extract_boundary_surface,
    generate_skin_layer,
)


def _box_mesh(size: tuple[float, float, float]) -> meshio.Mesh:
    x, y, z = size
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [x, 0.0, 0.0],
            [x, y, 0.0],
            [0.0, y, 0.0],
            [0.0, 0.0, z],
            [x, 0.0, z],
            [x, y, z],
            [0.0, y, z],
        ]
    )
    return meshio.Mesh(
        points,
        [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
        cell_data={
            "property_id": [np.array([100])],
            "material_id": [np.array([7])],
            "element_id": [np.array([501])],
        },
    )


def _two_box_mesh() -> meshio.Mesh:
    thin = _box_mesh((1.0, 1.0, 0.1))
    thick = _box_mesh((1.0, 1.0, 1.0))
    thick_points = thick.points + np.array([2.0, 0.0, 0.0])
    return meshio.Mesh(
        np.vstack((thin.points, thick_points)),
        [
            (
                "hexahedron",
                np.vstack((thin.cells[0].data, thick.cells[0].data + 8)),
            )
        ],
        cell_data={
            "property_id": [np.array([100, 100])],
            "material_id": [np.array([7, 7])],
            "element_id": [np.array([501, 502])],
        },
    )
def _config(policy: str) -> SkinLayerConfig:
    return SkinLayerConfig(
        thickness=0.08,
        volume_property_ids=(100,),
        layer_count=3,
        core_morphing=CoreMorphingConfig(method="none"),
        collision=SkinLayerCollisionConfig(policy=policy),
        skin_layer_property_id=200,
        inner_surface_property_id=201,
    )


def test_collision_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="collision policy"):
        SkinLayerCollisionConfig(policy="automatic")
    with pytest.raises(ValueError, match="between 0 and 0.5"):
        SkinLayerCollisionConfig(opposing_surface_safety_factor=0.5)
    with pytest.raises(ValueError, match="minimum_thickness_ratio"):
        SkinLayerCollisionConfig(minimum_thickness_ratio=0.0)


def test_narrow_gap_fails_before_generating_an_invalid_layer() -> None:
    mesh = _box_mesh((1.0, 1.0, 0.1))

    with pytest.raises(SkinLayerCollisionError, match="opposing_surface"):
        generate_skin_layer(mesh, _config("fail"))


def test_narrow_gap_can_apply_explicit_local_thickness_reduction() -> None:
    mesh = _box_mesh((1.0, 1.0, 0.1))

    result = generate_skin_layer(mesh, _config("reduce"))

    assert result.report.requested_thickness == pytest.approx(0.08)
    assert result.report.minimum_applied_thickness == pytest.approx(0.04)
    assert result.report.maximum_applied_thickness == pytest.approx(0.04)
    assert result.report.thickness_limit_node_count == 8
    assert result.report.thickness_reduced_node_count == 8
    assert result.report.opposing_surface_risk_node_count == 8
    assert result.report.minimum_opposing_clearance == pytest.approx(
        np.sqrt(3.0) * 0.1
    )
    assert result.report.actual_normal_thickness_min == pytest.approx(0.04)
    assert result.report.actual_normal_thickness_max == pytest.approx(0.04)
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0
    assert np.allclose(
        result.local_thickness.applied_thicknesses[
            result.local_thickness.active_nodes
        ],
        0.04,
    )
    assert any("local thickness reduced" in warning for warning in result.report.warnings)


def test_allow_policy_reports_risk_without_silent_reduction() -> None:
    mesh = _box_mesh((1.0, 1.0, 0.1))
    surface = extract_boundary_surface(mesh, (100,))
    classification = classify_boundary_faces(mesh, surface)
    offset = compute_inward_offset(
        mesh,
        surface,
        0.08,
        classification=classification,
    )

    result = compute_local_thickness_limits(
        mesh,
        surface,
        classification,
        offset,
        0.08,
        SkinLayerCollisionConfig(policy="allow"),
    )

    assert result.risk_node_count == 8
    assert result.reduced_node_count == 0
    assert np.all(result.applied_thicknesses[result.active_nodes] == 0.08)


def test_reduction_is_local_to_the_narrow_component() -> None:
    result = generate_skin_layer(_two_box_mesh(), _config("reduce"))

    assert result.report.minimum_applied_thickness == pytest.approx(0.04)
    assert result.report.maximum_applied_thickness == pytest.approx(0.08)
    assert result.report.thickness_reduced_node_count == 8
    assert np.allclose(result.local_thickness.applied_thicknesses[:8], 0.04)
    assert np.allclose(result.local_thickness.applied_thicknesses[8:], 0.08)
