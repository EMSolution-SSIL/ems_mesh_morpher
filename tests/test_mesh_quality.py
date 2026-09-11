from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_mesh_morpher import (
    QualityGate,
    apply_mesh_motion,
    config_from_dict,
    evaluate_mesh_quality,
    limit_displacement_by_quality,
)


def _triangle_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )


def test_quality_report_supports_3d_and_identifies_worst_element() -> None:
    mesh = meshio.Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    target = mesh.points.copy()
    target[3, 2] = 0.25

    report = evaluate_mesh_quality(mesh, target)

    assert report.counts_by_type == {"tetra": 1}
    assert report.minimum_measure_ratio == pytest.approx(0.25)
    assert report.minimum_scaled_jacobian is not None
    assert report.orientation_flip_count == 0
    assert report.worst_element == (0, 0)


def test_quality_report_supports_pyramid_and_detects_orientation_flip() -> None:
    mesh = meshio.Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        ),
        [("pyramid", np.array([[0, 1, 2, 3, 4]]))],
    )
    target = mesh.points.copy()
    target[4, 2] = -1.0

    report = evaluate_mesh_quality(mesh, target)

    assert report.counts_by_type == {"pyramid": 1}
    assert report.orientation_flip_count == 1
    assert report.minimum_measure_ratio == pytest.approx(1.0)


def test_displacement_is_limited_to_quality_threshold() -> None:
    mesh = _triangle_mesh()
    displacement = np.zeros_like(mesh.points)
    displacement[2, 1] = -2.0

    result = limit_displacement_by_quality(
        mesh,
        displacement,
        QualityGate(minimum_measure_ratio=0.25),
        scale_tolerance=1.0e-8,
    )

    assert result.scale == pytest.approx(0.375, abs=1.0e-7)
    assert result.report.minimum_measure_ratio == pytest.approx(0.25, abs=1.0e-7)
    assert result.report.orientation_flip_count == 0


def test_mesh_motion_can_scale_requested_motion_instead_of_failing() -> None:
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [2]}},
            "motion": {"type": "translation", "displacement": [0.0, -2.0]},
            "morphing": {"type": "none"},
            "quality": {
                "minimum_measure_ratio": 0.25,
                "failure_policy": "scale",
                "scale_tolerance": 1.0e-8,
            },
        }
    )

    result = apply_mesh_motion(_triangle_mesh(), config)

    assert result.applied_displacement_scale == pytest.approx(0.375, abs=1.0e-7)
    assert result.mesh.points[2, 1] == pytest.approx(0.25, abs=1.0e-7)
    assert result.relative_quality is not None

