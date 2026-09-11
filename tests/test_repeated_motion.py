from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_mesh_morpher import (
    LaplaceConfig,
    MotionSchedule,
    RepeatedMotionRunner,
    benchmark_repeated_motion,
    config_from_dict,
    prepare_laplace_interpolator,
)


def _line_mesh() -> meshio.Mesh:
    return meshio.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        [("line", np.array([[0, 1], [1, 2]]))],
    )


def _config(method: str):
    return config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0]},
                "deformable": {"point_indices": [1]},
                "fixed": {"point_indices": [2]},
            },
            "motion": {"type": "translation", "displacement": [0.0, 0.0]},
            "morphing": {"type": method},
            "quality": {"enabled": False},
        }
    )


def test_prepared_laplace_reuses_factorization_for_new_boundary_values() -> None:
    prepared = prepare_laplace_interpolator(
        _line_mesh(), [1], [0, 2], LaplaceConfig(weighting="inverse_distance")
    )

    np.testing.assert_allclose(prepared.solve(np.array([[0.0], [3.0]])), [[1.0]])
    np.testing.assert_allclose(prepared.solve(np.array([[1.0], [4.0]])), [[2.0]])


def test_element_aware_laplace_protects_short_elements_more_strongly() -> None:
    prepared = prepare_laplace_interpolator(
        _line_mesh(),
        [1],
        [0, 2],
        LaplaceConfig(weighting="inverse_distance_element"),
    )

    np.testing.assert_allclose(prepared.solve(np.array([[0.0], [3.0]])), [[0.6]])


def test_repeated_motion_uses_reference_mesh_and_returns_without_drift() -> None:
    schedule = MotionSchedule.translations(
        [0.0, 0.5, 1.0],
        [[0.0, 0.0], [0.3, 0.0], [0.0, 0.0]],
    )

    result = RepeatedMotionRunner(_line_mesh(), _config("weighted_laplace")).run(
        schedule
    )

    assert result.reused_laplace_factorization
    np.testing.assert_allclose(result.frames[1].result.mesh.points[:, 0], [0.3, 1.2, 3.0])
    np.testing.assert_allclose(result.frames[2].result.mesh.points, _line_mesh().points)
    assert result.summary()["frame_count"] == 3


def test_benchmark_compares_multiple_methods_with_one_schedule() -> None:
    schedule = MotionSchedule.translations([0.0, 1.0], [[0.0, 0.0], [0.2, 0.0]])

    reports = benchmark_repeated_motion(
        _line_mesh(),
        schedule,
        {"idw": _config("idw"), "laplace": _config("weighted_laplace")},
    )

    assert set(reports) == {"idw", "laplace"}
    assert reports["idw"]["frame_count"] == 2
    assert reports["laplace"]["reused_laplace_factorization"] is True


def test_release_quality_and_element_weighting_config_is_parsed() -> None:
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0]}},
            "morphing": {
                "type": "weighted_laplace",
                "weighting": "inverse_distance_element",
                "element_size_power": 2.0,
                "element_aspect_power": 0.5,
            },
            "quality": {
                "minimum_measure_ratio": 0.1,
                "minimum_scaled_jacobian": 0.01,
                "failure_policy": "scale",
                "minimum_scale": 0.2,
            },
        }
    )

    assert config.morphing.weighting == "inverse_distance_element"
    assert config.morphing.element_size_power == 2.0
    assert config.morphing.element_aspect_power == 0.5
    assert config.quality.minimum_measure_ratio == 0.1
    assert config.quality.minimum_scaled_jacobian == 0.01
    assert config.quality.failure_policy == "scale"
    assert config.quality.minimum_scale == 0.2


def test_motion_schedule_rejects_mismatched_frame_data() -> None:
    with pytest.raises(ValueError, match="equal length"):
        MotionSchedule.translations([0.0, 1.0], [[0.0, 0.0]])
