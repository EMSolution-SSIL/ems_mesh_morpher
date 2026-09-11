from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_mesh_morpher import (
    LaplaceConfig,
    build_laplace_displacement_field,
    laplace_interpolate,
)


def _line_mesh(x_coordinates: list[float]) -> meshio.Mesh:
    points = np.array([[x, 0.0, 0.0] for x in x_coordinates], dtype=float)
    cells = [("line", np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points)))))]
    return meshio.Mesh(points, cells)


def test_inverse_distance_laplace_reproduces_linear_physical_displacement() -> None:
    mesh = _line_mesh([0.0, 1.0, 3.0])

    weighted = laplace_interpolate(
        mesh,
        [1],
        [0, 2],
        np.array([[0.0], [3.0]]),
        LaplaceConfig(weighting="inverse_distance"),
    )
    uniform = laplace_interpolate(
        mesh,
        [1],
        [0, 2],
        np.array([[0.0], [3.0]]),
        LaplaceConfig(weighting="uniform"),
    )

    np.testing.assert_allclose(weighted, [[1.0]])
    np.testing.assert_allclose(uniform, [[1.5]])


def test_laplace_field_accepts_target_points_and_preserves_mesh_metadata() -> None:
    mesh = _line_mesh([0.0, 1.0, 3.0])
    mesh.point_data["id"] = np.array([101, 102, 103])
    mesh.cell_data["element_id"] = [np.array([201, 202])]

    field = build_laplace_displacement_field(
        mesh,
        [0],
        control_target_points=np.array([[0.3, 0.0, 0.0]]),
        fixed_indices=[2],
    )
    result = field.apply_to_mesh(mesh)

    np.testing.assert_allclose(field.displacement[:, 0], [0.3, 0.2, 0.0])
    np.testing.assert_allclose(result.points[:, 0], [0.3, 1.2, 3.0])
    np.testing.assert_array_equal(result.point_data["id"], [101, 102, 103])
    np.testing.assert_array_equal(result.cell_data["element_id"][0], [201, 202])


def test_laplace_deduplicates_edges_from_volume_and_surface_cells() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = meshio.Mesh(
        points,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )

    result = laplace_interpolate(
        mesh,
        [0],
        [1, 2, 3],
        np.array([[1.0], [2.0], [3.0]]),
    )

    np.testing.assert_allclose(result, [[2.0]])


def test_laplace_rejects_unanchored_component() -> None:
    mesh = meshio.Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        [("line", np.array([[0, 1], [2, 3]]))],
    )

    with pytest.raises(ValueError, match="unanchored component"):
        laplace_interpolate(mesh, [1, 2, 3], [0], np.array([[1.0, 0.0]]))


def test_laplace_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="weighting"):
        LaplaceConfig(weighting="unknown")
    with pytest.raises(ValueError, match="minimum_distance"):
        LaplaceConfig(minimum_distance=0.0)
