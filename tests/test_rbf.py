from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_file_format_converter import read_mesh, write_mesh
from ems_mesh_morpher import (
    RBFConfig,
    build_rbf_displacement_field,
    rbf_interpolate,
)


def test_rbf_reproduces_affine_vector_field() -> None:
    controls = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    values = np.column_stack(
        (
            0.1 + 0.2 * controls[:, 0] - 0.3 * controls[:, 1],
            -0.2 + 0.4 * controls[:, 0] + 0.1 * controls[:, 1],
        )
    )
    query = np.array([[0.25, 0.75], [0.6, 0.2]])
    expected = np.column_stack(
        (
            0.1 + 0.2 * query[:, 0] - 0.3 * query[:, 1],
            -0.2 + 0.4 * query[:, 0] + 0.1 * query[:, 1],
        )
    )

    result = rbf_interpolate(query, controls, values)

    np.testing.assert_allclose(result, expected, atol=1.0e-11)


def test_rbf_preserves_exact_control_values() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.2, -0.1], [0.0, 0.3]])

    interpolated = rbf_interpolate(controls, controls, values)

    np.testing.assert_allclose(interpolated, values)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("nearest", 2.0), ("zero", 0.0)],
)
def test_rbf_radius_outside_policy(policy: str, expected: float) -> None:
    result = rbf_interpolate(
        np.array([[10.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[2.0, 0.0]]),
        RBFConfig(
            radius=0.1,
            polynomial_degree=-1,
            outside_policy=policy,
        ),
    )

    assert result[0, 0] == pytest.approx(expected)


def test_rbf_radius_can_fail_for_uncontrolled_queries() -> None:
    with pytest.raises(ValueError, match="no RBF controls within radius"):
        rbf_interpolate(
            np.array([[10.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[2.0, 0.0]]),
            RBFConfig(
                radius=0.1,
                polynomial_degree=-1,
                outside_policy="error",
            ),
        )


def test_rbf_field_accepts_target_points_and_preserves_mesh_metadata() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    mesh = meshio.Mesh(
        points,
        [("line", np.array([[0, 1], [1, 2]]))],
        point_data={"id": np.array([101, 102, 103])},
        cell_data={"element_id": [np.array([201, 202])]},
    )

    field = build_rbf_displacement_field(
        points,
        [0],
        control_target_points=np.array([[0.2, 0.0, 0.0]]),
        fixed_indices=[2],
    )
    morphed = field.apply_to_mesh(mesh)

    np.testing.assert_allclose(field.displacement[:, 0], [0.2, 0.1, 0.0])
    np.testing.assert_allclose(morphed.points[:, 0], [0.2, 0.6, 1.0])
    np.testing.assert_array_equal(morphed.point_data["id"], [101, 102, 103])
    np.testing.assert_array_equal(morphed.cell_data["element_id"][0], [201, 202])
    np.testing.assert_array_equal(morphed.cells[0].data, [[0, 1], [1, 2]])
    assert field.resolved_radius > 0.0


def test_rbf_rejects_duplicate_control_coordinates() -> None:
    with pytest.raises(ValueError, match="duplicate coordinates"):
        rbf_interpolate(
            np.array([[0.5, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
        )


def test_rbf_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="radius_scale"):
        RBFConfig(radius_scale=1.0)
    with pytest.raises(ValueError, match="polynomial_degree"):
        RBFConfig(polynomial_degree=2)


def test_rbf_morphed_mesh_round_trips_through_gmsh4(tmp_path) -> None:
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
        point_data={"id": np.array([101, 102, 103, 104])},
        cell_data={
            "property_id": [np.array([10])],
            "material_id": [np.array([20])],
            "element_id": [np.array([501])],
        },
        field_data={"actuator": np.array([10, 3])},
    )
    field = build_rbf_displacement_field(
        mesh.points,
        [0],
        control_displacements=np.array([[0.05, 0.0, 0.0]]),
        fixed_indices=[1, 2, 3],
    )
    output = tmp_path / "rbf_actuator.msh"

    write_mesh(output, field.apply_to_mesh(mesh), file_format="gmsh4")
    restored = read_mesh(output, file_format="gmsh4")

    np.testing.assert_allclose(restored.points, field.target_points)
    np.testing.assert_array_equal(
        restored.point_data["gmsh:node_tags"], [101, 102, 103, 104]
    )
    np.testing.assert_array_equal(restored.cell_data["gmsh:element_tags"][0], [501])
    np.testing.assert_array_equal(restored.cells[0].data, [[0, 1, 2, 3]])
