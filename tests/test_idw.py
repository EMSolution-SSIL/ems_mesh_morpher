from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_file_format_converter import read_mesh, write_mesh
from ems_mesh_morpher import (
    IDWConfig,
    build_idw_displacement_field,
    idw_interpolate,
)


def test_idw_interpolates_vector_values_symmetrically() -> None:
    values = idw_interpolate(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]]),
        IDWConfig(neighbors=None),
    )

    np.testing.assert_allclose(values, [[1.0, 2.0, 0.0]])


def test_idw_preserves_exact_control_values() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.2, -0.1], [0.0, 0.3]])

    interpolated = idw_interpolate(controls, controls, values)

    np.testing.assert_allclose(interpolated, values)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("nearest", 2.0), ("zero", 0.0)],
)
def test_idw_radius_outside_policy(policy: str, expected: float) -> None:
    result = idw_interpolate(
        np.array([[10.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[2.0, 0.0]]),
        IDWConfig(radius=0.1, outside_policy=policy),
    )

    assert result[0, 0] == pytest.approx(expected)


def test_idw_radius_can_fail_for_uncontrolled_queries() -> None:
    with pytest.raises(ValueError, match="no IDW controls within radius"):
        idw_interpolate(
            np.array([[10.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[2.0, 0.0]]),
            IDWConfig(radius=0.1, outside_policy="error"),
        )


def test_displacement_field_accepts_target_points_and_preserves_mesh_metadata() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    mesh = meshio.Mesh(
        points,
        [("line", np.array([[0, 1], [1, 2]]))],
        point_data={"id": np.array([101, 102, 103])},
        cell_data={"element_id": [np.array([201, 202])]},
    )

    field = build_idw_displacement_field(
        points,
        [0],
        control_target_points=np.array([[0.2, 0.0, 0.0]]),
        fixed_indices=[2],
        config=IDWConfig(neighbors=None),
    )
    morphed = field.apply_to_mesh(mesh)

    np.testing.assert_allclose(field.displacement[:, 0], [0.2, 0.1, 0.0])
    np.testing.assert_allclose(morphed.points[:, 0], [0.2, 0.6, 1.0])
    np.testing.assert_array_equal(morphed.point_data["id"], [101, 102, 103])
    np.testing.assert_array_equal(morphed.cell_data["element_id"][0], [201, 202])
    np.testing.assert_array_equal(morphed.cells[0].data, [[0, 1], [1, 2]])


def test_displacement_field_rejects_ambiguous_or_overlapping_inputs() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="exactly one"):
        build_idw_displacement_field(points, [0])
    with pytest.raises(ValueError, match="must not overlap"):
        build_idw_displacement_field(
            points,
            [0],
            control_displacements=np.array([[1.0, 0.0]]),
            fixed_indices=[0],
        )


def test_idw_morphed_mesh_round_trips_through_gmsh4(tmp_path) -> None:
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
    field = build_idw_displacement_field(
        mesh.points,
        [0],
        control_displacements=np.array([[0.05, 0.0, 0.0]]),
        fixed_indices=[1, 2, 3],
    )
    output = tmp_path / "idw_actuator.msh"

    write_mesh(output, field.apply_to_mesh(mesh), file_format="gmsh4")
    restored = read_mesh(output, file_format="gmsh4")

    np.testing.assert_allclose(restored.points, field.target_points)
    np.testing.assert_array_equal(
        restored.point_data["gmsh:node_tags"], [101, 102, 103, 104]
    )
    np.testing.assert_array_equal(restored.cell_data["gmsh:element_tags"][0], [501])
    np.testing.assert_array_equal(restored.cells[0].data, [[0, 1, 2, 3]])
