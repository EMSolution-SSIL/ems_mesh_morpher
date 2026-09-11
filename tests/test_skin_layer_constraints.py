import numpy as np

from ems_mesh_morpher.skin_layer import PlaneConstraint, apply_directional_constraints


def test_plane_constraints_allow_only_tangent_displacement():
    displacement = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_plane = PlaneConstraint((1.0, 0.0, 0.0), 0.0)
    y_plane = PlaneConstraint((0.0, 1.0, 0.0), 0.0)

    constrained = apply_directional_constraints(
        displacement,
        {0: (x_plane,), 1: (x_plane, y_plane)},
    )

    np.testing.assert_allclose(constrained[0], [0.0, 2.0, 3.0])
    np.testing.assert_allclose(constrained[1], [0.0, 0.0, 6.0])


def test_three_independent_planes_fix_a_node():
    displacement = np.array([[1.0, 2.0, 3.0]])
    planes = tuple(
        PlaneConstraint(tuple(float(index == axis) for index in range(3)), 0.0)
        for axis in range(3)
    )

    constrained = apply_directional_constraints(displacement, {0: planes})

    np.testing.assert_allclose(constrained[0], 0.0, atol=1.0e-12)
