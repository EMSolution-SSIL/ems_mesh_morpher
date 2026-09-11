import numpy as np

from ems_mesh_morpher.morph import radial_blend_weights


def test_radial_blend_weights_support_inner_moving_boundary():
    points = np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]])

    weights = radial_blend_weights(points, (0.0, 0.0), moving_radius=1.0, fixed_radius=2.0, blend="linear")

    np.testing.assert_allclose(weights, [1.0, 0.5, 0.0])


def test_radial_blend_weights_support_outer_moving_boundary():
    points = np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]])

    weights = radial_blend_weights(points, (0.0, 0.0), moving_radius=2.0, fixed_radius=1.0, blend="linear")

    np.testing.assert_allclose(weights, [0.0, 0.5, 1.0])
