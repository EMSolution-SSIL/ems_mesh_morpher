import numpy as np

from ems_mesh_morpher.morph import distance_blend_weights, nearest_reference


def test_distance_blend_uses_nearest_moving_and_fixed_nodes():
    points = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [1.0, 0.0]])
    moving = np.array([True, False, False, False])
    deformable = np.array([False, True, True, False])
    fixed = np.array([False, False, False, True])

    weights = distance_blend_weights(
        points, deformable, moving, fixed, blend="linear", chunk_size=2
    )

    np.testing.assert_allclose(weights, [0.0, 0.75, 0.5, 0.0])


def test_nearest_reference_is_chunk_independent():
    query = np.array([[0.1, 0.0], [1.8, 0.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    distances, indices = nearest_reference(query, reference, chunk_size=1)

    np.testing.assert_allclose(distances, [0.1, 0.2])
    np.testing.assert_array_equal(indices, [0, 2])
