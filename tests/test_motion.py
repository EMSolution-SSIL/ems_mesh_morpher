import numpy as np

from ems_mesh_morpher.config import MotionConfig, PointDisplacementConfig
from ems_mesh_morpher.motion import motion_displacement_field


def test_rotation_displacement_field_rotates_about_center():
    points = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    motion = MotionConfig(type="rotation", center=(1.0, 1.0, 0.0), angle_degrees=90.0)

    displacement, assigned = motion_displacement_field(points, motion)

    assert assigned is None
    np.testing.assert_allclose(points + displacement, [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]], atol=1.0e-12)


def test_scaling_displacement_field_supports_axis_factors():
    points = np.array([[2.0, 3.0, 0.0]])
    motion = MotionConfig(type="scaling", center=(1.0, 1.0, 0.0), scale=(2.0, 0.5))

    displacement, _ = motion_displacement_field(points, motion)

    np.testing.assert_allclose(points + displacement, [[3.0, 2.0, 0.0]])


def test_prescribed_displacement_marks_assigned_points():
    points = np.zeros((3, 3))
    motion = MotionConfig(
        type="prescribed",
        point_displacements=(PointDisplacementConfig(1, (0.1, -0.2)),),
    )

    displacement, assigned = motion_displacement_field(points, motion)

    np.testing.assert_array_equal(assigned, [False, True, False])
    np.testing.assert_allclose(displacement[1], [0.1, -0.2, 0.0])
