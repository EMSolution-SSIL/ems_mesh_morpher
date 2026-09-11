import meshio
import numpy as np

from ems_mesh_morpher import apply_mesh_motion, config_from_dict
from ems_mesh_morpher.config import MorphConfig, MorphingConfig, QualityConfig, RegionConfig


def _point_mesh(x_coordinates):
    points = np.array([[x, 0.0, 0.0] for x in x_coordinates], dtype=float)
    cells = [("vertex", np.arange(len(points), dtype=int)[:, None])]
    return meshio.Mesh(points=points, cells=cells)


def test_distance_blend_translation_in_local_morphing_zone():
    mesh = _point_mesh([0.0, 0.25, 0.5, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0]},
                "deformable": {"point_indices": [1, 2]},
                "fixed": {"point_indices": [3]},
                "morphing_zone": {
                    "geometry": {"type": "box", "min": [0.0, -0.1], "max": [0.3, 0.1]}
                },
            },
            "motion": {"type": "translation", "displacement": [0.2, 0.0]},
            "morphing": {"type": "distance_blend", "blend": "linear"},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(result.mesh.points[:, 0], [0.2, 0.4, 0.5, 1.0])
    np.testing.assert_array_equal(result.classification.deformable, [False, True, False, False])


def test_rotation_without_propagation_moves_only_selected_nodes():
    mesh = meshio.Mesh(
        points=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]),
        cells=[("vertex", np.array([[0], [1], [2]]))],
    )
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0, 1]}},
            "motion": {"type": "rotation", "center": [0.0, 0.0], "angle_degrees": 90.0},
            "morphing": {"type": "none"},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(
        result.mesh.points,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        atol=1.0e-12,
    )


def test_prescribed_motion_infers_moving_nodes_and_propagates():
    mesh = _point_mesh([0.0, 0.5, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "deformable": {"point_indices": [1]},
                "fixed": {"point_indices": [2]},
            },
            "motion": {
                "type": "prescribed",
                "point_displacements": [{"point_index": 0, "displacement": [0.2, 0.0]}],
            },
            "morphing": {"type": "distance_blend", "blend": "linear"},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(result.mesh.points[:, 0], [0.2, 0.6, 1.0])
    np.testing.assert_array_equal(result.classification.moving, [True, False, False])


def test_protected_nodes_override_moving_and_deformable_selection():
    mesh = _point_mesh([0.0, 0.5, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0, 1]},
                "deformable": {"geometry": {"type": "all"}},
                "protected": {"point_indices": [1]},
            },
            "motion": {"type": "translation", "displacement": [0.2, 0.0]},
            "morphing": {"type": "none"},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(result.mesh.points[:, 0], [0.2, 0.5, 1.0])
    np.testing.assert_array_equal(result.classification.protected, [False, True, False])


def test_legacy_region_config_python_api_remains_supported():
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0]]),
        cells=[("vertex", np.array([[0]]))],
        cell_data={"property_id": [np.array([10])]},
    )
    config = MorphConfig(
        mode="mesh_motion",
        regions=RegionConfig(moving_property_ids=(10,)),
        morphing=MorphingConfig(type="none"),
        quality=QualityConfig(enabled=False),
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(result.mesh.points, mesh.points)


def test_external_field_allows_zero_displacement_at_selected_moving_node():
    mesh = _point_mesh([0.0, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0, 1]}},
            "morphing": {"type": "none"},
            "quality": {"enabled": False},
        }
    )
    displacement = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    result = apply_mesh_motion(mesh, config, prescribed_displacement=displacement)

    np.testing.assert_allclose(result.mesh.points[:, 0], [0.0, 1.1])


def test_idw_propagates_between_moving_and_fixed_nodes() -> None:
    mesh = _point_mesh([0.0, 0.25, 0.5, 0.75, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0]},
                "deformable": {"point_indices": [1, 2, 3]},
                "fixed": {"point_indices": [4]},
            },
            "motion": {"type": "translation", "displacement": [0.2, 0.0]},
            "morphing": {"type": "idw", "power": 2.0, "neighbors": None},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(
        result.displacement[:, 0],
        [0.2, 0.18, 0.1, 0.02, 0.0],
    )


def test_rbf_propagates_between_moving_and_fixed_nodes() -> None:
    mesh = _point_mesh([0.0, 0.25, 0.5, 0.75, 1.0])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0]},
                "deformable": {"point_indices": [1, 2, 3]},
                "fixed": {"point_indices": [4]},
            },
            "motion": {"type": "translation", "displacement": [0.2, 0.0]},
            "morphing": {"type": "rbf", "neighbors": None},
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(
        result.displacement[:, 0],
        [0.2, 0.15, 0.1, 0.05, 0.0],
        atol=1.0e-12,
    )


def test_weighted_laplace_follows_mesh_connectivity() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    mesh = meshio.Mesh(points, [("line", np.array([[0, 1], [1, 2]]))])
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": [0]},
                "deformable": {"point_indices": [1]},
                "fixed": {"point_indices": [2]},
            },
            "motion": {"type": "translation", "displacement": [0.3, 0.0]},
            "morphing": {
                "type": "weighted_laplace",
                "weighting": "inverse_distance",
                "distance_power": 1.0,
            },
            "quality": {"enabled": False},
        }
    )

    result = apply_mesh_motion(mesh, config)

    np.testing.assert_allclose(result.displacement[:, 0], [0.3, 0.2, 0.0])
