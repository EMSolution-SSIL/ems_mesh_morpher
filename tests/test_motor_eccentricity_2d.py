import meshio
import numpy as np

from ems_mesh_morpher.config import config_from_dict
from ems_mesh_morpher.motor_eccentricity_2d import apply_motor_eccentricity_2d


def test_motor_eccentricity_moves_and_blends_selected_nodes():
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        cells=[
            ("vertex", np.array([[0], [1], [2]])),
        ],
        cell_data={"property_id": [np.array([10, 20, 30])]},
    )
    config = config_from_dict(
        {
            "geometry": {"moving_radius": 1.0, "fixed_radius": 2.0},
            "regions": {
                "moving_property_ids": [10],
                "deformable_property_ids": [20],
                "fixed_property_ids": [30],
            },
            "motion": {"displacement": [0.1, 0.0]},
            "morphing": {"blend": "linear"},
        }
    )

    result = apply_motor_eccentricity_2d(mesh, config)

    np.testing.assert_allclose(result.mesh.points[:, 0], [1.1, 1.55, 2.0])
    assert result.classification.counts == {"moving": 1, "fixed": 1, "deformable": 1}
    assert result.orientation_flip_count == 0
