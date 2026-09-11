import meshio
import numpy as np

from ems_mesh_morpher.config import GeometricSelectionConfig, SelectionConfig
from ems_mesh_morpher.selection import geometric_node_mask, select_nodes


def test_box_and_annulus_selection():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    box = GeometricSelectionConfig(type="box", minimum=(0.5, -0.1), maximum=(1.6, 0.1))
    annulus = GeometricSelectionConfig(
        type="annulus", center=(0.0, 0.0), inner_radius=1.0, outer_radius=1.5
    )

    np.testing.assert_array_equal(geometric_node_mask(points, box), [False, True, True, False])
    np.testing.assert_array_equal(
        geometric_node_mask(points, annulus), [False, True, True, False]
    )


def test_finite_cylinder_selection_uses_axis_and_caps():
    points = np.array(
        [[0.5, 0.25, 0.0], [0.5, 2.0, 0.0], [-0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]
    )
    cylinder = GeometricSelectionConfig(
        type="cylinder",
        axis_start=(0.0, 0.0, 0.0),
        axis_end=(1.0, 0.0, 0.0),
        radius=0.5,
    )

    np.testing.assert_array_equal(
        geometric_node_mask(points, cylinder), [True, False, False, False]
    )


def test_selection_can_intersect_property_and_geometry():
    mesh = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cells=[("vertex", np.array([[0], [1], [2]]))],
        cell_data={"property_id": [np.array([10, 10, 20])]},
    )
    selection = SelectionConfig(
        property_ids=(10,),
        geometries=(
            GeometricSelectionConfig(type="box", minimum=(0.5, -1.0), maximum=(2.0, 1.0)),
        ),
        combine="intersection",
    )

    np.testing.assert_array_equal(select_nodes(mesh, selection), [False, True, False])
