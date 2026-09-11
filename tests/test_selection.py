import pytest
import meshio
import numpy as np

from ems_mesh_morpher.config import RegionConfig, SelectionConfig
from ems_mesh_morpher.selection import classify_nodes, classify_nodes_from_regions


def test_classify_nodes_uses_moving_fixed_deformable_priority():
    mesh = meshio.Mesh(
        points=np.zeros((4, 3)),
        cells=[("triangle", np.array([[0, 1, 2], [1, 2, 3]]))],
        cell_data={"property_id": [np.array([10, 20])]},
    )

    classification = classify_nodes(
        mesh,
        moving_property_ids=(10,),
        deformable_property_ids=(20,),
        fixed_property_ids=(20,),
    )

    np.testing.assert_array_equal(classification.moving, [True, True, True, False])
    np.testing.assert_array_equal(classification.fixed, [False, False, False, True])
    np.testing.assert_array_equal(classification.deformable, [False, False, False, False])


def test_classify_nodes_accepts_gmsh_physical_data():
    mesh = meshio.Mesh(
        points=np.zeros((3, 3)),
        cells=[("triangle", np.array([[0, 1, 2]]))],
        cell_data={"gmsh:physical": [np.array([41])]},
    )

    classification = classify_nodes(
        mesh,
        moving_property_ids=(),
        deformable_property_ids=(41,),
        fixed_property_ids=(),
    )

    np.testing.assert_array_equal(classification.deformable, [True, True, True])


def test_conflict_policy_can_reject_overlapping_region_assignments():
    mesh = meshio.Mesh(
        points=np.zeros((2, 3)),
        cells=[("vertex", np.array([[0], [1]]))],
    )
    regions = RegionConfig(
        moving=SelectionConfig(point_indices=(0,)),
        protected=SelectionConfig(point_indices=(0,)),
        conflict_policy="error",
    )

    with pytest.raises(ValueError, match="conflicting region assignments"):
        classify_nodes_from_regions(mesh, regions)
