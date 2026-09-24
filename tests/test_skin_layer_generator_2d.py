from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_mesh_morpher.skin_layer import (
    PlaneConstraint,
    PlaneSelector,
    SkinLayer2DConfig,
    extract_boundary_edges_2d,
    generate_skin_layer_2d,
)


def _shared_quad_and_hex_mesh() -> meshio.Mesh:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    return meshio.Mesh(
        points,
        [
            ("quad", np.array([[0, 1, 2, 3]], dtype=int)),
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)),
        ],
        point_data={"id": np.arange(1, 9, dtype=int)},
        cell_data={
            "property_id": [np.array([1]), np.array([14])],
            "element_id": [np.array([101]), np.array([201])],
        },
    )


def _triangulated_square_mesh() -> meshio.Mesh:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    triangles = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int
    )
    return meshio.Mesh(
        points,
        [("triangle", triangles)],
        point_data={"id": np.arange(1, 6, dtype=int)},
        cell_data={
            "property_id": [np.full(4, 7, dtype=int)],
            "element_id": [np.arange(1, 5, dtype=int)],
        },
    )


def _quad_square_mesh() -> meshio.Mesh:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return meshio.Mesh(
        points,
        [("quad", np.array([[0, 1, 2, 3]], dtype=int))],
        point_data={"id": np.arange(1, 5, dtype=int)},
        cell_data={
            "property_id": [np.array([1])],
            "element_id": [np.array([101])],
        },
    )


def test_2d_generator_rejects_mesh_with_volume_cells() -> None:
    source = _shared_quad_and_hex_mesh()
    with pytest.raises(ValueError, match="requires a 2D-only mesh"):
        generate_skin_layer_2d(
            source,
            SkinLayer2DConfig(thickness=0.1, region_property_ids=(1,)),
        )


def test_quad_skin_layer_generates_three_layers() -> None:
    source = _quad_square_mesh()
    result = generate_skin_layer_2d(
        source,
        SkinLayer2DConfig(
            thickness=0.1,
            region_property_ids=(1,),
            layer_count=3,
            growth_ratio=1.0,
            skin_layer_property_id=101,
        ),
    )

    assert result.report.boundary_edge_count == 4
    assert result.report.layer_element_count == 12
    assert result.report.layer_thicknesses == pytest.approx((1 / 30, 1 / 30, 1 / 30))
    assert result.report.core_orientation_flip_count == 0
    assert result.report.layer_inverted_count == 0
    assert result.report.minimum_core_measure_ratio == pytest.approx(0.64)
    assert np.allclose(result.mesh.points[: len(source.points)], source.points)

    layer_properties = result.mesh.cell_data["property_id"][result.layer_block_index]
    assert np.all(layer_properties == 101)
    assert np.array_equal(
        result.mesh.cell_data["skin_layer_index"][result.layer_block_index],
        np.repeat(np.arange(1, 4), 4),
    )


def test_triangle_skin_layer_uses_laplace_for_interior_node() -> None:
    source = _triangulated_square_mesh()
    result = generate_skin_layer_2d(
        source,
        SkinLayer2DConfig(
            thickness=0.1,
            region_property_ids=(7,),
            layer_count=2,
            growth_ratio=2.0,
            skin_layer_property_id=107,
        ),
    )

    assert result.report.selected_element_count == 4
    assert result.report.boundary_edge_count == 4
    assert result.report.layer_element_count == 8
    assert result.report.layer_thicknesses == pytest.approx((1 / 30, 2 / 30))
    assert result.report.core_morphing_method == "weighted_laplace"
    assert result.report.core_orientation_flip_count == 0
    assert result.report.core_degenerate_count == 0
    assert result.report.layer_inverted_count == 0
    assert result.report.layer_degenerate_count == 0
    copied_center = result.core_node_map[4]
    assert np.allclose(result.mesh.points[copied_center], source.points[4])


def test_symmetry_edges_are_excluded_and_junctions_slide_on_planes() -> None:
    source = _quad_square_mesh()
    result = generate_skin_layer_2d(
        source,
        SkinLayer2DConfig(
            thickness=0.1,
            region_property_ids=(1,),
            layer_count=2,
            excluded_planes=(
                PlaneSelector(
                    PlaneConstraint(normal=(1.0, 0.0, 0.0), offset=0.0)
                ),
                PlaneSelector(
                    PlaneConstraint(normal=(0.0, 1.0, 0.0), offset=0.0)
                ),
            ),
            skin_layer_property_id=101,
        ),
    )

    assert result.report.boundary_edge_count == 4
    assert result.report.skin_edge_count == 2
    assert result.report.excluded_edge_count == 2
    assert result.report.constrained_node_count == 3
    assert result.report.layer_element_count == 4
    assert set(result.layer_node_maps[0]) == {1, 2, 3}
    assert np.allclose(result.mesh.points[result.core_node_map[0]], [0.0, 0.0, 0.0])
    assert result.mesh.points[result.core_node_map[3], 0] == pytest.approx(0.0)
    assert result.mesh.points[result.core_node_map[1], 1] == pytest.approx(0.0)
    assert np.allclose(result.mesh.points[result.core_node_map[2]], [0.9, 0.9, 0.0])
    assert result.report.minimum_core_measure_ratio == pytest.approx(0.81)
    assert result.report.core_orientation_flip_count == 0
    assert result.report.layer_inverted_count == 0


def test_2d_generator_rejects_when_every_boundary_edge_is_excluded() -> None:
    source = _quad_square_mesh()
    with pytest.raises(ValueError, match="all 2D boundary edges are excluded"):
        generate_skin_layer_2d(
            source,
            SkinLayer2DConfig(
                thickness=0.1,
                region_property_ids=(1,),
                excluded_planes=(
                    PlaneSelector(
                        PlaneConstraint(normal=(0.0, 0.0, 1.0), offset=0.0),
                        node_constraint="none",
                    ),
                ),
            ),
        )


def test_extract_boundary_edges_requires_planar_triangle_or_quad_region() -> None:
    source = _quad_square_mesh()
    with pytest.raises(ValueError, match="triangle or quadrilateral"):
        extract_boundary_edges_2d(source, (14,))
