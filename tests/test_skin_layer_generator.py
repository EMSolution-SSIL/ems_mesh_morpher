from __future__ import annotations

import meshio
import numpy as np
import pytest

from ems_file_format_converter import read_mesh
from ems_mesh_morpher import (
    BoundaryRoleConfig,
    CoreMorphingConfig,
    SkinLayerConfig,
    SkinLayerQualityConfig,
    compute_layer_thicknesses,
    generate_skin_layer,
)
from ems_mesh_morpher.skin_layer import evaluate_3d_quality


_CUBE_POINTS = np.array(
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


def _cube_hex_mesh(*, with_symmetry_surface: bool = False) -> meshio.Mesh:
    cells: list[tuple[str, np.ndarray]] = [
        ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))
    ]
    properties = [np.array([100])]
    materials = [np.array([7])]
    element_ids = [np.array([501])]
    if with_symmetry_surface:
        cells.append(("quad", np.array([[0, 3, 7, 4]])))
        properties.append(np.array([210]))
        materials.append(np.array([1]))
        element_ids.append(np.array([601]))
    return meshio.Mesh(
        _CUBE_POINTS.copy(),
        cells,
        point_data={"id": np.arange(101, 109)},
        cell_data={
            "property_id": properties,
            "material_id": materials,
            "element_id": element_ids,
        },
        field_data={"conductor": np.array([100, 3])},
    )


def _cube_tetra_mesh() -> meshio.Mesh:
    tetrahedra = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]
    )
    return meshio.Mesh(
        _CUBE_POINTS.copy(),
        [("tetra", tetrahedra)],
        point_data={"id": np.arange(101, 109)},
        cell_data={
            "property_id": [np.full(6, 100)],
            "material_id": [np.full(6, 7)],
            "element_id": [np.arange(501, 507)],
        },
        field_data={"conductor": np.array([100, 3])},
    )


def _structured_cube_hex_mesh() -> meshio.Mesh:
    points = np.array(
        [
            [x, y, z]
            for z in (0.0, 0.5, 1.0)
            for y in (0.0, 0.5, 1.0)
            for x in (0.0, 0.5, 1.0)
        ]
    )
    lookup = {tuple(point): index for index, point in enumerate(points)}
    cells = []
    for z in (0.0, 0.5):
        for y in (0.0, 0.5):
            for x in (0.0, 0.5):
                cells.append(
                    [
                        lookup[(x, y, z)],
                        lookup[(x + 0.5, y, z)],
                        lookup[(x + 0.5, y + 0.5, z)],
                        lookup[(x, y + 0.5, z)],
                        lookup[(x, y, z + 0.5)],
                        lookup[(x + 0.5, y, z + 0.5)],
                        lookup[(x + 0.5, y + 0.5, z + 0.5)],
                        lookup[(x, y + 0.5, z + 0.5)],
                    ]
                )
    count = len(cells)
    return meshio.Mesh(
        points,
        [("hexahedron", np.asarray(cells))],
        cell_data={
            "property_id": [np.full(count, 100)],
            "material_id": [np.full(count, 7)],
            "element_id": [np.arange(501, 501 + count)],
        },
    )


def _config(**kwargs: object) -> SkinLayerConfig:
    values: dict[str, object] = {
        "thickness": 0.1,
        "volume_property_ids": (100,),
        "core_morphing": CoreMorphingConfig(method="none"),
        "skin_layer_property_id": 200,
        "inner_surface_property_id": 201,
    }
    values.update(kwargs)
    return SkinLayerConfig(**values)


def _cell_count(mesh: meshio.Mesh, cell_type: str) -> int:
    return sum(len(block.data) for block in mesh.cells if block.type == cell_type)


def test_layer_thicknesses_support_equal_and_geometric_division() -> None:
    assert compute_layer_thicknesses(0.12, 3, 1.0) == pytest.approx(
        (0.04, 0.04, 0.04)
    )
    assert compute_layer_thicknesses(0.14, 3, 2.0) == pytest.approx(
        (0.02, 0.04, 0.08)
    )


def test_skin_layer_supports_generic_idw_core_morphing() -> None:
    result = generate_skin_layer(
        _structured_cube_hex_mesh(),
        _config(core_morphing=CoreMorphingConfig(method="idw")),
    )

    assert result.report.core_morphing_method == "idw"
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0


def test_skin_layer_supports_generic_rbf_core_morphing() -> None:
    result = generate_skin_layer(
        _structured_cube_hex_mesh(),
        _config(core_morphing=CoreMorphingConfig(method="rbf")),
    )

    assert result.report.core_morphing_method == "rbf"
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0


@pytest.mark.parametrize(
    ("layer_count", "growth_ratio"),
    [(0, 1.0), (3, 0.0), (3, -1.0)],
)
def test_invalid_layer_division_is_rejected(
    layer_count: int,
    growth_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        SkinLayerConfig(
            thickness=0.1,
            layer_count=layer_count,
            growth_ratio=growth_ratio,
        )


def test_generate_three_equal_hex_layers() -> None:
    source = _cube_hex_mesh()
    result = generate_skin_layer(
        source,
        _config(layer_count=3, growth_ratio=1.0),
    )

    assert result.report.layer_count == 3
    assert result.report.layer_thicknesses == pytest.approx((0.1 / 3.0,) * 3)
    assert result.report.actual_layer_thickness_min == pytest.approx(
        (0.1 / 3.0,) * 3
    )
    assert result.report.new_node_count == 24
    assert result.report.layer_element_count == 18
    assert result.report.inner_surface_element_count == 6
    assert _cell_count(result.mesh, "hexahedron") == 19
    assert len(result.layer_node_maps) == 3
    for layer_index, node_map in enumerate(result.layer_node_maps, start=1):
        np.testing.assert_allclose(
            result.mesh.points[node_map[0]],
            np.full(3, 0.1 * layer_index / 3.0),
        )
    layer_block = result.topology.layer_block_indices[0]
    np.testing.assert_array_equal(
        result.mesh.cell_data["skin_layer_index"][layer_block],
        np.repeat([1, 2, 3], 6),
    )
    assert all(
        len(element_ids) == 3
        for element_ids in result.topology.layer_face_to_element_ids.values()
    )
    assert result.report.quality_after_core.total_signed_measure + result.report.quality_layer.total_signed_measure == pytest.approx(1.0)


def test_generate_three_geometric_tetra_layers() -> None:
    result = generate_skin_layer(
        _cube_tetra_mesh(),
        _config(layer_count=3, growth_ratio=2.0),
    )

    assert result.report.layer_thicknesses == pytest.approx(
        (0.1 / 7.0, 0.2 / 7.0, 0.4 / 7.0)
    )
    assert result.report.actual_layer_thickness_mean == pytest.approx(
        (0.1 / 7.0, 0.2 / 7.0, 0.4 / 7.0)
    )
    assert result.report.layer_element_count == 36
    assert _cell_count(result.mesh, "wedge") == 36
    assert result.report.quality_layer.inverted_count == 0
    node_positions = [
        result.mesh.points[node_map[0], 0] for node_map in result.layer_node_maps
    ]
    assert node_positions == pytest.approx((0.1 / 7.0, 0.3 / 7.0, 0.1))


def test_three_layers_split_symmetry_side_surfaces() -> None:
    source = _cube_hex_mesh(with_symmetry_surface=True)
    result = generate_skin_layer(
        source,
        _config(
            layer_count=3,
            boundary_roles=BoundaryRoleConfig(
                excluded_surface_property_ids=(210,)
            ),
        ),
    )

    assert result.report.layer_element_count == 15
    assert result.report.side_surface_element_count == 12
    side_block = result.topology.side_surface_block_indices[0]
    np.testing.assert_array_equal(
        result.mesh.cell_data["skin_layer_index"][side_block],
        np.tile([1, 2, 3], 4),
    )


def test_generate_hex_skin_layer_preserves_metadata_and_volume() -> None:
    source = _cube_hex_mesh()

    result = generate_skin_layer(source, _config())

    assert result.report.skin_face_count == 6
    assert result.report.new_node_count == 8
    assert result.report.layer_element_count == 6
    assert result.report.inner_surface_element_count == 6
    assert result.report.side_surface_element_count == 0
    assert result.report.minimum_core_measure_ratio == pytest.approx(0.8**3)
    assert result.report.actual_normal_thickness_min == pytest.approx(0.1)
    assert result.report.actual_normal_thickness_max == pytest.approx(0.1)
    assert _cell_count(result.mesh, "hexahedron") == 7
    assert _cell_count(result.mesh, "quad") == 6
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0
    assert result.report.quality_after_core.total_signed_measure + result.report.quality_layer.total_signed_measure == pytest.approx(1.0)
    np.testing.assert_allclose(result.mesh.points[:8], source.points)

    ids = np.concatenate(result.mesh.cell_data["element_id"])
    assert len(ids) == len(np.unique(ids))
    assert result.mesh.cell_data["element_id"][0][0] == 501
    assert result.mesh.cell_data["property_id"][0][0] == 100
    assert np.all(result.mesh.cell_data["property_id"][1] == 200)
    assert tuple(result.mesh.field_data["skin_layer"]) == (200, 3)
    assert tuple(result.mesh.field_data["skin_inner_surface"]) == (201, 2)


def test_generate_tetra_skin_layer_creates_wedges() -> None:
    result = generate_skin_layer(_cube_tetra_mesh(), _config())

    assert result.report.skin_face_count == 12
    assert result.report.layer_element_count == 12
    assert result.report.inner_surface_element_count == 12
    assert _cell_count(result.mesh, "tetra") == 6
    assert _cell_count(result.mesh, "wedge") == 12
    assert _cell_count(result.mesh, "triangle") == 12
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0
    assert result.report.quality_after_core.total_signed_measure + result.report.quality_layer.total_signed_measure == pytest.approx(1.0)


def test_excluded_symmetry_surface_is_not_layered_and_is_closed_by_sides() -> None:
    source = _cube_hex_mesh(with_symmetry_surface=True)
    config = _config(
        boundary_roles=BoundaryRoleConfig(excluded_surface_property_ids=(210,))
    )

    result = generate_skin_layer(source, config)

    assert result.report.skin_face_count == 5
    assert result.report.excluded_face_count == 1
    assert result.report.layer_element_count == 5
    assert result.report.side_surface_element_count == 4
    assert _cell_count(result.mesh, "hexahedron") == 6
    side_block = result.topology.side_surface_block_indices[0]
    assert np.all(result.mesh.cell_data["property_id"][side_block] == 210)
    excluded_nodes = result.mesh.cells[1].data[0]
    assert all(node >= len(source.points) for node in excluded_nodes)
    assert evaluate_3d_quality(result.mesh).inverted_count == 0


def test_laplace_core_morphing_is_used_by_the_complete_generator() -> None:
    source = _structured_cube_hex_mesh()
    config = _config(core_morphing=CoreMorphingConfig(method="laplace"))

    result = generate_skin_layer(source, config)

    center = np.flatnonzero(np.all(source.points == 0.5, axis=1))[0]
    assert result.report.core_morphing_method == "laplace"
    np.testing.assert_allclose(result.mesh.points[center], [0.5, 0.5, 0.5])
    assert result.report.quality_after_core.inverted_count == 0
    assert result.report.quality_layer.inverted_count == 0


def test_auto_core_morphing_falls_back_when_distance_quality_is_insufficient() -> None:
    source = _structured_cube_hex_mesh()
    config = _config(
        core_morphing=CoreMorphingConfig(method="auto", zone_depth=1.0),
        quality=SkinLayerQualityConfig(minimum_core_scaled_jacobian=0.99),
    )

    result = generate_skin_layer(source, config)

    assert result.report.core_morphing_method == "laplace"
    assert result.report.quality_after_core.minimum_scaled_jacobian == pytest.approx(1.0)
    assert len(result.report.warnings) == 1
    assert result.report.warnings[0].startswith("distance_blend:")


def test_automatic_property_ids_do_not_collide_with_legacy_defaults() -> None:
    source = meshio.Mesh(
        _CUBE_POINTS.copy(),
        [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
    )

    result = generate_skin_layer(
        source,
        SkinLayerConfig(
            thickness=0.1,
            core_morphing=CoreMorphingConfig(method="none"),
        ),
    )

    properties = set(np.concatenate(result.mesh.cell_data["property_id"]).tolist())
    assert properties == {1, 2, 3}
    assert tuple(result.mesh.field_data["skin_layer"]) == (2, 3)
    assert tuple(result.mesh.field_data["skin_inner_surface"]) == (3, 2)


@pytest.mark.parametrize(
    ("extension", "file_format"),
    [("atl", "atlas"), ("unv", "unv"), ("neu", "femap"), ("msh", "gmsh4")],
)
def test_generated_three_layer_hex_round_trips_through_supported_formats(
    tmp_path,
    extension: str,
    file_format: str,
) -> None:
    result = generate_skin_layer(
        _cube_hex_mesh(),
        _config(layer_count=3, growth_ratio=1.5),
    )
    output = tmp_path / f"skin_layer.{extension}"

    result.write(output, file_format=file_format)
    restored = read_mesh(output, file_format=file_format)

    assert _cell_count(restored, "hexahedron") == 19
    assert _cell_count(restored, "quad") == 6
    assert evaluate_3d_quality(restored).inverted_count == 0
    property_key = (
        "property_id"
        if "property_id" in restored.cell_data
        else "gmsh:physical"
    )
    element_key = (
        "element_id"
        if "element_id" in restored.cell_data
        else "gmsh:element_tags"
    )
    assert {100, 200, 201}.issubset(
        set(np.concatenate(restored.cell_data[property_key]).tolist())
    )
    element_ids = np.concatenate(restored.cell_data[element_key])
    assert len(element_ids) == len(np.unique(element_ids))
