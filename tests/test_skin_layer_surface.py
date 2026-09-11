import meshio
import numpy as np

from ems_mesh_morpher.skin_layer import (
    BoundaryRole,
    BoundaryRoleConfig,
    BoxSelector,
    PlaneConstraint,
    PlaneSelector,
    classify_boundary_faces,
    compute_inward_offset,
    detect_surface_features,
    extract_boundary_surface,
    validate_surface_topology,
)


def _cube_points(x0: float = 0.0, x1: float = 1.0) -> np.ndarray:
    return np.array(
        [
            [x0, 0.0, 0.0],
            [x1, 0.0, 0.0],
            [x1, 1.0, 0.0],
            [x0, 1.0, 0.0],
            [x0, 0.0, 1.0],
            [x1, 0.0, 1.0],
            [x1, 1.0, 1.0],
            [x0, 1.0, 1.0],
        ]
    )


def _cube_mesh(with_symmetry_surface: bool = False) -> meshio.Mesh:
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    properties = [np.array([100])]
    if with_symmetry_surface:
        cells.append(("quad", np.array([[0, 3, 7, 4]])))
        properties.append(np.array([210]))
    return meshio.Mesh(
        points=_cube_points(),
        cells=cells,
        cell_data={"property_id": properties},
    )


def test_extracts_oriented_closed_tetra_surface():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        cell_data={"property_id": [np.array([11])]},
    )

    surface = extract_boundary_surface(mesh, [11])
    validate_surface_topology(surface)

    assert surface.triangle_count == 4
    assert surface.quad_count == 0
    assert surface.selected_volume_elements == 1
    for face in surface.faces:
        assert np.dot(face.normal, face.centroid - face.owner_centroid) > 0.0


def test_tetrahedral_cube_keeps_planar_faces_and_rectangular_corners():
    points = _cube_points()
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
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", tetrahedra)],
        cell_data={"property_id": [np.full(6, 100)]},
    )

    surface = extract_boundary_surface(mesh, [100])
    validate_surface_topology(surface)
    features = detect_surface_features(mesh, surface)
    offset = compute_inward_offset(mesh, surface, 0.1, features=features)

    assert surface.triangle_count == 12
    assert len(features.sharp_edges) == 12
    assert len(features.corner_vertices) == 8
    np.testing.assert_allclose(
        offset.target_points,
        np.where(points == 0.0, 0.1, 0.9),
        atol=1.0e-12,
    )


def test_extracts_hex_quads_and_matches_existing_surface_property():
    surface = extract_boundary_surface(_cube_mesh(with_symmetry_surface=True), [100])
    validate_surface_topology(surface, quad_planarity_tolerance=1.0e-12)

    assert surface.triangle_count == 0
    assert surface.quad_count == 6
    matched = [face for face in surface.faces if face.surface_property_id == 210]
    assert len(matched) == 1
    assert set(matched[0].node_indices) == {0, 3, 4, 7}


def test_cube_features_and_corner_offsets_preserve_rectangular_shape():
    mesh = _cube_mesh()
    surface = extract_boundary_surface(mesh, [100])
    features = detect_surface_features(mesh, surface, feature_angle_degrees=30.0)
    offset = compute_inward_offset(mesh, surface, 0.1, features=features)

    assert len(features.sharp_edges) == 12
    assert len(features.corner_vertices) == 8
    np.testing.assert_allclose(
        np.sort(offset.target_points, axis=0),
        np.sort(np.where(mesh.points == 0.0, 0.1, 0.9), axis=0),
        atol=1.0e-12,
    )
    assert set(offset.node_methods.values()) == {"offset_plane_corner"}
    assert np.isclose(max(offset.miter_ratios.values()), np.sqrt(3.0))


def test_symmetry_face_is_excluded_and_nodes_stay_on_plane():
    mesh = _cube_mesh(with_symmetry_surface=True)
    surface = extract_boundary_surface(mesh, [100])
    symmetry = PlaneConstraint(normal=(1.0, 0.0, 0.0), offset=0.0, name="symmetry_x")
    classification = classify_boundary_faces(
        mesh,
        surface,
        BoundaryRoleConfig(
            excluded_surface_property_ids=(210,),
            excluded_planes=(PlaneSelector(symmetry),),
        ),
    )
    features = detect_surface_features(mesh, surface)
    offset = compute_inward_offset(
        mesh,
        surface,
        0.1,
        classification=classification,
        features=features,
    )

    assert len(classification.excluded_face_indices) == 1
    excluded = surface.faces[classification.excluded_face_indices[0]]
    assert classification.roles[classification.excluded_face_indices[0]] == BoundaryRole.EXCLUDED
    assert all(offset.target_points[node, 0] == 0.0 for node in excluded.node_indices)
    assert all(node in classification.node_plane_constraints for node in excluded.node_indices)


def test_box_selector_can_protect_a_boundary_face():
    mesh = _cube_mesh()
    surface = extract_boundary_surface(mesh, [100])
    classification = classify_boundary_faces(
        mesh,
        surface,
        BoundaryRoleConfig(
            protected_boxes=(BoxSelector((0.999, 0.0, 0.0), (1.001, 1.0, 1.0)),)
        ),
    )

    assert len(classification.protected_face_indices) == 1
    protected = surface.faces[classification.protected_face_indices[0]]
    assert set(protected.node_indices) == {1, 2, 5, 6}
    assert set(protected.node_indices) <= classification.fixed_nodes


def test_half_model_symmetry_offset_matches_full_model_center_section():
    full_points = np.array(
        [
            [x, y, z]
            for z in (0.0, 1.0)
            for y in (0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    lookup = {tuple(point): index for index, point in enumerate(full_points)}

    def cell(x0: float, x1: float) -> list[int]:
        return [
            lookup[(x0, 0.0, 0.0)],
            lookup[(x1, 0.0, 0.0)],
            lookup[(x1, 1.0, 0.0)],
            lookup[(x0, 1.0, 0.0)],
            lookup[(x0, 0.0, 1.0)],
            lookup[(x1, 0.0, 1.0)],
            lookup[(x1, 1.0, 1.0)],
            lookup[(x0, 1.0, 1.0)],
        ]

    full = meshio.Mesh(
        points=full_points,
        cells=[("hexahedron", np.array([cell(-1.0, 0.0), cell(0.0, 1.0)]))],
        cell_data={"property_id": [np.array([100, 100])]},
    )
    full_surface = extract_boundary_surface(full, [100])
    full_offset = compute_inward_offset(full, full_surface, 0.1)

    half = _cube_mesh(with_symmetry_surface=True)
    half_surface = extract_boundary_surface(half, [100])
    symmetry = PlaneConstraint((1.0, 0.0, 0.0), 0.0)
    half_roles = classify_boundary_faces(
        half,
        half_surface,
        BoundaryRoleConfig(excluded_planes=(PlaneSelector(symmetry),)),
    )
    half_offset = compute_inward_offset(
        half,
        half_surface,
        0.1,
        classification=half_roles,
    )

    for half_index, point in enumerate(half.points):
        full_index = lookup[tuple(point)]
        np.testing.assert_allclose(
            half_offset.target_points[half_index],
            full_offset.target_points[full_index],
            atol=1.0e-12,
        )


def _l_shape_hex_mesh() -> meshio.Mesh:
    coordinates = [
        (x, y, z)
        for z in (0.0, 1.0)
        for y in (0.0, 1.0, 2.0)
        for x in (0.0, 1.0, 2.0)
    ]
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}

    def cell(x: int, y: int) -> list[int]:
        return [
            lookup[(x, y, 0.0)],
            lookup[(x + 1, y, 0.0)],
            lookup[(x + 1, y + 1, 0.0)],
            lookup[(x, y + 1, 0.0)],
            lookup[(x, y, 1.0)],
            lookup[(x + 1, y, 1.0)],
            lookup[(x + 1, y + 1, 1.0)],
            lookup[(x, y + 1, 1.0)],
        ]

    cells = np.array([cell(0, 0), cell(1, 0), cell(0, 1)])
    return meshio.Mesh(
        points=np.asarray(coordinates),
        cells=[("hexahedron", cells)],
        cell_data={"property_id": [np.array([100, 100, 100])]},
    )


def test_concave_l_shape_is_detected_and_rejected_by_default():
    mesh = _l_shape_hex_mesh()
    surface = extract_boundary_surface(mesh, [100])
    features = detect_surface_features(mesh, surface)

    assert features.concave_edges
    with np.testing.assert_raises_regex(ValueError, "concave sharp feature"):
        compute_inward_offset(mesh, surface, 0.1, features=features)
