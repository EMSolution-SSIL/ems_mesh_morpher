import meshio
import numpy as np

from ems_mesh_morpher.skin_layer.quality import (
    evaluate_3d_quality,
    evaluate_element_quality,
)


def test_unit_tetra_hex_and_wedge_have_positive_quality():
    tetra = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    hexahedron = np.array(
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
    wedge = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    for cell_type, coordinates in (
        ("tetra", tetra),
        ("hexahedron", hexahedron),
        ("wedge", wedge),
    ):
        quality = evaluate_element_quality(cell_type, coordinates)
        assert quality.signed_measure > 0.0
        assert quality.minimum_scaled_jacobian > 0.0


def test_inverted_tetra_is_reported():
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = meshio.Mesh(points=points, cells=[("tetra", np.array([[0, 1, 2, 3]]))])

    report = evaluate_3d_quality(mesh)

    assert report.element_count == 1
    assert report.inverted_count == 1
    assert report.minimum_scaled_jacobian < 0.0
