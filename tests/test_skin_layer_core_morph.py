import meshio
import numpy as np

from ems_mesh_morpher.skin_layer import (
    compute_inward_offset,
    extract_boundary_surface,
)
from ems_mesh_morpher.skin_layer.core_morph import (
    CoreMorphingConfig,
    compute_core_displacement,
)


def _structured_cube() -> meshio.Mesh:
    points = np.array(
        [[x, y, z] for z in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)]
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
    return meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.asarray(cells))],
        cell_data={"property_id": [np.full(len(cells), 100)]},
    )


def test_laplace_core_morph_keeps_symmetric_center_fixed():
    mesh = _structured_cube()
    surface = extract_boundary_surface(mesh, [100])
    offset = compute_inward_offset(mesh, surface, 0.1)

    result = compute_core_displacement(
        mesh,
        surface,
        offset,
        [100],
        CoreMorphingConfig(method="laplace"),
    )

    center = np.flatnonzero(np.all(mesh.points == 0.5, axis=1))[0]
    assert result.movable_nodes[center]
    np.testing.assert_allclose(result.displacements[center], 0.0, atol=1.0e-12)
