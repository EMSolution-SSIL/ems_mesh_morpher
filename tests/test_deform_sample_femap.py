from pathlib import Path

import numpy as np
import pytest

from ems_mesh_morpher.io import read_mesh_auto


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Deform_sample"


def test_deform_sample_femap_section_403_coordinates_and_topology() -> None:
    if not (DATA_DIR / "pre_geom2D.neu").exists():
        pytest.skip("Optional pyemsol DEFORM sample data is not distributed")
    mesh = read_mesh_auto(DATA_DIR / "pre_geom2D.neu")

    assert len(mesh.points) == 354
    np.testing.assert_allclose(np.min(mesh.points, axis=0), [-1.2, -1.2, 0.0])
    np.testing.assert_allclose(np.max(mesh.points, axis=0), [1.2, 1.2, 0.0])
    assert [(block.type, len(block.data)) for block in mesh.cells] == [
        ("quad", 336),
        ("triangle", 6),
    ]
    assert set(mesh.point_data) >= {"id"}
    assert set(mesh.cell_data) >= {"property_id", "element_id"}
    property_counts: dict[int, int] = {}
    for values in mesh.cell_data["property_id"]:
        for value in values:
            property_id = int(value)
            property_counts[property_id] = property_counts.get(property_id, 0) + 1
    assert property_counts == {1: 100, 2: 154, 3: 80, 11: 8}
