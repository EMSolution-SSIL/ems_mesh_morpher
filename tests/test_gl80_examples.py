from pathlib import Path

import meshio
import numpy as np

from ems_mesh_morpher.config import load_config
from ems_mesh_morpher.gmsh4 import read_gmsh4_node_coordinates
from ems_mesh_morpher.io import read_mesh_auto, write_mesh_auto
from ems_mesh_morpher.motor_eccentricity_2d import apply_motor_eccentricity_2d


ROOT = Path(__file__).resolve().parents[1]
GL80 = ROOT / "data" / "GL80"
EXAMPLES = ROOT / "examples" / "GL80"


def _run_gl80_case(input_name: str, config_name: str, tmp_path: Path):
    mesh = read_mesh_auto(GL80 / input_name)
    config = load_config(EXAMPLES / config_name)

    result = apply_motor_eccentricity_2d(mesh, config)
    out = tmp_path / input_name
    write_mesh_auto(out, result.mesh, file_format=config.output.file_format, template_path=GL80 / input_name)
    reread = meshio.read(out)
    text = out.read_text(encoding="utf-8")

    assert len(reread.points) == len(mesh.points)
    assert len(reread.field_data) == len(mesh.field_data)
    assert "gmsh:physical" in reread.cell_data
    assert text.startswith("$MeshFormat\n4.1 0 8\n")
    assert "$PhysicalNames" in text
    assert "$Entities" in text
    assert "\x00" not in text
    np.testing.assert_allclose(read_gmsh4_node_coordinates(out), result.mesh.points)
    assert result.quality_after is not None
    assert result.quality_before is not None
    assert result.quality_after.inverted_count == result.quality_before.inverted_count
    assert result.orientation_flip_count == 0
    return result


def test_gl80_static_stator_eccentricity_smoke(tmp_path: Path):
    result = _run_gl80_case("stator.msh", "static_stator_eccentricity_0p05mm.json", tmp_path)

    assert result.classification.counts["moving"] > 0
    assert result.classification.counts["deformable"] > 0


def test_gl80_dynamic_rotor_eccentricity_smoke(tmp_path: Path):
    result = _run_gl80_case("rotor.msh", "dynamic_rotor_eccentricity_0p05mm.json", tmp_path)

    assert result.classification.counts["moving"] > 0
    assert result.classification.counts["deformable"] > 0
