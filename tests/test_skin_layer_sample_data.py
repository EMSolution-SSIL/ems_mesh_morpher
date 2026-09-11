from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ems_file_format_converter import read_mesh
from ems_mesh_morpher import (
    BoundaryRoleConfig,
    CoreMorphingConfig,
    PlaneConstraint,
    PlaneSelector,
    SkinLayerConfig,
    generate_skin_layer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "skin_layer_sample"


def _domain_boundary_roles() -> BoundaryRoleConfig:
    return BoundaryRoleConfig(
        excluded_planes=tuple(
            PlaneSelector(
                PlaneConstraint(normal=normal, offset=0.0, tolerance=1.0e-10)
            )
            for normal in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    )


@pytest.mark.parametrize(
    (
        "relative_path",
        "file_format",
        "volume_property_id",
        "thickness",
        "expected_layer_type",
        "expected_skin_faces",
        "expected_excluded_faces",
        "expected_new_nodes",
        "expected_side_faces",
    ),
    [
        (
            "BlockConductorModel/post_geom_hexa.neu",
            "femap",
            1,
            0.01,
            "hexahedron",
            125,
            125,
            146,
            40,
        ),
        (
            "BlockConductorModel/post_geom_tetra.neu",
            "femap",
            1,
            0.01,
            "wedge",
            250,
            250,
            146,
            40,
        ),
        (
            "BlockConductorCoilModel/post_geom.atl",
            "atlas",
            1,
            0.0025,
            "hexahedron",
            125,
            125,
            146,
            40,
        ),
        (
            "BlockConductorCoilModel/post_geom.atl",
            "atlas",
            3,
            0.002,
            "hexahedron",
            195,
            75,
            224,
            56,
        ),
    ],
)
def test_real_sample_generates_and_round_trips(
    tmp_path: Path,
    relative_path: str,
    file_format: str,
    volume_property_id: int,
    thickness: float,
    expected_layer_type: str,
    expected_skin_faces: int,
    expected_excluded_faces: int,
    expected_new_nodes: int,
    expected_side_faces: int,
) -> None:
    source = DATA_ROOT / relative_path
    mesh = read_mesh(source, file_format=file_format)
    result = generate_skin_layer(
        mesh,
        SkinLayerConfig(
            thickness=thickness,
            volume_property_ids=(volume_property_id,),
            layer_count=3,
            growth_ratio=1.0,
            boundary_roles=_domain_boundary_roles(),
            core_morphing=CoreMorphingConfig(method="auto"),
            skin_layer_property_id=10,
            inner_surface_property_id=11,
            side_surface_property_id=12,
        ),
    )

    report = result.report
    assert report.skin_face_count == expected_skin_faces
    assert report.excluded_face_count == expected_excluded_faces
    assert report.layer_count == 3
    assert report.layer_thicknesses == pytest.approx((thickness / 3.0,) * 3)
    assert report.new_node_count == expected_new_nodes * 3
    assert report.layer_element_count == expected_skin_faces * 3
    assert report.side_surface_element_count == expected_side_faces * 3
    assert report.quality_after_core.inverted_count == 0
    assert report.quality_layer.inverted_count == 0
    assert report.actual_normal_thickness_min == pytest.approx(thickness, rel=0.02)

    output = tmp_path / ("skin.neu" if file_format == "femap" else "skin.atl")
    result.write(output, file_format=file_format)
    restored = read_mesh(output, file_format=file_format)
    layer_count = sum(
        int(np.count_nonzero(values == 10))
        for block, values in zip(
            restored.cells,
            restored.cell_data["property_id"],
            strict=True,
        )
        if block.type == expected_layer_type
    )
    assert layer_count == expected_skin_faces * 3
    element_ids = np.concatenate(restored.cell_data["element_id"])
    assert len(element_ids) == len(np.unique(element_ids))


def test_prism_reference_is_rejected_as_an_unsupported_input() -> None:
    mesh = read_mesh(DATA_ROOT / "BlockConductorModel" / "post_geom_prism.neu")

    with pytest.raises(
        ValueError,
        match="no tetrahedral or hexahedral volume elements were selected",
    ):
        generate_skin_layer(
            mesh,
            SkinLayerConfig(thickness=0.01, volume_property_ids=(1,)),
        )
