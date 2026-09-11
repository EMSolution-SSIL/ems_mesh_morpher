from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ems_file_format_converter import read_mesh
from ems_mesh_morpher import (
    BoundaryRoleConfig,
    CoreMorphingConfig,
    PlaneConstraint,
    PlaneSelector,
    SkinLayerCollisionConfig,
    SkinLayerConfig,
    generate_skin_layer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "skin_layer_sample"


@dataclass(frozen=True)
class Sample:
    name: str
    input_path: Path
    output_name: str
    file_format: str
    volume_property_id: int
    thickness: float
    layer_property_id: int = 10
    inner_surface_property_id: int = 11
    side_surface_property_id: int = 12


SAMPLES = (
    Sample(
        "block_hexa",
        DATA_ROOT / "BlockConductorModel" / "post_geom_hexa.neu",
        "block_conductor_hexa_skin.neu",
        "femap",
        1,
        0.01,
    ),
    Sample(
        "block_tetra",
        DATA_ROOT / "BlockConductorModel" / "post_geom_tetra.neu",
        "block_conductor_tetra_skin.neu",
        "femap",
        1,
        0.01,
    ),
    Sample(
        "atlas_block_conductor",
        DATA_ROOT / "BlockConductorCoilModel" / "post_geom.atl",
        "atlas_block_conductor_skin.atl",
        "atlas",
        1,
        0.0025,
    ),
    Sample(
        "atlas_curved_coil",
        DATA_ROOT / "BlockConductorCoilModel" / "post_geom.atl",
        "atlas_curved_coil_skin.atl",
        "atlas",
        3,
        0.002,
        20,
        21,
        22,
    ),
)


def domain_boundary_roles() -> BoundaryRoleConfig:
    planes = tuple(
        PlaneSelector(
            PlaneConstraint(normal=normal, offset=0.0, tolerance=1.0e-10)
        )
        for normal in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    return BoundaryRoleConfig(excluded_planes=planes)


def generate_sample(
    sample: Sample,
    output_dir: Path,
    *,
    thickness: float | None,
    layer_count: int,
    growth_ratio: float,
    collision_policy: str,
) -> dict[str, object]:
    effective_thickness = sample.thickness if thickness is None else thickness
    mesh = read_mesh(sample.input_path)
    result = generate_skin_layer(
        mesh,
        SkinLayerConfig(
            thickness=effective_thickness,
            volume_property_ids=(sample.volume_property_id,),
            layer_count=layer_count,
            growth_ratio=growth_ratio,
            collision=SkinLayerCollisionConfig(policy=collision_policy),
            boundary_roles=domain_boundary_roles(),
            core_morphing=CoreMorphingConfig(method="auto"),
            skin_layer_property_id=sample.layer_property_id,
            inner_surface_property_id=sample.inner_surface_property_id,
            side_surface_property_id=sample.side_surface_property_id,
        ),
    )
    output_path = output_dir / sample.output_name
    result.write(output_path, file_format=sample.file_format)
    report = result.report.to_dict()
    report.update(
        {
            "name": sample.name,
            "input": str(sample.input_path),
            "output": str(output_path),
            "sample_default_thickness": sample.thickness,
            "volume_property_id": sample.volume_property_id,
            "skin_layer_property_id": sample.layer_property_id,
            "inner_surface_property_id": sample.inner_surface_property_id,
            "side_surface_property_id": sample.side_surface_property_id,
        }
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the skin-layer sample meshes")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=PROJECT_ROOT / "generated" / "skin_layer_sample",
    )
    parser.add_argument(
        "--sample",
        action="append",
        choices=tuple(sample.name for sample in SAMPLES),
        help="sample to generate; repeat to select multiple samples (default: all)",
    )
    parser.add_argument(
        "--thickness",
        type=_positive_float,
        help="total skin-layer thickness; requires exactly one --sample",
    )
    parser.add_argument(
        "--layer_count",
        "--layer-count",
        dest="layer_count",
        type=_positive_int,
        default=3,
        help="number of layers (default: 3)",
    )
    parser.add_argument(
        "--growth_ratio",
        "--growth-ratio",
        dest="growth_ratio",
        type=_positive_float,
        default=1.0,
        help="inner/outer adjacent-layer thickness ratio (default: 1.0)",
    )
    parser.add_argument(
        "--collision_policy",
        "--collision-policy",
        dest="collision_policy",
        choices=("fail", "reduce", "allow"),
        default="fail",
        help="local thickness risk policy (default: fail)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_samples = tuple(
        sample
        for sample in SAMPLES
        if args.sample is None or sample.name in args.sample
    )
    if args.thickness is not None and len(selected_samples) != 1:
        parser.error("--thickness requires exactly one --sample")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = [
        generate_sample(
            sample,
            args.output_dir,
            thickness=args.thickness,
            layer_count=args.layer_count,
            growth_ratio=args.growth_ratio,
            collision_policy=args.collision_policy,
        )
        for sample in selected_samples
    ]
    report_path = args.output_dir / "reports.json"
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for report in reports:
        print(
            f"{report['name']}: {report['layer_count']} layers, "
            f"{report['layer_element_count']} layer elements -> "
            f"{report['output']}"
        )
    print(f"reports: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
