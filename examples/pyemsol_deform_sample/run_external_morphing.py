from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ems_mesh_morpher import (
    MorphingQualityConfig,
    PyEMSolConstraintConfig,
    PyEMSolMorphingAdapter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pyemsol DEFORM sample through ems_mesh_morpher."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing input.json and the EMSolution mesh files.",
    )
    parser.add_argument(
        "--method",
        choices=("linear", "idw", "rbf", "weighted_laplace"),
        default="weighted_laplace",
    )
    parser.add_argument(
        "--input_file",
        default="input.json",
        help="Input JSON filename inside model_dir (for example input_xy.json).",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--delta_time", type=float, default=0.001)
    parser.add_argument("--no_quality_check", action="store_true")
    parser.add_argument("--no_anchor_inference", action="store_true")
    return parser


def _time_function_value(time_function: dict[str, object], time: float) -> float:
    amplitude = float(time_function["AMPLITUDE"])
    period = float(time_function["TCYCLE"])
    phase = math.radians(float(time_function.get("PHASE", 0.0)))
    return amplitude * math.cos(2.0 * math.pi * time / period + phase)


def main() -> None:
    args = _parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.delta_time <= 0.0:
        raise ValueError("delta_time must be positive")

    model_dir = args.model_dir.resolve()
    with (model_dir / args.input_file).open(encoding="utf-8") as stream:
        input_json = json.load(stream)

    time_functions = {
        int(item["TIME_ID"]): item for item in input_json["18_Time_Function"]
    }
    deform_motions = {
        int(item["motion_data"]["MOTION_ID"]): item
        for item in input_json["19_Motion"]
        if item["type"] == "DEFORM_MOTION"
    }

    try:
        import pyemsol
    except ImportError as exc:
        raise RuntimeError(
            "pyemsol is required to run this example; use the ems-py311 environment"
        ) from exc

    maximum_coordinate_error = 0.0
    minimum_measure_ratio = 1.0
    with pyemsol.CoupledSession(input_json, str(model_dir)) as session:
        description = session.describe_deformation()
        topology = session.describe_deformation_mesh()
        adapter = PyEMSolMorphingAdapter.prepare(
            description,
            topology=topology,
            method=args.method,
            quality_config=MorphingQualityConfig(enabled=not args.no_quality_check),
            constraint_config=PyEMSolConstraintConfig(
                infer_zero_displacement_anchors=not args.no_anchor_inference
            ),
        )

        description_regions = description["regions"]

        for step_no in range(1, args.steps + 1):
            end_time = step_no * args.delta_time
            positions: dict[int, float] = {}
            translations: dict[int, list[float]] = {}
            for region in description_regions:
                deform_id = int(region["deform_id"])
                motion_id = int(region["motion_id"])
                motion_data = deform_motions[motion_id]["motion_data"]
                if int(motion_data.get("COORD_ID", 0)) != 0:
                    raise ValueError("this sample runner supports only global COORD_ID=0")
                if int(motion_data.get("PHI_TIME_ID", 0)) != 0:
                    raise ValueError(
                        "rotational time functions require explicit rigid_targets"
                    )
                time_ids = [
                    int(motion_data.get(name, 0))
                    for name in ("X_TIME_ID", "Y_TIME_ID", "Z_TIME_ID")
                ]
                vector = [
                    0.0
                    if time_id == 0
                    else _time_function_value(time_functions[time_id], end_time)
                    for time_id in time_ids
                ]
                if args.method == "linear":
                    active = [value for value, time_id in zip(vector, time_ids) if time_id]
                    if len(active) != 1:
                        raise ValueError(
                            "linear mode supports one active translation component; "
                            "use idw, rbf, or weighted_laplace for independent XYZ motion"
                        )
                    positions[deform_id] = active[0]
                else:
                    translations[deform_id] = vector
            morphing = adapter.evaluate_result(
                positions if args.method == "linear" else None,
                rigid_translations=(
                    None if args.method == "linear" else translations
                ),
                mesh_revision=step_no,
            )
            result = session.step(
                time=end_time,
                delta_time=args.delta_time,
                deformation=morphing.payload,
            )
            if not result["success"]:
                raise RuntimeError(f"EMSolution failed at step {step_no}")

            expected = morphing.payload["regions"][0]["coordinates"]
            current = session.describe_deformation()["regions"][0][
                "current_coordinates"
            ]
            error = max(
                abs(float(actual[axis]) - float(wanted[axis]))
                for actual, wanted in zip(current, expected, strict=True)
                for axis in range(3)
            )
            maximum_coordinate_error = max(maximum_coordinate_error, error)
            for report in morphing.quality_reports.values():
                if report is not None and report.minimum_measure_ratio is not None:
                    minimum_measure_ratio = min(
                        minimum_measure_ratio, report.minimum_measure_ratio
                    )

    summaries = adapter.constraint_summaries
    inferred_count = sum(item.inferred_fixed_count for item in summaries.values())
    print(
        f"PASS: input={args.input_file}, method={args.method}, steps={args.steps}, "
        f"max XYZ error={maximum_coordinate_error:.3e} m, "
        f"minimum measure ratio={minimum_measure_ratio:.6g}, "
        f"inferred fixed anchors={inferred_count}"
    )


if __name__ == "__main__":
    main()
