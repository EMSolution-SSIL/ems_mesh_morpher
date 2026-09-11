from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np

from ems_mesh_morpher import MotionSchedule, benchmark_repeated_motion, config_from_dict


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark repeated mesh motion methods")
    parser.add_argument("--model", choices=("block", "annular", "all"), default="all")
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _structured_block(nx: int = 21, ny: int = 7) -> tuple[meshio.Mesh, list[int], list[int]]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(-0.2, 0.2, ny)
    points = np.array([[x_value, y_value, 0.0] for y_value in y for x_value in x])
    cells = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower = row * nx + column
            cells.append([lower, lower + 1, lower + nx + 1, lower + nx])
    moving = [row * nx for row in range(ny)]
    fixed = [row * nx + nx - 1 for row in range(ny)]
    return meshio.Mesh(points, [("quad", np.asarray(cells))]), moving, fixed


def _annular_mesh(
    radial_count: int = 6, angular_count: int = 48
) -> tuple[meshio.Mesh, list[int], list[int]]:
    radii = np.linspace(0.5, 1.0, radial_count)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_count, endpoint=False)
    points = np.array(
        [
            [radius * np.cos(angle), radius * np.sin(angle), 0.0]
            for radius in radii
            for angle in angles
        ]
    )
    cells = []
    for radial in range(radial_count - 1):
        for angular in range(angular_count):
            following = (angular + 1) % angular_count
            lower = radial * angular_count
            upper = (radial + 1) * angular_count
            cells.append(
                [lower + angular, upper + angular, upper + following, lower + following]
            )
    moving = list(range(angular_count))
    fixed = list(range((radial_count - 1) * angular_count, radial_count * angular_count))
    return meshio.Mesh(points, [("quad", np.asarray(cells))]), moving, fixed


def _config(moving: list[int], fixed: list[int], method: str):
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {"point_indices": moving},
                "fixed": {"point_indices": fixed},
                "deformable": {"geometry": {"type": "all"}},
            },
            "motion": {"type": "translation", "displacement": [0.0, 0.0]},
            "morphing": {
                "type": method,
                "neighbors": 32,
                "radius_scale": 2.0,
                "weighting": "inverse_distance_element",
            },
            "quality": {
                "minimum_measure_ratio": 0.05,
                "failure_policy": "error",
            },
        }
    )
    if method != "weighted_laplace":
        config = replace(
            config,
            morphing=replace(config.morphing, weighting="inverse_distance"),
        )
    return config


def _run_block(frame_count: int) -> dict[str, object]:
    mesh, moving, fixed = _structured_block()
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count)
    schedule = MotionSchedule.translations(
        np.linspace(0.0, 1.0, frame_count),
        [[0.12 * np.sin(value), 0.0] for value in phase],
    )
    configs = {
        method: _config(moving, fixed, method)
        for method in ("idw", "rbf", "weighted_laplace")
    }
    return benchmark_repeated_motion(mesh, schedule, configs)


def _run_annular(frame_count: int) -> dict[str, object]:
    mesh, moving, fixed = _annular_mesh()
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count)
    schedule = MotionSchedule.rotations(
        np.linspace(0.0, 1.0, frame_count),
        8.0 * np.sin(phase),
        center=(0.0, 0.0, 0.0),
    )
    configs = {
        method: _config(moving, fixed, method)
        for method in ("idw", "rbf", "weighted_laplace")
    }
    return benchmark_repeated_motion(mesh, schedule, configs)


def main() -> None:
    args = _arguments()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    report: dict[str, object] = {}
    if args.model in {"block", "all"}:
        report["block_translation"] = _run_block(args.frames)
    if args.model in {"annular", "all"}:
        report["annular_rotation"] = _run_annular(args.frames)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
        for case_name, methods in report.items():
            for method, summary in methods.items():
                print(
                    f"{case_name} {method}: {summary['frame_count']} frames, "
                    f"{summary['total_seconds']:.6f} s, "
                    f"min ratio={summary['minimum_measure_ratio']:.6g}"
                )
    else:
        print(text)


if __name__ == "__main__":
    main()
