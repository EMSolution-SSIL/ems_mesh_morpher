from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import read_mesh_auto, write_mesh_auto
from .mesh_motion import apply_mesh_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--config", required=True)
    parser.add_argument("--informat")
    parser.add_argument("--outformat")
    args = parser.parse_args()

    config = load_config(args.config)
    mesh = read_mesh_auto(args.input, file_format=args.informat)
    result = apply_mesh_motion(mesh, config)

    outformat = args.outformat or config.output.file_format
    write_mesh_auto(Path(args.output), result.mesh, file_format=outformat, template_path=args.input)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
