from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "examples" / "skin_layer_sample" / "generate_samples.py"


def test_cli_accepts_underscore_options_and_thickness(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sample",
            "block_hexa",
            "--thickness",
            "0.012",
            "--layer_count",
            "3",
            "--growth_ratio",
            "1.0",
            "--collision_policy",
            "fail",
            "--output_dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    reports = json.loads((tmp_path / "reports.json").read_text(encoding="utf-8"))
    assert len(reports) == 1
    assert reports[0]["name"] == "block_hexa"
    assert reports[0]["requested_thickness"] == pytest.approx(0.012)
    assert reports[0]["layer_count"] == 3
    assert reports[0]["layer_thicknesses"] == pytest.approx([0.004] * 3)
    assert (tmp_path / "block_conductor_hexa_skin.neu").is_file()


def test_cli_rejects_ambiguous_thickness_override(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--thickness",
            "0.012",
            "--output_dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--thickness requires exactly one --sample" in completed.stderr
    assert not (tmp_path / "reports.json").exists()
    assert not any(tmp_path.glob("*.neu"))
