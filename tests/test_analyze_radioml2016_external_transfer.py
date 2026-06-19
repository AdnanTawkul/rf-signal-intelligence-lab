from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "analyze_radioml2016_external_transfer.py"
)


def write_metrics(
    path: Path,
    *,
    offset: float,
) -> None:
    runs = []

    for run_index, seed in enumerate(
        (
            2026,
            2027,
        )
    ):
        base = (
            0.50
            + 0.05 * run_index
            + offset
        )

        runs.append(
            {
                "seed": seed,
                "overall_accuracy": base,
                "class_accuracy": {
                    "BPSK": base + 0.1,
                    "QPSK": base - 0.1,
                },
                "accuracy_by_snr": {
                    "-4.0": base - 0.1,
                    "0.0": base,
                    "4.0": base + 0.1,
                },
            }
        )

    path.write_text(
        json.dumps(
            {
                "experiment_name": path.stem,
                "class_names": [
                    "BPSK",
                    "QPSK",
                ],
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_external_transfer_script_creates_package(
    tmp_path: Path,
) -> None:
    reference_path = (
        tmp_path / "reference.json"
    )
    candidate_path = (
        tmp_path / "candidate.json"
    )
    synthetic_path = (
        tmp_path / "synthetic.json"
    )

    write_metrics(
        reference_path,
        offset=0.0,
    )
    write_metrics(
        candidate_path,
        offset=0.02,
    )
    write_metrics(
        synthetic_path,
        offset=-0.01,
    )

    summary_path = (
        tmp_path / "summary.json"
    )
    overall_figure = (
        tmp_path / "overall.png"
    )
    snr_figure = tmp_path / "snr.png"
    class_figure = (
        tmp_path / "classes.png"
    )

    config = {
        "experiment_name": (
            "external_transfer_test"
        ),
        "shared_snr_grid_db": [
            -4,
            0,
            4,
        ],
        "models": {
            "reference": {
                "display_name": "Reference",
                "aggregate_metrics": str(
                    reference_path
                ),
            },
            "candidate": {
                "display_name": "Candidate",
                "aggregate_metrics": str(
                    candidate_path
                ),
            },
        },
        "display_order": [
            "reference",
            "candidate",
        ],
        "detail_order": [
            "reference",
            "candidate",
        ],
        "paired_comparisons": {
            "candidate_vs_reference": {
                "display_name": (
                    "Candidate minus reference"
                ),
                "reference": "reference",
                "candidate": "candidate",
            }
        },
        "synthetic_controls": {
            "candidate": {
                "display_name": (
                    "Candidate external "
                    "minus synthetic"
                ),
                "synthetic_metrics": str(
                    synthetic_path
                ),
                "external_model": "candidate",
            }
        },
        "output": {
            "summary_json": str(
                summary_path
            ),
            "overall_figure": str(
                overall_figure
            ),
            "snr_figure": str(
                snr_figure
            ),
            "class_figure": str(
                class_figure
            ),
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    for path in (
        summary_path,
        overall_figure,
        snr_figure,
        class_figure,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    comparison = summary[
        "paired_comparisons"
    ]["candidate_vs_reference"]["overall"]

    assert comparison[
        "mean"
    ] == pytest.approx(0.02)
    assert comparison[
        "improved_seed_count"
    ] == 2
