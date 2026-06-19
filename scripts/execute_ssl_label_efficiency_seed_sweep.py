from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rfsil.training.ssl_sweep_execution import (
    SweepRunExpectation,
    load_completed_run,
    load_run_expectation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse execution and filtering arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute or resume a generated SSL "
            "label-efficiency sweep."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip runs whose summaries and "
            "artifacts validate successfully."
        ),
    )
    parser.add_argument(
        "--fraction",
        action="append",
        default=[],
        help=(
            "Run only this fraction identifier. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Run only this method. May be repeated."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help=(
            "Run only this seed. May be repeated."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Limit the number of selected runs."
        ),
    )
    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON mapping."""
    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"JSON content must be a mapping: {path}"
        )

    return content


def run_training(
    expectation: SweepRunExpectation,
) -> None:
    """Launch one supervised training process."""
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "train_baseline.py"
        ),
        "--config",
        str(
            expectation.generated_config_path
        ),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def is_selected(
    expectation: SweepRunExpectation,
    *,
    fractions: set[str],
    methods: set[str],
    seeds: set[int],
) -> bool:
    """Apply optional command-line filters."""
    if (
        fractions
        and expectation.fraction_identifier
        not in fractions
    ):
        return False

    if (
        methods
        and expectation.method not in methods
    ):
        return False

    return not (
        seeds
        and expectation.seed not in seeds
    )


def write_execution_manifest(
    *,
    path: Path,
    source_manifest: Path,
    selected_run_count: int,
    resume: bool,
    records: list[dict[str, object]],
) -> None:
    """Persist current execution progress."""
    completed_count = sum(
        record["status"] == "completed"
        for record in records
    )
    skipped_count = sum(
        record["status"] == "skipped"
        for record in records
    )

    content = {
        "format_version": 1,
        "source_manifest": (
            source_manifest.resolve().as_posix()
        ),
        "resume": resume,
        "selected_run_count": (
            selected_run_count
        ),
        "processed_run_count": len(records),
        "completed_run_count": (
            completed_count
        ),
        "skipped_run_count": skipped_count,
        "runs": records,
    }

    path.write_text(
        json.dumps(
            content,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Execute the requested sweep subset."""
    arguments = parse_arguments()

    if (
        arguments.max_runs is not None
        and arguments.max_runs <= 0
    ):
        raise ValueError(
            "--max-runs must be positive."
        )

    manifest_path = resolve_project_path(
        arguments.manifest
    )
    manifest = load_json_mapping(
        manifest_path
    )

    raw_runs = manifest.get("runs")

    if not isinstance(raw_runs, list):
        raise ValueError(
            "Manifest runs must be a list."
        )

    if manifest.get("run_count") != len(
        raw_runs
    ):
        raise ValueError(
            "Manifest run_count does not "
            "match its run list."
        )

    expectations = [
        load_run_expectation(
            record,
            project_root=PROJECT_ROOT,
        )
        for record in raw_runs
    ]

    fractions = set(arguments.fraction)
    methods = set(arguments.method)
    seeds = set(arguments.seed)

    selected = [
        expectation
        for expectation in expectations
        if is_selected(
            expectation,
            fractions=fractions,
            methods=methods,
            seeds=seeds,
        )
    ]

    if arguments.max_runs is not None:
        selected = selected[
            : arguments.max_runs
        ]

    if not selected:
        raise ValueError(
            "No runs match the requested filters."
        )

    execution_manifest_path = (
        manifest_path.parent
        / "execution_manifest.json"
    )
    execution_records: list[
        dict[str, object]
    ] = []

    print(
        f"Selected runs: {len(selected)}"
    )
    print(
        f"Resume enabled: {arguments.resume}"
    )

    for run_index, expectation in enumerate(
        selected,
        start=1,
    ):
        print()
        print("=" * 72)
        print(
            f"Run {run_index}/{len(selected)} | "
            f"{expectation.fraction_identifier} | "
            f"{expectation.method} | "
            f"seed={expectation.seed}"
        )
        print("=" * 72)

        completed = load_completed_run(
            expectation,
            project_root=PROJECT_ROOT,
        )

        if completed is not None:
            if not arguments.resume:
                raise FileExistsError(
                    "A validated result already "
                    "exists. Re-run with --resume "
                    "to skip it."
                )

            status = "skipped"
            print(
                "Validated completed run found; "
                "skipping."
            )
        else:
            run_training(expectation)

            completed = load_completed_run(
                expectation,
                project_root=PROJECT_ROOT,
            )

            if completed is None:
                raise RuntimeError(
                    "Training returned successfully "
                    "without a summary."
                )

            status = "completed"

        record = {
            **asdict(completed),
            "summary_path": (
                completed.summary_path.resolve().as_posix()
            ),
            "checkpoint_path": (
                completed.checkpoint_path.resolve().as_posix()
            ),
            "history_path": (
                completed.history_path.resolve().as_posix()
            ),
            "status": status,
        }
        execution_records.append(record)

        write_execution_manifest(
            path=execution_manifest_path,
            source_manifest=manifest_path,
            selected_run_count=len(selected),
            resume=arguments.resume,
            records=execution_records,
        )

        print(
            "Best validation accuracy: "
            f"{completed.best_validation_accuracy:.4f}"
        )
        print(
            f"Execution status: {status}"
        )

    print()
    print(
        "Execution manifest: "
        f"{execution_manifest_path}"
    )


if __name__ == "__main__":
    main()
