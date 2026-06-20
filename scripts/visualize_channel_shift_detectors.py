from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rfsil.evaluation.shift_detector_visualization import (
    create_detector_comparison_figures,
    load_detector_comparison_visualization_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create channel-shift detector "
            "comparison figures."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml_mapping(
    path: Path,
) -> dict[str, Any]:
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    input_path = resolve_project_path(
        content["input"]["summary_path"]
    )
    output_directory = resolve_project_path(
        content["output"]["directory"]
    )
    summary_path = (
        output_directory
        / str(
            content["output"][
                "summary_filename"
            ]
        )
    )
    dpi = int(
        content["visualization"]["dpi"]
    )

    data = (
        load_detector_comparison_visualization_data(
            input_path
        )
    )
    figure_paths = (
        create_detector_comparison_figures(
            data,
            output_directory,
            dpi=dpi,
        )
    )

    summary = {
        "format_version": 1,
        "experiment_name": str(
            content["experiment_name"]
        ),
        "config_path": (
            config_path
            .resolve()
            .as_posix()
        ),
        "input_summary_path": (
            input_path
            .resolve()
            .as_posix()
        ),
        "visualization_data": (
            data.to_dict()
        ),
        "figures": {
            name: path.resolve().as_posix()
            for name, path
            in figure_paths.items()
        },
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Conditions: "
        f"{len(data.conditions)}"
    )
    print(
        "Detector systems: "
        f"{len(data.systems)}"
    )

    for name, path in (
        figure_paths.items()
    ):
        print(f"{name}: {path}")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
