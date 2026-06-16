from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.data.synthetic import MODULATION_CLASSES
from rfsil.evaluation.error_analysis import (
    compute_class_snr_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze baseline accuracy by class and SNR.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analyze_baseline_v1.yaml"),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError("Analysis configuration must be a YAML mapping.")

    return content


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    predictions_path = resolve_project_path(
        content["predictions_path"]
    )

    with np.load(predictions_path, allow_pickle=False) as data:
        labels = data["labels"].astype(np.int64)
        predictions = data["predictions"].astype(np.int64)
        snr_db = data["snr_db"].astype(np.float32)

    class_names = [
        modulation.value.upper()
        for modulation in MODULATION_CLASSES
    ]

    analysis = compute_class_snr_analysis(
        labels=labels,
        predictions=predictions,
        snr_db=snr_db,
        num_classes=len(class_names),
    )

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    heatmap_path = resolve_project_path(
        output_content["class_snr_heatmap"]
    )
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))

    image = axis.imshow(
        analysis.accuracy,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    axis.set_title("Baseline CNN Accuracy by Class and SNR")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("True modulation class")

    axis.set_xticks(
        np.arange(len(analysis.snr_values_db)),
        labels=[
            f"{float(value):.0f}"
            for value in analysis.snr_values_db
        ],
    )
    axis.set_yticks(
        np.arange(len(class_names)),
        labels=class_names,
    )

    for class_index in range(len(class_names)):
        for snr_index in range(len(analysis.snr_values_db)):
            value = analysis.accuracy[
                class_index,
                snr_index,
            ]

            label = (
                "N/A"
                if np.isnan(value)
                else f"{float(value):.2f}"
            )

            axis.text(
                snr_index,
                class_index,
                label,
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Accuracy",
    )
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=160)
    plt.close(figure)

    finite_accuracy = np.where(
        np.isnan(analysis.accuracy),
        np.inf,
        analysis.accuracy,
    )
    worst_flat_index = int(np.argmin(finite_accuracy))
    worst_class_index, worst_snr_index = np.unravel_index(
        worst_flat_index,
        analysis.accuracy.shape,
    )

    worst_class_name = class_names[worst_class_index]
    worst_snr_db = float(
        analysis.snr_values_db[worst_snr_index]
    )
    worst_accuracy = float(
        analysis.accuracy[
            worst_class_index,
            worst_snr_index,
        ]
    )

    report = {
        "format_version": 1,
        "experiment_name": str(content["experiment_name"]),
        "class_names": class_names,
        "snr_values_db": [
            float(value)
            for value in analysis.snr_values_db
        ],
        "accuracy_matrix": analysis.accuracy.tolist(),
        "count_matrix": analysis.counts.tolist(),
        "error_count_matrix": analysis.error_counts.tolist(),
        "worst_group": {
            "class_name": worst_class_name,
            "snr_db": worst_snr_db,
            "accuracy": worst_accuracy,
            "example_count": int(
                analysis.counts[
                    worst_class_index,
                    worst_snr_index,
                ]
            ),
            "error_count": int(
                analysis.error_counts[
                    worst_class_index,
                    worst_snr_index,
                ]
            ),
        },
    }

    report_path = output_directory / "class_snr_analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Predictions: {predictions_path}")
    print(f"Examples analyzed: {len(labels)}")

    for class_index, class_name in enumerate(class_names):
        values = ", ".join(
            (
                f"{float(snr_value):.0f} dB="
                f"{float(accuracy):.3f}"
            )
            for snr_value, accuracy in zip(
                analysis.snr_values_db,
                analysis.accuracy[class_index],
                strict=True,
            )
        )
        print(f"{class_name}: {values}")

    print(
        "Worst class-SNR group: "
        f"{worst_class_name} at {worst_snr_db:.1f} dB, "
        f"accuracy={worst_accuracy:.3f}"
    )
    print(f"Report: {report_path}")
    print(f"Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
