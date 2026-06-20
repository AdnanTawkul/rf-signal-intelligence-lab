from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from rfsil.deployment.benchmark_analysis import (
    BenchmarkPoint,
    build_benchmark_summary,
    load_benchmark_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark-analysis arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze fresh-process deployment "
            "benchmark results."
        )
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/"
            "deployment_benchmark_v1/"
            "analysis_summary.json"
        ),
    )
    parser.add_argument(
        "--latency-figure",
        type=Path,
        default=Path(
            "reports/figures/"
            "deployment_inference_latency_v1.png"
        ),
    )
    parser.add_argument(
        "--throughput-figure",
        type=Path,
        default=Path(
            "reports/figures/"
            "deployment_inference_throughput_v1.png"
        ),
    )
    parser.add_argument(
        "--memory-figure",
        type=Path,
        default=Path(
            "reports/figures/"
            "deployment_inference_memory_v1.png"
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a project-relative path."""
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def serialize_path(path: Path) -> str:
    """Serialize project paths relatively when possible."""
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()

    try:
        return resolved.relative_to(
            project_root
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def points_for_device(
    points: tuple[BenchmarkPoint, ...],
    device: str,
) -> list[BenchmarkPoint]:
    """Return batch-sorted points for one device."""
    return sorted(
        (
            point
            for point in points
            if point.device == device
        ),
        key=lambda point: point.batch_size,
    )


def save_latency_figure(
    points: tuple[BenchmarkPoint, ...],
    output_path: Path,
) -> None:
    """Plot steady-state batch latency."""
    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    for device in ("cpu", "cuda"):
        device_points = points_for_device(
            points,
            device,
        )

        if not device_points:
            continue

        batches = [
            point.batch_size
            for point in device_points
        ]

        axis.plot(
            batches,
            [
                point.mean_latency_ms
                for point in device_points
            ],
            marker="o",
            label=f"{device.upper()} mean",
        )
        axis.plot(
            batches,
            [
                point.p95_latency_ms
                for point in device_points
            ],
            marker="x",
            linestyle="--",
            label=f"{device.upper()} p95",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Batch latency (ms)")
    axis.set_title(
        "Steady-State IQ Inference Latency"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_throughput_figure(
    points: tuple[BenchmarkPoint, ...],
    output_path: Path,
) -> None:
    """Plot inference throughput."""
    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    for device in ("cpu", "cuda"):
        device_points = points_for_device(
            points,
            device,
        )

        if not device_points:
            continue

        axis.plot(
            [
                point.batch_size
                for point in device_points
            ],
            [
                point
                .throughput_windows_per_second
                for point in device_points
            ],
            marker="o",
            label=device.upper(),
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("IQ windows per second")
    axis.set_title(
        "IQ Inference Throughput Scaling"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_memory_figure(
    points: tuple[BenchmarkPoint, ...],
    output_path: Path,
) -> None:
    """Plot CUDA peak allocated memory."""
    cuda_points = [
        point
        for point in points_for_device(
            points,
            "cuda",
        )
        if (
            point.cuda_peak_memory_bytes
            is not None
        )
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    if cuda_points:
        axis.plot(
            [
                point.batch_size
                for point in cuda_points
            ],
            [
                (
                    point
                    .cuda_peak_memory_bytes
                    / (1024**2)
                )
                for point in cuda_points
                if (
                    point
                    .cuda_peak_memory_bytes
                    is not None
                )
            ],
            marker="o",
        )
        axis.set_xscale(
            "log",
            base=2,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No CUDA memory measurements",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_xlabel("Batch size")
    axis.set_ylabel(
        "Peak allocated CUDA memory (MiB)"
    )
    axis.set_title(
        "CUDA Memory Scaling"
    )
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    """Create benchmark summary and figures."""
    arguments = parse_arguments()

    input_directory = resolve_path(
        arguments.input_directory
    )

    if not input_directory.is_dir():
        raise NotADirectoryError(
            input_directory
        )

    paths = sorted(
        input_directory.glob("*.json")
    )
    points = load_benchmark_points(paths)

    summary_output = resolve_path(
        arguments.summary_output
    )
    latency_figure = resolve_path(
        arguments.latency_figure
    )
    throughput_figure = resolve_path(
        arguments.throughput_figure
    )
    memory_figure = resolve_path(
        arguments.memory_figure
    )

    for path in (
        summary_output,
        latency_figure,
        throughput_figure,
        memory_figure,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    derived = build_benchmark_summary(
        points
    )

    summary: dict[str, Any] = {
        "format_version": 1,
        "analysis_name": (
            "deployment_benchmark_analysis_v1"
        ),
        "source_directory": (
            input_directory
            .resolve()
            .as_posix()
        ),
        "source_files": [
            path.resolve().as_posix()
            for path in paths
        ],
        "points": [
            point.to_dict()
            for point in points
        ],
        "summary": derived,
        "figures": {
            "latency": serialize_path(
                latency_figure
            ),
            "throughput": serialize_path(
                throughput_figure
            ),
            "memory": serialize_path(
                memory_figure
            ),
        },
    }

    summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    save_latency_figure(
        points,
        latency_figure,
    )
    save_throughput_figure(
        points,
        throughput_figure,
    )
    save_memory_figure(
        points,
        memory_figure,
    )

    print("Benchmark cases")
    print("=" * 96)
    print(
        "Device | Batch | Load ms | First ms | "
        "Mean ms | P95 ms | Windows/s | CUDA MiB"
    )
    print("-" * 96)

    for point in points:
        memory = (
            "-"
            if (
                point.cuda_peak_memory_bytes
                is None
            )
            else (
                f"{point.cuda_peak_memory_bytes / (1024**2):.1f}"
            )
        )

        print(
            f"{point.device:6s} | "
            f"{point.batch_size:5d} | "
            f"{point.model_load_ms:7.2f} | "
            f"{point.first_inference_ms:8.2f} | "
            f"{point.mean_latency_ms:7.2f} | "
            f"{point.p95_latency_ms:6.2f} | "
            f"{point.throughput_windows_per_second:9.1f} | "
            f"{memory:>8s}"
        )

    print()
    print(f"Summary: {summary_output}")
    print(f"Latency figure: {latency_figure}")
    print(
        f"Throughput figure: "
        f"{throughput_figure}"
    )
    print(f"Memory figure: {memory_figure}")


if __name__ == "__main__":
    main()
