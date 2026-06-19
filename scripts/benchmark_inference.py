from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from rfsil.deployment import (
    benchmark_checkpoint,
    benchmark_environment,
    build_benchmark_batch,
    load_iq_file,
    write_prediction_document,
)


def parse_arguments() -> argparse.Namespace:
    """Parse inference benchmark arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RF IQ inference latency "
            "and throughput."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--array-key",
        default="iq",
    )
    parser.add_argument(
        "--device",
        action="append",
        choices=(
            "cpu",
            "cuda",
        ),
        default=None,
        help=(
            "Device to benchmark. Repeat to "
            "benchmark multiple devices. "
            "The default benchmarks CPU and "
            "CUDA when CUDA is available."
        ),
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        type=int,
        default=None,
        help=(
            "Batch size to benchmark. Repeat "
            "for multiple batch sizes."
        ),
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--measurement-iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "deployment_benchmark_v1/"
            "benchmark.json"
        ),
    )

    return parser.parse_args()


def resolve_devices(
    requested: list[str] | None,
) -> list[str]:
    """Resolve the device benchmark matrix."""
    if requested:
        devices = list(
            dict.fromkeys(requested)
        )
    else:
        devices = ["cpu"]

        if torch.cuda.is_available():
            devices.append("cuda")

    if (
        "cuda" in devices
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not "
            "available."
        )

    return devices


def resolve_batch_sizes(
    requested: list[int] | None,
) -> list[int]:
    """Resolve and validate batch sizes."""
    values = (
        [1, 8, 32, 128]
        if requested is None
        else requested
    )
    batch_sizes = list(
        dict.fromkeys(values)
    )

    if any(
        isinstance(value, bool)
        or value <= 0
        for value in batch_sizes
    ):
        raise ValueError(
            "Every batch size must be positive."
        )

    return batch_sizes


def run_benchmark(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Execute the complete benchmark matrix."""
    loaded = load_iq_file(
        arguments.input,
        array_key=arguments.array_key,
        expected_sample_count=(
            arguments.expected_samples
        ),
    )

    devices = resolve_devices(
        arguments.device
    )
    batch_sizes = resolve_batch_sizes(
        arguments.batch_size
    )

    results = []

    for device in devices:
        for batch_size in batch_sizes:
            batch = build_benchmark_batch(
                loaded.iq,
                batch_size=batch_size,
            )

            print(
                f"Benchmarking device={device}, "
                f"batch_size={batch_size}",
                file=sys.stderr,
            )

            result = benchmark_checkpoint(
                checkpoint_path=(
                    arguments.checkpoint
                ),
                inputs=batch,
                device=device,
                input_scale=(
                    arguments.input_scale
                ),
                expected_sample_count=(
                    arguments.expected_samples
                ),
                warmup_iterations=(
                    arguments
                    .warmup_iterations
                ),
                measurement_iterations=(
                    arguments
                    .measurement_iterations
                ),
            )

            results.append(
                result.to_dict()
            )

    return {
        "format_version": 1,
        "benchmark_name": (
            "deployment_inference_v1"
        ),
        "checkpoint_path": (
            arguments.checkpoint
            .resolve()
            .as_posix()
        ),
        "input_path": (
            arguments.input
            .resolve()
            .as_posix()
        ),
        "input_scale": (
            arguments.input_scale
        ),
        "expected_sample_count": (
            arguments.expected_samples
        ),
        "environment": {
            **benchmark_environment(),
            "python_version": (
                platform.python_version()
            ),
            "platform": platform.platform(),
        },
        "configuration": {
            "devices": devices,
            "batch_sizes": batch_sizes,
            "warmup_iterations": (
                arguments.warmup_iterations
            ),
            "measurement_iterations": (
                arguments
                .measurement_iterations
            ),
        },
        "results": results,
    }


def main() -> None:
    """Run inference benchmarking."""
    arguments = parse_arguments()

    try:
        document = run_benchmark(
            arguments
        )
    except (
        FileNotFoundError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    write_prediction_document(
        arguments.output,
        document,
    )

    print(
        json.dumps(
            document,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
