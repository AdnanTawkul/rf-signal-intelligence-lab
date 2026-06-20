from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkPoint:
    """Validated benchmark result for one case."""

    source_path: str
    device: str
    batch_size: int
    sample_count: int
    model_load_ms: float
    first_inference_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_windows_per_second: float
    cuda_peak_memory_bytes: int | None

    @property
    def mean_latency_per_window_ms(
        self,
    ) -> float:
        """Return average latency per window."""
        return (
            self.mean_latency_ms
            / self.batch_size
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to JSON-compatible data."""
        content = asdict(self)
        content[
            "mean_latency_per_window_ms"
        ] = self.mean_latency_per_window_ms

        return content


def _require_mapping(
    value: object,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def _positive_float(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be positive and finite."
        ) from error

    if (
        not math.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    return validated


def _positive_integer(
    value: object,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    try:
        validated = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive integer."
        ) from error

    if validated <= 0 or validated != value:
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return validated


def parse_benchmark_document(
    document: Mapping[str, Any],
    *,
    source_path: Path,
) -> tuple[BenchmarkPoint, ...]:
    """Parse one benchmark JSON document."""
    raw_results = document.get("results")

    if (
        isinstance(raw_results, (str, bytes))
        or not isinstance(
            raw_results,
            Sequence,
        )
        or not raw_results
    ):
        raise ValueError(
            "Benchmark results must be a "
            "non-empty sequence."
        )

    points = []

    for index, raw_result in enumerate(
        raw_results
    ):
        result = _require_mapping(
            raw_result,
            f"results[{index}]",
        )
        latency = _require_mapping(
            result.get(
                "steady_state_latency"
            ),
            (
                f"results[{index}]."
                "steady_state_latency"
            ),
        )

        device = str(
            result.get("device", "")
        ).strip().lower()

        if device not in {"cpu", "cuda"}:
            raise ValueError(
                "Benchmark device must be "
                "'cpu' or 'cuda'."
            )

        memory_value = result.get(
            "cuda_peak_memory_bytes"
        )

        if memory_value is None:
            memory = None
        else:
            memory = _positive_integer(
                memory_value,
                "cuda_peak_memory_bytes",
            )

        point = BenchmarkPoint(
            source_path=(
                source_path.resolve()
                .as_posix()
            ),
            device=device,
            batch_size=_positive_integer(
                result.get("batch_size"),
                "batch_size",
            ),
            sample_count=_positive_integer(
                result.get("sample_count"),
                "sample_count",
            ),
            model_load_ms=_positive_float(
                result.get("model_load_ms"),
                "model_load_ms",
            ),
            first_inference_ms=(
                _positive_float(
                    result.get(
                        "first_inference_ms"
                    ),
                    "first_inference_ms",
                )
            ),
            mean_latency_ms=_positive_float(
                latency.get("mean_ms"),
                "mean_ms",
            ),
            p95_latency_ms=_positive_float(
                latency.get("p95_ms"),
                "p95_ms",
            ),
            throughput_windows_per_second=(
                _positive_float(
                    result.get(
                        "throughput_windows_per_second"
                    ),
                    (
                        "throughput_windows_"
                        "per_second"
                    ),
                )
            ),
            cuda_peak_memory_bytes=memory,
        )

        points.append(point)

    return tuple(points)


def load_benchmark_points(
    paths: Sequence[Path],
) -> tuple[BenchmarkPoint, ...]:
    """Load benchmark points from JSON files."""
    if not paths:
        raise ValueError(
            "At least one benchmark file "
            "is required."
        )

    points = []
    seen_cases: set[
        tuple[str, int]
    ] = set()

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        document = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        mapping = _require_mapping(
            document,
            f"benchmark document {path}",
        )

        for point in parse_benchmark_document(
            mapping,
            source_path=path,
        ):
            case = (
                point.device,
                point.batch_size,
            )

            if case in seen_cases:
                raise ValueError(
                    "Duplicate benchmark case: "
                    f"{point.device}, "
                    f"batch {point.batch_size}."
                )

            seen_cases.add(case)
            points.append(point)

    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.device,
                point.batch_size,
            ),
        )
    )


def build_benchmark_summary(
    points: Sequence[BenchmarkPoint],
) -> dict[str, object]:
    """Build derived benchmark comparisons."""
    if not points:
        raise ValueError(
            "At least one benchmark point "
            "is required."
        )

    device_summaries = {}

    for device in sorted(
        {
            point.device
            for point in points
        }
    ):
        device_points = [
            point
            for point in points
            if point.device == device
        ]

        peak_throughput = max(
            device_points,
            key=lambda point: (
                point
                .throughput_windows_per_second
            ),
        )
        lowest_batch_latency = min(
            device_points,
            key=lambda point: (
                point.mean_latency_ms
            ),
        )
        lowest_per_window_latency = min(
            device_points,
            key=lambda point: (
                point
                .mean_latency_per_window_ms
            ),
        )

        device_summaries[device] = {
            "peak_throughput": {
                "batch_size": (
                    peak_throughput.batch_size
                ),
                "windows_per_second": (
                    peak_throughput
                    .throughput_windows_per_second
                ),
            },
            "lowest_batch_latency": {
                "batch_size": (
                    lowest_batch_latency
                    .batch_size
                ),
                "mean_ms": (
                    lowest_batch_latency
                    .mean_latency_ms
                ),
            },
            "lowest_per_window_latency": {
                "batch_size": (
                    lowest_per_window_latency
                    .batch_size
                ),
                "mean_ms": (
                    lowest_per_window_latency
                    .mean_latency_per_window_ms
                ),
            },
        }

    lookup = {
        (
            point.device,
            point.batch_size,
        ): point
        for point in points
    }

    common_batches = sorted(
        {
            point.batch_size
            for point in points
            if (
                ("cpu", point.batch_size)
                in lookup
                and (
                    "cuda",
                    point.batch_size,
                )
                in lookup
            )
        }
    )

    comparisons = []

    for batch_size in common_batches:
        cpu = lookup[
            ("cpu", batch_size)
        ]
        cuda = lookup[
            ("cuda", batch_size)
        ]

        comparisons.append(
            {
                "batch_size": batch_size,
                "cuda_throughput_speedup": (
                    cuda
                    .throughput_windows_per_second
                    / cpu
                    .throughput_windows_per_second
                ),
                "cuda_batch_latency_ratio": (
                    cuda.mean_latency_ms
                    / cpu.mean_latency_ms
                ),
                "cpu_mean_latency_ms": (
                    cpu.mean_latency_ms
                ),
                "cuda_mean_latency_ms": (
                    cuda.mean_latency_ms
                ),
            }
        )

    return {
        "benchmark_case_count": len(points),
        "devices": device_summaries,
        "cpu_cuda_comparisons": comparisons,
    }


__all__ = [
    "BenchmarkPoint",
    "build_benchmark_summary",
    "load_benchmark_points",
    "parse_benchmark_document",
]
