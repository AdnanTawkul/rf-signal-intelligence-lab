from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from rfsil.deployment.inference import (
    IQInferenceEngine,
    resolve_device,
)

Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Summary statistics for repeated inference calls."""

    iteration_count: int
    mean_ms: float
    standard_deviation_ms: float
    minimum_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    maximum_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One checkpoint, device, and batch-size benchmark."""

    device: str
    batch_size: int
    sample_count: int
    warmup_iterations: int
    measurement_iterations: int
    model_load_ms: float
    first_inference_ms: float
    steady_state_latency: LatencySummary
    throughput_windows_per_second: float
    cuda_peak_memory_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        """Convert the result to JSON-compatible data."""
        return asdict(self)


def _validate_positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return validated


def _validate_nonnegative_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    validated = int(value)

    if validated < 0:
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    return validated


def summarize_latencies(
    durations_seconds: list[float]
    | tuple[float, ...]
    | np.ndarray,
) -> LatencySummary:
    """Summarize measured inference durations."""
    values = np.asarray(
        durations_seconds,
        dtype=np.float64,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "At least one latency measurement "
            "is required."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Latency measurements must be finite."
        )

    if np.any(values < 0.0):
        raise ValueError(
            "Latency measurements must not "
            "be negative."
        )

    milliseconds = 1000.0 * values

    return LatencySummary(
        iteration_count=int(
            milliseconds.size
        ),
        mean_ms=float(
            np.mean(milliseconds)
        ),
        standard_deviation_ms=float(
            np.std(milliseconds)
        ),
        minimum_ms=float(
            np.min(milliseconds)
        ),
        median_ms=float(
            np.median(milliseconds)
        ),
        p90_ms=float(
            np.percentile(
                milliseconds,
                90.0,
            )
        ),
        p95_ms=float(
            np.percentile(
                milliseconds,
                95.0,
            )
        ),
        maximum_ms=float(
            np.max(milliseconds)
        ),
    )


def build_benchmark_batch(
    iq: np.ndarray,
    *,
    batch_size: int,
) -> Float32Array:
    """Build a deterministic batch from loaded IQ windows."""
    validated_batch_size = (
        _validate_positive_integer(
            batch_size,
            name="batch_size",
        )
    )
    array = np.asarray(iq)

    if (
        array.ndim != 3
        or array.shape[0] <= 0
        or array.shape[1] != 2
        or array.shape[2] <= 0
    ):
        raise ValueError(
            "iq must have shape "
            "[batch, 2, samples]."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise ValueError(
            "iq must contain numeric values."
        )

    if np.iscomplexobj(array):
        raise ValueError(
            "iq must use two real-valued channels."
        )

    converted = np.asarray(
        array,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            "iq must contain only finite values."
        )

    source_indices = (
        np.arange(
            validated_batch_size,
            dtype=np.int64,
        )
        % converted.shape[0]
    )

    return np.ascontiguousarray(
        converted[source_indices]
    )


def _synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_call(
    engine: IQInferenceEngine,
    inputs: Float32Array,
) -> float:
    _synchronize_device(engine.device)

    start = perf_counter()
    engine.predict_batch(inputs)
    _synchronize_device(engine.device)

    return perf_counter() - start


def benchmark_checkpoint(
    *,
    checkpoint_path: str | Path,
    inputs: np.ndarray,
    device: str | torch.device,
    input_scale: float = 1.0,
    expected_sample_count: int = 2048,
    warmup_iterations: int = 20,
    measurement_iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark one checkpoint and input batch."""
    validated_warmup = (
        _validate_nonnegative_integer(
            warmup_iterations,
            name="warmup_iterations",
        )
    )
    validated_measurements = (
        _validate_positive_integer(
            measurement_iterations,
            name="measurement_iterations",
        )
    )

    resolved_device = resolve_device(device)
    input_array = np.asarray(inputs)

    if (
        input_array.ndim != 3
        or input_array.shape[0] <= 0
    ):
        raise ValueError(
            "inputs must be a non-empty batch."
        )

    batch_size = int(
        input_array.shape[0]
    )

    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            resolved_device
        )
        _synchronize_device(
            resolved_device
        )

    load_start = perf_counter()

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        input_scale=input_scale,
        expected_sample_count=(
            expected_sample_count
        ),
    )

    _synchronize_device(engine.device)

    model_load_ms = (
        1000.0
        * (
            perf_counter()
            - load_start
        )
    )

    first_inference_ms = (
        1000.0
        * _measure_call(
            engine,
            np.asarray(
                input_array,
                dtype=np.float32,
            ),
        )
    )

    for _ in range(validated_warmup):
        engine.predict_batch(input_array)

    _synchronize_device(engine.device)

    durations = [
        _measure_call(
            engine,
            input_array,
        )
        for _ in range(
            validated_measurements
        )
    ]

    latency = summarize_latencies(
        durations
    )

    if (
        not math.isfinite(latency.mean_ms)
        or latency.mean_ms <= 0.0
    ):
        raise RuntimeError(
            "Measured mean latency is invalid."
        )

    throughput = (
        batch_size
        / (latency.mean_ms / 1000.0)
    )

    cuda_peak_memory = None

    if engine.device.type == "cuda":
        cuda_peak_memory = int(
            torch.cuda.max_memory_allocated(
                engine.device
            )
        )

    return BenchmarkResult(
        device=str(engine.device),
        batch_size=batch_size,
        sample_count=int(
            input_array.shape[2]
        ),
        warmup_iterations=(
            validated_warmup
        ),
        measurement_iterations=(
            validated_measurements
        ),
        model_load_ms=float(
            model_load_ms
        ),
        first_inference_ms=float(
            first_inference_ms
        ),
        steady_state_latency=latency,
        throughput_windows_per_second=float(
            throughput
        ),
        cuda_peak_memory_bytes=(
            cuda_peak_memory
        ),
    )


def benchmark_environment() -> dict[str, Any]:
    """Return relevant runtime environment metadata."""
    cuda_available = (
        torch.cuda.is_available()
    )

    environment: dict[str, Any] = {
        "python_torch_version": (
            torch.__version__
        ),
        "cuda_available": cuda_available,
        "torch_cuda_version": (
            torch.version.cuda
        ),
    }

    if cuda_available:
        environment.update(
            {
                "cuda_device_count": (
                    torch.cuda.device_count()
                ),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                ),
            }
        )

    return environment


__all__ = [
    "BenchmarkResult",
    "LatencySummary",
    "benchmark_checkpoint",
    "benchmark_environment",
    "build_benchmark_batch",
    "summarize_latencies",
]
