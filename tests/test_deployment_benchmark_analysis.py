from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rfsil.deployment.benchmark_analysis import (
    build_benchmark_summary,
    load_benchmark_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "analyze_deployment_benchmark.py"
)


def write_document(
    path: Path,
    *,
    device: str,
    batch_size: int,
    mean_ms: float,
    throughput: float,
    memory: int | None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "device": device,
                        "batch_size": batch_size,
                        "sample_count": 2048,
                        "model_load_ms": 5.0,
                        "first_inference_ms": 10.0,
                        "steady_state_latency": {
                            "mean_ms": mean_ms,
                            "p95_ms": (
                                mean_ms * 1.2
                            ),
                        },
                        "throughput_windows_per_second": (
                            throughput
                        ),
                        "cuda_peak_memory_bytes": (
                            memory
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_loads_and_sorts_points(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path / "cuda.json",
        device="cuda",
        batch_size=8,
        mean_ms=1.0,
        throughput=8000.0,
        memory=16_000_000,
    )
    write_document(
        tmp_path / "cpu.json",
        device="cpu",
        batch_size=1,
        mean_ms=2.0,
        throughput=500.0,
        memory=None,
    )

    points = load_benchmark_points(
        sorted(tmp_path.glob("*.json"))
    )

    assert [
        (
            point.device,
            point.batch_size,
        )
        for point in points
    ] == [
        ("cpu", 1),
        ("cuda", 8),
    ]


def test_rejects_duplicate_cases(
    tmp_path: Path,
) -> None:
    for name in ("first", "second"):
        write_document(
            tmp_path / f"{name}.json",
            device="cpu",
            batch_size=1,
            mean_ms=2.0,
            throughput=500.0,
            memory=None,
        )

    with pytest.raises(
        ValueError,
        match="Duplicate benchmark case",
    ):
        load_benchmark_points(
            sorted(tmp_path.glob("*.json"))
        )


def test_builds_speedup_summary(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path / "cpu.json",
        device="cpu",
        batch_size=8,
        mean_ms=4.0,
        throughput=2000.0,
        memory=None,
    )
    write_document(
        tmp_path / "cuda.json",
        device="cuda",
        batch_size=8,
        mean_ms=1.0,
        throughput=8000.0,
        memory=16_000_000,
    )

    points = load_benchmark_points(
        sorted(tmp_path.glob("*.json"))
    )
    summary = build_benchmark_summary(
        points
    )

    comparison = summary[
        "cpu_cuda_comparisons"
    ][0]

    assert comparison[
        "cuda_throughput_speedup"
    ] == pytest.approx(4.0)
    assert comparison[
        "cuda_batch_latency_ratio"
    ] == pytest.approx(0.25)


def test_rejects_invalid_latency(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path / "invalid.json",
        device="cpu",
        batch_size=1,
        mean_ms=0.0,
        throughput=500.0,
        memory=None,
    )

    with pytest.raises(
        ValueError,
        match="mean_ms",
    ):
        load_benchmark_points(
            [tmp_path / "invalid.json"]
        )


def test_script_creates_analysis_package(
    tmp_path: Path,
) -> None:
    input_directory = (
        tmp_path / "cases"
    )
    input_directory.mkdir()

    write_document(
        input_directory / "cpu.json",
        device="cpu",
        batch_size=1,
        mean_ms=2.0,
        throughput=500.0,
        memory=None,
    )
    write_document(
        input_directory / "cuda.json",
        device="cuda",
        batch_size=1,
        mean_ms=1.0,
        throughput=1000.0,
        memory=12_000_000,
    )

    summary_path = (
        tmp_path / "summary.json"
    )
    latency_path = (
        tmp_path / "latency.png"
    )
    throughput_path = (
        tmp_path / "throughput.png"
    )
    memory_path = (
        tmp_path / "memory.png"
    )

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-directory",
            str(input_directory),
            "--summary-output",
            str(summary_path),
            "--latency-figure",
            str(latency_path),
            "--throughput-figure",
            str(throughput_path),
            "--memory-figure",
            str(memory_path),
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
        latency_path,
        throughput_path,
        memory_path,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["summary"][
        "benchmark_case_count"
    ] == 2
