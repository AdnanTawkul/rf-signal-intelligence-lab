from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExternalSeedSweep:
    """Validated metrics from one external evaluation seed sweep."""

    name: str
    class_names: tuple[str, ...]
    seeds: tuple[int, ...]
    snr_values_db: tuple[float, ...]
    overall_accuracy: Float64Array
    class_accuracy: Float64Array
    snr_accuracy: Float64Array


@dataclass(frozen=True, slots=True)
class AccuracySummary:
    """Mean and spread across training seeds."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    per_seed: Float64Array


@dataclass(frozen=True, slots=True)
class PairedAccuracyChange:
    """Candidate-minus-reference accuracy changes by seed."""

    seeds: tuple[int, ...]
    mean: float
    standard_deviation: float
    improved_seed_count: int
    per_seed: Float64Array


def _load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Aggregate metrics must contain "
            "a JSON mapping."
        )

    return content


def _validate_accuracy(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite accuracy."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite accuracy."
        ) from error

    if (
        not np.isfinite(result)
        or result < 0.0
        or result > 1.0
    ):
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return result


def load_external_seed_sweep(
    path: str | Path,
    *,
    name: str | None = None,
) -> ExternalSeedSweep:
    """Load and validate one aggregate_metrics.json file."""
    resolved_path = Path(path)
    content = _load_json_mapping(
        resolved_path
    )

    raw_class_names = content.get(
        "class_names"
    )

    if (
        not isinstance(raw_class_names, list)
        or not raw_class_names
        or not all(
            isinstance(value, str)
            and value.strip()
            for value in raw_class_names
        )
    ):
        raise ValueError(
            "class_names must be a non-empty "
            "list of strings."
        )

    class_names = tuple(
        value.strip()
        for value in raw_class_names
    )

    raw_runs = content.get("runs")

    if (
        not isinstance(raw_runs, list)
        or not raw_runs
    ):
        raise ValueError(
            "runs must be a non-empty list."
        )

    parsed_runs: list[
        tuple[
            int,
            float,
            list[float],
            dict[float, float],
        ]
    ] = []

    observed_seeds: set[int] = set()
    reference_snr_values: tuple[
        float,
        ...,
    ] | None = None

    for run_index, raw_run in enumerate(
        raw_runs
    ):
        if not isinstance(raw_run, dict):
            raise ValueError(
                f"runs[{run_index}] must "
                "be a mapping."
            )

        raw_seed = raw_run.get("seed")

        if (
            isinstance(raw_seed, bool)
            or not isinstance(raw_seed, int)
        ):
            raise ValueError(
                f"runs[{run_index}].seed "
                "must be an integer."
            )

        seed = int(raw_seed)

        if seed in observed_seeds:
            raise ValueError(
                f"Duplicate seed: {seed}."
            )

        observed_seeds.add(seed)

        overall_accuracy = (
            _validate_accuracy(
                raw_run.get(
                    "overall_accuracy"
                ),
                (
                    f"runs[{run_index}]"
                    ".overall_accuracy"
                ),
            )
        )

        raw_class_accuracy = raw_run.get(
            "class_accuracy"
        )

        if not isinstance(
            raw_class_accuracy,
            dict,
        ):
            raise ValueError(
                f"runs[{run_index}]"
                ".class_accuracy must "
                "be a mapping."
            )

        if set(raw_class_accuracy) != set(
            class_names
        ):
            raise ValueError(
                f"runs[{run_index}] class "
                "names do not match."
            )

        class_values = [
            _validate_accuracy(
                raw_class_accuracy[
                    class_name
                ],
                (
                    f"runs[{run_index}]"
                    f".class_accuracy."
                    f"{class_name}"
                ),
            )
            for class_name in class_names
        ]

        raw_snr_accuracy = raw_run.get(
            "accuracy_by_snr"
        )

        if not isinstance(
            raw_snr_accuracy,
            dict,
        ):
            raise ValueError(
                f"runs[{run_index}]"
                ".accuracy_by_snr must "
                "be a mapping."
            )

        parsed_snr: dict[float, float] = {}

        for raw_snr, raw_accuracy in (
            raw_snr_accuracy.items()
        ):
            try:
                snr = float(raw_snr)
            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "SNR mapping keys must "
                    "be numeric."
                ) from error

            if not np.isfinite(snr):
                raise ValueError(
                    "SNR values must be finite."
                )

            if snr in parsed_snr:
                raise ValueError(
                    f"Duplicate SNR value: {snr}."
                )

            parsed_snr[snr] = (
                _validate_accuracy(
                    raw_accuracy,
                    (
                        f"runs[{run_index}]"
                        f".accuracy_by_snr."
                        f"{raw_snr}"
                    ),
                )
            )

        ordered_snr_values = tuple(
            sorted(parsed_snr)
        )

        if not ordered_snr_values:
            raise ValueError(
                "accuracy_by_snr must "
                "not be empty."
            )

        if reference_snr_values is None:
            reference_snr_values = (
                ordered_snr_values
            )
        elif (
            ordered_snr_values
            != reference_snr_values
        ):
            raise ValueError(
                "All runs must contain "
                "identical SNR values."
            )

        parsed_runs.append(
            (
                seed,
                overall_accuracy,
                class_values,
                parsed_snr,
            )
        )

    if reference_snr_values is None:
        raise RuntimeError(
            "No SNR values were loaded."
        )

    parsed_runs.sort(
        key=lambda value: value[0]
    )

    seeds = tuple(
        run[0]
        for run in parsed_runs
    )

    overall = np.asarray(
        [
            run[1]
            for run in parsed_runs
        ],
        dtype=np.float64,
    )
    class_accuracy = np.asarray(
        [
            run[2]
            for run in parsed_runs
        ],
        dtype=np.float64,
    )
    snr_accuracy = np.asarray(
        [
            [
                run[3][snr]
                for snr in (
                    reference_snr_values
                )
            ]
            for run in parsed_runs
        ],
        dtype=np.float64,
    )

    selected_name = (
        name
        if name is not None
        else str(
            content.get(
                "experiment_name",
                resolved_path.parent.name,
            )
        )
    )

    return ExternalSeedSweep(
        name=selected_name,
        class_names=class_names,
        seeds=seeds,
        snr_values_db=(
            reference_snr_values
        ),
        overall_accuracy=overall,
        class_accuracy=class_accuracy,
        snr_accuracy=snr_accuracy,
    )


def summarize_accuracy(
    sweep: ExternalSeedSweep,
    *,
    snr_values_db: tuple[
        float,
        ...,
    ] | None = None,
) -> AccuracySummary:
    """Summarize overall or selected-SNR accuracy."""
    if snr_values_db is None:
        values = sweep.overall_accuracy
    else:
        if not snr_values_db:
            raise ValueError(
                "snr_values_db must "
                "not be empty."
            )

        lookup = {
            value: index
            for index, value in enumerate(
                sweep.snr_values_db
            )
        }

        missing = [
            float(value)
            for value in snr_values_db
            if float(value) not in lookup
        ]

        if missing:
            raise ValueError(
                "Requested SNR values are "
                f"missing: {missing}."
            )

        indices = [
            lookup[float(value)]
            for value in snr_values_db
        ]

        values = np.mean(
            sweep.snr_accuracy[
                :,
                indices,
            ],
            axis=1,
        )

    selected = np.asarray(
        values,
        dtype=np.float64,
    )

    return AccuracySummary(
        mean=float(np.mean(selected)),
        standard_deviation=float(
            np.std(selected)
        ),
        minimum=float(np.min(selected)),
        maximum=float(np.max(selected)),
        per_seed=selected,
    )


def summarize_class_accuracy(
    sweep: ExternalSeedSweep,
) -> dict[
    str,
    AccuracySummary,
]:
    """Summarize every class over training seeds."""
    return {
        class_name: AccuracySummary(
            mean=float(
                np.mean(
                    sweep.class_accuracy[
                        :,
                        class_index,
                    ]
                )
            ),
            standard_deviation=float(
                np.std(
                    sweep.class_accuracy[
                        :,
                        class_index,
                    ]
                )
            ),
            minimum=float(
                np.min(
                    sweep.class_accuracy[
                        :,
                        class_index,
                    ]
                )
            ),
            maximum=float(
                np.max(
                    sweep.class_accuracy[
                        :,
                        class_index,
                    ]
                )
            ),
            per_seed=np.asarray(
                sweep.class_accuracy[
                    :,
                    class_index,
                ],
                dtype=np.float64,
            ),
        )
        for class_index, class_name
        in enumerate(sweep.class_names)
    }


def compute_paired_accuracy_change(
    reference: ExternalSeedSweep,
    candidate: ExternalSeedSweep,
    *,
    snr_values_db: tuple[
        float,
        ...,
    ] | None = None,
) -> PairedAccuracyChange:
    """Calculate candidate-minus-reference paired changes."""
    if reference.seeds != candidate.seeds:
        raise ValueError(
            "Paired sweeps must use "
            "identical seeds."
        )

    if (
        reference.class_names
        != candidate.class_names
    ):
        raise ValueError(
            "Paired sweeps must use "
            "identical class names."
        )

    reference_summary = summarize_accuracy(
        reference,
        snr_values_db=snr_values_db,
    )
    candidate_summary = summarize_accuracy(
        candidate,
        snr_values_db=snr_values_db,
    )

    differences = (
        candidate_summary.per_seed
        - reference_summary.per_seed
    )

    return PairedAccuracyChange(
        seeds=reference.seeds,
        mean=float(
            np.mean(differences)
        ),
        standard_deviation=float(
            np.std(differences)
        ),
        improved_seed_count=int(
            np.sum(differences > 0.0)
        ),
        per_seed=np.asarray(
            differences,
            dtype=np.float64,
        ),
    )


__all__ = [
    "AccuracySummary",
    "ExternalSeedSweep",
    "PairedAccuracyChange",
    "compute_paired_accuracy_change",
    "load_external_seed_sweep",
    "summarize_accuracy",
    "summarize_class_accuracy",
]
