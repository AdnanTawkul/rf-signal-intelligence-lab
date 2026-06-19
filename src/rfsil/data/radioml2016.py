from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.data.dataset import (
    SyntheticDatasetSplit,
)
from rfsil.data.synthetic import (
    MODULATION_TO_LABEL,
)
from rfsil.dsp.modulations import Modulation

Float32Array = NDArray[np.float32]

RADIOML_TO_PROJECT: tuple[
    tuple[str, Modulation],
    ...,
] = (
    ("BPSK", Modulation.BPSK),
    ("QPSK", Modulation.QPSK),
    ("8PSK", Modulation.PSK8),
    ("QAM16", Modulation.QAM16),
)


class RestrictedRadioMLUnpickler(
    pickle.Unpickler
):
    """Load only the NumPy objects expected in RadioML 2016.10A."""

    allowed_globals = {
        (
            "numpy.core.multiarray",
            "_reconstruct",
        ),
        (
            "numpy._core.multiarray",
            "_reconstruct",
        ),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
    }

    def find_class(
        self,
        module: str,
        name: str,
    ) -> Any:
        """Reject globals outside the expected NumPy pickle format."""
        if (
            module,
            name,
        ) not in self.allowed_globals:
            raise pickle.UnpicklingError(
                "Blocked unsupported pickle global: "
                f"{module}.{name}"
            )

        return super().find_class(
            module,
            name,
        )


def load_radioml2016_dictionary(
    input_path: str | Path,
) -> dict[tuple[str, int], Float32Array]:
    """Load and validate a RadioML 2016.10A dictionary."""
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as file:
        content = RestrictedRadioMLUnpickler(
            file,
            fix_imports=True,
            encoding="latin1",
        ).load()

    if not isinstance(content, Mapping):
        raise TypeError(
            "RadioML content must be a mapping."
        )

    normalized: dict[
        tuple[str, int],
        Float32Array,
    ] = {}

    for raw_key, raw_value in content.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
        ):
            raise ValueError(
                "RadioML keys must be "
                "(modulation, SNR) tuples."
            )

        raw_modulation, raw_snr = raw_key

        if isinstance(raw_modulation, bytes):
            modulation = raw_modulation.decode(
                "latin1"
            )
        elif isinstance(raw_modulation, str):
            modulation = raw_modulation
        else:
            raise TypeError(
                "RadioML modulation names must "
                "be strings or bytes."
            )

        if isinstance(raw_snr, bool):
            raise TypeError(
                "RadioML SNR values must be integers."
            )

        try:
            snr = int(raw_snr)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "RadioML SNR values must be integers."
            ) from error

        if not isinstance(raw_value, np.ndarray):
            raise TypeError(
                f"Group {(modulation, snr)!r} "
                "must contain a NumPy array."
            )

        if raw_value.ndim != 3:
            raise ValueError(
                f"Group {(modulation, snr)!r} "
                "must have shape "
                "[examples, 2, samples]."
            )

        if raw_value.shape[1] != 2:
            raise ValueError(
                f"Group {(modulation, snr)!r} "
                "must contain two IQ channels."
            )

        if raw_value.dtype != np.float32:
            raise ValueError(
                f"Group {(modulation, snr)!r} "
                "must use float32."
            )

        if not np.all(np.isfinite(raw_value)):
            raise ValueError(
                f"Group {(modulation, snr)!r} "
                "contains non-finite values."
            )

        normalized[
            (modulation, snr)
        ] = np.ascontiguousarray(raw_value)

    if not normalized:
        raise ValueError(
            "RadioML dictionary must not be empty."
        )

    return normalized


def _validate_split_counts(
    split_counts: Mapping[str, int],
) -> dict[str, int]:
    expected_names = {
        "train",
        "validation",
        "test",
    }

    if set(split_counts) != expected_names:
        raise ValueError(
            "split_counts must contain exactly "
            "train, validation, and test."
        )

    validated: dict[str, int] = {}

    for name in (
        "train",
        "validation",
        "test",
    ):
        value = split_counts[name]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"split_counts[{name!r}] "
                "must be a positive integer."
            )

        validated[name] = value

    return validated


def _neutral_metadata(
    example_count: int,
    compatibility_seeds: NDArray[np.uint32],
) -> dict[str, np.ndarray]:
    return {
        "frequency_offset_hz": np.zeros(
            example_count,
            dtype=np.float32,
        ),
        "phase_offset_rad": np.zeros(
            example_count,
            dtype=np.float32,
        ),
        "amplitude_scale": np.ones(
            example_count,
            dtype=np.float32,
        ),
        "time_shift_samples": np.zeros(
            example_count,
            dtype=np.int32,
        ),
        "rayleigh_fading": np.zeros(
            example_count,
            dtype=np.bool_,
        ),
        "example_seed": compatibility_seeds,
    }


def build_radioml2016_four_class_splits(
    groups: Mapping[
        tuple[str, int],
        Float32Array,
    ],
    *,
    split_counts: Mapping[str, int],
    seed: int,
) -> tuple[
    dict[str, SyntheticDatasetSplit],
    tuple[int, ...],
]:
    """Create deterministic balanced four-class RadioML splits."""
    counts = _validate_split_counts(
        split_counts
    )

    required_source_names = tuple(
        source_name
        for source_name, _ in RADIOML_TO_PROJECT
    )

    snr_sets = {
        source_name: {
            snr
            for modulation, snr in groups
            if modulation == source_name
        }
        for source_name in required_source_names
    }

    for source_name, snr_values in snr_sets.items():
        if not snr_values:
            raise ValueError(
                f"Missing RadioML class "
                f"{source_name!r}."
            )

    reference_snr_values = next(
        iter(snr_sets.values())
    )

    for source_name, snr_values in snr_sets.items():
        if snr_values != reference_snr_values:
            raise ValueError(
                "All selected RadioML classes "
                "must contain identical SNR values. "
                f"Mismatch found for {source_name!r}."
            )

    ordered_snr_values = tuple(
        sorted(reference_snr_values)
    )

    total_per_group = sum(counts.values())

    iq_chunks: dict[
        str,
        list[Float32Array],
    ] = {
        name: []
        for name in counts
    }
    label_chunks: dict[
        str,
        list[NDArray[np.int64]],
    ] = {
        name: []
        for name in counts
    }
    snr_chunks: dict[
        str,
        list[NDArray[np.float32]],
    ] = {
        name: []
        for name in counts
    }
    seed_chunks: dict[
        str,
        list[NDArray[np.uint32]],
    ] = {
        name: []
        for name in counts
    }

    reference_shape: tuple[int, int, int] | None = None
    group_index = 0

    for source_name, project_modulation in (
        RADIOML_TO_PROJECT
    ):
        label = MODULATION_TO_LABEL[
            project_modulation
        ]

        for snr in ordered_snr_values:
            key = (
                source_name,
                snr,
            )

            if key not in groups:
                raise ValueError(
                    f"Missing RadioML group {key!r}."
                )

            group = groups[key]

            if reference_shape is None:
                reference_shape = tuple(
                    int(value)
                    for value in group.shape
                )
            elif group.shape != reference_shape:
                raise ValueError(
                    "All selected RadioML groups "
                    "must use an identical shape."
                )

            if group.shape[0] != total_per_group:
                raise ValueError(
                    f"Group {key!r} contains "
                    f"{group.shape[0]} examples, "
                    f"but split counts require "
                    f"{total_per_group}."
                )

            group_generator = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(seed),
                        int(group_index),
                    ]
                )
            )

            permutation = group_generator.permutation(
                group.shape[0]
            )
            compatibility_seeds = (
                group_generator.integers(
                    low=0,
                    high=np.iinfo(np.uint32).max,
                    size=group.shape[0],
                    dtype=np.uint32,
                )
            )

            start = 0

            for split_name in (
                "train",
                "validation",
                "test",
            ):
                stop = (
                    start
                    + counts[split_name]
                )
                selected = permutation[
                    start:stop
                ]

                iq_chunks[split_name].append(
                    np.ascontiguousarray(
                        group[selected]
                    )
                )
                label_chunks[split_name].append(
                    np.full(
                        selected.size,
                        label,
                        dtype=np.int64,
                    )
                )
                snr_chunks[split_name].append(
                    np.full(
                        selected.size,
                        float(snr),
                        dtype=np.float32,
                    )
                )
                seed_chunks[split_name].append(
                    compatibility_seeds[selected]
                )

                start = stop

            group_index += 1

    result: dict[
        str,
        SyntheticDatasetSplit,
    ] = {}

    for split_index, split_name in enumerate(
        (
            "train",
            "validation",
            "test",
        )
    ):
        iq = np.concatenate(
            iq_chunks[split_name],
            axis=0,
        )
        labels = np.concatenate(
            label_chunks[split_name],
            axis=0,
        )
        snr_db = np.concatenate(
            snr_chunks[split_name],
            axis=0,
        )
        example_seed = np.concatenate(
            seed_chunks[split_name],
            axis=0,
        )

        split_generator = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    0x524D4C,
                    int(split_index),
                ]
            )
        )
        permutation = split_generator.permutation(
            iq.shape[0]
        )

        iq = np.ascontiguousarray(
            iq[permutation]
        )
        labels = np.ascontiguousarray(
            labels[permutation]
        )
        snr_db = np.ascontiguousarray(
            snr_db[permutation]
        )
        example_seed = np.ascontiguousarray(
            example_seed[permutation]
        )

        metadata = _neutral_metadata(
            iq.shape[0],
            example_seed,
        )

        result[split_name] = SyntheticDatasetSplit(
            iq=iq,
            labels=labels,
            snr_db=snr_db,
            frequency_offset_hz=metadata[
                "frequency_offset_hz"
            ],
            phase_offset_rad=metadata[
                "phase_offset_rad"
            ],
            amplitude_scale=metadata[
                "amplitude_scale"
            ],
            time_shift_samples=metadata[
                "time_shift_samples"
            ],
            rayleigh_fading=metadata[
                "rayleigh_fading"
            ],
            example_seed=metadata[
                "example_seed"
            ],
        )

    return result, ordered_snr_values


__all__ = [
    "RADIOML_TO_PROJECT",
    "RestrictedRadioMLUnpickler",
    "build_radioml2016_four_class_splits",
    "load_radioml2016_dictionary",
]
