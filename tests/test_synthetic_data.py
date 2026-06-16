from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rfsil.data.synthetic import (
    MODULATION_CLASSES,
    MODULATION_TO_LABEL,
    SyntheticExampleConfig,
    generate_synthetic_example,
)
from rfsil.dsp.modulations import Modulation


def test_synthetic_example_has_fixed_shape_dtype_and_metadata() -> None:
    configuration = SyntheticExampleConfig(
        sample_count=2_048,
        snr_db=12.0,
    )

    example = generate_synthetic_example(
        modulation=Modulation.QPSK,
        configuration=configuration,
        seed=42,
    )

    assert example.samples.shape == (2_048,)
    assert example.samples.dtype == np.complex64
    assert example.modulation == Modulation.QPSK
    assert example.label == MODULATION_TO_LABEL[Modulation.QPSK]
    assert example.configuration == configuration
    assert example.seed == 42
    assert 0 <= example.symbol_sample_offset < configuration.samples_per_symbol
    assert np.all(np.isfinite(example.samples))


@pytest.mark.parametrize(
    ("modulation", "expected_label"),
    [
        (Modulation.BPSK, 0),
        (Modulation.QPSK, 1),
        (Modulation.PSK8, 2),
        (Modulation.QAM16, 3),
    ],
)
def test_modulation_labels_are_stable(
    modulation: Modulation,
    expected_label: int,
) -> None:
    configuration = SyntheticExampleConfig(
        sample_count=512,
        snr_db=None,
    )

    example = generate_synthetic_example(
        modulation=modulation,
        configuration=configuration,
        seed=1,
    )

    assert example.label == expected_label
    assert MODULATION_CLASSES[expected_label] == modulation


def test_synthetic_example_is_reproducible() -> None:
    configuration = SyntheticExampleConfig(
        sample_count=1_024,
        snr_db=8.0,
        frequency_offset_hz=12_000.0,
        phase_offset_rad=0.3,
        amplitude_scale=0.8,
        time_shift_samples=5,
        apply_rayleigh_fading=True,
    )

    example_a = generate_synthetic_example(
        modulation="16qam",
        configuration=configuration,
        seed=123,
    )
    example_b = generate_synthetic_example(
        modulation="16qam",
        configuration=configuration,
        seed=123,
    )

    np.testing.assert_array_equal(
        example_a.samples,
        example_b.samples,
    )
    assert example_a.symbol_sample_offset == example_b.symbol_sample_offset


def test_different_seeds_produce_different_examples() -> None:
    configuration = SyntheticExampleConfig(
        sample_count=1_024,
        snr_db=10.0,
    )

    example_a = generate_synthetic_example(
        "8psk",
        configuration=configuration,
        seed=1,
    )
    example_b = generate_synthetic_example(
        "8psk",
        configuration=configuration,
        seed=2,
    )

    assert not np.array_equal(
        example_a.samples,
        example_b.samples,
    )


def test_configured_awgn_matches_requested_snr() -> None:
    clean_configuration = SyntheticExampleConfig(
        sample_count=16_384,
        snr_db=None,
    )
    noisy_configuration = replace(
        clean_configuration,
        snr_db=10.0,
    )

    clean = generate_synthetic_example(
        "qpsk",
        configuration=clean_configuration,
        seed=77,
    )
    noisy = generate_synthetic_example(
        "qpsk",
        configuration=noisy_configuration,
        seed=77,
    )

    noise = noisy.samples - clean.samples
    signal_power = float(np.mean(np.abs(clean.samples) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    measured_snr_db = 10.0 * np.log10(
        signal_power / noise_power
    )

    assert measured_snr_db == pytest.approx(
        10.0,
        abs=0.2,
    )


def test_positive_time_shift_creates_leading_zeros_without_noise() -> None:
    configuration = SyntheticExampleConfig(
        sample_count=1_024,
        snr_db=None,
        time_shift_samples=11,
    )

    example = generate_synthetic_example(
        "bpsk",
        configuration=configuration,
        seed=42,
    )

    np.testing.assert_array_equal(
        example.samples[:11],
        np.zeros(11, dtype=np.complex64),
    )


@pytest.mark.parametrize(
    "invalid_sample_count",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_synthetic_example_rejects_invalid_sample_count(
    invalid_sample_count: object,
) -> None:
    configuration = SyntheticExampleConfig(
        sample_count=invalid_sample_count,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        generate_synthetic_example(
            "qpsk",
            configuration=configuration,
            seed=42,
        )
