from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.spectral import compute_spectrogram


def test_compute_spectrogram_shape_and_dtype() -> None:
    samples = np.ones(1_024, dtype=np.complex64)

    result = compute_spectrogram(
        samples,
        sample_rate_hz=1_000_000.0,
        window_size=256,
        hop_size=64,
        fft_size=512,
    )

    expected_frame_count = 1 + (1_024 - 256) // 64

    assert result.frequencies_hz.shape == (512,)
    assert result.times_s.shape == (expected_frame_count,)
    assert result.power_db.shape == (512, expected_frame_count)
    assert result.frequencies_hz.dtype == np.float64
    assert result.times_s.dtype == np.float64
    assert result.power_db.dtype == np.float32


def test_compute_spectrogram_detects_known_complex_tone() -> None:
    sample_rate_hz = 1_024_000.0
    tone_frequency_hz = 128_000.0
    sample_count = 4_096

    time_s = np.arange(sample_count) / sample_rate_hz
    samples = np.exp(
        1j * 2.0 * np.pi * tone_frequency_hz * time_s
    ).astype(np.complex64)

    result = compute_spectrogram(
        samples,
        sample_rate_hz=sample_rate_hz,
        window_size=256,
        hop_size=128,
        fft_size=256,
    )

    mean_spectrum_db = np.mean(result.power_db, axis=1)
    peak_frequency_hz = result.frequencies_hz[
        int(np.argmax(mean_spectrum_db))
    ]

    assert peak_frequency_hz == pytest.approx(
        tone_frequency_hz,
        abs=1e-6,
    )


def test_spectrogram_peak_is_normalized_to_zero_db() -> None:
    samples = np.ones(1_024, dtype=np.complex64)

    result = compute_spectrogram(
        samples,
        sample_rate_hz=1_000_000.0,
    )

    assert float(np.max(result.power_db)) == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_spectrogram_time_axis_uses_frame_centers() -> None:
    samples = np.ones(1_024, dtype=np.complex64)
    sample_rate_hz = 1_000.0
    window_size = 100
    hop_size = 25

    result = compute_spectrogram(
        samples,
        sample_rate_hz=sample_rate_hz,
        window_size=window_size,
        hop_size=hop_size,
    )

    expected_first_time_s = (window_size - 1) / (2.0 * sample_rate_hz)
    expected_second_time_s = (
        hop_size + (window_size - 1) / 2.0
    ) / sample_rate_hz

    assert result.times_s[0] == pytest.approx(expected_first_time_s)
    assert result.times_s[1] == pytest.approx(expected_second_time_s)


@pytest.mark.parametrize(
    "invalid_samples",
    [
        np.array([], dtype=np.complex64),
        np.zeros(256, dtype=np.complex64),
        np.ones((16, 16), dtype=np.complex64),
        np.array(
            [1.0 + 0.0j, complex(float("nan"), 0.0)],
            dtype=np.complex64,
        ),
    ],
)
def test_compute_spectrogram_rejects_invalid_samples(
    invalid_samples: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        compute_spectrogram(
            invalid_samples,
            sample_rate_hz=1_000_000.0,
        )


@pytest.mark.parametrize(
    "invalid_sample_rate",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_compute_spectrogram_rejects_invalid_sample_rate(
    invalid_sample_rate: float,
) -> None:
    samples = np.ones(512, dtype=np.complex64)

    with pytest.raises(ValueError):
        compute_spectrogram(
            samples,
            sample_rate_hz=invalid_sample_rate,
        )


@pytest.mark.parametrize(
    "invalid_configuration",
    [
        {"window_size": 0},
        {"window_size": 1.5},
        {"window_size": True},
        {"window_size": 1_024},
        {"window_size": 128, "hop_size": 0},
        {"window_size": 128, "hop_size": 256},
        {"window_size": 128, "fft_size": 64},
        {"dynamic_range_db": 0.0},
    ],
)
def test_compute_spectrogram_rejects_invalid_configuration(
    invalid_configuration: dict[str, object],
) -> None:
    samples = np.ones(512, dtype=np.complex64)

    with pytest.raises(ValueError):
        compute_spectrogram(
            samples,
            sample_rate_hz=1_000_000.0,
            **invalid_configuration,  # type: ignore[arg-type]
        )
