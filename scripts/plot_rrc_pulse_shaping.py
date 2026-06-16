from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.modulations import Modulation, generate_iq_signal
from rfsil.dsp.pulse_shaping import apply_root_raised_cosine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalized_spectrum_db(
    samples: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    """Return a normalized, shifted magnitude spectrum in decibels."""
    spectrum = np.fft.fftshift(
        np.fft.fft(samples, n=fft_size)
    )
    magnitude = np.abs(spectrum)
    reference = float(np.max(magnitude))
    tiny = np.finfo(np.float64).tiny

    return 20.0 * np.log10(
        np.maximum(magnitude, tiny) / reference
    )


def normalize_average_power(samples: np.ndarray) -> np.ndarray:
    """Normalize a complex waveform to unit average power."""
    power = float(np.mean(np.abs(samples) ** 2))

    return samples / np.sqrt(power)


def main() -> None:
    samples_per_symbol = 8
    rolloff = 0.35
    span_symbols = 8
    sample_rate_hz = 1_000_000.0

    symbol_signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=512,
        samples_per_symbol=1,
        sample_rate_hz=sample_rate_hz,
        seed=42,
    )

    rectangular_samples = np.repeat(
        symbol_signal.samples,
        samples_per_symbol,
    ).astype(np.complex64)

    shaped = apply_root_raised_cosine(
        symbol_signal.samples,
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        span_symbols=span_symbols,
    )

    aligned_rrc_samples = shaped.samples[
        shaped.group_delay_samples:
        shaped.group_delay_samples + len(rectangular_samples)
    ]

    rectangular_display = normalize_average_power(
        rectangular_samples,
    )
    rrc_display = normalize_average_power(
        aligned_rrc_samples,
    )

    fft_size = 16_384
    frequency_axis_khz = (
        np.fft.fftshift(
            np.fft.fftfreq(
                fft_size,
                d=1.0 / sample_rate_hz,
            )
        )
        / 1_000.0
    )

    rectangular_spectrum_db = normalized_spectrum_db(
        rectangular_display,
        fft_size=fft_size,
    )
    rrc_spectrum_db = normalized_spectrum_db(
        rrc_display,
        fft_size=fft_size,
    )

    samples_to_plot = 160
    sample_indices = np.arange(samples_to_plot)

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "rrc_pulse_shaping.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(
        sample_indices,
        rectangular_display.real[:samples_to_plot],
        label="Rectangular",
    )
    axes[0].plot(
        sample_indices,
        rrc_display.real[:samples_to_plot],
        label="RRC shaped",
    )
    axes[0].set_title("QPSK In-Phase Waveform")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        frequency_axis_khz,
        rectangular_spectrum_db,
        label="Rectangular",
    )
    axes[1].plot(
        frequency_axis_khz,
        rrc_spectrum_db,
        label=f"RRC, rolloff={rolloff:.2f}",
    )
    axes[1].set_title("Normalized Magnitude Spectrum")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Relative magnitude (dB)")
    axes[1].set_xlim(-300.0, 300.0)
    axes[1].set_ylim(-80.0, 5.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("Root-Raised-Cosine Pulse Shaping")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Modulation: {symbol_signal.modulation}")
    print(f"Symbols: {len(symbol_signal.symbols)}")
    print(f"Samples per symbol: {samples_per_symbol}")
    print(f"Rolloff: {rolloff:.2f}")
    print(f"Filter span symbols: {span_symbols}")
    print(f"Filter taps: {len(shaped.taps)}")
    print(f"Group delay samples: {shaped.group_delay_samples}")
    print(f"Rectangular samples: {len(rectangular_samples)}")
    print(f"Full RRC samples: {len(shaped.samples)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
