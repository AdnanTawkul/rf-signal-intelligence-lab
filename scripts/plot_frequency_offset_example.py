from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import apply_frequency_offset
from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sample_rate_hz = 1_000_000.0
    frequency_offset_hz = 20_000.0

    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=2_000,
        samples_per_symbol=8,
        sample_rate_hz=sample_rate_hz,
        seed=42,
    )

    shifted_samples = apply_frequency_offset(
        signal.samples,
        frequency_offset_hz=frequency_offset_hz,
        sample_rate_hz=sample_rate_hz,
    )

    symbol_indices = np.arange(
        0,
        len(signal.samples),
        signal.samples_per_symbol,
    )

    clean_symbols = signal.samples[symbol_indices]
    shifted_symbols = shifted_samples[symbol_indices]

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "frequency_offset_constellation.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(
        clean_symbols.real,
        clean_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[0].set_title("Clean QPSK")

    axes[1].scatter(
        shifted_symbols.real,
        shifted_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[1].set_title(f"QPSK with {frequency_offset_hz / 1_000:.1f} kHz Offset")

    for axis in axes:
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)

    figure.suptitle("Carrier Frequency Offset")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    magnitude_error = float(
        np.max(np.abs(np.abs(shifted_samples) - np.abs(signal.samples)))
    )

    print(f"Frequency offset Hz: {frequency_offset_hz:.1f}")
    print(f"Sample rate Hz: {sample_rate_hz:.1f}")
    print(f"Number of IQ samples: {len(signal.samples)}")
    print(f"Maximum magnitude error: {magnitude_error:.8f}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
