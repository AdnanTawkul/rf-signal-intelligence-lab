from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import apply_amplitude_scaling
from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    amplitude_scale = 0.55

    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=2_000,
        samples_per_symbol=8,
        sample_rate_hz=1_000_000.0,
        seed=42,
    )

    scaled_samples = apply_amplitude_scaling(
        signal.samples,
        amplitude_scale=amplitude_scale,
    )

    symbol_indices = np.arange(
        0,
        len(signal.samples),
        signal.samples_per_symbol,
    )

    clean_symbols = signal.samples[symbol_indices]
    scaled_symbols = scaled_samples[symbol_indices]

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "amplitude_scaling_constellation.png"
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
        scaled_symbols.real,
        scaled_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[1].set_title(f"QPSK with Amplitude Scale {amplitude_scale:.2f}")

    for axis in axes:
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)

    figure.suptitle("Linear Amplitude Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    original_power = float(np.mean(np.abs(signal.samples) ** 2))
    scaled_power = float(np.mean(np.abs(scaled_samples) ** 2))
    measured_power_ratio = scaled_power / original_power
    expected_power_ratio = amplitude_scale**2

    print(f"Amplitude scale: {amplitude_scale:.3f}")
    print(f"Expected power ratio: {expected_power_ratio:.6f}")
    print(f"Measured power ratio: {measured_power_ratio:.6f}")
    print(f"Number of IQ samples: {len(signal.samples)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
