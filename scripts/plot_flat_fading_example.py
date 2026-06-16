from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import apply_flat_rayleigh_fading
from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    seed = 42

    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=2_000,
        samples_per_symbol=8,
        sample_rate_hz=1_000_000.0,
        seed=123,
    )

    faded_samples = apply_flat_rayleigh_fading(
        signal.samples,
        seed=seed,
    )

    symbol_indices = np.arange(
        0,
        len(signal.samples),
        signal.samples_per_symbol,
    )

    clean_symbols = signal.samples[symbol_indices]
    faded_symbols = faded_samples[symbol_indices]

    nonzero_index = int(np.flatnonzero(signal.samples)[0])
    channel_coefficient = (
        faded_samples[nonzero_index]
        / signal.samples[nonzero_index]
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "flat_rayleigh_fading_constellation.png"
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
        faded_symbols.real,
        faded_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[1].set_title("QPSK with Flat Rayleigh Fading")

    axis_limit = max(
        1.2,
        float(np.max(np.abs(faded_symbols.real))) * 1.3,
        float(np.max(np.abs(faded_symbols.imag))) * 1.3,
    )

    for axis in axes:
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)

    figure.suptitle("Flat Rayleigh Block Fading")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    input_power = float(np.mean(np.abs(signal.samples) ** 2))
    output_power = float(np.mean(np.abs(faded_samples) ** 2))
    measured_power_ratio = output_power / input_power
    expected_power_ratio = float(np.abs(channel_coefficient) ** 2)
    phase_degrees = float(np.rad2deg(np.angle(channel_coefficient)))

    print(f"Channel coefficient: {channel_coefficient}")
    print(f"Channel magnitude: {abs(channel_coefficient):.6f}")
    print(f"Channel phase degrees: {phase_degrees:.6f}")
    print(f"Expected power ratio: {expected_power_ratio:.6f}")
    print(f"Measured power ratio: {measured_power_ratio:.6f}")
    print(f"Number of IQ samples: {len(signal.samples)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
