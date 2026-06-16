from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import add_awgn
from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calculate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Estimate SNR from a known clean signal and its noisy version."""
    noise = noisy - clean
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)

    return float(10.0 * np.log10(signal_power / noise_power))


def main() -> None:
    requested_snr_db = 10.0

    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=2_000,
        samples_per_symbol=8,
        sample_rate_hz=1_000_000.0,
        seed=42,
    )

    noisy_samples = add_awgn(
        signal.samples,
        snr_db=requested_snr_db,
        seed=123,
    )

    measured_snr_db = calculate_snr_db(signal.samples, noisy_samples)

    symbol_indices = np.arange(0, len(signal.samples), signal.samples_per_symbol)
    clean_symbols = signal.samples[symbol_indices]
    noisy_symbols = noisy_samples[symbol_indices]

    output_path = PROJECT_ROOT / "reports" / "figures" / "awgn_constellation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(
        clean_symbols.real,
        clean_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[0].set_title("Clean QPSK")
    axes[0].set_xlabel("In-phase (I)")
    axes[0].set_ylabel("Quadrature (Q)")

    axes[1].scatter(
        noisy_symbols.real,
        noisy_symbols.imag,
        s=10,
        alpha=0.45,
    )
    axes[1].set_title(f"QPSK with AWGN at {requested_snr_db:.1f} dB")
    axes[1].set_xlabel("In-phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")

    for axis in axes:
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.7, 1.7)
        axis.set_ylim(-1.7, 1.7)

    figure.suptitle("Controlled AWGN Impairment")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Requested SNR dB: {requested_snr_db:.3f}")
    print(f"Measured SNR dB: {measured_snr_db:.3f}")
    print(f"Number of IQ samples: {len(signal.samples)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
