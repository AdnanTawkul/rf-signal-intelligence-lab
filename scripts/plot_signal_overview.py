from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import add_awgn
from rfsil.dsp.modulations import Modulation, generate_iq_signal
from rfsil.dsp.spectral import compute_spectrogram

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sample_rate_hz = 1_000_000.0
    snr_db = 14.0

    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=4_096,
        samples_per_symbol=8,
        sample_rate_hz=sample_rate_hz,
        seed=42,
    )

    impaired_samples = add_awgn(
        signal.samples,
        snr_db=snr_db,
        seed=123,
    )

    spectrogram = compute_spectrogram(
        impaired_samples,
        sample_rate_hz=sample_rate_hz,
        window_size=256,
        hop_size=64,
        fft_size=512,
        dynamic_range_db=80.0,
    )

    symbol_indices = np.arange(
        0,
        len(impaired_samples),
        signal.samples_per_symbol,
    )
    constellation_samples = impaired_samples[symbol_indices]

    time_sample_count = 192
    time_axis_us = (
        np.arange(time_sample_count)
        / sample_rate_hz
        * 1_000_000.0
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "rf_signal_overview.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(
        time_axis_us,
        impaired_samples.real[:time_sample_count],
        label="I",
    )
    axes[0].plot(
        time_axis_us,
        impaired_samples.imag[:time_sample_count],
        label="Q",
    )
    axes[0].set_title("IQ Time-Domain Samples")
    axes[0].set_xlabel("Time (?s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(
        constellation_samples.real,
        constellation_samples.imag,
        s=8,
        alpha=0.35,
    )
    axes[1].set_title(f"QPSK Constellation at {snr_db:.1f} dB")
    axes[1].set_xlabel("In-phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].axvline(0.0, linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(-1.4, 1.4)
    axes[1].set_ylim(-1.4, 1.4)

    spectral_image = axes[2].pcolormesh(
        spectrogram.times_s * 1_000.0,
        spectrogram.frequencies_hz / 1_000.0,
        spectrogram.power_db,
        shading="auto",
        vmin=-80.0,
        vmax=0.0,
    )
    axes[2].set_title("IQ Spectrogram")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Frequency (kHz)")
    figure.colorbar(
        spectral_image,
        ax=axes[2],
        label="Relative power (dB)",
    )

    figure.suptitle("Synthetic RF IQ Signal Overview")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Modulation: {signal.modulation}")
    print(f"SNR dB: {snr_db:.1f}")
    print(f"IQ samples: {len(impaired_samples)}")
    print(f"Spectrogram frequency bins: {len(spectrogram.frequencies_hz)}")
    print(f"Spectrogram time frames: {len(spectrogram.times_s)}")
    print(f"Spectrogram shape: {spectrogram.power_db.shape}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
