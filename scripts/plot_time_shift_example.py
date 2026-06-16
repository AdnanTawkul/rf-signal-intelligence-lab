from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.impairments import apply_time_shift
from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    shift_samples = 20

    signal = generate_iq_signal(
        modulation=Modulation.BPSK,
        num_symbols=24,
        samples_per_symbol=8,
        sample_rate_hz=1_000_000.0,
        seed=42,
    )

    shifted_samples = apply_time_shift(
        signal.samples,
        shift_samples=shift_samples,
    )

    samples_to_plot = 128
    sample_indices = np.arange(samples_to_plot)

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "time_shift_waveform.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 5))

    axis.plot(
        sample_indices,
        signal.samples.real[:samples_to_plot],
        drawstyle="steps-post",
        label="Original BPSK I component",
    )
    axis.plot(
        sample_indices,
        shifted_samples.real[:samples_to_plot],
        drawstyle="steps-post",
        label=f"Delayed by {shift_samples} samples",
    )

    axis.axvline(
        shift_samples,
        linestyle="--",
        linewidth=1,
        label="Delay boundary",
    )
    axis.set_title("Zero-Padded IQ Time Shift")
    axis.set_xlabel("Sample index")
    axis.set_ylabel("In-phase amplitude")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    delay_seconds = shift_samples / signal.sample_rate_hz
    leading_zero_count = int(np.count_nonzero(shifted_samples[:shift_samples] == 0))

    print(f"Shift samples: {shift_samples}")
    print(f"Sample rate Hz: {signal.sample_rate_hz:.1f}")
    print(f"Delay seconds: {delay_seconds:.8f}")
    print(f"Leading zero samples: {leading_zero_count}")
    print(f"Output length: {len(shifted_samples)}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
