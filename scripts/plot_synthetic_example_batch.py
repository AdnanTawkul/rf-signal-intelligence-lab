from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.data.synthetic import (
    MODULATION_CLASSES,
    SyntheticExampleConfig,
    generate_synthetic_example,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    configuration = SyntheticExampleConfig(
        sample_count=4_096,
        sample_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
        snr_db=12.0,
        frequency_offset_hz=0.0,
        phase_offset_rad=np.deg2rad(18.0),
        amplitude_scale=0.85,
        time_shift_samples=5,
        apply_rayleigh_fading=False,
    )

    display_names = {
        "bpsk": "BPSK",
        "qpsk": "QPSK",
        "8psk": "8PSK",
        "16qam": "16QAM",
    }

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "synthetic_example_batch.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(11, 11))

    for index, (axis, modulation) in enumerate(
        zip(
            axes.flat,
            MODULATION_CLASSES,
            strict=True,
        )
    ):
        example = generate_synthetic_example(
            modulation=modulation,
            configuration=configuration,
            seed=100 + index,
        )

        symbol_indices = np.arange(
            example.symbol_sample_offset,
            len(example.samples),
            configuration.samples_per_symbol,
        )
        symbol_samples = example.samples[symbol_indices]

        axis.scatter(
            symbol_samples.real,
            symbol_samples.imag,
            s=10,
            alpha=0.35,
        )
        axis.set_title(
            f"{display_names[modulation.value]} | label={example.label}"
        )
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.6, 1.6)
        axis.set_ylim(-1.6, 1.6)

        print(
            f"{modulation.value}: "
            f"label={example.label}, "
            f"shape={example.samples.shape}, "
            f"dtype={example.samples.dtype}, "
            f"symbol offset={example.symbol_sample_offset}"
        )

    figure.suptitle(
        "Fixed-Length Synthetic IQ Classification Examples"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
