from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    modulations = [
        Modulation.BPSK,
        Modulation.QPSK,
        Modulation.PSK8,
        Modulation.QAM16,
    ]

    display_names = {
        Modulation.BPSK: "BPSK",
        Modulation.QPSK: "QPSK",
        Modulation.PSK8: "8PSK",
        Modulation.QAM16: "16QAM",
    }

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "modulation_constellation_catalog.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10, 10))

    for axis, modulation in zip(
        axes.flat,
        modulations,
        strict=True,
    ):
        signal = generate_iq_signal(
            modulation=modulation,
            num_symbols=4_096,
            samples_per_symbol=1,
            seed=42,
        )

        unique_points = np.unique(signal.samples)
        average_power = float(
            np.mean(np.abs(unique_points) ** 2)
        )

        axis.scatter(
            signal.samples.real,
            signal.samples.imag,
            s=12,
            alpha=0.35,
        )
        axis.set_title(display_names[modulation])
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")
        axis.axhline(0.0, linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-1.5, 1.5)

        print(
            f"{display_names[modulation]}: "
            f"{len(unique_points)} points, "
            f"average power {average_power:.6f}"
        )

    figure.suptitle("Normalized Digital Modulation Constellations")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
