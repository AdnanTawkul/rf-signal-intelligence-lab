from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rfsil.dsp.modulations import Modulation, generate_iq_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    signal = generate_iq_signal(
        modulation=Modulation.QPSK,
        num_symbols=512,
        samples_per_symbol=8,
        sample_rate_hz=1_000_000.0,
        seed=42,
    )

    output_path = PROJECT_ROOT / "reports" / "figures" / "first_iq_constellation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(signal.samples.real, signal.samples.imag, s=10, alpha=0.45)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.title("Synthetic QPSK IQ Constellation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Generated modulation: {signal.modulation}")
    print(f"Number of symbols: {len(signal.symbols)}")
    print(f"Number of IQ samples: {len(signal.samples)}")
    print(f"Samples per symbol: {signal.samples_per_symbol}")
    print(f"Sample rate Hz: {signal.sample_rate_hz}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
