from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rfsil.data.torch_dataset import NPZIQDataset
from rfsil.data.transforms import normalize_iq_rms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def complex_average_power(inputs: torch.Tensor) -> torch.Tensor:
    """Return average complex power for each IQ example."""
    return (
        inputs.square()
        .sum(dim=1)
        .mean(dim=1)
    )


def main() -> None:
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "rf_modulation_baseline_v1"
        / "train.npz"
    )

    dataset = NPZIQDataset(dataset_path)
    original_iq = dataset.iq
    normalized_iq = normalize_iq_rms(original_iq)

    original_power = complex_average_power(original_iq)
    normalized_power = complex_average_power(normalized_iq)

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "rms_normalization_diagnostic.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        original_power.numpy(),
        bins=50,
    )
    axes[0].set_title("Original Example Power")
    axes[0].set_xlabel("Average complex power")
    axes[0].set_ylabel("Example count")
    axes[0].set_xlim(left=0.0)
    axes[0].grid(True, alpha=0.3)

    normalized_power_error = (
        normalized_power - 1.0
    ) * 1e7
    error_limit = max(
        float(torch.max(torch.abs(normalized_power_error))),
        1.0,
    )
    error_bin_edges = torch.linspace(
        -1.1 * error_limit,
        1.1 * error_limit,
        steps=51,
    ).numpy()

    axes[1].hist(
        normalized_power_error.numpy(),
        bins=error_bin_edges,
    )
    axes[1].axvline(
        0.0,
        linewidth=1,
        linestyle="--",
    )
    axes[1].set_title("Normalized Power Error")
    axes[1].set_xlabel(r"Power error ($\times 10^{-7}$)")
    axes[1].set_ylabel("Example count")
    axes[1].grid(True, alpha=0.3)

    figure.suptitle("Per-Example Complex RMS Normalization")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Dataset examples: {len(dataset)}")
    print(
        "Original power mean: "
        f"{float(torch.mean(original_power)):.6f}"
    )
    print(
        "Original power standard deviation: "
        f"{float(torch.std(original_power)):.6f}"
    )
    print(
        "Original power range: "
        f"{float(torch.min(original_power)):.6f} to "
        f"{float(torch.max(original_power)):.6f}"
    )
    print(
        "Normalized power mean: "
        f"{float(torch.mean(normalized_power)):.6f}"
    )
    print(
        "Normalized power standard deviation: "
        f"{float(torch.std(normalized_power)):.8f}"
    )
    print(
        "Maximum unit-power error: "
        f"{float(torch.max(torch.abs(normalized_power - 1.0))):.8f}"
    )
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
