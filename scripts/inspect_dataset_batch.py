from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfsil.data.synthetic import MODULATION_CLASSES
from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect one PyTorch batch from a generated IQ dataset.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "data/processed/rf_modulation_smoke_v1/train.npz"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    split_path = arguments.split

    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path

    dataset = NPZIQDataset(split_path)

    loader = create_data_loader(
        dataset,
        DataLoaderConfig(
            batch_size=arguments.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            seed=2026,
        ),
    )

    batch = next(iter(loader))

    iq = batch["iq"]
    labels = batch["label"]
    snr_db = batch["snr_db"]

    label_counts = torch.bincount(
        labels,
        minlength=len(MODULATION_CLASSES),
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "dataset_batch_inspection.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    example_count = min(4, iq.shape[0])

    for index in range(example_count):
        axis = axes.flat[index]
        in_phase = iq[index, 0].numpy()
        quadrature = iq[index, 1].numpy()
        sample_axis = np.arange(min(256, iq.shape[-1]))

        axis.plot(
            sample_axis,
            in_phase[: len(sample_axis)],
            label="I",
        )
        axis.plot(
            sample_axis,
            quadrature[: len(sample_axis)],
            label="Q",
        )

        modulation = MODULATION_CLASSES[
            int(labels[index])
        ].value

        axis.set_title(
            f"{modulation} | label={int(labels[index])} | "
            f"SNR={float(snr_db[index]):.1f} dB"
        )
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("PyTorch IQ Dataset Batch Inspection")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Dataset split: {split_path}")
    print(f"Dataset examples: {len(dataset)}")
    print(f"Batch IQ shape: {tuple(iq.shape)}")
    print(f"Batch IQ dtype: {iq.dtype}")
    print(f"Batch label shape: {tuple(labels.shape)}")
    print(f"Batch label dtype: {labels.dtype}")
    print(f"Pin memory configured: {torch.cuda.is_available()}")

    for label, modulation in enumerate(MODULATION_CLASSES):
        print(
            f"Batch count {modulation.value}: "
            f"{int(label_counts[label])}"
        )

    print(
        "Batch SNR range dB: "
        f"{float(torch.min(snr_db)):.1f} to "
        f"{float(torch.max(snr_db)):.1f}"
    )
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
