from __future__ import annotations

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
from rfsil.models.baseline_cnn import (
    BaselineIQCNN,
    count_trainable_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(2026)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "rf_modulation_smoke_v1"
        / "train.npz"
    )

    dataset = NPZIQDataset(dataset_path)

    loader = create_data_loader(
        dataset,
        DataLoaderConfig(
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            seed=2026,
        ),
    )

    batch = next(iter(loader))
    inputs = batch["iq"].to(
        device,
        non_blocking=True,
    )
    labels = batch["label"].to(
        device,
        non_blocking=True,
    )

    model = BaselineIQCNN().to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(inputs)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    first_probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    true_label = int(labels[0].item())
    predicted_label = int(
        torch.argmax(probabilities[0]).item()
    )

    class_names = [
        modulation.value.upper()
        for modulation in MODULATION_CLASSES
    ]

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "baseline_cnn_forward_pass.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        class_names,
        first_probabilities,
    )
    axis.set_title(
        "Untrained Baseline CNN Class Probabilities"
    )
    axis.set_xlabel("Modulation class")
    axis.set_ylabel("Probability")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Device: {device}")
    print(f"Dataset examples: {len(dataset)}")
    print(f"Batch input shape: {tuple(inputs.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model)}"
    )
    print(
        "First probability sum: "
        f"{float(np.sum(first_probabilities)):.6f}"
    )
    print(
        "First true class: "
        f"{class_names[true_label]}"
    )
    print(
        "First untrained predicted class: "
        f"{class_names[predicted_label]}"
    )
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
