from __future__ import annotations

import math

import pytest
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.ssl.augmentations import (
    IQAugmentationConfig,
    RandomIQAugmentation,
)
from rfsil.ssl.contrastive import (
    ProjectionHeadConfig,
    SimCLRModel,
)
from rfsil.ssl.training import (
    run_contrastive_evaluation_epoch,
    run_contrastive_training_epoch,
)


def create_model() -> SimCLRModel:
    encoder = BaselineIQCNN(
        BaselineCNNConfig(
            channels=(8, 16, 32),
            normalization="group",
            group_norm_groups=8,
        )
    )

    return SimCLRModel(
        encoder=encoder,
        projection_configuration=ProjectionHeadConfig(
            hidden_dimension=32,
            output_dimension=16,
        ),
    )


def create_identity_augmentation() -> RandomIQAugmentation:
    return RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=0.0,
            amplitude_scale_probability=0.0,
            time_shift_probability=0.0,
            max_time_shift_samples=0,
            awgn_probability=0.0,
        )
    )


def create_loader(
    example_count: int = 8,
    batch_size: int = 4,
) -> DataLoader:
    examples = [
        {
            "iq": torch.randn(2, 64),
            "label": torch.tensor(
                index % 4,
                dtype=torch.int64,
            ),
        }
        for index in range(example_count)
    ]

    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
    )


def create_optimizer(
    model: SimCLRModel,
) -> SGD:
    parameters = [
        parameter
        for name, parameter
        in model.named_parameters()
        if not name.startswith(
            "encoder.classifier"
        )
    ]

    return SGD(
        parameters,
        lr=0.01,
    )


def test_training_epoch_returns_finite_metrics() -> None:
    torch.manual_seed(2026)

    model = create_model()
    optimizer = create_optimizer(model)

    metrics = run_contrastive_training_epoch(
        model=model,
        data_loader=create_loader(),
        optimizer=optimizer,
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
        temperature=0.2,
    )

    assert metrics.example_count == 8
    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0.0
    assert math.isfinite(
        metrics.positive_cosine_similarity
    )
    assert -1.0 <= (
        metrics.positive_cosine_similarity
    ) <= 1.0
    assert math.isfinite(
        metrics.projection_standard_deviation
    )
    assert (
        metrics.projection_standard_deviation
        >= 0.0
    )


def test_training_epoch_updates_parameters() -> None:
    torch.manual_seed(2026)

    model = create_model()
    optimizer = create_optimizer(model)

    parameter = (
        model.projection_head.layers[0].weight
    )
    before = parameter.detach().clone()

    run_contrastive_training_epoch(
        model=model,
        data_loader=create_loader(),
        optimizer=optimizer,
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
    )

    after = parameter.detach()

    assert not torch.equal(before, after)


def test_evaluation_epoch_does_not_update_parameters() -> None:
    torch.manual_seed(2026)

    model = create_model()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    metrics = run_contrastive_evaluation_epoch(
        model=model,
        data_loader=create_loader(),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
    )

    assert metrics.example_count == 8

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            before[name],
        )


def test_training_ignores_class_labels() -> None:
    examples = [
        {
            "iq": torch.randn(2, 64),
        }
        for _ in range(4)
    ]
    loader = DataLoader(
        examples,
        batch_size=4,
    )

    model = create_model()

    metrics = run_contrastive_training_epoch(
        model=model,
        data_loader=loader,
        optimizer=create_optimizer(model),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
    )

    assert metrics.example_count == 4


def test_missing_iq_tensor_is_rejected() -> None:
    examples = [
        {
            "label": torch.tensor(0),
        },
        {
            "label": torch.tensor(1),
        },
    ]
    loader = DataLoader(
        examples,
        batch_size=2,
    )

    model = create_model()

    with pytest.raises(KeyError):
        run_contrastive_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_single_example_batch_is_rejected() -> None:
    loader = create_loader(
        example_count=1,
        batch_size=1,
    )
    model = create_model()

    with pytest.raises(ValueError):
        run_contrastive_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_invalid_iq_batch_shape_is_rejected() -> None:
    examples = [
        {
            "iq": torch.randn(64),
        },
        {
            "iq": torch.randn(64),
        },
    ]
    loader = DataLoader(
        examples,
        batch_size=2,
    )
    model = create_model()

    with pytest.raises(ValueError):
        run_contrastive_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_empty_loader_is_rejected() -> None:
    loader = DataLoader(
        [],
        batch_size=4,
    )
    model = create_model()

    with pytest.raises(ValueError):
        run_contrastive_evaluation_epoch(
            model=model,
            data_loader=loader,
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )
