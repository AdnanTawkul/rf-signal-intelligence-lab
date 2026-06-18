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
)
from rfsil.ssl.vicreg import (
    VICRegLossConfig,
    VICRegModel,
)
from rfsil.ssl.vicreg_training import (
    run_vicreg_evaluation_epoch,
    run_vicreg_training_epoch,
)


def create_model() -> VICRegModel:
    encoder = BaselineIQCNN(
        BaselineCNNConfig(
            channels=(8, 16, 32),
            normalization="group",
            group_norm_groups=8,
        )
    )

    return VICRegModel(
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
    model: VICRegModel,
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
        lr=0.001,
    )


def test_training_epoch_returns_finite_metrics() -> None:
    torch.manual_seed(2026)

    model = create_model()

    metrics = run_vicreg_training_epoch(
        model=model,
        data_loader=create_loader(),
        optimizer=create_optimizer(model),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
        loss_configuration=VICRegLossConfig(),
    )

    assert metrics.example_count == 8

    values = (
        metrics.total_loss,
        metrics.invariance_loss,
        metrics.variance_loss,
        metrics.covariance_loss,
        metrics.positive_cosine_similarity,
        metrics.projection_standard_deviation,
    )

    assert all(
        math.isfinite(value)
        for value in values
    )
    assert metrics.total_loss >= 0.0
    assert metrics.invariance_loss >= 0.0
    assert metrics.variance_loss >= 0.0
    assert metrics.covariance_loss >= 0.0
    assert -1.0 <= (
        metrics.positive_cosine_similarity
    ) <= 1.0
    assert (
        metrics.projection_standard_deviation
        >= 0.0
    )


def test_training_epoch_updates_parameters() -> None:
    torch.manual_seed(2026)

    model = create_model()
    parameter = (
        model.projection_head.layers[0].weight
    )
    before = parameter.detach().clone()

    run_vicreg_training_epoch(
        model=model,
        data_loader=create_loader(),
        optimizer=create_optimizer(model),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
    )

    assert not torch.equal(
        before,
        parameter.detach(),
    )


def test_evaluation_does_not_update_parameters() -> None:
    torch.manual_seed(2026)

    model = create_model()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    metrics = run_vicreg_evaluation_epoch(
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


def test_training_ignores_labels() -> None:
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

    metrics = run_vicreg_training_epoch(
        model=model,
        data_loader=loader,
        optimizer=create_optimizer(model),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
    )

    assert metrics.example_count == 4


def test_custom_loss_configuration_is_used() -> None:
    torch.manual_seed(2026)

    model = create_model()
    configuration = VICRegLossConfig(
        invariance_weight=1.0,
        variance_weight=2.0,
        covariance_weight=3.0,
    )

    metrics = run_vicreg_evaluation_epoch(
        model=model,
        data_loader=create_loader(),
        augmentation=create_identity_augmentation(),
        device=torch.device("cpu"),
        loss_configuration=configuration,
    )

    expected_total = (
        configuration.invariance_weight
        * metrics.invariance_loss
        + configuration.variance_weight
        * metrics.variance_loss
        + configuration.covariance_weight
        * metrics.covariance_loss
    )

    assert metrics.total_loss == pytest.approx(
        expected_total,
        rel=1e-5,
        abs=1e-6,
    )


def test_missing_iq_is_rejected() -> None:
    loader = DataLoader(
        [
            {
                "label": torch.tensor(0),
            },
            {
                "label": torch.tensor(1),
            },
        ],
        batch_size=2,
    )

    model = create_model()

    with pytest.raises(KeyError):
        run_vicreg_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_single_example_batch_is_rejected() -> None:
    model = create_model()

    with pytest.raises(ValueError):
        run_vicreg_training_epoch(
            model=model,
            data_loader=create_loader(
                example_count=1,
                batch_size=1,
            ),
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_invalid_iq_shape_is_rejected() -> None:
    loader = DataLoader(
        [
            {
                "iq": torch.randn(64),
            },
            {
                "iq": torch.randn(64),
            },
        ],
        batch_size=2,
    )

    model = create_model()

    with pytest.raises(ValueError):
        run_vicreg_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=create_optimizer(model),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )


def test_empty_loader_is_rejected() -> None:
    model = create_model()

    with pytest.raises(ValueError):
        run_vicreg_evaluation_epoch(
            model=model,
            data_loader=DataLoader(
                [],
                batch_size=4,
            ),
            augmentation=create_identity_augmentation(),
            device=torch.device("cpu"),
        )
