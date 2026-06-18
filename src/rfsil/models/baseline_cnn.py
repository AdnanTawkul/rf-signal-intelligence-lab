from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from torch import Tensor, nn

from rfsil.data.transforms import (
    create_channel_aware_iq_representation,
    normalize_iq_rms,
)


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


@dataclass(frozen=True, slots=True)
class BaselineCNNConfig:
    """Configuration for the baseline one-dimensional IQ classifier."""

    in_channels: int = 2
    num_classes: int = 4
    channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 7
    dropout: float = 0.20
    normalize_input_rms: bool = False
    input_representation: str = "iq"
    normalization: str = "batch"
    group_norm_groups: int = 8

    def __post_init__(self) -> None:
        """Validate model architecture settings."""
        _validate_positive_integer(self.in_channels, "in_channels")
        _validate_positive_integer(self.num_classes, "num_classes")

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if not self.channels:
            raise ValueError("channels must not be empty.")

        for channel_count in self.channels:
            _validate_positive_integer(channel_count, "channel count")

        _validate_positive_integer(self.kernel_size, "kernel_size")

        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        if not isinstance(self.normalize_input_rms, bool):
            raise ValueError("normalize_input_rms must be a boolean.")

        if not isinstance(self.input_representation, str):
            raise ValueError(
                "input_representation must be a string."
            )

        normalized_representation = (
            self.input_representation.strip().lower()
        )

        if normalized_representation not in {
            "iq",
            "iq_magnitude",
            "iq_dphase",
            "iq_magnitude_dphase",
        }:
            raise ValueError(
                "input_representation must be one of "
                "'iq', 'iq_magnitude', 'iq_dphase', "
                "or 'iq_magnitude_dphase'."
            )

        object.__setattr__(
            self,
            "input_representation",
            normalized_representation,
        )

        if (
            normalized_representation != "iq"
            and self.in_channels != 2
        ):
            raise ValueError(
                "Derived IQ representations require "
                "exactly two raw IQ input channels."
            )

        if (
            self.normalize_input_rms
            and self.in_channels != 2
        ):
            raise ValueError(
                "normalize_input_rms requires exactly "
                "two raw IQ input channels."
            )

        if self.normalization not in {"batch", "group"}:
            raise ValueError(
                "normalization must be either 'batch' or 'group'."
            )

        _validate_positive_integer(
            self.group_norm_groups,
            "group_norm_groups",
        )

        if self.normalization == "group":
            for channel_count in self.channels:
                if channel_count % self.group_norm_groups != 0:
                    raise ValueError(
                        "Every channel count must be divisible by "
                        "group_norm_groups when using GroupNorm."
                    )


    @property
    def feature_channels(self) -> int:
        """Return the channel count entering the first convolution."""
        if self.input_representation == "iq":
            return self.in_channels

        if self.input_representation in {
            "iq_magnitude",
            "iq_dphase",
        }:
            return 3

        return 4

    @classmethod
    def from_mapping(
        cls,
        content: Mapping[str, object],
    ) -> BaselineCNNConfig:
        """Construct a configuration from serialized values."""
        channels_value = content.get(
            "channels",
            (32, 64, 128),
        )

        if not isinstance(
            channels_value,
            (list, tuple),
        ):
            raise ValueError(
                "channels must be a list or tuple."
            )

        return cls(
            in_channels=int(
                content.get("in_channels", 2)
            ),
            num_classes=int(
                content.get("num_classes", 4)
            ),
            channels=tuple(
                int(value)
                for value in channels_value
            ),
            kernel_size=int(
                content.get("kernel_size", 7)
            ),
            dropout=float(
                content.get("dropout", 0.2)
            ),
            normalize_input_rms=bool(
                content.get(
                    "normalize_input_rms",
                    False,
                )
            ),
            input_representation=str(
                content.get(
                    "input_representation",
                    "iq",
                )
            ),
            normalization=str(
                content.get(
                    "normalization",
                    "batch",
                )
            ),
            group_norm_groups=int(
                content.get(
                    "group_norm_groups",
                    8,
                )
            ),
        )


def _create_normalization_layer(
    normalization: str,
    channel_count: int,
    group_norm_groups: int,
) -> nn.Module:
    """Create the configured channel-normalization layer."""
    if normalization == "batch":
        return nn.BatchNorm1d(channel_count)

    if normalization == "group":
        return nn.GroupNorm(
            num_groups=group_norm_groups,
            num_channels=channel_count,
        )

    raise ValueError(f"Unsupported normalization: {normalization}")


class _ConvBlock(nn.Module):
    """Convolution, normalization, activation, and temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalization: str,
        group_norm_groups: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            _create_normalization_layer(
                normalization=normalization,
                channel_count=out_channels,
                group_norm_groups=group_norm_groups,
            ),
            nn.GELU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply one convolutional feature-extraction block."""
        return self.layers(inputs)


class BaselineIQCNN(nn.Module):
    """Compact one-dimensional CNN for RF modulation classification.

    Raw input tensors use shape ``[batch, in_channels, samples]``.

    The default representation passes I/Q directly to the convolutional
    encoder. The optional channel-aware representation derives normalized
    magnitude and differential phase inside the model.
    """

    def __init__(
        self,
        configuration: BaselineCNNConfig | None = None,
    ) -> None:
        super().__init__()

        self.configuration = configuration or BaselineCNNConfig()

        blocks: list[nn.Module] = []
        current_channels = self.configuration.feature_channels

        for output_channels in self.configuration.channels:
            blocks.append(
                _ConvBlock(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=self.configuration.kernel_size,
                    normalization=self.configuration.normalization,
                    group_norm_groups=(
                        self.configuration.group_norm_groups
                    ),
                )
            )
            current_channels = output_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(self.configuration.dropout),
            nn.Linear(
                current_channels,
                self.configuration.num_classes,
            ),
        )

    def extract_features(self, inputs: Tensor) -> Tensor:
        """Return the pooled feature embedding before classification."""
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch, channels, samples]."
            )

        if inputs.shape[1] != self.configuration.in_channels:
            raise ValueError(
                "input channel count does not match the model configuration."
            )

        minimum_sample_count = 2 ** len(self.configuration.channels)

        if inputs.shape[2] < minimum_sample_count:
            raise ValueError(
                f"inputs must contain at least {minimum_sample_count} samples."
            )

        model_inputs = (
            normalize_iq_rms(inputs)
            if self.configuration.normalize_input_rms
            else inputs
        )

        representation = (
            self.configuration.input_representation
        )

        if representation != "iq":
            model_inputs = (
                create_channel_aware_iq_representation(
                    model_inputs,
                    include_magnitude=(
                        representation
                        in {
                            "iq_magnitude",
                            "iq_magnitude_dphase",
                        }
                    ),
                    include_differential_phase=(
                        representation
                        in {
                            "iq_dphase",
                            "iq_magnitude_dphase",
                        }
                    ),
                )
            )

        features = self.features(model_inputs)

        return self.global_pool(features).squeeze(-1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class logits."""
        return self.classifier(
            self.extract_features(inputs)
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "BaselineCNNConfig",
    "BaselineIQCNN",
    "count_trainable_parameters",
]
