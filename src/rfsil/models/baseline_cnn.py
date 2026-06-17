from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from torch import Tensor, nn

from rfsil.data.transforms import normalize_iq_rms


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

    Input tensors must use the shape:

        [batch, 2, samples]

    Channel zero contains the in-phase component and channel one contains the
    quadrature component.
    """

    def __init__(
        self,
        configuration: BaselineCNNConfig | None = None,
    ) -> None:
        super().__init__()

        self.configuration = configuration or BaselineCNNConfig()

        blocks: list[nn.Module] = []
        current_channels = self.configuration.in_channels

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
