from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from torch import Tensor, nn

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)


def _positive_integer(
    value: object,
    name: str,
) -> int:
    """Validate and return one positive integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    result = int(value)

    if result <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _normalization_layer(
    normalization: str,
    channel_count: int,
    group_count: int,
) -> nn.Module:
    """Create one equalizer normalization layer."""
    if normalization == "batch":
        return nn.BatchNorm1d(channel_count)

    if normalization == "group":
        return nn.GroupNorm(
            num_groups=group_count,
            num_channels=channel_count,
        )

    raise ValueError(
        f"Unsupported normalization: {normalization}"
    )


@dataclass(frozen=True, slots=True)
class ResidualEqualizerCNNConfig:
    """Configuration for an IQ equalizer followed by the baseline CNN."""

    model_type: str = "residual_equalizer_iq_cnn"
    in_channels: int = 2
    num_classes: int = 4
    channels: tuple[int, ...] = (
        32,
        64,
        128,
    )
    kernel_size: int = 7
    dropout: float = 0.20
    normalize_input_rms: bool = False
    normalization: str = "group"
    group_norm_groups: int = 8
    equalizer_hidden_channels: int = 16
    equalizer_kernel_size: int = 9
    equalizer_normalization: str = "group"
    equalizer_group_norm_groups: int = 8

    def __post_init__(self) -> None:
        """Validate model and equalizer settings."""
        model_type = (
            self.model_type.strip().lower()
            if isinstance(
                self.model_type,
                str,
            )
            else ""
        )

        if model_type != (
            "residual_equalizer_iq_cnn"
        ):
            raise ValueError(
                "model_type must be "
                "'residual_equalizer_iq_cnn'."
            )

        object.__setattr__(
            self,
            "model_type",
            model_type,
        )

        normalization = (
            self.normalization.strip().lower()
            if isinstance(
                self.normalization,
                str,
            )
            else ""
        )
        equalizer_normalization = (
            self.equalizer_normalization
            .strip()
            .lower()
            if isinstance(
                self.equalizer_normalization,
                str,
            )
            else ""
        )

        object.__setattr__(
            self,
            "normalization",
            normalization,
        )
        object.__setattr__(
            self,
            "equalizer_normalization",
            equalizer_normalization,
        )

        # Reuse the baseline configuration validation.
        BaselineCNNConfig(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            normalize_input_rms=(
                self.normalize_input_rms
            ),
            input_representation="iq",
            normalization=normalization,
            group_norm_groups=(
                self.group_norm_groups
            ),
        )

        if self.in_channels != 2:
            raise ValueError(
                "The residual equalizer requires "
                "exactly two raw IQ channels."
            )

        _positive_integer(
            self.equalizer_hidden_channels,
            "equalizer_hidden_channels",
        )
        _positive_integer(
            self.equalizer_kernel_size,
            "equalizer_kernel_size",
        )
        _positive_integer(
            self.equalizer_group_norm_groups,
            "equalizer_group_norm_groups",
        )

        if self.equalizer_kernel_size % 2 == 0:
            raise ValueError(
                "equalizer_kernel_size must be odd."
            )

        if equalizer_normalization not in {
            "batch",
            "group",
        }:
            raise ValueError(
                "equalizer_normalization must be "
                "either 'batch' or 'group'."
            )

        if (
            equalizer_normalization == "group"
            and self.equalizer_hidden_channels
            % self.equalizer_group_norm_groups
            != 0
        ):
            raise ValueError(
                "equalizer_hidden_channels must be "
                "divisible by "
                "equalizer_group_norm_groups."
            )

    def create_backbone_configuration(
        self,
    ) -> BaselineCNNConfig:
        """Return the unchanged baseline CNN configuration."""
        return BaselineCNNConfig(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            normalize_input_rms=(
                self.normalize_input_rms
            ),
            input_representation="iq",
            normalization=self.normalization,
            group_norm_groups=(
                self.group_norm_groups
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        content: Mapping[str, object],
    ) -> ResidualEqualizerCNNConfig:
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
            model_type=str(
                content.get(
                    "model_type",
                    "residual_equalizer_iq_cnn",
                )
            ),
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
                content.get("dropout", 0.20)
            ),
            normalize_input_rms=bool(
                content.get(
                    "normalize_input_rms",
                    False,
                )
            ),
            normalization=str(
                content.get(
                    "normalization",
                    "group",
                )
            ),
            group_norm_groups=int(
                content.get(
                    "group_norm_groups",
                    8,
                )
            ),
            equalizer_hidden_channels=int(
                content.get(
                    "equalizer_hidden_channels",
                    16,
                )
            ),
            equalizer_kernel_size=int(
                content.get(
                    "equalizer_kernel_size",
                    9,
                )
            ),
            equalizer_normalization=str(
                content.get(
                    "equalizer_normalization",
                    "group",
                )
            ),
            equalizer_group_norm_groups=int(
                content.get(
                    "equalizer_group_norm_groups",
                    8,
                )
            ),
        )


class _EqualizerBlock(nn.Module):
    """Length-preserving convolutional equalizer block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalization: str,
        group_count: int,
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
            _normalization_layer(
                normalization=normalization,
                channel_count=out_channels,
                group_count=group_count,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Apply one equalizer feature block."""
        return self.layers(inputs)


class ResidualIQEqualizer(nn.Module):
    """Predict a residual correction for raw two-channel IQ."""

    def __init__(
        self,
        configuration: ResidualEqualizerCNNConfig,
    ) -> None:
        super().__init__()

        hidden_channels = (
            configuration.equalizer_hidden_channels
        )
        kernel_size = (
            configuration.equalizer_kernel_size
        )
        normalization = (
            configuration.equalizer_normalization
        )
        group_count = (
            configuration
            .equalizer_group_norm_groups
        )

        self.input_block = _EqualizerBlock(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            group_count=group_count,
        )
        self.hidden_block = _EqualizerBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            normalization=normalization,
            group_count=group_count,
        )
        self.output = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        # Start as an exact identity transformation.
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _validate_inputs(
        inputs: Tensor,
    ) -> None:
        """Validate raw IQ tensor shape."""
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape "
                "[batch, 2, samples]."
            )

        if inputs.shape[1] != 2:
            raise ValueError(
                "inputs must contain exactly "
                "two raw IQ channels."
            )

        if inputs.shape[2] == 0:
            raise ValueError(
                "inputs must contain at least "
                "one sample."
            )

    def predict_correction(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return the learned residual IQ correction."""
        self._validate_inputs(inputs)

        hidden = self.input_block(inputs)
        hidden = self.hidden_block(hidden)

        return self.output(hidden)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return residual-corrected IQ."""
        return (
            inputs
            + self.predict_correction(inputs)
        )


class ResidualEqualizerIQCNN(nn.Module):
    """Apply a learnable residual equalizer before the baseline CNN."""

    def __init__(
        self,
        configuration: (
            ResidualEqualizerCNNConfig | None
        ) = None,
    ) -> None:
        super().__init__()

        self.configuration = (
            configuration
            or ResidualEqualizerCNNConfig()
        )
        self.equalizer = ResidualIQEqualizer(
            self.configuration
        )
        self.backbone = BaselineIQCNN(
            self.configuration
            .create_backbone_configuration()
        )

    @property
    def classifier(self) -> nn.Module:
        """Expose the baseline classifier for compatibility."""
        return self.backbone.classifier

    def equalize_iq(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return the corrected two-channel IQ tensor."""
        return self.equalizer(inputs)

    def extract_features(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return baseline features from corrected IQ."""
        corrected = self.equalize_iq(inputs)

        return self.backbone.extract_features(
            corrected
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return unnormalized class logits."""
        return self.classifier(
            self.extract_features(inputs)
        )


__all__ = [
    "ResidualEqualizerCNNConfig",
    "ResidualEqualizerIQCNN",
    "ResidualIQEqualizer",
]
