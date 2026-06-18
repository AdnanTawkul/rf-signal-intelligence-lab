from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn

from rfsil.data.transforms import (
    create_channel_aware_iq_representation,
    normalize_iq_rms,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)


def _positive_integer(
    value: object,
    name: str,
) -> int:
    """Validate a strictly positive integer."""
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


@dataclass(frozen=True, slots=True)
class LateFusionCNNConfig:
    """Configuration for separate IQ and differential-phase branches."""

    model_type: str = "late_fusion_iq_dphase"
    in_channels: int = 2
    num_classes: int = 4
    branch_channels: tuple[int, ...] = (
        16,
        32,
        64,
    )
    kernel_size: int = 7
    dropout: float = 0.20
    normalize_input_rms: bool = False
    normalization: str = "group"
    group_norm_groups: int = 8
    fusion_hidden: int = 256

    def __post_init__(self) -> None:
        """Validate late-fusion architecture settings."""
        model_type = (
            self.model_type.strip().lower()
            if isinstance(
                self.model_type,
                str,
            )
            else ""
        )

        if model_type != "late_fusion_iq_dphase":
            raise ValueError(
                "model_type must be "
                "'late_fusion_iq_dphase'."
            )

        object.__setattr__(
            self,
            "model_type",
            model_type,
        )

        _positive_integer(
            self.in_channels,
            "in_channels",
        )
        _positive_integer(
            self.num_classes,
            "num_classes",
        )
        _positive_integer(
            self.kernel_size,
            "kernel_size",
        )
        _positive_integer(
            self.group_norm_groups,
            "group_norm_groups",
        )
        _positive_integer(
            self.fusion_hidden,
            "fusion_hidden",
        )

        if self.in_channels != 2:
            raise ValueError(
                "Late fusion requires exactly two "
                "raw IQ input channels."
            )

        if self.num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        if not self.branch_channels:
            raise ValueError(
                "branch_channels must not be empty."
            )

        for channel_count in (
            self.branch_channels
        ):
            _positive_integer(
                channel_count,
                "branch channel count",
            )

        if self.kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if not isinstance(
            self.normalize_input_rms,
            bool,
        ):
            raise ValueError(
                "normalize_input_rms must be "
                "a boolean."
            )

        normalized_normalization = (
            self.normalization.strip().lower()
            if isinstance(
                self.normalization,
                str,
            )
            else ""
        )

        if normalized_normalization not in {
            "batch",
            "group",
        }:
            raise ValueError(
                "normalization must be either "
                "'batch' or 'group'."
            )

        object.__setattr__(
            self,
            "normalization",
            normalized_normalization,
        )

        if normalized_normalization == "group":
            for channel_count in (
                self.branch_channels
            ):
                if (
                    channel_count
                    % self.group_norm_groups
                    != 0
                ):
                    raise ValueError(
                        "Every branch channel count "
                        "must be divisible by "
                        "group_norm_groups."
                    )

    @property
    def branch_embedding_dimension(self) -> int:
        """Return each branch's pooled feature dimension."""
        return self.branch_channels[-1]

    @property
    def fused_input_dimension(self) -> int:
        """Return the concatenated branch dimension."""
        return (
            2
            * self.branch_embedding_dimension
        )

    @classmethod
    def from_mapping(
        cls,
        content: Mapping[str, object],
    ) -> LateFusionCNNConfig:
        """Build a configuration from serialized values."""
        channels_value = content.get(
            "branch_channels",
            (16, 32, 64),
        )

        if not isinstance(
            channels_value,
            (list, tuple),
        ):
            raise ValueError(
                "branch_channels must be "
                "a list or tuple."
            )

        return cls(
            model_type=str(
                content.get(
                    "model_type",
                    "late_fusion_iq_dphase",
                )
            ),
            in_channels=int(
                content.get(
                    "in_channels",
                    2,
                )
            ),
            num_classes=int(
                content.get(
                    "num_classes",
                    4,
                )
            ),
            branch_channels=tuple(
                int(value)
                for value in channels_value
            ),
            kernel_size=int(
                content.get(
                    "kernel_size",
                    7,
                )
            ),
            dropout=float(
                content.get(
                    "dropout",
                    0.20,
                )
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
            fusion_hidden=int(
                content.get(
                    "fusion_hidden",
                    256,
                )
            ),
        )


class LateFusionIQDPhaseCNN(nn.Module):
    """Fuse separately encoded raw IQ and differential phase."""

    def __init__(
        self,
        configuration: (
            LateFusionCNNConfig | None
        ) = None,
    ) -> None:
        super().__init__()

        self.configuration = (
            configuration
            or LateFusionCNNConfig()
        )

        common_configuration = {
            "num_classes": (
                self.configuration.num_classes
            ),
            "channels": (
                self.configuration.branch_channels
            ),
            "kernel_size": (
                self.configuration.kernel_size
            ),
            "dropout": (
                self.configuration.dropout
            ),
            "normalize_input_rms": False,
            "input_representation": "iq",
            "normalization": (
                self.configuration.normalization
            ),
            "group_norm_groups": (
                self.configuration
                .group_norm_groups
            ),
        }

        self.iq_encoder = BaselineIQCNN(
            BaselineCNNConfig(
                in_channels=2,
                **common_configuration,
            )
        )
        self.dphase_encoder = BaselineIQCNN(
            BaselineCNNConfig(
                in_channels=1,
                **common_configuration,
            )
        )

        # Only the encoders and global pools are used.
        self.iq_encoder.classifier = (
            nn.Identity()
        )
        self.dphase_encoder.classifier = (
            nn.Identity()
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                self.configuration
                .fused_input_dimension,
                self.configuration
                .fusion_hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                self.configuration.dropout
            ),
        )
        self.classifier = nn.Linear(
            self.configuration.fusion_hidden,
            self.configuration.num_classes,
        )

    def _validate_inputs(
        self,
        inputs: Tensor,
    ) -> None:
        """Validate raw IQ input shape."""
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

        minimum_samples = (
            2
            ** len(
                self.configuration
                .branch_channels
            )
        )

        if inputs.shape[2] < minimum_samples:
            raise ValueError(
                "inputs must contain at least "
                f"{minimum_samples} samples."
            )

    def extract_branch_features(
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return pooled IQ and differential-phase embeddings."""
        self._validate_inputs(inputs)

        model_inputs = (
            normalize_iq_rms(inputs)
            if self.configuration
            .normalize_input_rms
            else inputs
        )

        derived = (
            create_channel_aware_iq_representation(
                model_inputs,
                include_magnitude=False,
                include_differential_phase=True,
            )
        )
        differential_phase = derived[
            :,
            2:3,
            :,
        ]

        iq_features = (
            self.iq_encoder.extract_features(
                model_inputs
            )
        )
        dphase_features = (
            self.dphase_encoder
            .extract_features(
                differential_phase
            )
        )

        return (
            iq_features,
            dphase_features,
        )

    def extract_features(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return the fused embedding before classification."""
        (
            iq_features,
            dphase_features,
        ) = self.extract_branch_features(
            inputs
        )

        fused_inputs = torch.cat(
            (
                iq_features,
                dphase_features,
            ),
            dim=1,
        )

        return self.fusion(fused_inputs)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return unnormalized class logits."""
        return self.classifier(
            self.extract_features(inputs)
        )


__all__ = [
    "LateFusionCNNConfig",
    "LateFusionIQDPhaseCNN",
]
