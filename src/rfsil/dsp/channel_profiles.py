from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from rfsil.dsp.modulations import ComplexArray


@dataclass(frozen=True, slots=True)
class MultipathChannelProfile:
    """Average-power profile for a tapped-delay-line channel."""

    name: str
    tap_delays_samples: tuple[int, ...]
    average_powers_db: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and normalize profile metadata."""
        if not isinstance(self.name, str):
            raise ValueError("name must be a string.")

        normalized_name = self.name.strip().lower()

        if not normalized_name:
            raise ValueError("name must not be empty.")

        try:
            raw_delays = tuple(
                self.tap_delays_samples
            )
        except TypeError as error:
            raise ValueError(
                "tap_delays_samples must be iterable."
            ) from error

        try:
            raw_powers = tuple(
                self.average_powers_db
            )
        except TypeError as error:
            raise ValueError(
                "average_powers_db must be iterable."
            ) from error

        if not raw_delays:
            raise ValueError(
                "tap_delays_samples must not be empty."
            )

        if len(raw_delays) != len(raw_powers):
            raise ValueError(
                "tap_delays_samples and "
                "average_powers_db must have "
                "matching lengths."
            )

        validated_delays: list[int] = []

        for delay in raw_delays:
            if (
                isinstance(delay, bool)
                or not isinstance(delay, Integral)
            ):
                raise ValueError(
                    "tap_delays_samples must contain "
                    "integers."
                )

            validated_delay = int(delay)

            if validated_delay < 0:
                raise ValueError(
                    "tap_delays_samples must be "
                    "nonnegative."
                )

            validated_delays.append(
                validated_delay
            )

        if validated_delays[0] != 0:
            raise ValueError(
                "The first tap delay must be zero."
            )

        if any(
            current >= following
            for current, following in zip(
                validated_delays,
                validated_delays[1:],
                strict=False,
            )
        ):
            raise ValueError(
                "tap_delays_samples must be "
                "strictly increasing."
            )

        validated_powers = tuple(
            float(value)
            for value in raw_powers
        )

        if not np.all(
            np.isfinite(validated_powers)
        ):
            raise ValueError(
                "average_powers_db must contain "
                "only finite values."
            )

        object.__setattr__(
            self,
            "name",
            normalized_name,
        )
        object.__setattr__(
            self,
            "tap_delays_samples",
            tuple(validated_delays),
        )
        object.__setattr__(
            self,
            "average_powers_db",
            validated_powers,
        )


MILD_MULTIPATH_PROFILE = MultipathChannelProfile(
    name="mild",
    tap_delays_samples=(0, 1, 3),
    average_powers_db=(0.0, -6.0, -12.0),
)

MODERATE_MULTIPATH_PROFILE = MultipathChannelProfile(
    name="moderate",
    tap_delays_samples=(0, 2, 5, 9),
    average_powers_db=(0.0, -3.0, -8.0, -14.0),
)

SEVERE_MULTIPATH_PROFILE = MultipathChannelProfile(
    name="severe",
    tap_delays_samples=(0, 3, 8, 16, 28),
    average_powers_db=(0.0, -2.0, -5.0, -9.0, -15.0),
)

MULTIPATH_PROFILES: Mapping[
    str,
    MultipathChannelProfile,
] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            MILD_MULTIPATH_PROFILE,
            MODERATE_MULTIPATH_PROFILE,
            SEVERE_MULTIPATH_PROFILE,
        )
    }
)


def get_multipath_profile(
    profile: str | MultipathChannelProfile,
) -> MultipathChannelProfile:
    """Resolve a predefined or custom multipath profile."""
    if isinstance(
        profile,
        MultipathChannelProfile,
    ):
        return profile

    if not isinstance(profile, str):
        raise ValueError(
            "profile must be a profile name or "
            "MultipathChannelProfile."
        )

    normalized_name = profile.strip().lower()

    if not normalized_name:
        raise ValueError(
            "profile name must not be empty."
        )

    try:
        return MULTIPATH_PROFILES[
            normalized_name
        ]
    except KeyError as error:
        available = ", ".join(
            sorted(MULTIPATH_PROFILES)
        )

        raise ValueError(
            f"Unknown multipath profile "
            f"{profile!r}. Available profiles: "
            f"{available}."
        ) from error


def sample_multipath_tap_gains(
    profile: str | MultipathChannelProfile,
    seed: int | None = None,
    normalize_total_power: bool = True,
) -> ComplexArray:
    """Sample independent complex Rayleigh gains for a profile.

    Each tap follows a zero-mean complex Gaussian distribution whose
    expected power is determined by the profile's average power in dB.
    """
    selected_profile = get_multipath_profile(
        profile
    )

    if (
        seed is not None
        and (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
        )
    ):
        raise ValueError(
            "seed must be an integer or null."
        )

    if not isinstance(
        normalize_total_power,
        (bool, np.bool_),
    ):
        raise ValueError(
            "normalize_total_power must be "
            "a boolean."
        )

    average_powers_db = np.asarray(
        selected_profile.average_powers_db,
        dtype=np.float64,
    )
    average_powers_linear = np.power(
        10.0,
        average_powers_db / 10.0,
    )

    if (
        not np.all(
            np.isfinite(
                average_powers_linear
            )
        )
        or np.sum(
            average_powers_linear
        )
        <= 0.0
    ):
        raise ValueError(
            "Profile powers cannot be converted "
            "to valid linear powers."
        )

    component_standard_deviation = np.sqrt(
        average_powers_linear / 2.0
    )

    generator = np.random.default_rng(
        None if seed is None else int(seed)
    )

    gains = (
        generator.normal(
            loc=0.0,
            scale=component_standard_deviation,
        )
        + 1j
        * generator.normal(
            loc=0.0,
            scale=component_standard_deviation,
        )
    ).astype(np.complex64)

    instantaneous_power = float(
        np.sum(
            np.abs(
                gains.astype(np.complex128)
            )
            ** 2
        )
    )

    if (
        not np.isfinite(
            instantaneous_power
        )
        or instantaneous_power <= 0.0
    ):
        raise RuntimeError(
            "Sampled multipath gains have "
            "invalid total power."
        )

    if bool(normalize_total_power):
        gains = (
            gains
            / np.float32(
                np.sqrt(
                    instantaneous_power
                )
            )
        ).astype(np.complex64)

    return gains


__all__ = [
    "MILD_MULTIPATH_PROFILE",
    "MODERATE_MULTIPATH_PROFILE",
    "MULTIPATH_PROFILES",
    "MultipathChannelProfile",
    "SEVERE_MULTIPATH_PROFILE",
    "get_multipath_profile",
    "sample_multipath_tap_gains",
]
