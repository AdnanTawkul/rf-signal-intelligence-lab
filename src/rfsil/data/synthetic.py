from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from numbers import Integral

import numpy as np

from rfsil.dsp.channel_profiles import (
    get_multipath_profile,
    sample_multipath_tap_gains,
)
from rfsil.dsp.impairments import (
    add_awgn,
    apply_amplitude_scaling,
    apply_flat_rayleigh_fading,
    apply_frequency_offset,
    apply_phase_offset,
    apply_tapped_delay_line,
    apply_time_shift,
)
from rfsil.dsp.modulations import ComplexArray, Modulation, generate_iq_signal
from rfsil.dsp.pulse_shaping import apply_root_raised_cosine

MODULATION_CLASSES: tuple[Modulation, ...] = (
    Modulation.BPSK,
    Modulation.QPSK,
    Modulation.PSK8,
    Modulation.QAM16,
)

MODULATION_TO_LABEL: dict[Modulation, int] = {
    modulation: label
    for label, modulation in enumerate(MODULATION_CLASSES)
}


@dataclass(frozen=True, slots=True)
class SyntheticExampleConfig:
    """Configuration for one fixed-length synthetic IQ example."""

    sample_count: int = 4_096
    sample_rate_hz: float = 1_000_000.0
    samples_per_symbol: int = 8
    rolloff: float = 0.35
    span_symbols: int = 8
    snr_db: float | None = 15.0
    frequency_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0
    amplitude_scale: float = 1.0
    time_shift_samples: int = 0
    apply_rayleigh_fading: bool = False
    multipath_profile: str | None = None


@dataclass(frozen=True, slots=True)
class SyntheticIQExample:
    """Fixed-length synthetic IQ example with classification metadata."""

    samples: ComplexArray
    modulation: Modulation
    label: int
    configuration: SyntheticExampleConfig
    seed: int | None
    symbol_sample_offset: int


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


def _draw_child_seed(rng: np.random.Generator) -> int:
    """Draw a deterministic child seed from a parent generator."""
    return int(
        rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            dtype=np.uint32,
        )
    )


def generate_synthetic_example(
    modulation: Modulation | str,
    configuration: SyntheticExampleConfig,
    seed: int | None = None,
) -> SyntheticIQExample:
    """Generate one fixed-length labeled synthetic RF IQ example.

    Processing order:

    1. Generate random modulation symbols.
    2. Apply root-raised-cosine transmit pulse shaping.
    3. Crop a deterministic fixed-length waveform.
    4. Apply amplitude scaling.
    5. Apply optional flat Rayleigh fading.
    6. Apply optional frequency-selective multipath fading.
    7. Apply carrier frequency and phase offsets.
    8. Apply zero-padded integer time shift.
    9. Add AWGN at the requested SNR.

    AWGN is applied last so the configured SNR describes the final impaired
    signal before receiver noise.

    Args:
        modulation: Supported digital modulation class.
        configuration: Signal-generation and impairment configuration.
        seed: Optional master random seed.

    Returns:
        Fixed-length SyntheticIQExample with stable label metadata.

    Raises:
        ValueError: If the modulation or configuration is invalid.
    """
    selected_modulation = Modulation(modulation)

    sample_count = _validate_positive_integer(
        configuration.sample_count,
        "sample_count",
    )
    samples_per_symbol = _validate_positive_integer(
        configuration.samples_per_symbol,
        "samples_per_symbol",
    )
    span_symbols = _validate_positive_integer(
        configuration.span_symbols,
        "span_symbols",
    )

    if not isinstance(configuration.apply_rayleigh_fading, bool):
        raise ValueError("apply_rayleigh_fading must be a boolean.")

    selected_multipath_profile = None

    if configuration.multipath_profile is not None:
        selected_multipath_profile = (
            get_multipath_profile(
                configuration.multipath_profile
            )
        )

    rng = np.random.default_rng(seed)

    symbol_seed = _draw_child_seed(rng)
    fading_seed = _draw_child_seed(rng)
    noise_seed = _draw_child_seed(rng)

    required_symbol_count = (
        ceil(sample_count / samples_per_symbol)
        + 2 * span_symbols
        + 4
    )

    symbol_signal = generate_iq_signal(
        modulation=selected_modulation,
        num_symbols=required_symbol_count,
        samples_per_symbol=1,
        sample_rate_hz=configuration.sample_rate_hz,
        seed=symbol_seed,
    )

    shaped = apply_root_raised_cosine(
        symbol_signal.samples,
        samples_per_symbol=samples_per_symbol,
        rolloff=configuration.rolloff,
        span_symbols=span_symbols,
    )

    group_delay = shaped.group_delay_samples

    stable_samples = (
        shaped.samples[group_delay:-group_delay]
        if group_delay > 0
        else shaped.samples
    )

    maximum_crop_start = len(stable_samples) - sample_count

    if maximum_crop_start < 0:
        raise RuntimeError(
            "Pulse-shaped signal is shorter than the requested sample count."
        )

    crop_start = int(
        rng.integers(
            low=0,
            high=maximum_crop_start + 1,
        )
    )

    samples = stable_samples[
        crop_start:crop_start + sample_count
    ].astype(np.complex64)

    symbol_sample_offset = (
        -crop_start
    ) % samples_per_symbol

    multipath_seed = (
        _draw_child_seed(rng)
        if selected_multipath_profile is not None
        else None
    )

    samples = apply_amplitude_scaling(
        samples,
        amplitude_scale=configuration.amplitude_scale,
    )

    if configuration.apply_rayleigh_fading:
        samples = apply_flat_rayleigh_fading(
            samples,
            seed=fading_seed,
        )

    if selected_multipath_profile is not None:
        multipath_tap_gains = (
            sample_multipath_tap_gains(
                selected_multipath_profile,
                seed=multipath_seed,
                normalize_total_power=True,
            )
        )

        samples = apply_tapped_delay_line(
            samples,
            tap_delays_samples=(
                selected_multipath_profile
                .tap_delays_samples
            ),
            tap_gains=multipath_tap_gains,
            normalize_tap_power=False,
        )

    samples = apply_frequency_offset(
        samples,
        frequency_offset_hz=configuration.frequency_offset_hz,
        sample_rate_hz=configuration.sample_rate_hz,
    )

    samples = apply_phase_offset(
        samples,
        phase_offset_rad=configuration.phase_offset_rad,
    )

    samples = apply_time_shift(
        samples,
        shift_samples=configuration.time_shift_samples,
    )

    symbol_sample_offset = (
        symbol_sample_offset
        + int(configuration.time_shift_samples)
    ) % samples_per_symbol

    if configuration.snr_db is not None:
        samples = add_awgn(
            samples,
            snr_db=configuration.snr_db,
            seed=noise_seed,
        )

    return SyntheticIQExample(
        samples=samples.astype(np.complex64),
        modulation=selected_modulation,
        label=MODULATION_TO_LABEL[selected_modulation],
        configuration=configuration,
        seed=seed,
        symbol_sample_offset=int(symbol_sample_offset),
    )


__all__ = [
    "MODULATION_CLASSES",
    "MODULATION_TO_LABEL",
    "SyntheticExampleConfig",
    "SyntheticIQExample",
    "generate_synthetic_example",
]
