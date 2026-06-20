from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from rfsil.deployment import (
    IQInferenceEngine,
    LoadedIQ,
    build_prediction_document,
    load_iq_file,
)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Validated Streamlit-demo configuration."""

    experiment_name: str
    checkpoint_search_root: Path
    preferred_checkpoint: Path | None
    expected_sample_count: int
    input_scale: float
    default_device: str
    top_k: int
    default_sample_rate_hz: float
    maximum_waveform_points: int
    maximum_constellation_points: int
    spectrum_fft_size: int


@dataclass(frozen=True, slots=True)
class CheckpointOption:
    """One checkpoint shown by the GUI."""

    path: Path
    label: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SignalViewData:
    """Downsampled signal arrays for visualization."""

    waveform_time_seconds: Float64Array
    waveform_i: Float32Array
    waveform_q: Float32Array
    constellation_i: Float32Array
    constellation_q: Float32Array
    spectrum_frequency_hz: Float64Array
    spectrum_power_db: Float64Array
    original_sample_count: int


@dataclass(frozen=True, slots=True)
class DemoPrediction:
    """One GUI prediction and export document."""

    document: dict[str, object]
    predicted_record: dict[str, object]

    def to_json(self) -> str:
        """Serialize the prediction document."""
        return json.dumps(
            self.document,
            indent=2,
        ) + "\n"


def _mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def _nonempty_string(
    value: object,
    *,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(value)


def _positive_float(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    number = float(value)

    if (
        not math.isfinite(number)
        or number <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    return number


def _resolve_project_path(
    value: object,
    *,
    project_root: Path,
    name: str,
) -> Path:
    raw = _nonempty_string(
        value,
        name=name,
    )
    path = Path(raw)

    if path.is_absolute():
        return path

    return project_root / path


def load_demo_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DemoConfig:
    """Load and validate the Streamlit configuration."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Demo configuration does not exist: "
            f"{path}"
        )

    resolved_project_root = (
        Path(project_root)
        if project_root is not None
        else path.resolve().parents[1]
    )

    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    root = _mapping(
        content,
        name="configuration",
    )
    checkpoint = _mapping(
        root.get("checkpoint"),
        name="checkpoint",
    )
    inference = _mapping(
        root.get("inference"),
        name="inference",
    )
    visualization = _mapping(
        root.get("visualization"),
        name="visualization",
    )

    preferred_value = checkpoint.get(
        "preferred_path"
    )
    preferred_checkpoint = (
        None
        if preferred_value in (None, "")
        else _resolve_project_path(
            preferred_value,
            project_root=resolved_project_root,
            name="checkpoint.preferred_path",
        )
    )

    default_device = _nonempty_string(
        inference.get("default_device"),
        name="inference.default_device",
    ).lower()

    if default_device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "inference.default_device must be "
            "auto, cpu, or cuda."
        )

    return DemoConfig(
        experiment_name=_nonempty_string(
            root.get("experiment_name"),
            name="experiment_name",
        ),
        checkpoint_search_root=(
            _resolve_project_path(
                checkpoint.get("search_root"),
                project_root=resolved_project_root,
                name="checkpoint.search_root",
            )
        ),
        preferred_checkpoint=(
            preferred_checkpoint
        ),
        expected_sample_count=(
            _positive_integer(
                inference.get(
                    "expected_sample_count"
                ),
                name=(
                    "inference."
                    "expected_sample_count"
                ),
            )
        ),
        input_scale=_positive_float(
            inference.get("input_scale"),
            name="inference.input_scale",
        ),
        default_device=default_device,
        top_k=_positive_integer(
            inference.get("top_k"),
            name="inference.top_k",
        ),
        default_sample_rate_hz=(
            _positive_float(
                visualization.get(
                    "default_sample_rate_hz"
                ),
                name=(
                    "visualization."
                    "default_sample_rate_hz"
                ),
            )
        ),
        maximum_waveform_points=(
            _positive_integer(
                visualization.get(
                    "maximum_waveform_points"
                ),
                name=(
                    "visualization."
                    "maximum_waveform_points"
                ),
            )
        ),
        maximum_constellation_points=(
            _positive_integer(
                visualization.get(
                    "maximum_constellation_points"
                ),
                name=(
                    "visualization."
                    "maximum_constellation_points"
                ),
            )
        ),
        spectrum_fft_size=(
            _positive_integer(
                visualization.get(
                    "spectrum_fft_size"
                ),
                name=(
                    "visualization."
                    "spectrum_fft_size"
                ),
            )
        ),
    )


def discover_checkpoints(
    search_root: str | Path,
) -> tuple[CheckpointOption, ...]:
    """Discover local best-model checkpoints."""
    root = Path(search_root)

    if not root.is_dir():
        return ()

    options = []

    for path in sorted(
        root.rglob("best_model.pt")
    ):
        if not path.is_file():
            continue

        options.append(
            CheckpointOption(
                path=path.resolve(),
                label=(
                    path.relative_to(root)
                    .as_posix()
                ),
                size_bytes=path.stat().st_size,
            )
        )

    return tuple(options)


def load_uploaded_iq(
    *,
    filename: str,
    content: bytes | bytearray | memoryview,
    array_key: str = "iq",
    expected_sample_count: int | None = None,
) -> LoadedIQ:
    """Load uploaded NPY or NPZ bytes through the deployment loader."""
    safe_name = Path(
        _nonempty_string(
            filename,
            name="filename",
        )
    ).name

    suffix = Path(safe_name).suffix.lower()

    if suffix not in {
        ".npy",
        ".npz",
    }:
        raise ValueError(
            "Uploaded IQ file must use the "
            ".npy or .npz extension."
        )

    raw_content = bytes(content)

    if not raw_content:
        raise ValueError(
            "Uploaded IQ file must not be empty."
        )

    with TemporaryDirectory() as directory:
        temporary_path = (
            Path(directory) / safe_name
        )
        temporary_path.write_bytes(
            raw_content
        )

        loaded = load_iq_file(
            temporary_path,
            array_key=array_key,
            expected_sample_count=(
                expected_sample_count
            ),
        )

    return LoadedIQ(
        source_path=Path(safe_name),
        array_key=loaded.array_key,
        iq=np.ascontiguousarray(
            loaded.iq.copy()
        ),
        sample_indices=np.ascontiguousarray(
            loaded.sample_indices.copy()
        ),
        labels=(
            None
            if loaded.labels is None
            else np.ascontiguousarray(
                loaded.labels.copy()
            )
        ),
        snr_db=(
            None
            if loaded.snr_db is None
            else np.ascontiguousarray(
                loaded.snr_db.copy()
            )
        ),
    )


def select_loaded_window(
    loaded: LoadedIQ,
    position: object,
) -> LoadedIQ:
    """Select one window from a loaded batch."""
    if (
        isinstance(position, (bool, np.bool_))
        or not isinstance(position, Integral)
    ):
        raise ValueError(
            "position must be an integer."
        )

    index = int(position)

    if not 0 <= index < loaded.batch_size:
        raise IndexError(
            "position is outside the IQ batch."
        )

    selection = slice(
        index,
        index + 1,
    )

    return LoadedIQ(
        source_path=loaded.source_path,
        array_key=loaded.array_key,
        iq=np.ascontiguousarray(
            loaded.iq[selection]
        ),
        sample_indices=np.ascontiguousarray(
            loaded.sample_indices[selection]
        ),
        labels=(
            None
            if loaded.labels is None
            else np.ascontiguousarray(
                loaded.labels[selection]
            )
        ),
        snr_db=(
            None
            if loaded.snr_db is None
            else np.ascontiguousarray(
                loaded.snr_db[selection]
            )
        ),
    )


def _sample_positions(
    sample_count: int,
    maximum_count: int,
) -> Int64Array:
    count = min(
        sample_count,
        maximum_count,
    )

    return np.unique(
        np.linspace(
            0,
            sample_count - 1,
            num=count,
            dtype=np.int64,
        )
    )


def build_signal_view_data(
    iq: object,
    *,
    sample_rate_hz: object,
    maximum_waveform_points: object = 2048,
    maximum_constellation_points: object = 4096,
    spectrum_fft_size: object = 4096,
) -> SignalViewData:
    """Build plotting arrays for one IQ window."""
    array = np.asarray(iq)

    if (
        array.ndim == 3
        and array.shape[0] == 1
    ):
        array = array[0]

    if (
        array.ndim != 2
        or array.shape[0] != 2
        or array.shape[1] < 2
    ):
        raise ValueError(
            "iq must have shape [2, samples] "
            "or [1, 2, samples]."
        )

    if (
        np.iscomplexobj(array)
        or not np.issubdtype(
            array.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "iq must contain real I and Q channels."
        )

    normalized = np.asarray(
        array,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "iq must contain only finite values."
        )

    validated_sample_rate = (
        _positive_float(
            sample_rate_hz,
            name="sample_rate_hz",
        )
    )
    waveform_limit = _positive_integer(
        maximum_waveform_points,
        name="maximum_waveform_points",
    )
    constellation_limit = (
        _positive_integer(
            maximum_constellation_points,
            name=(
                "maximum_constellation_points"
            ),
        )
    )
    fft_size = _positive_integer(
        spectrum_fft_size,
        name="spectrum_fft_size",
    )

    sample_count = int(
        normalized.shape[1]
    )
    waveform_indices = _sample_positions(
        sample_count,
        waveform_limit,
    )
    constellation_indices = (
        _sample_positions(
            sample_count,
            constellation_limit,
        )
    )

    complex_iq = (
        normalized[0].astype(np.float64)
        + 1j
        * normalized[1].astype(np.float64)
    )

    analysis_count = min(
        sample_count,
        fft_size,
    )
    analysis_signal = (
        complex_iq[:analysis_count]
    )
    window = np.hanning(
        analysis_count
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            analysis_signal * window,
            n=fft_size,
        )
    )
    power = np.abs(spectrum) ** 2
    reference_power = float(
        np.max(power)
    )

    if reference_power <= 0.0:
        power_db = np.full(
            fft_size,
            -120.0,
            dtype=np.float64,
        )
    else:
        tiny = np.finfo(
            np.float64
        ).tiny
        power_db = (
            10.0
            * np.log10(
                np.maximum(power, tiny)
                / reference_power
            )
        )
        power_db = np.maximum(
            power_db,
            -120.0,
        )

    frequencies_hz = np.fft.fftshift(
        np.fft.fftfreq(
            fft_size,
            d=1.0
            / validated_sample_rate,
        )
    ).astype(np.float64)

    return SignalViewData(
        waveform_time_seconds=(
            waveform_indices.astype(
                np.float64
            )
            / validated_sample_rate
        ),
        waveform_i=np.ascontiguousarray(
            normalized[
                0,
                waveform_indices,
            ]
        ),
        waveform_q=np.ascontiguousarray(
            normalized[
                1,
                waveform_indices,
            ]
        ),
        constellation_i=(
            np.ascontiguousarray(
                normalized[
                    0,
                    constellation_indices,
                ]
            )
        ),
        constellation_q=(
            np.ascontiguousarray(
                normalized[
                    1,
                    constellation_indices,
                ]
            )
        ),
        spectrum_frequency_hz=(
            frequencies_hz
        ),
        spectrum_power_db=(
            np.ascontiguousarray(
                power_db
            )
        ),
        original_sample_count=sample_count,
    )


def build_public_prediction_document(
    document: Mapping[str, object],
    *,
    source_name: str,
    checkpoint_reference: str,
) -> dict[str, object]:
    """Remove machine-specific paths from a GUI export."""
    validated_source = Path(
        _nonempty_string(
            source_name,
            name="source_name",
        )
    ).name
    validated_checkpoint = (
        _nonempty_string(
            checkpoint_reference,
            name="checkpoint_reference",
        )
    )

    exported = deepcopy(
        dict(document)
    )
    model = exported.get("model")
    input_metadata = exported.get("input")

    if not isinstance(model, dict):
        raise ValueError(
            "Prediction document model metadata "
            "must be a mapping."
        )

    if not isinstance(
        input_metadata,
        dict,
    ):
        raise ValueError(
            "Prediction document input metadata "
            "must be a mapping."
        )

    model["checkpoint_path"] = (
        validated_checkpoint
    )
    input_metadata["source_path"] = (
        validated_source
    )

    return exported


def run_single_window_prediction(
    *,
    engine: IQInferenceEngine,
    loaded: LoadedIQ,
    position: object,
    checkpoint_path: str | Path,
    top_k: int,
) -> DemoPrediction:
    """Run one GUI prediction through the deployment API."""
    selected = select_loaded_window(
        loaded,
        position,
    )
    prediction = engine.predict_batch(
        selected.iq
    )

    document = build_prediction_document(
        loaded=selected,
        prediction=prediction,
        checkpoint_path=checkpoint_path,
        device=str(engine.device),
        input_scale=engine.input_scale,
        top_k=top_k,
        checkpoint_metadata=(
            engine.checkpoint_metadata
        ),
    )

    raw_predictions = document.get(
        "predictions"
    )

    if (
        not isinstance(
            raw_predictions,
            list,
        )
        or len(raw_predictions) != 1
        or not isinstance(
            raw_predictions[0],
            dict,
        )
    ):
        raise RuntimeError(
            "Prediction document did not "
            "contain exactly one prediction."
        )

    return DemoPrediction(
        document=document,
        predicted_record=(
            raw_predictions[0]
        ),
    )


__all__ = [
    "CheckpointOption",
    "DemoConfig",
    "DemoPrediction",
    "SignalViewData",
    "build_public_prediction_document",
    "build_signal_view_data",
    "discover_checkpoints",
    "load_demo_config",
    "load_uploaded_iq",
    "run_single_window_prediction",
    "select_loaded_window",
]
