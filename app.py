from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from rfsil.demo import (
    build_public_prediction_document,
    build_signal_view_data,
    discover_checkpoints,
    load_demo_config,
    load_uploaded_iq,
    run_single_window_prediction,
    select_loaded_window,
)
from rfsil.deployment import IQInferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "streamlit_demo_v1.yaml"
)


@st.cache_resource(
    show_spinner=False,
)
def load_inference_engine(
    checkpoint_path: str,
    device: str,
    input_scale: float,
    expected_sample_count: int,
) -> IQInferenceEngine:
    """Load and cache one checkpoint-backed model."""
    return IQInferenceEngine.from_checkpoint(
        checkpoint_path,
        device=device,
        input_scale=input_scale,
        expected_sample_count=(
            expected_sample_count
        ),
    )


def default_checkpoint_index(
    paths: list[Path],
    preferred: Path | None,
) -> int:
    """Return the preferred checkpoint index."""
    if preferred is None:
        return 0

    preferred_resolved = (
        preferred.resolve()
    )

    for index, path in enumerate(paths):
        if path.resolve() == preferred_resolved:
            return index

    return 0


def render_signal_plots(
    view,
) -> None:
    """Render waveform, constellation, and spectrum."""
    st.subheader("Signal overview")

    waveform_figure, waveform_axis = (
        plt.subplots(
            figsize=(10.5, 3.5)
        )
    )
    time_ms = (
        view.waveform_time_seconds
        * 1000.0
    )

    waveform_axis.plot(
        time_ms,
        view.waveform_i,
        label="I",
        linewidth=1.0,
    )
    waveform_axis.plot(
        time_ms,
        view.waveform_q,
        label="Q",
        linewidth=1.0,
    )
    waveform_axis.set_xlabel(
        "Time (ms)"
    )
    waveform_axis.set_ylabel(
        "Amplitude"
    )
    waveform_axis.set_title(
        "IQ waveform"
    )
    waveform_axis.grid(
        alpha=0.25
    )
    waveform_axis.legend(
        frameon=False
    )
    waveform_figure.tight_layout()

    st.pyplot(
        waveform_figure,
        clear_figure=True,
    )
    plt.close(waveform_figure)

    left_column, right_column = (
        st.columns(2)
    )

    with left_column:
        constellation_figure, (
            constellation_axis
        ) = plt.subplots(
            figsize=(5.2, 4.5)
        )
        constellation_axis.scatter(
            view.constellation_i,
            view.constellation_q,
            s=8,
            alpha=0.45,
        )
        constellation_axis.set_xlabel(
            "In-phase"
        )
        constellation_axis.set_ylabel(
            "Quadrature"
        )
        constellation_axis.set_title(
            "Constellation"
        )
        constellation_axis.grid(
            alpha=0.25
        )
        constellation_axis.set_aspect(
            "equal",
            adjustable="box",
        )
        constellation_figure.tight_layout()

        st.pyplot(
            constellation_figure,
            clear_figure=True,
        )
        plt.close(
            constellation_figure
        )

    with right_column:
        spectrum_figure, spectrum_axis = (
            plt.subplots(
                figsize=(5.2, 4.5)
            )
        )
        spectrum_axis.plot(
            (
                view
                .spectrum_frequency_hz
                / 1000.0
            ),
            view.spectrum_power_db,
            linewidth=1.0,
        )
        spectrum_axis.set_xlabel(
            "Frequency (kHz)"
        )
        spectrum_axis.set_ylabel(
            "Relative power (dB)"
        )
        spectrum_axis.set_title(
            "Power spectrum"
        )
        spectrum_axis.grid(
            alpha=0.25
        )
        spectrum_figure.tight_layout()

        st.pyplot(
            spectrum_figure,
            clear_figure=True,
        )
        plt.close(spectrum_figure)


def main() -> None:
    """Render the RF Signal Intelligence Lab GUI."""
    st.set_page_config(
        page_title=(
            "RF Signal Intelligence Lab"
        ),
        page_icon="??",
        layout="wide",
    )

    config = load_demo_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )

    st.title(
        "RF Signal Intelligence Lab"
    )
    st.caption(
        "Interactive modulation recognition "
        "from raw I/Q samples"
    )

    checkpoints = discover_checkpoints(
        config.checkpoint_search_root
    )

    if not checkpoints:
        st.error(
            "No best_model.pt checkpoints were "
            "found under the configured results "
            "directory."
        )
        st.stop()

    checkpoint_paths = [
        option.path
        for option in checkpoints
    ]
    checkpoint_labels = [
        option.label
        for option in checkpoints
    ]

    with st.sidebar:
        st.header("Model")

        checkpoint_index = st.selectbox(
            "Checkpoint",
            options=range(
                len(checkpoints)
            ),
            index=default_checkpoint_index(
                checkpoint_paths,
                config.preferred_checkpoint,
            ),
            format_func=lambda index: (
                checkpoint_labels[index]
            ),
        )
        checkpoint_path = (
            checkpoint_paths[
                checkpoint_index
            ]
        )

        device_options = [
            "auto",
            "cpu",
            "cuda",
        ]
        device = st.selectbox(
            "Inference device",
            options=device_options,
            index=device_options.index(
                config.default_device
            ),
        )

        st.header("Input")

        array_key = st.text_input(
            "NPZ IQ array key",
            value="iq",
        )
        sample_rate_hz = st.number_input(
            "Sample rate (Hz)",
            min_value=1.0,
            value=float(
                config
                .default_sample_rate_hz
            ),
            step=1000.0,
            format="%.1f",
        )

        st.header("Inference")

        st.write(
            "Expected samples:",
            config.expected_sample_count,
        )
        st.write(
            "Input scale:",
            config.input_scale,
        )
        st.write(
            "Top-k:",
            config.top_k,
        )

    uploaded_file = st.file_uploader(
        "Upload an IQ file",
        type=[
            "npy",
            "npz",
        ],
        help=(
            "Supported layouts include complex "
            "[samples], complex [batch, samples], "
            "real [2, samples], and real "
            "[batch, 2, samples]."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload a .npy or .npz IQ file "
            "to begin."
        )
        st.stop()

    uploaded_bytes = (
        uploaded_file.getvalue()
    )

    try:
        loaded = load_uploaded_iq(
            filename=uploaded_file.name,
            content=uploaded_bytes,
            array_key=array_key,
            expected_sample_count=(
                config.expected_sample_count
            ),
        )
    except (
        FileNotFoundError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(str(error))
        st.stop()

    metadata_columns = st.columns(4)

    metadata_columns[0].metric(
        "Windows",
        loaded.batch_size,
    )
    metadata_columns[1].metric(
        "Channels",
        loaded.channel_count,
    )
    metadata_columns[2].metric(
        "Samples/window",
        loaded.sample_count,
    )
    metadata_columns[3].metric(
        "File size",
        f"{len(uploaded_bytes) / 1024:.1f} KiB",
    )

    sample_position = int(
        st.number_input(
            "Window position",
            min_value=0,
            max_value=(
                loaded.batch_size - 1
            ),
            value=0,
            step=1,
            help=(
                "Position within the uploaded "
                "batch, not necessarily the "
                "original dataset sample index."
            ),
        )
    )

    selected = select_loaded_window(
        loaded,
        sample_position,
    )

    if selected.labels is not None:
        st.write(
            "Ground-truth class index:",
            int(selected.labels[0]),
        )

    if selected.snr_db is not None:
        st.write(
            "SNR:",
            f"{float(selected.snr_db[0]):.2f} dB",
        )

    try:
        signal_view = (
            build_signal_view_data(
                selected.iq,
                sample_rate_hz=(
                    sample_rate_hz
                ),
                maximum_waveform_points=(
                    config
                    .maximum_waveform_points
                ),
                maximum_constellation_points=(
                    config
                    .maximum_constellation_points
                ),
                spectrum_fft_size=(
                    config
                    .spectrum_fft_size
                ),
            )
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    render_signal_plots(signal_view)

    st.subheader(
        "Modulation prediction"
    )

    file_digest = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()
    prediction_key = (
        file_digest,
        str(checkpoint_path),
        device,
        sample_position,
        array_key,
    )

    if st.button(
        "Run modulation classification",
        type="primary",
    ):
        try:
            with st.spinner(
                "Loading model and running "
                "inference..."
            ):
                engine = (
                    load_inference_engine(
                        str(checkpoint_path),
                        device,
                        config.input_scale,
                        config
                        .expected_sample_count,
                    )
                )
                result = (
                    run_single_window_prediction(
                        engine=engine,
                        loaded=loaded,
                        position=(
                            sample_position
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                        top_k=config.top_k,
                    )
                )

            st.session_state[
                "demo_prediction_key"
            ] = prediction_key
            st.session_state[
                "demo_prediction"
            ] = result
        except (
            FileNotFoundError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            st.error(str(error))

    stored_key = st.session_state.get(
        "demo_prediction_key"
    )
    result = st.session_state.get(
        "demo_prediction"
    )

    if (
        stored_key == prediction_key
        and result is not None
    ):
        record = result.predicted_record

        prediction_columns = st.columns(3)

        prediction_columns[0].metric(
            "Predicted modulation",
            str(
                record[
                    "predicted_label"
                ]
            ),
        )
        prediction_columns[1].metric(
            "Confidence",
            (
                f"{100.0 * float(
                    record['confidence']
                ):.2f}%"
            ),
        )
        prediction_columns[2].metric(
            "Sample index",
            int(
                record["sample_index"]
            ),
        )

        top_k = record["top_k"]
        probability_table = pd.DataFrame(
            top_k
        )

        st.bar_chart(
            probability_table.set_index(
                "label"
            )["probability"]
        )

        st.dataframe(
            probability_table,
            width="stretch",
            hide_index=True,
        )

        export_document = (
            build_public_prediction_document(
                result.document,
                source_name=(
                    uploaded_file.name
                ),
                checkpoint_reference=(
                    checkpoint_labels[
                        checkpoint_index
                    ]
                ),
            )
        )
        export_json = (
            json.dumps(
                export_document,
                indent=2,
            )
            + "\n"
        )

        st.download_button(
            "Download prediction JSON",
            data=export_json,
            file_name=(
                "rf_iq_prediction.json"
            ),
            mime="application/json",
        )

        with st.expander(
            "Prediction details"
        ):
            st.json(export_document)

    st.subheader(
        "Channel-shift assessment"
    )
    st.info(
        "The IQ shift-detector model has not "
        "yet been serialized as a deployment "
        "artifact. This panel will be enabled "
        "in the next GUI implementation unit."
    )


if __name__ == "__main__":
    main()
