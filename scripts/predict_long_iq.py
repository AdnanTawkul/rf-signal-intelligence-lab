from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfsil.deployment import (
    IQInferenceEngine,
    aggregate_window_predictions,
    load_iq_file,
    predict_window_batches,
    rank_probabilities,
    window_iq_signal,
    write_prediction_document,
)


def parse_arguments() -> argparse.Namespace:
    """Parse long-signal prediction arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify a long RF IQ signal with "
            "sliding-window aggregation."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--array-key",
        default="iq",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Select one signal from an NPZ or "
            "batched NPY input."
        ),
    )
    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Sliding-window stride. Defaults "
            "to the window size."
        ),
    )
    parser.add_argument(
        "--remainder",
        choices=(
            "drop",
            "pad",
            "error",
        ),
        default="drop",
    )
    parser.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def run_prediction(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Run sliding-window IQ inference."""
    loaded = load_iq_file(
        arguments.input,
        array_key=arguments.array_key,
        sample_index=arguments.sample_index,
        expected_sample_count=None,
    )

    if loaded.batch_size != 1:
        raise ValueError(
            "Long-signal inference requires "
            "exactly one signal. Use "
            "--sample-index for batched input."
        )

    windowed = window_iq_signal(
        loaded.iq,
        window_size=arguments.window_size,
        stride=arguments.stride,
        remainder_policy=(
            arguments.remainder
        ),
        pad_value=arguments.pad_value,
    )

    engine = IQInferenceEngine.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        input_scale=arguments.input_scale,
        expected_sample_count=(
            arguments.window_size
        ),
    )

    window_predictions = (
        predict_window_batches(
            engine,
            windowed.windows,
            batch_size=arguments.batch_size,
        )
    )

    aggregate = (
        aggregate_window_predictions(
            window_predictions,
            weights=(
                windowed.valid_sample_counts
            ),
        )
    )

    top_k_count = (
        len(engine.class_names)
        if arguments.top_k is None
        else arguments.top_k
    )

    window_records = []

    for index in range(
        windowed.window_count
    ):
        start_sample = int(
            windowed.start_indices[index]
        )
        valid_sample_count = int(
            windowed.valid_sample_counts[
                index
            ]
        )
        predicted_index = int(
            window_predictions
            .predicted_indices[index]
        )

        window_records.append(
            {
                "window_index": index,
                "start_sample": start_sample,
                "stop_sample_exclusive": (
                    start_sample
                    + valid_sample_count
                ),
                "valid_sample_count": (
                    valid_sample_count
                ),
                "predicted_index": (
                    predicted_index
                ),
                "predicted_label": (
                    window_predictions
                    .predicted_labels[index]
                ),
                "confidence": float(
                    window_predictions
                    .confidences[index]
                ),
                "probabilities": [
                    float(value)
                    for value in (
                        window_predictions
                        .probabilities[index]
                    )
                ],
                "top_k": rank_probabilities(
                    window_predictions
                    .probabilities[index],
                    class_names=(
                        engine.class_names
                    ),
                    top_k=top_k_count,
                ),
            }
        )

    aggregate_record: dict[
        str,
        object,
    ] = {
        "predicted_index": (
            aggregate.predicted_index
        ),
        "predicted_label": (
            aggregate.predicted_label
        ),
        "confidence": aggregate.confidence,
        "probabilities": list(
            aggregate.probabilities
        ),
        "top_k": rank_probabilities(
            aggregate.probabilities,
            class_names=engine.class_names,
            top_k=top_k_count,
        ),
    }

    if loaded.labels is not None:
        true_index = int(
            loaded.labels[0]
        )

        if not (
            0
            <= true_index
            < len(engine.class_names)
        ):
            raise ValueError(
                "Ground-truth class index is "
                "out of range."
            )

        aggregate_record.update(
            {
                "true_index": true_index,
                "true_label": (
                    engine.class_names[
                        true_index
                    ]
                ),
                "correct": (
                    aggregate.predicted_index
                    == true_index
                ),
            }
        )

    if loaded.snr_db is not None:
        aggregate_record["snr_db"] = float(
            loaded.snr_db[0]
        )

    document = {
        "format_version": 1,
        "model": {
            "checkpoint_path": (
                arguments.checkpoint
                .resolve()
                .as_posix()
            ),
            "device": str(engine.device),
            "input_scale": (
                engine.input_scale
            ),
            "class_names": list(
                engine.class_names
            ),
            "checkpoint_metadata": (
                engine.checkpoint_metadata
            ),
        },
        "input": {
            "source_path": (
                loaded.source_path
                .resolve()
                .as_posix()
            ),
            "array_key": loaded.array_key,
            "source_sample_index": int(
                loaded.sample_indices[0]
            ),
            "channel_count": (
                loaded.channel_count
            ),
            "original_sample_count": (
                windowed
                .original_sample_count
            ),
        },
        "windowing": {
            "window_size": (
                windowed.window_size
            ),
            "stride": windowed.stride,
            "remainder_policy": (
                windowed.remainder_policy
            ),
            "pad_value": (
                windowed.pad_value
            ),
            "window_count": (
                windowed.window_count
            ),
            "aggregation": (
                "valid-sample-weighted "
                "mean probability"
            ),
        },
        "configuration": {
            "batch_size": (
                arguments.batch_size
            ),
            "top_k": top_k_count,
        },
        "aggregate_prediction": (
            aggregate_record
        ),
        "window_predictions": (
            window_records
        ),
    }

    if arguments.output is not None:
        write_prediction_document(
            arguments.output,
            document,
        )

    return document


def main() -> None:
    """Run long-signal prediction."""
    arguments = parse_arguments()

    try:
        document = run_prediction(
            arguments
        )
    except (
        FileNotFoundError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    print(
        json.dumps(
            document,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
