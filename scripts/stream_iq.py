from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfsil.deployment import (
    IQInferenceEngine,
    StreamingIQClassifier,
    load_iq_file,
    rank_probabilities,
    write_prediction_document,
)


def parse_arguments() -> argparse.Namespace:
    """Parse streaming simulation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Simulate online RF IQ inference "
            "from an NPY or NPZ signal."
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
        "--hop-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--inference-batch-size",
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


def run_stream(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Run file-backed streaming inference."""
    if arguments.chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    loaded = load_iq_file(
        arguments.input,
        array_key=arguments.array_key,
        sample_index=arguments.sample_index,
        expected_sample_count=None,
    )

    if loaded.batch_size != 1:
        raise ValueError(
            "Streaming inference requires "
            "exactly one signal. Use "
            "--sample-index for batched input."
        )

    engine = IQInferenceEngine.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        input_scale=arguments.input_scale,
        expected_sample_count=(
            arguments.window_size
        ),
    )

    classifier = StreamingIQClassifier(
        engine=engine,
        window_size=arguments.window_size,
        hop_size=arguments.hop_size,
        sample_rate_hz=(
            arguments.sample_rate_hz
        ),
        inference_batch_size=(
            arguments.inference_batch_size
        ),
    )

    signal = loaded.iq[0]
    events = []

    for start in range(
        0,
        loaded.sample_count,
        arguments.chunk_size,
    ):
        stop = min(
            start + arguments.chunk_size,
            loaded.sample_count,
        )

        events.extend(
            classifier.push(
                signal[:, start:stop]
            )
        )

    if not events:
        raise ValueError(
            "No complete streaming windows "
            "were produced."
        )

    top_k_count = (
        len(engine.class_names)
        if arguments.top_k is None
        else arguments.top_k
    )

    prediction_records = []

    for event in events:
        prediction_records.append(
            {
                "sequence_index": (
                    event.sequence_index
                ),
                "start_sample": (
                    event.start_sample
                ),
                "stop_sample_exclusive": (
                    event
                    .stop_sample_exclusive
                ),
                "start_time_seconds": (
                    event.start_time_seconds
                ),
                "end_time_seconds": (
                    event.end_time_seconds
                ),
                "center_time_seconds": (
                    event.center_time_seconds
                ),
                "predicted_index": (
                    event.predicted_index
                ),
                "predicted_label": (
                    event.predicted_label
                ),
                "confidence": (
                    event.confidence
                ),
                "logits": list(
                    event.logits
                ),
                "probabilities": list(
                    event.probabilities
                ),
                "top_k": rank_probabilities(
                    event.probabilities,
                    class_names=(
                        engine.class_names
                    ),
                    top_k=top_k_count,
                ),
            }
        )

    last_stop = int(
        events[-1].stop_sample_exclusive
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
            "sample_count": (
                loaded.sample_count
            ),
        },
        "streaming": {
            "window_size": (
                classifier.window_size
            ),
            "hop_size": (
                classifier.hop_size
            ),
            "chunk_size": (
                arguments.chunk_size
            ),
            "sample_rate_hz": (
                classifier.sample_rate_hz
            ),
            "inference_batch_size": (
                classifier
                .inference_batch_size
            ),
            "prediction_count": len(
                prediction_records
            ),
            "last_complete_window_stop": (
                last_stop
            ),
            "trailing_sample_count": (
                loaded.sample_count
                - last_stop
            ),
        },
        "configuration": {
            "top_k": top_k_count,
        },
        "predictions": prediction_records,
    }

    if loaded.labels is not None:
        true_index = int(
            loaded.labels[0]
        )

        document["input"][
            "true_index"
        ] = true_index
        document["input"][
            "true_label"
        ] = engine.class_names[
            true_index
        ]

    if loaded.snr_db is not None:
        document["input"]["snr_db"] = float(
            loaded.snr_db[0]
        )

    if arguments.output is not None:
        write_prediction_document(
            arguments.output,
            document,
        )

    return document


def main() -> None:
    """Run streaming inference."""
    arguments = parse_arguments()

    try:
        document = run_stream(arguments)
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
