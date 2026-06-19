from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfsil.deployment import (
    IQInferenceEngine,
    build_prediction_document,
    load_iq_file,
    write_prediction_document,
)


def parse_arguments() -> argparse.Namespace:
    """Parse IQ prediction arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify fixed-length RF IQ "
            "windows with a trained checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained checkpoint path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .npy or .npz IQ file.",
    )
    parser.add_argument(
        "--array-key",
        default="iq",
        help=(
            "IQ array key for NPZ files "
            "(default: iq)."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Predict one dataset sample. "
            "Omit to predict the complete batch."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        help=(
            "Inference device "
            "(default: auto)."
        ),
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplicative IQ input scale "
            "(default: 1.0)."
        ),
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=2048,
        help=(
            "Required samples per IQ window "
            "(default: 2048)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of ranked classes to "
            "include. The default includes all."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. "
            "JSON is always also written to "
            "standard output."
        ),
    )

    return parser.parse_args()


def run_prediction(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Run file loading and batch inference."""
    engine = IQInferenceEngine.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        input_scale=arguments.input_scale,
        expected_sample_count=(
            arguments.expected_samples
        ),
    )

    loaded = load_iq_file(
        arguments.input,
        array_key=arguments.array_key,
        sample_index=arguments.sample_index,
        expected_sample_count=(
            arguments.expected_samples
        ),
    )

    prediction = engine.predict_batch(
        loaded.iq
    )

    document = build_prediction_document(
        loaded=loaded,
        prediction=prediction,
        checkpoint_path=(
            arguments.checkpoint
        ),
        device=str(engine.device),
        input_scale=engine.input_scale,
        top_k=arguments.top_k,
        checkpoint_metadata=(
            engine.checkpoint_metadata
        ),
    )

    if arguments.output is not None:
        write_prediction_document(
            arguments.output,
            document,
        )

    return document


def main() -> None:
    """Run the command-line prediction workflow."""
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
