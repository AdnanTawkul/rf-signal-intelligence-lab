from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "convert_radioml2016.py"
)


def create_groups(
    examples_per_group: int = 10,
) -> dict[tuple[str, int], np.ndarray]:
    groups: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}

    for class_index, modulation in enumerate(
        (
            "BPSK",
            "QPSK",
            "8PSK",
            "QAM16",
        )
    ):
        for snr_index, snr in enumerate(
            (-2, 2)
        ):
            rng = np.random.default_rng(
                class_index * 100
                + snr_index
            )

            groups[
                (modulation, snr)
            ] = rng.normal(
                size=(
                    examples_per_group,
                    2,
                    128,
                )
            ).astype(np.float32)

    return groups


def write_test_inputs(
    tmp_path: Path,
    *,
    expected_md5: str | None = None,
) -> tuple[Path, Path]:
    archive_path = (
        tmp_path / "RML2016.10a.tar.bz2"
    )
    pickle_path = (
        tmp_path
        / "RML2016.10a_dict_optimized.pkl"
    )
    output_directory = (
        tmp_path / "processed"
    )

    archive_path.write_bytes(
        b"test archive bytes"
    )

    with pickle_path.open("wb") as file:
        pickle.dump(
            create_groups(),
            file,
            protocol=2,
        )

    actual_md5 = hashlib.md5(
        archive_path.read_bytes()
    ).hexdigest()

    config = {
        "dataset_name": (
            "radioml2016_test"
        ),
        "seed": 2026,
        "source": {
            "dataset_name": (
                "RadioML 2016.10A"
            ),
            "archive_path": str(
                archive_path
            ),
            "pickle_path": str(
                pickle_path
            ),
            "download_url": (
                "https://example.invalid/"
                "radioml.tar.bz2"
            ),
            "expected_archive_md5": (
                expected_md5
                if expected_md5 is not None
                else actual_md5
            ),
            "license": (
                "CC BY-NC-SA 4.0"
            ),
        },
        "output_directory": str(
            output_directory
        ),
        "splits": {
            "train": 6,
            "validation": 2,
            "test": 2,
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return config_path, output_directory


def run_converter(
    config_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CONVERTER_PATH),
            "--config",
            str(config_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_converter_creates_balanced_canonical_splits(
    tmp_path: Path,
) -> None:
    config_path, output_directory = (
        write_test_inputs(tmp_path)
    )

    result = run_converter(config_path)

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
    assert (
        "RadioML 2016 four-class "
        "conversion: OK"
        in result.stdout
    )

    expected_sizes = {
        "train": 48,
        "validation": 16,
        "test": 16,
    }

    for split_name, expected_size in (
        expected_sizes.items()
    ):
        path = (
            output_directory
            / f"{split_name}.npz"
        )

        assert path.is_file()

        with np.load(
            path,
            allow_pickle=False,
        ) as content:
            assert content["iq"].shape == (
                expected_size,
                2,
                128,
            )
            assert (
                content["iq"].dtype
                == np.float32
            )
            assert (
                content["labels"].dtype
                == np.int64
            )

            expected_group_count = (
                6
                if split_name == "train"
                else 2
            )

            for label in range(4):
                for snr in (-2, 2):
                    count = int(
                        np.sum(
                            (
                                content["labels"]
                                == label
                            )
                            & (
                                content["snr_db"]
                                == snr
                            )
                        )
                    )

                    assert count == (
                        expected_group_count
                    )

    manifest = json.loads(
        (
            output_directory
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["selection"][
        "selected_example_count"
    ] == 80
    assert manifest["selection"][
        "selected_group_count"
    ] == 8
    assert manifest["selection"][
        "snr_values_db"
    ] == [-2, 2]

    assert manifest[
        "metadata_availability"
    ]["example_seed"].startswith(
        "deterministic compatibility"
    )


def test_converter_rejects_archive_hash_mismatch(
    tmp_path: Path,
) -> None:
    config_path, output_directory = (
        write_test_inputs(
            tmp_path,
            expected_md5="0" * 32,
        )
    )

    result = run_converter(config_path)

    assert result.returncode != 0
    assert (
        "Archive MD5 mismatch"
        in result.stderr
    )
    assert not output_directory.exists()
