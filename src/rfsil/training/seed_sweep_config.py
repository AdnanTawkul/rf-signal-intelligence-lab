from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def expand_seed_templates(
    value: Any,
    seed: int,
) -> Any:
    """Replace {seed} recursively inside configuration strings."""
    if isinstance(value, str):
        return value.replace(
            "{seed}",
            str(seed),
        )

    if isinstance(value, list):
        return [
            expand_seed_templates(item, seed)
            for item in value
        ]

    if isinstance(value, Mapping):
        return {
            key: expand_seed_templates(
                item,
                seed,
            )
            for key, item in value.items()
        }

    return value


def prepare_seed_training_config(
    *,
    base_content: Mapping[str, Any],
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    """Create one self-contained training config for a seed."""
    expanded = expand_seed_templates(
        dict(base_content),
        seed,
    )

    if not isinstance(expanded, dict):
        raise TypeError(
            "Expanded configuration must be "
            "a dictionary."
        )

    expanded["seed"] = int(seed)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        yaml.safe_dump(
            expanded,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return expanded


__all__ = [
    "expand_seed_templates",
    "prepare_seed_training_config",
]
