from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Config file must define a mapping at the top level: {config_path}")

    return config


def apply_overrides(
    config: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    updated_config = deepcopy(config)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")

        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_dotted_value(updated_config, dotted_key, value)

    return updated_config


def set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config

    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        elif not isinstance(next_value, dict):
            raise TypeError(f"Cannot override nested key '{dotted_key}' because '{key}' is not a mapping.")

        cursor = next_value

    cursor[keys[-1]] = value
