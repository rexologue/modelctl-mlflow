"""Utilities for loading and normalizing user-provided metadata tags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

MAX_TAG_VALUE_LENGTH = 4500


class TagError(ValueError):
    """Raised when tag input cannot be parsed into a dictionary."""


def read_json_dict(path: str | Path | None) -> dict[str, Any]:
    """Read a JSON file that must contain a top-level object.

    Parameters
    ----------
    path:
        Path to a JSON file. ``None`` means "no tags" and returns an empty dict.

    Returns
    -------
    dict[str, Any]
        Parsed dictionary.

    Raises
    ------
    TagError
        If the file does not contain a JSON object.
    """

    if path is None:
        return {}
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TagError(f"JSON tags file must contain an object: {json_path}")
    return data


def parse_key_value_items(items: Iterable[str] | None) -> dict[str, Any]:
    """Parse repeated ``key=value`` CLI arguments into a dictionary.

    Values are decoded as JSON when possible, so these inputs are valid::

        category=primary
        score=0.91
        labels=["positive","negative"]
        enabled=true

    If JSON decoding fails, the raw string is used.
    """

    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise TagError(f"Tag must have key=value format: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise TagError(f"Tag key is empty: {item}")
        result[key] = _decode_cli_value(raw_value.strip())
    return result


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries from left to right without mutating the inputs."""

    merged: dict[str, Any] = {}
    for item in dicts:
        merged.update(item)
    return merged


def flatten_for_mlflow_tags(prefix: str, data: dict[str, Any]) -> dict[str, str]:
    """Flatten a nested dictionary into MLflow-safe string tags.

    MLflow tags are best used for short searchable values. The full metadata is
    logged as JSON artifacts; this function only produces a flattened searchable
    projection. Nested keys are joined with dots and prefixed with a namespace,
    for example ``general.task`` or ``training.dataset.version``.

    Non-scalar values are JSON-serialized. Very long values are skipped to avoid
    backend tag length issues.
    """

    flattened: dict[str, str] = {}
    for key, value in _walk(data):
        tag_key = f"{prefix}.{key}"
        tag_value = _stringify_tag_value(value)
        if len(tag_value) <= MAX_TAG_VALUE_LENGTH:
            flattened[tag_key] = tag_value
    return flattened


def _walk(data: dict[str, Any], parent: str = "") -> Iterable[tuple[str, Any]]:
    """Yield flattened ``(key, value)`` pairs from a nested dictionary."""

    for key, value in data.items():
        safe_key = str(key).strip().replace(" ", "_")
        full_key = f"{parent}.{safe_key}" if parent else safe_key
        if isinstance(value, dict):
            yield from _walk(value, full_key)
        else:
            yield full_key, value


def _decode_cli_value(raw_value: str) -> Any:
    """Decode a CLI value as JSON if possible; otherwise return the raw string."""

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _stringify_tag_value(value: Any) -> str:
    """Convert a metadata value into a stable MLflow tag string."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
