"""Small MLflow Model Registry utility.

The package exposes a command line interface via ``python -m modelctl`` and the
``modelctl`` console script declared in ``pyproject.toml``.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

_DISTRIBUTION_NAME = "modelctl-mlflow"


def _version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject_path.is_file():
        return None

    in_project_section = False
    project_name = None
    project_version = None
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[project]":
            in_project_section = True
            continue
        if in_project_section and line.startswith("["):
            break
        if not in_project_section:
            continue

        key, separator, value = line.partition("=")
        if not separator:
            continue

        normalized_key = key.strip()
        normalized_value = value.strip().strip("\"'")
        if normalized_key == "name":
            project_name = normalized_value
        elif normalized_key == "version":
            project_version = normalized_value

    if project_name == _DISTRIBUTION_NAME:
        return project_version
    return None


def _read_version() -> str:
    local_version = _version_from_pyproject()
    if local_version is not None:
        return local_version

    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _read_version()

__all__ = ["__version__"]
