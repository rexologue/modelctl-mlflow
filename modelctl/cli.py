"""Command line interface for modelctl."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .core import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    VerifyResult,
    get_model_info,
    list_model_versions,
    promote_alias,
    pull_model,
    register_model_directory,
    verify_model,
)
from .tags import TagError, merge_dicts, parse_key_value_items, read_json_dict


TOP_LEVEL_HELP = """\
Small MLflow Model Registry utility for arbitrary model payloads.

modelctl stores a directory as an opaque payload in MLflow artifacts and keeps
registry metadata in MLflow Model Registry: versions, aliases, tags and source
URIs. It does not require the payload to be a Python, sklearn, torch or any
other framework-specific model.
"""

TOP_LEVEL_EPILOG = """\
Quick start:
  modelctl register ./model my-model
  modelctl list my-model
  modelctl info my-model@champion
  modelctl pull my-model@champion ./downloaded-model
  modelctl verify my-model@champion ./downloaded-model
  modelctl promote my-model 3 champion

Model refs:
  my-model@champion           Resolve by alias
  my-model:3                  Resolve by version
  models:/my-model@champion   MLflow alias URI form
  models:/my-model/3          MLflow version URI form

Connection:
  By default modelctl uses http://localhost:5000.
  Override with --host/--port or pass --tracking-uri to any command.
  MLflow auth is handled by MLflow environment variables, for example:
  MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD.

Registration:
  The first version gets aliases baseline and champion by default.
  Later versions get alias candidate by default.
  Pass --alias one or more times to set explicit aliases.
  Optional metadata can be passed with --general-tag, --training-tag,
  --general-tags-json and --training-tags-json.

Pull and verify:
  pull downloads payload only by default. Use --full-package to download the
  whole modelctl package with manifest and metadata files.
  pull verifies the payload hash by default. Use --no-verify only when you
  intentionally want to skip that check.
  verify exits with 0 on match, 2 on hash mismatch and 1 on command failure.

Output:
  Commands print machine-readable JSON to stdout.
  Human-readable progress and errors go to stderr.

More help:
  modelctl <command> --help
"""


def main(argv: list[str] | None = None) -> int:
    """Run the modelctl CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = dispatch(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print clear errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result is not None:
        print_json(result)

    if isinstance(result, VerifyResult) and not result.matches:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser and subcommands."""

    parser = argparse.ArgumentParser(
        prog="modelctl",
        description=TOP_LEVEL_HELP,
        epilog=TOP_LEVEL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"modelctl {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_register_parser(subparsers)
    add_promote_parser(subparsers)
    add_pull_parser(subparsers)
    add_verify_parser(subparsers)
    add_list_parser(subparsers)
    add_info_parser(subparsers)
    return parser


def add_common_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add MLflow connection flags shared by all commands."""

    parser.add_argument("--host", default=DEFAULT_HOST, help="MLflow host. Default: localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MLflow port. Default: 5000.")
    parser.add_argument("--tracking-uri", default=None, help="Full MLflow tracking URI. Overrides --host and --port.")


def add_register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``register`` command parser."""

    parser = subparsers.add_parser("register", help="Register a payload directory as a new MLflow model version.")
    parser.add_argument("source_dir", help="Payload directory to register.")
    parser.add_argument("name", help="Registered model name.")
    parser.add_argument("--alias", action="append", default=None, help="Alias to set on the new version. Can be repeated.")
    parser.add_argument("--general-tags-json", default=None, help="Path to JSON object with general metadata.")
    parser.add_argument("--training-tags-json", default=None, help="Path to JSON object with training metadata.")
    parser.add_argument("--general-tag", action="append", default=None, help="Inline general tag key=value. Can be repeated.")
    parser.add_argument("--training-tag", action="append", default=None, help="Inline training tag key=value. Can be repeated.")
    parser.add_argument("--description", default=None, help="Optional model version description.")
    add_common_connection_args(parser)


def add_promote_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``promote`` command parser."""

    parser = subparsers.add_parser("promote", help="Point an alias to an existing model version.")
    parser.add_argument("name", help="Registered model name.")
    parser.add_argument("version", help="Model version number.")
    parser.add_argument("alias", help="Alias to set.")
    add_common_connection_args(parser)


def add_pull_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``pull`` command parser."""

    parser = subparsers.add_parser("pull", help="Download a model version or alias.")
    parser.add_argument("ref", help="Model ref: name@alias, name:version, models:/name@alias or models:/name/version.")
    parser.add_argument("output_dir", help="Destination directory.")
    parser.add_argument("--full-package", action="store_true", help="Download the full modelctl package instead of payload only.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists.")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-download payload hash verification.")
    add_common_connection_args(parser)


def add_verify_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``verify`` command parser."""

    parser = subparsers.add_parser("verify", help="Compare a directory with the registry payload hash.")
    parser.add_argument("ref", help="Model ref: name@alias, name:version, models:/name@alias or models:/name/version.")
    parser.add_argument("path", help="Directory to verify. Payload directories and full modelctl packages are both accepted.")
    add_common_connection_args(parser)


def add_list_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``list`` command parser."""

    parser = subparsers.add_parser("list", help="List versions of a registered model.")
    parser.add_argument("name", help="Registered model name.")
    add_common_connection_args(parser)


def add_info_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``info`` command parser."""

    parser = subparsers.add_parser("info", help="Show JSON info for one model ref.")
    parser.add_argument("ref", help="Model ref: name@alias, name:version, models:/name@alias or models:/name/version.")
    add_common_connection_args(parser)


def dispatch(args: argparse.Namespace) -> Any:
    """Dispatch parsed CLI arguments to the corresponding core function."""

    if args.command == "register":
        general_tags = load_tags(args.general_tags_json, args.general_tag)
        training_tags = load_tags(args.training_tags_json, args.training_tag)
        return register_model_directory(
            args.source_dir,
            args.name,
            aliases=args.alias,
            general_tags=general_tags,
            training_tags=training_tags,
            description=args.description,
            host=args.host,
            port=args.port,
            tracking_uri=args.tracking_uri,
        )

    if args.command == "promote":
        return promote_alias(
            args.name,
            args.version,
            args.alias,
            host=args.host,
            port=args.port,
            tracking_uri=args.tracking_uri,
        )

    if args.command == "pull":
        return pull_model(
            args.ref,
            args.output_dir,
            full_package=args.full_package,
            overwrite=args.overwrite,
            verify=not args.no_verify,
            host=args.host,
            port=args.port,
            tracking_uri=args.tracking_uri,
        )

    if args.command == "verify":
        return verify_model(args.ref, args.path, host=args.host, port=args.port, tracking_uri=args.tracking_uri)

    if args.command == "list":
        return list_model_versions(args.name, host=args.host, port=args.port, tracking_uri=args.tracking_uri)

    if args.command == "info":
        return get_model_info(args.ref, host=args.host, port=args.port, tracking_uri=args.tracking_uri)

    raise ValueError(f"Unknown command: {args.command}")


def load_tags(json_path: str | None, inline_items: list[str] | None) -> dict[str, Any]:
    """Load optional JSON and inline tags, then merge them."""

    try:
        return merge_dicts(read_json_dict(json_path), parse_key_value_items(inline_items))
    except TagError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize CLI error text.
        raise TagError(str(exc)) from exc


def print_json(value: Any) -> None:
    """Print dataclasses, lists and dictionaries as pretty UTF-8 JSON."""

    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses recursively into JSON-serializable objects."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
