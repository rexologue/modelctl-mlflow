"""Core implementation for the ``modelctl`` command line utility.

modelctl stores arbitrary model payloads in an MLflow artifact store and exposes
stable registry references through MLflow Model Registry. The implementation is
intentionally narrow: a registered artifact is always an opaque payload directory
plus small metadata files. No framework-specific MLflow flavors are used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .tags import flatten_for_mlflow_tags

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000
DEFAULT_EXPERIMENT_NAME = "__model_registry_uploads__"
DEFAULT_MODEL_ARTIFACT_NAME = "model"
DEFAULT_SCHEMA_VERSION = "1.0"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_HASH_SCOPE = "payload_tree_v1"
PAYLOAD_DIR_NAME = "payload"
PAYLOAD_HASH_FILE_NAME = "payload.sha256.json"


@dataclass(frozen=True)
class RegisterResult:
    """Result returned after a successful registration."""

    name: str
    version: str
    aliases: list[str]
    run_id: str
    model_uri: str
    source_uri: str
    payload_hash: str
    tracking_uri: str


@dataclass(frozen=True)
class PullResult:
    """Result returned after a successful pull."""

    ref: str
    model_uri: str
    output_path: str
    full_package: bool
    payload_hash: str | None
    download_size_bytes: int | None
    payload_size_bytes: int | None
    verified: bool
    replaced_existing: bool


@dataclass(frozen=True)
class VerifyResult:
    """Result returned after comparing a local payload with a registry version."""

    ref: str
    model_uri: str
    path: str
    expected_payload_hash: str
    actual_payload_hash: str
    matches: bool


@dataclass(frozen=True)
class ModelVersionSummary:
    """Small printable summary for one MLflow model version."""

    name: str
    version: str
    aliases: list[str]
    status: str | None
    run_id: str | None
    source: str | None
    payload_hash: str | None
    created_at: str | None


def configure_mlflow(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, tracking_uri: str | None = None) -> str:
    """Configure the MLflow Tracking URI used by the utility."""

    effective_uri = tracking_uri or f"http://{host}:{port}"
    mlflow.set_tracking_uri(effective_uri)
    return effective_uri


def register_model_directory(
    source_dir: str | Path,
    name: str,
    *,
    aliases: Iterable[str] | None = None,
    general_tags: dict[str, Any] | None = None,
    training_tags: dict[str, Any] | None = None,
    description: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> RegisterResult:
    """Register a directory as a new MLflow Model Registry version.

    The directory is treated as an opaque payload. modelctl computes a stable
    payload hash, stores the payload in the artifact store under ``model/payload``,
    creates a registry version, and writes the hash to Model Version tags.
    """

    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_path}")
    if not name.strip():
        raise ValueError("Registered model name cannot be empty")

    effective_uri = configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)

    general_tags = general_tags or {}
    training_tags = training_tags or {}

    emit_status(f"hashing payload directory: {source_path}")
    payload_hash = hash_directory(source_path, progress=True)
    emit_status(f"payload hash computed: {payload_hash}")
    created_at = utc_now_iso()

    ensure_registered_model(client, name)
    selected_aliases = list(aliases) if aliases is not None else default_aliases_for_next_version(client, name)

    manifest = build_manifest(
        model_name=name,
        payload_hash=payload_hash,
        created_at=created_at,
        general_tags=general_tags,
        training_tags=training_tags,
    )
    payload_hash_document = build_payload_hash_document(payload_hash=payload_hash, created_at=created_at)
    version_tags = build_version_tags(
        payload_hash=payload_hash,
        created_at=created_at,
        general_tags=general_tags,
        training_tags=training_tags,
    )

    run_name = f"register:{name}"
    emit_status(f"starting MLflow run: {run_name}")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags(build_run_tags(name=name, payload_hash=payload_hash))
        mlflow.log_params({"model_name": name, "payload_hash": payload_hash})
        mlflow.log_dict(general_tags, "modelctl_metadata/general_tags.json")
        mlflow.log_dict(training_tags, "modelctl_metadata/training_tags.json")
        mlflow.log_dict(manifest, "modelctl_metadata/manifest.json")
        mlflow.log_dict(payload_hash_document, f"modelctl_metadata/{PAYLOAD_HASH_FILE_NAME}")

        log_payload_package(
            source_path=source_path,
            manifest=manifest,
            payload_hash_document=payload_hash_document,
            general_tags=general_tags,
            training_tags=training_tags,
        )

        source_uri = f"runs:/{run_id}/{DEFAULT_MODEL_ARTIFACT_NAME}"
        version_tags["modelctl.source_uri"] = source_uri
        emit_status(f"creating MLflow model version: name={name}, source={source_uri}")
        model_version = client.create_model_version(
            name=name,
            source=source_uri,
            run_id=run_id,
            tags=version_tags,
            description=description,
        )

    emit_status(f"setting model version tags: name={name}, version={model_version.version}")
    for key, value in version_tags.items():
        client.set_model_version_tag(name=name, version=str(model_version.version), key=key, value=value)

    emit_status(f"setting aliases: {selected_aliases}")
    for alias in selected_aliases:
        client.set_registered_model_alias(name=name, alias=alias, version=str(model_version.version))

    model_uri = f"models:/{name}/{model_version.version}"
    emit_status(f"registered model version: name={name}, version={model_version.version}")
    return RegisterResult(
        name=name,
        version=str(model_version.version),
        aliases=selected_aliases,
        run_id=run_id,
        model_uri=model_uri,
        source_uri=source_uri,
        payload_hash=payload_hash,
        tracking_uri=effective_uri,
    )


def promote_alias(
    name: str,
    version: str,
    alias: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Point an alias at an existing model version."""

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(name=name, alias=alias, version=str(version))
    return {"name": name, "version": str(version), "alias": alias}


def pull_model(
    ref: str,
    output_dir: str | Path,
    *,
    full_package: bool = False,
    overwrite: bool = False,
    verify: bool = True,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> PullResult:
    """Download a registry ref into an output directory.

    Downloads always land in a staging directory on the same filesystem as the
    final output path. Existing output is replaced only after the new download and
    optional hash verification have succeeded.
    """

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    model_uri = normalize_model_ref(ref)
    output_path = Path(output_dir).expanduser().resolve()
    replaced_existing = output_path.exists()
    if replaced_existing and not overwrite:
        raise FileExistsError(f"Destination already exists. Use --overwrite: {output_path}")

    client = MlflowClient()
    model_version = resolve_model_version(client, ref)
    tags = dict(getattr(model_version, "tags", {}) or {})
    payload_hash = get_required_payload_hash(tags, ref=ref)
    source_uri = tags.get("modelctl.source_uri") or str(getattr(model_version, "source", None) or model_uri)

    artifact_uri = source_uri if full_package else f"{source_uri}/{PAYLOAD_DIR_NAME}"
    download_size_bytes = estimate_artifact_tree_size(client, artifact_uri)
    payload_size_bytes = (
        estimate_artifact_tree_size(client, f"{source_uri}/{PAYLOAD_DIR_NAME}") if full_package else download_size_bytes
    )
    staging_dir = make_output_staging_dir(output_path)
    downloaded_candidate: Path | None = None
    try:
        downloaded_candidate = download_artifact_to_staging(
            artifact_uri,
            staging_dir,
            total_bytes=download_size_bytes,
        )
        source_to_install = choose_downloaded_source(downloaded_candidate, full_package=full_package)

        verified = False
        if verify:
            verify_path = source_to_install / PAYLOAD_DIR_NAME if full_package else source_to_install
            actual_hash = hash_directory(verify_path, progress=True, total_bytes=payload_size_bytes)
            if actual_hash != payload_hash:
                raise ValueError(
                    f"Downloaded payload hash mismatch: expected {payload_hash}, got {actual_hash}"
                )
            verified = True

        replace_output_from_staging(source_to_install, output_path, overwrite=overwrite)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return PullResult(
        ref=ref,
        model_uri=model_uri,
        output_path=str(output_path),
        full_package=full_package,
        payload_hash=payload_hash,
        download_size_bytes=download_size_bytes,
        payload_size_bytes=payload_size_bytes,
        verified=verified,
        replaced_existing=replaced_existing,
    )


def verify_model(
    ref: str,
    path: str | Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> VerifyResult:
    """Compare a local payload directory against the registry payload hash."""

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    model_version = resolve_model_version(client, ref)
    tags = dict(getattr(model_version, "tags", {}) or {})
    expected_hash = get_required_payload_hash(tags, ref=ref)
    model_uri = normalize_model_ref(ref)

    candidate_path = Path(path).expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate_path}")
    if not candidate_path.is_dir():
        raise ValueError(f"Path must be a directory: {candidate_path}")

    actual_path = candidate_path / PAYLOAD_DIR_NAME if is_full_package(candidate_path) else candidate_path
    emit_status(f"hashing local payload directory: {actual_path}")
    actual_hash = hash_directory(actual_path, progress=True)
    matches = actual_hash == expected_hash

    return VerifyResult(
        ref=ref,
        model_uri=model_uri,
        path=str(candidate_path),
        expected_payload_hash=expected_hash,
        actual_payload_hash=actual_hash,
        matches=matches,
    )


def list_model_versions(
    name: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> list[ModelVersionSummary]:
    """Return all versions of a registered model sorted newest first."""

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{name}'"))
    versions.sort(key=lambda item: int(item.version), reverse=True)
    aliases_by_version = collect_aliases_by_version(client, name)

    summaries: list[ModelVersionSummary] = []
    for item in versions:
        full_version = fetch_model_version(client, name, str(item.version))
        summaries.append(summarize_model_version(full_version, aliases_by_version=aliases_by_version))
    return summaries


def get_model_info(
    ref: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Return registry information for a model reference."""

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    name, version_or_alias, ref_kind = split_registry_ref(ref)
    if ref_kind == "alias":
        mv = client.get_model_version_by_alias(name, version_or_alias)
    else:
        mv = client.get_model_version(name, version_or_alias)
    aliases_by_version = collect_aliases_by_version(client, name)
    summary = summarize_model_version(mv, aliases_by_version=aliases_by_version)
    return asdict(summary) | {"tags": dict(mv.tags or {})}


def log_payload_package(
    *,
    source_path: Path,
    manifest: dict[str, Any],
    payload_hash_document: dict[str, Any],
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
) -> dict[str, str]:
    """Log small package metadata and then the large payload directory."""

    with tempfile.TemporaryDirectory(prefix="modelctl_meta_") as temp_dir:
        model_dir = Path(temp_dir) / DEFAULT_MODEL_ARTIFACT_NAME
        metadata_dir = model_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        write_json(model_dir / "manifest.json", manifest)
        write_json(model_dir / PAYLOAD_HASH_FILE_NAME, payload_hash_document)
        write_json(metadata_dir / "general_tags.json", general_tags)
        write_json(metadata_dir / "training_tags.json", training_tags)
        write_text(model_dir / "MLmodel", build_mlmodel_text(manifest))

        emit_status("logging package metadata files")
        mlflow.log_artifacts(str(model_dir), artifact_path=DEFAULT_MODEL_ARTIFACT_NAME)

    emit_status(f"logging payload directory: {source_path} -> {DEFAULT_MODEL_ARTIFACT_NAME}/{PAYLOAD_DIR_NAME}")
    mlflow.log_artifacts(str(source_path), artifact_path=f"{DEFAULT_MODEL_ARTIFACT_NAME}/{PAYLOAD_DIR_NAME}")
    emit_status("payload logged")

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else ""
    return {
        "artifact_path": DEFAULT_MODEL_ARTIFACT_NAME,
        "model_uri": f"runs:/{run_id}/{DEFAULT_MODEL_ARTIFACT_NAME}" if run_id else DEFAULT_MODEL_ARTIFACT_NAME,
        "layout": "modelctl_payload_package",
    }


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create a registered model if it does not already exist."""

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def default_aliases_for_next_version(client: MlflowClient, name: str) -> list[str]:
    """Choose aliases when the user did not pass ``--alias``."""

    versions = list(client.search_model_versions(f"name='{name}'"))
    if not versions:
        return ["baseline", "champion"]
    return ["candidate"]


def build_manifest(
    *,
    model_name: str,
    payload_hash: str,
    created_at: str,
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable manifest stored next to every registered payload."""

    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "created_by": "modelctl",
        "created_at": created_at,
        "model_name": model_name,
        "artifact_layout": "modelctl_payload_package",
        "payload_path": PAYLOAD_DIR_NAME,
        "payload_hash": payload_hash,
        "hash_algorithm": DEFAULT_HASH_ALGORITHM,
        "hash_scope": DEFAULT_HASH_SCOPE,
        "payload_hash_path": PAYLOAD_HASH_FILE_NAME,
        "general_tags_path": "metadata/general_tags.json",
        "training_tags_path": "metadata/training_tags.json",
        "general_tags": general_tags,
        "training_tags": training_tags,
    }


def build_payload_hash_document(*, payload_hash: str, created_at: str) -> dict[str, Any]:
    """Build the small hash document stored in the artifact package."""

    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "created_by": "modelctl",
        "created_at": created_at,
        "payload_hash": payload_hash,
        "hash_algorithm": DEFAULT_HASH_ALGORITHM,
        "hash_scope": DEFAULT_HASH_SCOPE,
    }


def build_run_tags(name: str, payload_hash: str) -> dict[str, str]:
    """Build tags for the technical MLflow run created by modelctl."""

    return {
        "modelctl.managed": "true",
        "modelctl.operation": "register",
        "modelctl.registry_only": "true",
        "modelctl.model_name": name,
        "modelctl.payload_hash": payload_hash,
        "modelctl.hash_algorithm": DEFAULT_HASH_ALGORITHM,
        "modelctl.hash_scope": DEFAULT_HASH_SCOPE,
    }


def build_version_tags(
    *,
    payload_hash: str,
    created_at: str,
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
) -> dict[str, str]:
    """Build searchable MLflow Model Version tags."""

    tags = {
        "modelctl.managed": "true",
        "modelctl.schema_version": DEFAULT_SCHEMA_VERSION,
        "modelctl.payload_hash": payload_hash,
        "modelctl.hash_algorithm": DEFAULT_HASH_ALGORITHM,
        "modelctl.hash_scope": DEFAULT_HASH_SCOPE,
        "modelctl.created_at": created_at,
    }
    tags.update(flatten_for_mlflow_tags("general", general_tags))
    tags.update(flatten_for_mlflow_tags("training", training_tags))
    return tags


def get_required_payload_hash(tags: dict[str, str], *, ref: str) -> str:
    """Read the required payload hash from Model Version tags."""

    payload_hash = tags.get("modelctl.payload_hash")
    if not payload_hash:
        raise ValueError(f"Model version has no required tag modelctl.payload_hash: {ref}")
    return payload_hash


def hash_directory(path: Path, *, progress: bool = False, total_bytes: int | None = None) -> str:
    """Compute a stable SHA256 hash for all files in a directory.

    The hash includes relative file paths and file bytes. Directory mtimes,
    owners, groups, permissions and absolute paths are intentionally ignored.
    """

    digest = hashlib.sha256()
    processed_bytes = 0
    report_step_bytes = progress_report_step(total_bytes)
    next_report_bytes = report_step_bytes
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
                if progress:
                    processed_bytes += len(chunk)
                    if processed_bytes >= next_report_bytes:
                        emit_status(format_byte_progress("hashed", processed_bytes, total_bytes))
                        next_report_bytes += report_step_bytes
        digest.update(b"\0")
    if progress:
        emit_status(format_byte_progress("hashed total", processed_bytes, total_bytes))
    return f"{DEFAULT_HASH_ALGORITHM}:{digest.hexdigest()}"


def normalize_model_ref(ref: str) -> str:
    """Normalize user-friendly model references into MLflow model URIs."""

    if ref.startswith("models:/"):
        return ref
    name, value, ref_kind = split_registry_ref(ref)
    if ref_kind == "alias":
        return f"models:/{name}@{value}"
    return f"models:/{name}/{value}"


def split_registry_ref(ref: str) -> tuple[str, str, Literal["alias", "version"]]:
    """Split ``name@alias`` or ``name:version`` into components."""

    if ref.startswith("models:/"):
        stripped = ref.removeprefix("models:/")
        if "@" in stripped:
            name, alias = stripped.split("@", 1)
            return name, alias.split("/", 1)[0], "alias"
        parts = stripped.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1].split("/", 1)[0], "version"
        raise ValueError(f"Unsupported model URI: {ref}")

    if "@" in ref:
        name, alias = ref.split("@", 1)
        return name, alias, "alias"
    if ":" in ref:
        name, version = ref.rsplit(":", 1)
        return name, version, "version"
    raise ValueError("Model ref must be name@alias, name:version or models:/... URI")


def resolve_model_version(client: MlflowClient, ref: str) -> Any:
    """Resolve a user-facing reference to a concrete MLflow ModelVersion."""

    name, value, ref_kind = split_registry_ref(ref)
    if ref_kind == "alias":
        return client.get_model_version_by_alias(name=name, alias=value)
    return client.get_model_version(name=name, version=str(value))


def estimate_artifact_tree_size(client: MlflowClient, artifact_uri: str) -> int | None:
    """Best-effort byte size estimate for a runs:/ artifact tree."""

    parsed = split_runs_artifact_uri(artifact_uri)
    if parsed is None:
        emit_status(f"download size unavailable for non-runs artifact URI: {artifact_uri}")
        return None

    run_id, artifact_path = parsed
    try:
        total_bytes = sum_artifact_tree_size(client, run_id, artifact_path)
    except Exception as exc:  # noqa: BLE001 - artifact stores differ in listing support.
        emit_status(f"download size unavailable: {exc}")
        return None

    if total_bytes is None:
        emit_status("download size unavailable: artifact store did not report all file sizes")
        return None
    return total_bytes


def split_runs_artifact_uri(artifact_uri: str) -> tuple[str, str | None] | None:
    """Return run id and artifact path for a runs:/ URI."""

    if not artifact_uri.startswith("runs:/"):
        return None
    stripped = artifact_uri.removeprefix("runs:/").lstrip("/")
    if not stripped:
        return None
    parts = stripped.split("/", 1)
    run_id = parts[0]
    artifact_path = parts[1] if len(parts) == 2 and parts[1] else None
    return run_id, artifact_path


def sum_artifact_tree_size(client: MlflowClient, run_id: str, artifact_path: str | None) -> int | None:
    """Recursively sum MLflow artifact file sizes when the backend exposes them."""

    total_bytes = 0
    stack: list[str | None] = [artifact_path]
    while stack:
        current_path = stack.pop()
        for info in client.list_artifacts(run_id, current_path):
            if info.is_dir:
                stack.append(info.path)
                continue
            file_size = getattr(info, "file_size", None)
            if file_size is None or file_size < 0:
                return None
            total_bytes += int(file_size)
    return total_bytes


def make_output_staging_dir(output_path: Path) -> Path:
    """Create a temporary staging directory on the output filesystem."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / f".modelctl_download_{output_path.name}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    return staging_dir


def download_artifact_to_staging(artifact_uri: str, staging_dir: Path, *, total_bytes: int | None = None) -> Path:
    """Download one artifact into staging and return the downloaded path."""

    emit_status(f"downloading artifact: {artifact_uri}")
    if total_bytes is None:
        emit_status("download size: unknown")
    else:
        emit_status(f"download size: {format_bytes(total_bytes)}")

    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=report_download_progress,
        args=(staging_dir, total_bytes, stop_event),
        daemon=True,
    )
    progress_thread.start()
    try:
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=str(staging_dir))).resolve()
    finally:
        stop_event.set()
        progress_thread.join(timeout=1)
        emit_status(format_byte_progress("downloaded total", local_tree_size_bytes(staging_dir), total_bytes))


def choose_downloaded_source(downloaded: Path, *, full_package: bool) -> Path:
    """Choose the directory that should be installed from a downloaded artifact."""

    if full_package:
        if is_full_package(downloaded):
            return downloaded
        for candidate in downloaded.rglob("manifest.json"):
            package_dir = candidate.parent
            if is_full_package(package_dir):
                return package_dir
        raise FileNotFoundError("Downloaded artifact does not look like a modelctl package")

    if downloaded.is_dir():
        return downloaded
    raise ValueError(f"Downloaded payload is not a directory: {downloaded}")


def replace_output_from_staging(source_to_install: Path, output_path: Path, *, overwrite: bool) -> None:
    """Move a staged download into place without deleting existing output early."""

    backup_path: Path | None = None
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists. Use --overwrite: {output_path}")
        backup_path = output_path.parent / f".modelctl_replace_backup_{output_path.name}_{uuid.uuid4().hex}"
        output_path.rename(backup_path)

    try:
        shutil.move(str(source_to_install), str(output_path))
    except Exception:
        if backup_path is not None and backup_path.exists():
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path, ignore_errors=True)
                else:
                    output_path.unlink(missing_ok=True)
            backup_path.rename(output_path)
        raise
    else:
        if backup_path is not None:
            if backup_path.is_dir():
                shutil.rmtree(backup_path, ignore_errors=True)
            else:
                backup_path.unlink(missing_ok=True)


def is_full_package(path: Path) -> bool:
    """Return true when a directory has the modelctl package layout."""

    return (
        path.is_dir()
        and (path / "manifest.json").is_file()
        and (path / PAYLOAD_HASH_FILE_NAME).is_file()
        and (path / PAYLOAD_DIR_NAME).is_dir()
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as pretty UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_mlmodel_text(manifest: dict[str, Any]) -> str:
    """Return a small MLmodel descriptor for modelctl artifacts."""

    model_name = str(manifest.get("model_name") or "")
    payload_hash = str(manifest.get("payload_hash") or "")
    return (
        f"artifact_path: {DEFAULT_MODEL_ARTIFACT_NAME}\n"
        "flavors:\n"
        "  modelctl_payload:\n"
        f"    schema_version: {DEFAULT_SCHEMA_VERSION}\n"
        f"    payload_path: {PAYLOAD_DIR_NAME}\n"
        "    manifest_path: manifest.json\n"
        f"    payload_hash_path: {PAYLOAD_HASH_FILE_NAME}\n"
        f"    model_name: {json.dumps(model_name, ensure_ascii=False)}\n"
        f"    payload_hash: {json.dumps(payload_hash, ensure_ascii=False)}\n"
        f"modelctl_schema_version: {DEFAULT_SCHEMA_VERSION}\n"
        "modelctl_artifact_layout: modelctl_payload_package\n"
    )


def fetch_model_version(client: MlflowClient, name: str, version: str) -> Any:
    """Fetch one fully populated model version from the registry."""

    return client.get_model_version(name=name, version=str(version))


def collect_aliases_by_version(client: MlflowClient, name: str) -> dict[str, list[str]]:
    """Return a reverse mapping ``version -> [aliases]`` for a model."""

    try:
        registered_model = client.get_registered_model(name)
    except MlflowException:
        return {}

    alias_map = getattr(registered_model, "aliases", {}) or {}
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in dict(alias_map).items():
        aliases_by_version.setdefault(str(version), []).append(str(alias))

    for aliases in aliases_by_version.values():
        aliases.sort()
    return aliases_by_version


def summarize_model_version(mv: Any, *, aliases_by_version: dict[str, list[str]] | None = None) -> ModelVersionSummary:
    """Convert an MLflow ModelVersion entity into a small summary."""

    tags = dict(mv.tags or {})
    version = str(mv.version)
    version_aliases = list(getattr(mv, "aliases", []) or [])
    if aliases_by_version is not None:
        version_aliases = aliases_by_version.get(version, version_aliases)

    return ModelVersionSummary(
        name=str(mv.name),
        version=version,
        aliases=version_aliases,
        status=str(getattr(mv, "status", "")) or None,
        run_id=getattr(mv, "run_id", None),
        source=getattr(mv, "source", None),
        payload_hash=tags.get("modelctl.payload_hash"),
        created_at=tags.get("modelctl.created_at") or timestamp_ms_to_iso(getattr(mv, "creation_timestamp", None)),
    )


def timestamp_ms_to_iso(timestamp_ms: int | None) -> str | None:
    """Convert an MLflow millisecond timestamp to ISO-8601 UTC text."""

    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def emit_status(message: str) -> None:
    """Write a human-readable modelctl status line to stderr."""

    print(f"[modelctl] {message}", file=sys.stderr, flush=True)


def report_download_progress(staging_dir: Path, total_bytes: int | None, stop_event: threading.Event) -> None:
    """Periodically report bytes observed in the staging directory."""

    last_reported_bytes = -1
    while not stop_event.wait(5):
        current_bytes = local_tree_size_bytes(staging_dir)
        if current_bytes != last_reported_bytes:
            emit_status(format_byte_progress("downloaded", current_bytes, total_bytes))
            last_reported_bytes = current_bytes


def local_tree_size_bytes(path: Path) -> int:
    """Return the total size of regular files currently visible under a path."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total_bytes = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total_bytes += item.stat().st_size
        except OSError:
            continue
    return total_bytes


def progress_report_step(total_bytes: int | None) -> int:
    """Choose a progress report step suitable for large model payloads."""

    if total_bytes is None or total_bytes <= 0:
        return 5 * 1024**3
    one_percent = max(total_bytes // 100, 1)
    return max(512 * 1024**2, min(5 * 1024**3, one_percent))


def format_byte_progress(label: str, current_bytes: int, total_bytes: int | None) -> str:
    """Format byte progress with a percent when total size is known."""

    if total_bytes is None:
        return f"{label} {format_bytes(current_bytes)}"
    if total_bytes <= 0:
        return f"{label} {format_bytes(current_bytes)} / {format_bytes(total_bytes)} (100.0%)"
    percent = min((current_bytes / total_bytes) * 100, 100.0)
    return f"{label} {format_bytes(current_bytes)} / {format_bytes(total_bytes)} ({percent:.1f}%)"


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""

    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
