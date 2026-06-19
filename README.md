# modelctl-mlflow

`modelctl` is a small CLI wrapper around MLflow Model Registry for storing, verifying, promoting and pulling arbitrary model payloads.

The tool does not try to interpret a payload as a specific ML framework. A registered version is always:

```text
small metadata package + opaque payload directory
```

MLflow backend storage keeps metadata: runs, versions, aliases, tags and source URIs. MLflow artifact storage keeps the actual payload bytes.

## Installation

Install as a CLI tool with uv:

```bash
uv tool install modelctl-mlflow
```

Run once without installing the tool globally:

```bash
uvx --from modelctl-mlflow modelctl --help
```

Install with pip:

```bash
pip install modelctl-mlflow
```

For local development from a checkout:

```bash
pip install -e .
```

or:

```bash
uv pip install -e .
```

After installation, the CLI command is:

```bash
modelctl --help
```

The PyPI package name is `modelctl-mlflow`; the installed command is `modelctl`.

## Quick start

```bash
modelctl register ./model my-model
modelctl list my-model
modelctl info my-model@champion
modelctl pull my-model@champion ./downloaded-model
modelctl verify my-model@champion ./downloaded-model
modelctl promote my-model 3 champion
```

Commands print machine-readable JSON to stdout. Human-readable progress and errors are printed to stderr.

## MLflow connection

Default tracking URI:

```text
http://localhost:5000
```

Override host and port:

```bash
modelctl list <model_name> --host <host> --port <port>
```

Or pass a full tracking URI:

```bash
modelctl list <model_name> --tracking-uri <tracking_uri>
```

MLflow authentication is handled by MLflow itself through its usual environment variables, for example `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.

## Register

```bash
modelctl register <payload_dir> <model_name>
```

With explicit aliases:

```bash
modelctl register <payload_dir> <model_name> --alias <alias_a> --alias <alias_b>
```

Default alias behavior:

```text
first version -> baseline + champion
later versions -> candidate
```

During registration, `modelctl`:

```text
1. connects to MLflow
2. computes a stable SHA256 payload hash
3. starts a short technical MLflow run
4. writes metadata files
5. logs payload bytes to the artifact store under model/payload
6. creates a Model Registry version
7. writes modelctl.payload_hash to Model Version tags
8. attaches aliases
```

The registration result is printed as JSON. Human-readable progress is printed to stderr.

## Metadata tags

Two optional metadata namespaces are supported:

```text
general  - stable descriptive metadata
training - training, dataset, evaluation or build metadata
```

Full metadata dictionaries are stored as JSON artifacts. A flattened searchable projection is also written to MLflow Model Version tags.

Register with JSON metadata:

```bash
modelctl register <payload_dir> <model_name> --general-tags-json <general.json> --training-tags-json <training.json>
```

Register with inline metadata:

```bash
modelctl register <payload_dir> <model_name> --general-tag <key>=<value> --training-tag <key>=<value>
```

Inline values are parsed as JSON when possible, so numbers, booleans, objects and lists are supported.

## Artifact layout

Every registered version stores this package in the artifact store:

```text
model/
├── MLmodel
├── manifest.json
├── payload.sha256.json
├── metadata/
│   ├── general_tags.json
│   └── training_tags.json
└── payload/
    └── ... payload contents ...
```

`model/payload/` contains the actual registered payload.

`model/payload.sha256.json` contains the hash contract:

```json
{
  "created_by": "modelctl",
  "hash_algorithm": "sha256",
  "hash_scope": "payload_tree_v1",
  "payload_hash": "sha256:...",
  "schema_version": "1.0"
}
```

The same hash is always written to Model Version tags as:

```text
modelctl.payload_hash=sha256:...
modelctl.source_uri=runs:/<run_id>/model
```

That tag is the fast path for hash-only verification because it can be read from the registry without downloading the payload.

## Pull

Download payload only:

```bash
modelctl pull <model_ref> <output_dir>
```

Download the full package:

```bash
modelctl pull <model_ref> <output_dir> --full-package
```

Supported refs:

```text
<model_name>@<alias>
<model_name>:<version>
models:/<model_name>@<alias>
models:/<model_name>/<version>
```

By default, `pull` verifies the downloaded payload hash against `modelctl.payload_hash`. To skip this verification:

```bash
modelctl pull <model_ref> <output_dir> --no-verify
```

If the destination already exists, pass `--overwrite`:

```bash
modelctl pull <model_ref> <output_dir> --overwrite
```

`pull --overwrite` is intentionally safe: the existing destination is not removed before download. The new artifact is downloaded into a staging directory next to the destination, verified, and only then swapped into place. If download or verification fails, the previous destination is kept.

## Verify

Compare an existing directory with the payload hash stored in the registry:

```bash
modelctl verify <model_ref> <path>
```

Both payload-only directories and full modelctl packages are accepted. If `<path>` is a full package, `modelctl` verifies `<path>/payload`.

Exit codes:

```text
0 - hash matches
1 - command failed
2 - hash mismatch
```

Example JSON shape:

```json
{
  "actual_payload_hash": "sha256:...",
  "expected_payload_hash": "sha256:...",
  "matches": true,
  "model_uri": "models:/<model_name>@<alias>",
  "path": "<path>",
  "ref": "<model_ref>"
}
```

## List

```bash
modelctl list <model_name>
```

Shows versions, aliases, registry source URI, status, creation time and payload hash.

## Info

```bash
modelctl info <model_ref>
```

Shows one resolved registry version with all Model Version tags.

## Promote

```bash
modelctl promote <model_name> <version> <alias>
```

Promotion only moves an alias. It does not copy or modify artifacts.

## Hash semantics

The payload hash includes:

```text
relative file paths
file bytes
```

The payload hash ignores:

```text
absolute paths
file mtimes
owners
groups
permissions
```

This makes the digest stable across machines, mount points and container environments.
