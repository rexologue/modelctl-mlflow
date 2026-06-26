# modelctl artifact-store workflow

This document describes the storage model and command behavior of `modelctl`.

The project has one deliberately simple contract:

```text
any payload directory -> MLflow artifact store -> MLflow Model Registry version -> pull or verify by registry ref
```

No framework-specific assumptions are made about the payload contents.

## 1. Component roles

### MLflow backend store

The backend store keeps registry metadata:

```text
experiments
runs
registered models
model versions
aliases
tags
params
source URIs
statuses
timestamps
```

It is not the payload byte store.

### MLflow artifact store

The artifact store keeps the actual registered package:

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

### MLflow Model Registry

The registry gives stable references to concrete artifact packages:

```text
<model_name>:<version>
<model_name>@<alias>
```

The registry version points to an artifact source URI, usually shaped like:

```text
runs:/<run_id>/model
```

## 2. Registration

Command shape:

```bash
modelctl register <payload_dir> <model_name>
```

Registration steps:

```text
1. Validate input.
2. Configure MLflow tracking.
3. Compute a stable SHA256 hash over the payload tree.
4. Create or reuse the registered model name.
5. Start a short technical MLflow run.
6. Log metadata JSON files.
7. Log the payload directory to model/payload.
8. Create a Model Registry version with source=runs:/<run_id>/model.
9. Write required Model Version tags, including modelctl.payload_hash.
10. Attach aliases.
```

Default aliases:

```text
first version -> baseline + champion
later versions -> candidate
```

Explicit aliases can be provided with repeated `--alias` flags.

## 3. Required hash contract

Every registered version has this Model Version tag:

```text
modelctl.payload_hash=sha256:...
modelctl.source_uri=runs:/<run_id>/model
```

It also has these hash metadata tags:

```text
modelctl.hash_algorithm=sha256
modelctl.hash_scope=payload_tree_v1
```

The same hash is written inside the artifact package:

```text
model/payload.sha256.json
model/manifest.json
model/MLmodel
```

The registry tag is the fastest source for verification because reading it does not require downloading the artifact payload.

## 4. Hash algorithm

The payload tree hash includes each regular file in sorted relative-path order.

For each file, the hash stream receives:

```text
relative path
separator byte
file bytes
separator byte
```

The digest ignores:

```text
absolute path
mtime
owner
group
permissions
```

Therefore identical payload trees produce the same digest even if they are stored in different locations or mounted differently.

## 5. Artifact package files

### `model/payload/`

The actual payload bytes. Default `pull` installs this directory directly as the destination.

### `model/payload.sha256.json`

A small machine-readable hash document:

```json
{
  "created_by": "modelctl",
  "created_at": "<utc timestamp>",
  "hash_algorithm": "sha256",
  "hash_scope": "payload_tree_v1",
  "payload_hash": "sha256:...",
  "schema_version": "1.0"
}
```

### `model/manifest.json`

The artifact package manifest:

```json
{
  "artifact_layout": "modelctl_payload_package",
  "created_by": "modelctl",
  "created_at": "<utc timestamp>",
  "general_tags": {},
  "general_tags_path": "metadata/general_tags.json",
  "hash_algorithm": "sha256",
  "hash_scope": "payload_tree_v1",
  "model_name": "<model_name>",
  "payload_hash": "sha256:...",
  "payload_hash_path": "payload.sha256.json",
  "payload_path": "payload",
  "schema_version": "1.0",
  "training_tags": {},
  "training_tags_path": "metadata/training_tags.json"
}
```

### `model/MLmodel`

A small MLflow descriptor that makes the package self-describing in MLflow UI and manual inspection. It is not an inference flavor.

### `model/metadata/general_tags.json`

Full general metadata dictionary.

### `model/metadata/training_tags.json`

Full training metadata dictionary.

### `modelctl_metadata/*`

The same metadata is also logged at run level for easier run-level inspection:

```text
modelctl_metadata/general_tags.json
modelctl_metadata/training_tags.json
modelctl_metadata/manifest.json
modelctl_metadata/payload.sha256.json
```

## 6. Pull

Command shape:

```bash
modelctl pull <model_ref> <output_dir>
```

Default behavior:

```text
1. Resolve <model_ref> to a concrete Model Registry version.
2. Read modelctl.payload_hash from Model Version tags.
3. Estimate artifact bytes with MLflow artifact listing when the backend reports file sizes.
4. Download runs:/<run_id>/model/payload to staging and report observed staging bytes.
5. Hash the staged payload and report byte progress when total size is known.
6. Compare the staged hash with the registry hash.
7. Move staged payload into destination.
```

The default output is payload-only.

Full package behavior:

```bash
modelctl pull <model_ref> <output_dir> --full-package
```

This installs the whole package with `MLmodel`, manifest, hash document, metadata and payload.

## 7. Safe overwrite

If the destination exists, `pull` requires `--overwrite`.

With `--overwrite`, the current destination is not deleted before download. The replacement sequence is:

```text
1. Download to a staging directory next to the destination.
2. Verify the staged payload hash.
3. Rename the current destination to a temporary backup.
4. Move the staged result into the final destination.
5. Delete the backup only after successful install.
```

If download or verification fails, the existing destination remains in place.

## 8. Verify

Command shape:

```bash
modelctl verify <model_ref> <path>
```

Verification steps:

```text
1. Resolve <model_ref> to a concrete Model Registry version.
2. Read modelctl.payload_hash from Model Version tags.
3. Detect whether <path> is a full modelctl package or a payload-only directory.
4. Hash the payload tree.
5. Return JSON with expected hash, actual hash and matches=true/false.
```

Exit codes:

```text
0 - hash matches
1 - command failed
2 - hash mismatch
```

`verify` downloads no payload bytes. It only reads registry metadata and hashes the provided directory.

## 9. List and info

`list` reads versions for one registered model name:

```bash
modelctl list <model_name>
```

`info` resolves one model reference:

```bash
modelctl info <model_ref>
```

Both commands read registry metadata only. They do not download payload artifacts.

## 10. Promote

Command shape:

```bash
modelctl promote <model_name> <version> <alias>
```

Promotion updates only the alias pointer. It does not copy or rewrite artifacts.

## 11. Supported refs

All commands that take a model reference accept:

```text
<model_name>@<alias>
<model_name>:<version>
models:/<model_name>@<alias>
models:/<model_name>/<version>
```
