"""Shared IO and validation helpers for Stage 1 data generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_spec(config_path: Path) -> tuple[dict[str, Any], Path, str]:
    config_path = config_path.resolve()
    raw_text = config_path.read_text(encoding="utf-8")
    spec = yaml.safe_load(raw_text)
    if not isinstance(spec, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return spec, config_path.parent.parent, raw_text


def resolve_path(root: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else root / candidate


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def artifact_entry(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def verify_manifest_artifacts(manifest: dict[str, Any], root: Path) -> None:
    for artifact in manifest.get("artifacts", []):
        artifact_path = resolve_path(root, artifact["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact is missing: {artifact_path}")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != artifact.get("sha256"):
            raise ValueError(f"Manifest hash mismatch for {artifact_path}")


def validate_weights(entries: list[dict[str, Any]], label: str) -> None:
    if not entries:
        raise ValueError(f"{label} must not be empty")
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate ids")
    total = sum(float(entry["share"]) for entry in entries)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{label} shares must sum to 1.0, got {total}")
