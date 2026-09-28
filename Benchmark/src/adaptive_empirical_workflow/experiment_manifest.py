"""Collision-safe identities and manifests for adaptive workflow runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_EXECUTION_ONLY_KEYS = frozenset({"concurrency"})


def _algorithm_config(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _algorithm_config(item)
            for key, item in sorted(value.items())
            if str(key) not in _EXECUTION_ONLY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_algorithm_config(item) for item in value]
    return value


def config_hash(resolved_config: dict[str, object]) -> str:
    canonical = json.dumps(
        _algorithm_config(resolved_config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_component(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"{name} must be non-empty")
    if Path(normalized).name != normalized or any(char in normalized for char in "/\\"):
        raise ValueError(f"{name} must be one path component")
    return normalized


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    split_id: str
    arm_id: str
    run_id: str
    config_hash: str
    resolved_config: dict[str, object]
    run_directory: Path

    @property
    def manifest_path(self) -> Path:
        return self.run_directory / "run_manifest.json"


def build_run_identity(
    *,
    output_root: str | Path,
    experiment_id: str,
    split_id: str,
    arm_id: str,
    run_id: str,
    resolved_config: dict[str, object],
) -> RunIdentity:
    parts = (
        _safe_component("experiment_id", experiment_id),
        _safe_component("split_id", split_id),
        _safe_component("arm_id", arm_id),
        _safe_component("run_id", run_id),
    )
    return RunIdentity(
        experiment_id=parts[0],
        split_id=parts[1],
        arm_id=parts[2],
        run_id=parts[3],
        config_hash=config_hash(resolved_config),
        resolved_config=dict(resolved_config),
        run_directory=Path(output_root).joinpath(*parts),
    )


def prepare_run_directory(identity: RunIdentity) -> Path:
    directory = identity.run_directory
    manifest = identity.manifest_path
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing run manifest: {manifest}") from error
        if existing.get("config_hash") != identity.config_hash:
            raise ValueError("existing run configuration mismatch")
    arm_directory = directory.parent
    arm_directory.mkdir(parents=True, exist_ok=True)
    arm_manifest = arm_directory / "arm_manifest.json"
    arm_payload = (
        json.dumps(
            {"config_hash": identity.config_hash},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    fd, candidate_name = tempfile.mkstemp(
        prefix=".arm-manifest-", suffix=".json.tmp", dir=arm_directory
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(arm_payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(candidate, arm_manifest)
        except FileExistsError:
            try:
                existing_arm = json.loads(arm_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid existing arm manifest: {arm_manifest}"
                ) from error
            if existing_arm.get("config_hash") != identity.config_hash:
                raise ValueError("arm configuration mismatch")
    finally:
        if candidate.exists():
            candidate.unlink()
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing run manifest: {manifest}") from error
        if existing.get("config_hash") != identity.config_hash:
            raise ValueError("existing run configuration mismatch")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("existing run directory has no matching manifest")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_run_manifest(identity: RunIdentity, manifest: dict[str, Any]) -> Path:
    if manifest.get("config_hash") != identity.config_hash:
        raise ValueError("run manifest config_hash does not match identity")
    prepare_run_directory(identity)
    payload = {
        **manifest,
        "experiment_id": identity.experiment_id,
        "split_id": identity.split_id,
        "arm_id": identity.arm_id,
        "run_id": identity.run_id,
        "config_hash": identity.config_hash,
        "resolved_config": identity.resolved_config,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=".run-manifest-",
        suffix=".json.tmp",
        dir=identity.run_directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(identity.manifest_path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return identity.manifest_path
