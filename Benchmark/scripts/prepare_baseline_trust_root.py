"""Offline preparation of a candidate Baseline trust manifest.

The default writes a candidate only. --activate-local explicitly registers its
digest in ignored Benchmark/cache; predictions and run metadata stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def register_baseline_trust_root(
    artifact: Path,
    *,
    output: Path,
    artifact_id: str,
    domain: str,
    baseline_code_sha256: str,
) -> str:
    artifact = artifact.resolve()
    relative = artifact.relative_to(ROOT).as_posix()
    rows = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    configs = {row.get("config_hash") for row in rows}
    if len(configs) != 1:
        raise ValueError("Baseline artifact must have one config hash")
    source_config_hash = next(iter(configs))
    if not isinstance(source_config_hash, str) or len(source_config_hash) != 64:
        raise ValueError("Baseline artifact config hash is malformed")
    if len(baseline_code_sha256) != 64:
        raise ValueError("baseline code SHA256 is malformed")
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("Baseline artifact has malformed record IDs")
    manifest = {
        "schema_version": 1,
        "status": "candidate",
        "artifact_id": artifact_id,
        "domain": domain,
        "artifact": {
            "relative_path": relative,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "format": "jsonl",
        },
        "source_config_hash": source_config_hash,
        "record_set": {
            "count": len(record_ids),
            "record_ids_sha256": _canonical_sha(sorted(record_ids)),
            "hash_algorithm": "canonical_sorted_record_ids_sha256_v1",
        },
        "expected_shape": {
            "valid_count": sum(row.get("invalid") is False for row in rows),
            "invalid_count": sum(row.get("invalid") is True for row in rows),
        },
        "baseline_run": {
            "config_sha256": source_config_hash,
            "code_sha256": baseline_code_sha256,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return _canonical_sha(manifest)


def activate_local_trust_root(output: Path) -> str:
    """Explicit local approval; never register a supplied manifest implicitly."""
    output = output.resolve()
    cache = (ROOT / "Benchmark/cache").resolve()
    if not output.is_relative_to(cache):
        raise ValueError("Active local trust manifests must be under Benchmark/cache")
    manifest = json.loads(output.read_text(encoding="utf-8"))
    artifact = (ROOT / manifest["artifact"]["relative_path"]).resolve()
    if not artifact.is_relative_to(ROOT.resolve()) or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact"]["sha256"]:
        raise ValueError("Baseline artifact path/hash differs from candidate")
    manifest["status"] = "active"
    digest = _canonical_sha(manifest)
    registry_path = cache / "baseline_trust_registry.json"
    entries = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"schema_version": 1, "manifests": {}}
    entries["manifests"][output.relative_to(ROOT.resolve()).as_posix()] = digest
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--baseline-code-sha256", required=True)
    parser.add_argument("--activate-local", action="store_true",
                        help="Explicitly trust the candidate in ignored Benchmark/cache.")
    args = parser.parse_args()
    print(
        register_baseline_trust_root(
            args.artifact,
            output=args.output,
            artifact_id=args.artifact_id,
            domain=args.domain,
            baseline_code_sha256=args.baseline_code_sha256,
        )
    )
    if args.activate_local:
        print(activate_local_trust_root(args.output))


if __name__ == "__main__":
    main()
