"""Load decision-only Baseline anchors without treating predictions as evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .contracts import BaselineAnchor


_FORBIDDEN_GROUND_TRUTH_KEYS = frozenset(
    {
        "groundtruth",
        "gold",
        "gt",
        "goldlabel",
        "labelanswer",
        "answerlabel",
        "targetlabel",
        "referencelabel",
    }
)


@dataclass(frozen=True)
class BaselineAnchorBundle:
    """One content-addressed artifact and its loader-owned selected bindings."""

    anchors: Mapping[str, BaselineAnchor]
    source_config_hash: str
    source_predictions_sha256: str
    selected_anchor_set_sha256: str
    anchor_digests: Mapping[str, str]
    anchor_digest_map_sha256: str
    trust_manifest_sha256: str
    trust_artifact_id: str
    trust_manifest_relative_path: str


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative_registered_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            "Baseline trust manifest must be inside the repository"
        ) from error


def _registered_baseline_sha(relative: str) -> str | None:
    """Private trust lookup, separated so tests can register synthetic artifacts."""
    return MappingProxyType(
        {
            "Benchmark/configs/baseline_trust/"
            "ase2022_dev50_mas_evidence_anchored_v1.json": (
                "1e67dd653631d3fa5efd7dc96d91739ea49c01a54e36f143bdaa75ac9c6bc591"
            )
        }
    ).get(relative)


def _load_registered_trust_manifest(
    manifest_path: str | Path,
    *,
    artifact_path: str | Path,
    artifact_raw: bytes,
) -> tuple[dict[str, object], str, str]:
    source = Path(manifest_path)
    root = _repository_root().resolve()
    try:
        relative = _relative_registered_path(source, root)
    except ValueError:
        relative = ""
    expected_sha = _registered_baseline_sha(relative)
    if expected_sha is None:
        raise ValueError("Baseline trust manifest is not a pre-registered trust root")
    try:
        raw = source.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read Baseline trust manifest") from error
    actual_sha = _canonical_sha256(manifest)
    if actual_sha != expected_sha:
        raise ValueError("Baseline trust manifest hash does not match registry")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "status",
        "artifact_id",
        "domain",
        "artifact",
        "source_config_hash",
        "record_set",
        "expected_shape",
        "baseline_run",
    }:
        raise ValueError("Baseline trust manifest schema mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "active":
        raise ValueError("Baseline trust manifest is not active schema v1")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "relative_path",
        "sha256",
        "format",
    }:
        raise ValueError("Baseline trust artifact binding is malformed")
    trusted_artifact = (root / str(artifact.get("relative_path"))).resolve()
    if (
        artifact.get("format") != "jsonl"
        or trusted_artifact != Path(artifact_path).resolve()
        or _relative_registered_path(trusted_artifact, root)
        != str(artifact.get("relative_path"))
    ):
        raise ValueError("Baseline artifact path does not match its trust root")
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    if artifact_sha != artifact.get("sha256"):
        raise ValueError("Baseline artifact hash does not match trust root")
    for section in ("record_set", "expected_shape", "baseline_run"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"Baseline trust manifest {section} is malformed")
    if (
        not isinstance(manifest.get("artifact_id"), str)
        or not manifest["artifact_id"]
        or not isinstance(manifest.get("domain"), str)
        or not manifest["domain"]
        or set(manifest["record_set"])
        != {"count", "record_ids_sha256", "hash_algorithm"}
        or set(manifest["expected_shape"]) != {"valid_count", "invalid_count"}
        or set(manifest["baseline_run"]) != {"config_sha256", "code_sha256"}
    ):
        raise ValueError("Baseline trust manifest provenance schema mismatch")
    return manifest, actual_sha, relative


def baseline_anchor_hash(anchor: BaselineAnchor) -> str:
    """Return the canonical record-specific digest bound by a trusted manifest."""

    canonical = BaselineAnchor.model_validate(anchor.model_dump(mode="python"))
    payload = json.dumps(
        canonical.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_ground_truth_fields(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character for character in key.lower() if character.isalnum()
            )
            if normalized in _FORBIDDEN_GROUND_TRUTH_KEYS:
                raise ValueError(
                    f"Baseline artifact contains ground-truth field: {path}.{key}"
                )
            _reject_ground_truth_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_ground_truth_fields(item, path=f"{path}[{index}]")


def _taxonomy_labels(
    taxonomy: Mapping[str, Sequence[str]], dimension: str
) -> frozenset[str]:
    labels = taxonomy.get(dimension)
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or not labels
    ):
        raise ValueError(f"taxonomy must contain non-empty {dimension} labels")
    normalized = tuple(str(label) for label in labels)
    if any(not label for label in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise ValueError(f"taxonomy {dimension} labels must be unique and non-empty")
    return frozenset(normalized)


def _rows_from_bytes(raw: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("cannot decode Baseline predictions artifact") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid Baseline JSONL row {line_number}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Baseline JSONL row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("Baseline predictions artifact is empty")
    return rows


def load_baseline_anchors(
    path: str | Path,
    *,
    expected_record_ids: Sequence[str],
    taxonomy: Mapping[str, Sequence[str]],
) -> Mapping[str, BaselineAnchor]:
    """Bind a frozen Baseline JSONL artifact to an explicit ordered record set."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read Baseline predictions artifact: {source}"
        ) from error
    return load_baseline_anchors_from_bytes(
        raw,
        expected_record_ids=expected_record_ids,
        taxonomy=taxonomy,
    )


def load_baseline_anchors_from_bytes(
    raw: bytes,
    *,
    expected_record_ids: Sequence[str],
    taxonomy: Mapping[str, Sequence[str]],
) -> Mapping[str, BaselineAnchor]:
    """Bind immutable Baseline JSONL bytes to an explicit ordered record set."""

    expected = tuple(expected_record_ids)
    if (
        not expected
        or any(
            not isinstance(record_id, str) or not record_id for record_id in expected
        )
        or len(expected) != len(set(expected))
    ):
        raise ValueError("expected_record_ids must be non-empty and unique")
    allowed = {
        "symptom": _taxonomy_labels(taxonomy, "symptom"),
        "root_cause": _taxonomy_labels(taxonomy, "root_cause"),
    }
    rows = _rows_from_bytes(raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    indexed: dict[str, BaselineAnchor] = {}
    config_hashes: set[str] = set()
    for row in rows:
        _reject_ground_truth_fields(row, path="row")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Baseline prediction row is missing record_id")
        if record_id in indexed:
            raise ValueError(f"duplicate record_id in Baseline artifact: {record_id}")
        invalid = row.get("invalid")
        if type(invalid) is not bool:
            raise ValueError(f"Baseline row {record_id} has malformed invalid flag")
        config_hash = row.get("config_hash")
        if (
            not isinstance(config_hash, str)
            or len(config_hash) != 64
            or any(character not in "0123456789abcdef" for character in config_hash)
        ):
            raise ValueError(f"Baseline row {record_id} has malformed config_hash")
        config_hashes.add(config_hash)
        final = row.get("final_prediction")
        symptom_label: str | None = None
        root_cause_label: str | None = None
        if invalid:
            if final not in (None, {}):
                raise ValueError(
                    f"invalid Baseline row {record_id} cannot contain final_prediction"
                )
        else:
            if not isinstance(final, dict):
                raise ValueError(
                    f"valid Baseline row {record_id} requires final_prediction"
                )
            symptom_label = final.get("symptom")
            root_cause_label = final.get("root_cause")
            if symptom_label not in allowed["symptom"]:
                raise ValueError(
                    f"Baseline row {record_id} label outside symptom taxonomy"
                )
            if root_cause_label not in allowed["root_cause"]:
                raise ValueError(
                    f"Baseline row {record_id} label outside root_cause taxonomy"
                )
        indexed[record_id] = BaselineAnchor(
            record_id=record_id,
            valid=not invalid,
            symptom_label=symptom_label,
            root_cause_label=root_cause_label,
            source_config_hash=config_hash,
            source_predictions_sha256=source_sha,
        )
    if len(config_hashes) != 1:
        raise ValueError("Baseline artifact must use a single config_hash")
    missing = set(expected) - set(indexed)
    if missing:
        raise ValueError(
            "Baseline artifact missing expected record_ids: "
            + ", ".join(sorted(missing))
        )
    return MappingProxyType({record_id: indexed[record_id] for record_id in expected})


def load_baseline_anchor_bundle(
    path: str | Path,
    *,
    trust_manifest_path: str | Path,
    expected_domain: str,
    expected_record_ids: Sequence[str],
    taxonomy: Mapping[str, Sequence[str]],
) -> BaselineAnchorBundle:
    """Load anchors once and freeze every selected digest used by runtime gates."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read Baseline predictions artifact: {source}"
        ) from error
    trust, trust_sha, trust_relative = _load_registered_trust_manifest(
        trust_manifest_path,
        artifact_path=source,
        artifact_raw=raw,
    )
    if trust.get("domain") != expected_domain:
        raise ValueError("Baseline trust manifest domain mismatch")
    anchors = load_baseline_anchors_from_bytes(
        raw,
        expected_record_ids=expected_record_ids,
        taxonomy=taxonomy,
    )
    ordered = tuple(anchors)
    anchor_digests = MappingProxyType(
        {record_id: baseline_anchor_hash(anchors[record_id]) for record_id in ordered}
    )
    first = anchors[ordered[0]]
    rows = _rows_from_bytes(raw)
    record_ids = [str(row.get("record_id")) for row in rows]
    record_set = trust["record_set"]
    expected_shape = trust["expected_shape"]
    baseline_run = trust["baseline_run"]
    assert isinstance(record_set, dict)
    assert isinstance(expected_shape, dict)
    assert isinstance(baseline_run, dict)
    actual_shape = {
        "valid_count": sum(row.get("invalid") is False for row in rows),
        "invalid_count": sum(row.get("invalid") is True for row in rows),
    }
    if (
        record_set.get("hash_algorithm") != "canonical_sorted_record_ids_sha256_v1"
        or record_set.get("count") != len(record_ids)
        or record_set.get("record_ids_sha256") != _canonical_sha256(sorted(record_ids))
        or expected_shape != actual_shape
        or trust.get("source_config_hash") != first.source_config_hash
        or baseline_run.get("config_sha256") != first.source_config_hash
    ):
        raise ValueError("Baseline artifact content does not match trust manifest")
    code_sha = baseline_run.get("code_sha256")
    if (
        not isinstance(code_sha, str)
        or len(code_sha) != 64
        or any(character not in "0123456789abcdef" for character in code_sha)
    ):
        raise ValueError("Baseline trust run provenance is malformed")
    selected_payload = [
        anchors[record_id].model_dump(mode="json") for record_id in ordered
    ]
    return BaselineAnchorBundle(
        anchors=anchors,
        source_config_hash=first.source_config_hash,
        source_predictions_sha256=first.source_predictions_sha256,
        selected_anchor_set_sha256=_canonical_sha256(selected_payload),
        anchor_digests=anchor_digests,
        anchor_digest_map_sha256=_canonical_sha256(dict(anchor_digests)),
        trust_manifest_sha256=trust_sha,
        trust_artifact_id=str(trust["artifact_id"]),
        trust_manifest_relative_path=trust_relative,
    )
