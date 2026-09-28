"""Leakage-safe, frozen Stage 3 split preparation.

This module is intentionally independent from experiment runners.  Preparing a
split is the only operation here that may read labels; runner-facing cohorts
contain evidence fields only and the restricted gold artifact is not referenced
by the public manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import stat
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


SPLIT_SCHEMA_VERSION = 1
GROUP_ALGORITHM_VERSION = "github_entity_family_v1"
CLUSTER_ALGORITHM_VERSION = "normalized_field_token_jaccard_v1"
TEXT_HASH_ALGORITHM_VERSION = "canonical_utf8_lf_sha256_v1"
CONTAMINATION_SCANNER_VERSION = "semantic_leaf_plus_strict_text_ase_id_v2"
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.82

GOLD_FIELDS = frozenset(
    {
        "decision",
        "symptom",
        "root_cause",
        "bug_type",
        "component",
        "sub_component",
        "trigger_condition",
        "consequence",
        "fix_type",
        "severity_or_impact",
        "original_label_json",
        "source_file",
        "source_sheet",
        "source_row_index",
        "gold",
        "label",
        "labels",
    }
)

RUNNER_FIELDS = (
    "record_id",
    "paper_id",
    "source_project",
    "issue_url",
    "title",
    "body",
    "comments",
    "created_at",
    "updated_at",
    "state",
)


@dataclass(frozen=True)
class ContaminationSource:
    path: str
    sha256: str
    record_id_count: int


@dataclass(frozen=True)
class ContaminationAudit:
    record_ids: tuple[str, ...]
    sources: tuple[ContaminationSource, ...]
    roots: tuple[str, ...]
    candidate_file_count: int
    matched_source_count: int
    unmatched_candidate_count: int
    parse_failure_count: int
    candidate_files_sha256: str


@dataclass(frozen=True)
class SplitPreparationConfig:
    source_csv: Path
    taxonomy_path: Path
    contamination_roots: tuple[Path, ...]
    output_dir: Path
    restricted_gold_path: Path
    split_revision: int
    seed: int = 20260816
    validation_min_size: int = 100
    final_min_size: int = 100
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    supersedes_manifest: Path | None = None
    label_taxonomy_path: Path | None = None


@dataclass(frozen=True)
class PreparedSplitArtifacts:
    manifest_path: Path
    validation_cohort_path: Path
    final_cohort_path: Path
    restricted_gold_path: Path


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    text = (
        path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    )
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_hash(values: Iterable[str]) -> str:
    return _sha256_bytes(
        json.dumps(
            sorted(set(values)), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )


_STRICT_ASE_RECORD_ID = re.compile(
    r"(?<![A-Za-z0-9_])ase2022(?:_[a-z0-9_]+)?(?::[a-z0-9][a-z0-9_-]*){1,3}(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)


def _strict_ids_in_text(text: str, *, prefix: str) -> set[str]:
    return {
        match.group(0)
        for match in _STRICT_ASE_RECORD_ID.finditer(text)
        if match.group(0).startswith(prefix)
    }


def _record_ids_in_json(value: Any, *, prefix: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found.update(_record_ids_in_json(child, prefix=prefix))
    elif isinstance(value, str):
        found.update(_strict_ids_in_text(value, prefix=prefix))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.update(_record_ids_in_json(child, prefix=prefix))
    return found


def _record_ids_in_file(path: Path, *, prefix: str) -> set[str]:
    found: set[str] = set()
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            limit = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(limit)
                    break
                except OverflowError:
                    limit //= 10
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for value in row.values():
                        found.update(
                            _strict_ids_in_text(str(value or ""), prefix=prefix)
                        )
            return found
        if suffix == ".jsonl":
            with path.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        found.update(
                            _record_ids_in_json(json.loads(line), prefix=prefix)
                        )
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"malformed JSONL contamination artifact: {path}"
                        )
            return found
        if suffix == ".json":
            text = path.read_text(encoding="utf-8-sig")
            try:
                return _record_ids_in_json(json.loads(text), prefix=prefix)
            except json.JSONDecodeError:
                raise ValueError(f"malformed JSON contamination artifact: {path}")
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"unreadable contamination artifact: {path}") from error
    return found


def _candidate_paths(roots: Sequence[Path]) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix.lower() in {".csv", ".json", ".jsonl"}:
            candidates.add(root)
        elif root.is_dir():
            candidates.update(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"}
            )
    return tuple(sorted(candidates, key=lambda item: item.as_posix().lower()))


def _candidate_files_hash(paths: Sequence[Path]) -> str:
    return _canonical_hash(
        f"{path.as_posix()}|{_canonical_text_sha256(path)}" for path in paths
    )


def strict_text_contamination_scan(
    roots: Sequence[Path], *, record_id_prefix: str = "ase2022"
) -> ContaminationAudit:
    paths = _candidate_paths(roots)
    ids: set[str] = set()
    sources: list[ContaminationSource] = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        matched = _strict_ids_in_text(text, prefix=record_id_prefix)
        ids.update(matched)
        if matched:
            sources.append(
                ContaminationSource(
                    path=path.as_posix(),
                    sha256=_canonical_text_sha256(path),
                    record_id_count=len(matched),
                )
            )
    return ContaminationAudit(
        record_ids=tuple(sorted(ids)),
        sources=tuple(sources),
        roots=tuple(Path(root).as_posix() for root in roots),
        candidate_file_count=len(paths),
        matched_source_count=len(sources),
        unmatched_candidate_count=len(paths) - len(sources),
        parse_failure_count=0,
        candidate_files_sha256=_candidate_files_hash(paths),
    )


def collect_contaminated_record_ids(
    roots: Sequence[Path], *, record_id_prefix: str = "ase2022"
) -> ContaminationAudit:
    """Collect record IDs from frozen cohorts, manifests and prediction artifacts."""

    all_ids: set[str] = set()
    sources: list[ContaminationSource] = []
    candidate_paths = _candidate_paths(roots)
    for path in candidate_paths:
        ids = _record_ids_in_file(path, prefix=record_id_prefix)
        if not ids:
            continue
        all_ids.update(ids)
        sources.append(
            ContaminationSource(
                path=path.as_posix(),
                sha256=_canonical_text_sha256(path),
                record_id_count=len(ids),
            )
        )
    independent = strict_text_contamination_scan(
        roots, record_id_prefix=record_id_prefix
    )
    if all_ids != set(independent.record_ids):
        missing = sorted(set(independent.record_ids) - all_ids)
        extra = sorted(all_ids - set(independent.record_ids))
        raise ValueError(
            f"contamination scanner coverage mismatch: missing={missing}, extra={extra}"
        )
    return ContaminationAudit(
        record_ids=tuple(sorted(all_ids)),
        sources=tuple(sources),
        roots=tuple(Path(root).as_posix() for root in roots),
        candidate_file_count=len(candidate_paths),
        matched_source_count=len(sources),
        unmatched_candidate_count=len(candidate_paths) - len(sources),
        parse_failure_count=0,
        candidate_files_sha256=_candidate_files_hash(candidate_paths),
    )


_GITHUB_ENTITY = re.compile(
    r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/(issues|pull|commit)/([^/?#]+)",
    re.IGNORECASE,
)


def family_group_key(record: Mapping[str, Any]) -> str:
    """Return a canonical repository + issue/PR/commit family identifier."""

    url = str(record.get("issue_url", "")).strip()
    match = _GITHUB_ENTITY.match(url)
    if match:
        owner, repo, kind, identifier = match.groups()
        kind = "issue-or-pull" if kind.lower() in {"issues", "pull"} else "commit"
        return f"github:{owner.lower()}/{repo.lower()}:{kind}:{identifier.lower()}"
    project = str(record.get("source_project", "unknown/unknown")).strip().lower()
    record_id = str(record.get("record_id", "missing")).strip().lower()
    return f"fallback:{project}:{record_id}"


_URL = re.compile(r"https?://[^\s<>()\[\]{}]+", re.IGNORECASE)
_TOKEN = re.compile(r"url:https://[^\s]+|[a-z0-9_+-]+", re.IGNORECASE)


def _canonical_url(value: str) -> str:
    trimmed = value.rstrip(".,;:'\"!?/)")
    parsed = urlsplit(trimmed)
    host = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path).rstrip("/").lower()
    return urlunsplit(("https", host, path, parsed.query, ""))


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"```[a-z0-9_+-]*", " ", text)
    text = text.replace("```", " ").replace("`", " ")
    text = _URL.sub(lambda match: f" url:{_canonical_url(match.group(0))} ", text)
    return " ".join(_TOKEN.findall(text))


def _near_duplicate_features(record: Mapping[str, Any]) -> frozenset[str]:
    title_tokens = _normalized_text(record.get("title", "")).split()
    body_tokens = _normalized_text(record.get("body", "")).split()
    features: set[str] = {f"title:{token}" for token in title_tokens}
    features.update(f"body:{token}" for token in body_tokens)
    features.update(
        f"title2:{left}|{right}" for left, right in zip(title_tokens, title_tokens[1:])
    )
    features.update(
        f"body2:{left}|{right}" for left, right in zip(body_tokens, body_tokens[1:])
    )
    return frozenset(features or {"empty:"})


def near_duplicate_clusters(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, str]:
    """Cluster normalized title/body/code/URL variants by token-set Jaccard."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    ids = [str(record["record_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id in Stage 3 source")
    features = [_near_duplicate_features(record) for record in records]
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    inverted: dict[str, list[int]] = defaultdict(list)
    pair_overlap: Counter[tuple[int, int]] = Counter()
    for index, tokens in enumerate(features):
        for token in tokens:
            previous = inverted[token]
            for other in previous:
                pair_overlap[(other, index)] += 1
            previous.append(index)
    for (left, right), overlap in pair_overlap.items():
        union_size = len(features[left]) + len(features[right]) - overlap
        if union_size and overlap / union_size >= threshold:
            union(left, right)
    members: dict[int, list[str]] = defaultdict(list)
    for index, record_id in enumerate(ids):
        members[find(index)].append(record_id)
    result: dict[str, str] = {}
    for cluster_members in members.values():
        cluster_id = "ndc:" + _canonical_hash(cluster_members)[:20]
        for record_id in cluster_members:
            result[record_id] = cluster_id
    return result


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _leakage_units(
    records: Sequence[Mapping[str, Any]], clusters: Mapping[str, str]
) -> tuple[list[list[Mapping[str, Any]]], dict[str, str]]:
    ids = [str(record["record_id"]) for record in records]
    union = _UnionFind(ids)
    first_by_family: dict[str, str] = {}
    first_by_cluster: dict[str, str] = {}
    family_by_id: dict[str, str] = {}
    for record in records:
        record_id = str(record["record_id"])
        family = family_group_key(record)
        family_by_id[record_id] = family
        if family in first_by_family:
            union.union(record_id, first_by_family[family])
        else:
            first_by_family[family] = record_id
        cluster = clusters[record_id]
        if cluster in first_by_cluster:
            union.union(record_id, first_by_cluster[cluster])
        else:
            first_by_cluster[cluster] = record_id
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[union.find(str(record["record_id"]))].append(record)
    units = [
        sorted(unit, key=lambda row: str(row["record_id"])) for unit in grouped.values()
    ]
    return units, family_by_id


def _rank(seed: int, unit: Sequence[Mapping[str, Any]]) -> str:
    ids = ",".join(str(row["record_id"]) for row in unit)
    return _sha256_bytes(f"stage3-split-v1:{seed}:{ids}".encode("utf-8"))


def _assign_units(
    units: Sequence[Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    validation_min: int,
    final_min: int,
) -> dict[str, list[Mapping[str, Any]]]:
    ordered = sorted(units, key=lambda value: _rank(seed, value))
    states: dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {
        (0, 0): ((), ())
    }
    for index, unit in enumerate(ordered):
        size = len(unit)
        updated = dict(states)
        for (validation_size, final_size), (
            validation_units,
            final_units,
        ) in states.items():
            if validation_size + size <= validation_min:
                updated.setdefault(
                    (validation_size + size, final_size),
                    (validation_units + (index,), final_units),
                )
            if final_size + size <= final_min:
                updated.setdefault(
                    (validation_size, final_size + size),
                    (validation_units, final_units + (index,)),
                )
        states = updated
    selected = states.get((validation_min, final_min))
    if selected is None:
        raise ValueError(
            "insufficient leakage-isolated Stage 3 records for requested validation/final minimums"
        )
    validation_units, final_units = selected
    assigned: dict[str, list[Mapping[str, Any]]] = {
        "validation": [row for index in validation_units for row in ordered[index]],
        "final": [row for index in final_units for row in ordered[index]],
    }
    for split in assigned:
        assigned[split].sort(key=lambda row: str(row["record_id"]))
    return assigned


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None, "dirty_state_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_state_sha256": _sha256_bytes(status),
    }


def _repository_root() -> Path:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot resolve repository root for sealed gold") from error
    return Path(value).resolve()


def _apply_restricted_gold_security(path: Path) -> None:
    if os.name != "nt":
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return
    whoami = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sid = next(csv.reader([whoami]))[-1].strip()
    subprocess.run(
        ["icacls", str(path), "/setowner", f"*{sid}"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(R,W)",
            "*S-1-5-18:(F)",
            "*S-1-5-32-544:(F)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def validate_windows_acl_entries(
    *,
    owner_sid: str,
    entries: Sequence[Mapping[str, Any]],
    current_sid: str | None = None,
) -> dict[str, Any]:
    owner_sid = owner_sid.upper()
    if current_sid is not None and owner_sid != current_sid.upper():
        raise PermissionError("restricted gold owner does not match current evaluator")
    allowed = {owner_sid, "S-1-5-18", "S-1-5-32-544"}
    allowed_seen: set[str] = set()
    for entry in entries:
        sid = str(entry.get("sid", "")).upper()
        access_type = str(entry.get("type", ""))
        inherited = bool(entry.get("inherited", False))
        if access_type.lower() != "allow":
            continue
        if inherited:
            raise PermissionError("restricted gold contains inherited allow ACE")
        if sid not in allowed:
            raise PermissionError(
                f"unknown allow principal in restricted gold ACL: {sid}"
            )
        allowed_seen.add(sid)
    if owner_sid not in allowed_seen:
        raise PermissionError("restricted gold ACL does not grant the owner access")
    return {
        "verified": True,
        "policy_version": "owner_only_acl_v2",
        "allowed_sids": sorted(allowed_seen),
    }


def _windows_acl_snapshot(path: Path) -> tuple[str, list[dict[str, Any]]]:
    escaped_path = str(path).replace("'", "''")
    script = (
        f"$acl=Get-Acl -LiteralPath '{escaped_path}';"
        "$owner=$acl.Owner;"
        "$account=New-Object System.Security.Principal.NTAccount -ArgumentList $owner;"
        "$ownerSid=$account.Translate([System.Security.Principal.SecurityIdentifier]).Value;"
        "$entries=@($acl.Access | ForEach-Object {"
        "$sid=($_.IdentityReference).Translate([System.Security.Principal.SecurityIdentifier]).Value;"
        "[pscustomobject]@{sid=$sid;type=$_.AccessControlType.ToString();"
        "inherited=$_.IsInherited;rights=$_.FileSystemRights.ToString()}});"
        "[pscustomobject]@{owner_sid=$ownerSid;entries=$entries} | ConvertTo-Json -Depth 5"
    )
    raw = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    payload = json.loads(raw)
    entries = payload.get("entries", [])
    if isinstance(entries, dict):
        entries = [entries]
    return str(payload["owner_sid"]), [dict(entry) for entry in entries]


def verify_restricted_gold_security(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError("restricted gold is missing")
    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError("restricted gold grants group/other permissions")
        return {
            "verified": True,
            "policy_version": "owner_only_acl_v2",
            "platform": "posix",
        }
    whoami = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_sid = next(csv.reader([whoami]))[-1].strip()
    owner_sid, entries = _windows_acl_snapshot(path)
    result = validate_windows_acl_entries(
        owner_sid=owner_sid, entries=entries, current_sid=current_sid
    )
    return {**result, "platform": "windows"}


def load_restricted_gold_for_evaluation(
    path: Path, *, expected_sha256: str
) -> list[dict[str, str]]:
    path = Path(path)
    verify_restricted_gold_security(path)
    if _canonical_text_sha256(path) != expected_sha256:
        raise ValueError("restricted gold hash mismatch")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {"record_id", "split", "symptom", "root_cause"}
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError("restricted gold schema mismatch")
        rows = [dict(row) for row in reader]
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not record_id for record_id in ids):
        raise ValueError("restricted gold contains invalid record IDs")
    if any(row["split"] not in {"validation", "final"} for row in rows):
        raise ValueError("restricted gold contains invalid split names")
    return rows


def load_split_manifest_for_runner(
    manifest_path: Path,
    *,
    cohort_path: Path | None = None,
    domain: str | None = None,
    taxonomy_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", ""))
    if status != "active":
        reason = manifest.get("revocation", {}).get("reason", "not active")
        raise ValueError(f"split manifest is {status or 'invalid'}: {reason}")
    revision = manifest.get("split_revision")
    if not isinstance(revision, int) or revision <= 0:
        raise ValueError("active split manifest has invalid split_revision")
    split_kind = manifest.get("split_kind", "uncontaminated_holdout")
    if split_kind == "contaminated_development":
        expected_id = (
            "ase2022-stage3-contaminated-development-"
            f"seed{manifest.get('seed')}-revision{revision}"
        )
    else:
        expected_id = (
            f"ase2022-stage3-uncontaminated-seed{manifest.get('seed')}-"
            f"revision{revision}"
        )
    if manifest.get("split_id") != expected_id:
        raise ValueError("active split manifest identity does not match revision")
    if split_kind == "contaminated_development":
        if (
            revision != 1
            or manifest_path.parent.name != "ase2022_stage3_contaminated_dev50"
        ):
            raise ValueError("contaminated development split trust path is invalid")
        policy = manifest.get("label_access_policy", {})
        if policy != {
            "runner_may_load_labels": False,
            "runner_cohorts_are_evidence_only": True,
            "development_is_contaminated": True,
        }:
            raise ValueError("contaminated development split label policy is invalid")
    elif not manifest_path.parent.name.endswith(f"_revision{revision}"):
        raise ValueError("split manifest directory does not match split_revision")
    supersedes = manifest.get("supersedes")
    if revision > 1:
        if not supersedes:
            raise ValueError("active split manifest is missing supersedes")
        prior = json.loads(Path(supersedes).read_text(encoding="utf-8"))
        if prior.get("status") != "revoked":
            raise ValueError("superseded split must be revoked")
        if prior.get("split_revision") != revision - 1:
            raise ValueError("split revision chain is not monotonic")
    if domain is not None and manifest.get("domain") != domain:
        raise ValueError("split manifest domain does not match runtime domain")
    if taxonomy_path is not None:
        expected_taxonomy_hash = manifest.get("source", {}).get(
            "taxonomy_structure_sha256"
        )
        if _canonical_text_sha256(Path(taxonomy_path)) != expected_taxonomy_hash:
            raise ValueError("runtime taxonomy hash does not match split manifest")
    if cohort_path is not None:
        cohort_path = Path(cohort_path)
        matching = [
            split
            for split in manifest.get("splits", {}).values()
            if split.get("runner_cohort_file") == cohort_path.name
        ]
        if len(matching) != 1:
            raise ValueError("cohort is not bound by active split manifest")
        if matching[0].get("runner_cohort_sha256") != _canonical_text_sha256(
            cohort_path
        ):
            raise ValueError("runner cohort hash does not match split manifest")
    return manifest


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "symptom", "root_cause", "issue_url", "title", "body"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Stage 3 source missing required fields: {sorted(missing)}")
    if any(not row["symptom"].strip() or not row["root_cause"].strip() for row in rows):
        raise ValueError("Stage 3 source contains empty gold labels")
    return rows


def _manifest_split(
    records: Sequence[Mapping[str, Any]], cohort_path: Path
) -> dict[str, Any]:
    record_ids = [str(row["record_id"]) for row in records]
    return {
        "count": len(record_ids),
        "record_ids": record_ids,
        "record_ids_sha256": _canonical_hash(record_ids),
        "runner_cohort_file": cohort_path.name,
        "runner_cohort_sha256": _canonical_text_sha256(cohort_path),
    }


def _taxonomy_label_sets(path: Path) -> dict[str, set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("nodes"), list):
        result = {
            dimension: {
                str(node["label"])
                for node in raw["nodes"]
                if isinstance(node, dict)
                and node.get("dimension") == dimension
                and node.get("label")
            }
            for dimension in ("symptom", "root_cause")
        }
    elif isinstance(raw, dict):
        result = {
            dimension: {str(label) for label in raw.get(dimension, [])}
            for dimension in ("symptom", "root_cause")
        }
    else:
        result = {}
    if not all(result.get(dimension) for dimension in ("symptom", "root_cause")):
        raise ValueError("taxonomy must define symptom and root_cause labels")
    return result


def prepare_frozen_stage3_splits(
    config: SplitPreparationConfig,
) -> PreparedSplitArtifacts:
    """Prepare immutable validation/final inputs and an evaluator-only gold file."""

    source_csv = Path(config.source_csv)
    taxonomy_path = Path(config.taxonomy_path)
    label_taxonomy_path = Path(config.label_taxonomy_path or config.taxonomy_path)
    output_dir = Path(config.output_dir)
    restricted_gold_path = Path(config.restricted_gold_path)
    if config.split_revision <= 0:
        raise ValueError("split_revision must be a positive integer")
    structure_labels = _taxonomy_label_sets(taxonomy_path)
    label_taxonomy_labels = _taxonomy_label_sets(label_taxonomy_path)
    if structure_labels != label_taxonomy_labels:
        raise ValueError("taxonomy structure and label taxonomy label sets differ")
    if config.supersedes_manifest is not None:
        prior = json.loads(Path(config.supersedes_manifest).read_text(encoding="utf-8"))
        prior_revision = int(prior.get("split_revision", 0))
        if prior.get("status") != "revoked":
            raise ValueError("superseded split manifest must be revoked")
        if prior_revision + 1 != config.split_revision:
            raise ValueError("split_revision must increment superseded revision by one")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    if restricted_gold_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing restricted gold: {restricted_gold_path}"
        )
    if restricted_gold_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(
            "restricted gold must be outside the runner artifact directory"
        )
    if restricted_gold_path.resolve().is_relative_to(_repository_root()):
        raise ValueError("restricted gold must be outside the repository")
    for contamination_root in config.contamination_roots:
        root = Path(contamination_root)
        if root.is_dir() and output_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError("output directory must be outside contamination roots")
    if config.validation_min_size <= 0 or config.final_min_size <= 0:
        raise ValueError("split minimum sizes must be positive")
    rows = _source_rows(source_csv)
    contamination = collect_contaminated_record_ids(
        tuple(Path(root) for root in config.contamination_roots),
        record_id_prefix="ase2022",
    )
    contaminated_ids = set(contamination.record_ids)
    clusters = near_duplicate_clusters(rows, threshold=config.near_duplicate_threshold)
    units, family_by_id = _leakage_units(rows, clusters)
    clean_units = [
        unit
        for unit in units
        if not contaminated_ids.intersection(str(row["record_id"]) for row in unit)
    ]
    assigned = _assign_units(
        clean_units,
        seed=config.seed,
        validation_min=config.validation_min_size,
        final_min=config.final_min_size,
    )
    assigned_ids = {
        str(row["record_id"])
        for split_records in assigned.values()
        for row in split_records
    }
    clean_records = [row for unit in clean_units for row in unit]
    unused_ids = sorted(
        str(row["record_id"])
        for row in clean_records
        if str(row["record_id"]) not in assigned_ids
    )
    validation_ids = {str(row["record_id"]) for row in assigned["validation"]}
    final_ids = {str(row["record_id"]) for row in assigned["final"]}
    validation_families = {family_by_id[record_id] for record_id in validation_ids}
    final_families = {family_by_id[record_id] for record_id in final_ids}
    validation_clusters = {clusters[record_id] for record_id in validation_ids}
    final_clusters = {clusters[record_id] for record_id in final_ids}
    contaminated_source_ids = contaminated_ids & set(family_by_id)
    contaminated_families = {
        family_by_id[record_id] for record_id in contaminated_source_ids
    }
    contaminated_clusters = {
        clusters[record_id] for record_id in contaminated_source_ids
    }
    leakage_audit = {
        "contaminated_id_intersection_count": len(
            (validation_ids | final_ids) & contaminated_ids
        ),
        "contaminated_family_intersection_count": len(
            (validation_families | final_families) & contaminated_families
        ),
        "contaminated_near_duplicate_cluster_intersection_count": len(
            (validation_clusters | final_clusters) & contaminated_clusters
        ),
        "cross_split_family_intersection_count": len(
            validation_families & final_families
        ),
        "cross_split_near_duplicate_cluster_intersection_count": len(
            validation_clusters & final_clusters
        ),
    }
    if any(leakage_audit.values()):
        raise RuntimeError(f"split leakage audit failed: {leakage_audit}")
    strata = {
        split: Counter(
            (str(row["symptom"]), str(row["root_cause"])) for row in assigned[split]
        )
        for split in ("validation", "final")
    }
    stratum_labels = set(strata["validation"]) | set(strata["final"])
    max_stratum_difference = max(
        (
            abs(strata["validation"][label] - strata["final"][label])
            for label in stratum_labels
        ),
        default=0,
    )

    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True, exist_ok=False)
    restricted_gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_temp_path = restricted_gold_path.with_name(
        f".{restricted_gold_path.name}.tmp-{uuid.uuid4().hex}"
    )
    manifest_name = "split_manifest.json"
    validation_name = "validation_runner_cohort.csv"
    final_name = "final_runner_cohort.csv"
    try:
        validation_path = temp_dir / validation_name
        final_path = temp_dir / final_name
        _write_csv(validation_path, RUNNER_FIELDS, assigned["validation"])
        _write_csv(final_path, RUNNER_FIELDS, assigned["final"])
        gold_rows = [
            {
                "record_id": str(row["record_id"]),
                "split": split,
                "symptom": str(row["symptom"]),
                "root_cause": str(row["root_cause"]),
            }
            for split in ("validation", "final")
            for row in assigned[split]
        ]
        _write_csv(
            gold_temp_path,
            ("record_id", "split", "symptom", "root_cause"),
            gold_rows,
        )
        manifest = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "status": "active",
            "split_revision": config.split_revision,
            "split_id": (
                f"ase2022-stage3-uncontaminated-seed{config.seed}-"
                f"revision{config.split_revision}"
            ),
            "supersedes": (
                config.supersedes_manifest.as_posix()
                if config.supersedes_manifest is not None
                else None
            ),
            "domain": "ase2022",
            "seed": config.seed,
            "source": {
                "stage3_csv": source_csv.as_posix(),
                "stage3_csv_sha256": _canonical_text_sha256(source_csv),
                "taxonomy_structure": taxonomy_path.as_posix(),
                "taxonomy_structure_sha256": _canonical_text_sha256(taxonomy_path),
                "label_taxonomy": label_taxonomy_path.as_posix(),
                "label_taxonomy_sha256": _canonical_text_sha256(label_taxonomy_path),
                "text_hash_algorithm": TEXT_HASH_ALGORITHM_VERSION,
            },
            "contamination_exclusion": {
                "excluded_record_id_count": len(contamination.record_ids),
                "excluded_record_ids_sha256": _canonical_hash(contamination.record_ids),
                "sources": [source.__dict__ for source in contamination.sources],
                "roots": list(contamination.roots),
                "allowed_extensions": [".csv", ".json", ".jsonl"],
                "scanner_version": CONTAMINATION_SCANNER_VERSION,
                "candidate_file_count": contamination.candidate_file_count,
                "matched_source_count": contamination.matched_source_count,
                "unmatched_candidate_count": contamination.unmatched_candidate_count,
                "parse_failure_count": contamination.parse_failure_count,
                "candidate_files_sha256": contamination.candidate_files_sha256,
            },
            "isolation": {
                "family_group_algorithm": GROUP_ALGORITHM_VERSION,
                "near_duplicate_cluster_algorithm": CLUSTER_ALGORITHM_VERSION,
                "near_duplicate_threshold": config.near_duplicate_threshold,
                "rule_tuning_policy": "fixed_before_validation; no per-record gold inspection",
            },
            "splits": {
                "validation": _manifest_split(assigned["validation"], validation_path),
                "final": _manifest_split(assigned["final"], final_path),
            },
            "pool_audit": {
                "clean_record_count": len(clean_records),
                "clean_unit_count": len(clean_units),
                "original_target_total": 200,
                "pool_exhausted": len(clean_records) < 200,
                "requested_validation_size": config.validation_min_size,
                "requested_final_size": config.final_min_size,
                "effective_validation_size": len(assigned["validation"]),
                "effective_final_size": len(assigned["final"]),
                "unused_record_count": len(unused_ids),
                "unused_record_ids_sha256": _canonical_hash(unused_ids),
                "selection_algorithm": "sha256_rank_leakage_unit_no_labels_v2",
                "selection_uses_gold_labels": False,
                "preparation_only_joint_strata_count": len(stratum_labels),
                "preparation_only_max_joint_stratum_count_difference": max_stratum_difference,
            },
            "usage_policy": {
                "development_source": "contaminated_dev_50_only",
                "validation_mode": "aggregate_pipeline_only",
                "validation_per_case_gold_access": False,
                "final_mode": "single_use_locked",
                "performance_target": "stage3_joint_accuracy_plus_10pp",
                "statistical_gate_relaxed_for_pool_exhaustion": False,
            },
            "label_access_policy": {
                "runner_may_load_gold": False,
                "runner_cohorts_are_evidence_only": True,
                "restricted_gold_is_not_referenced_by_manifest": True,
                "restricted_gold_sha256": _canonical_text_sha256(gold_temp_path),
                "evaluation_requires_separate_explicit_gold_artifact": True,
                "outside_repository": True,
                "permission_policy_version": "owner_only_acl_v2",
            },
            "leakage_audit": leakage_audit,
            "generator": {
                **_git_provenance(),
                "module_sha256": _canonical_text_sha256(Path(__file__)),
            },
        }
        _write_json(temp_dir / manifest_name, manifest)
        os.replace(gold_temp_path, restricted_gold_path)
        _apply_restricted_gold_security(restricted_gold_path)
        verify_restricted_gold_security(restricted_gold_path)
        os.replace(temp_dir, output_dir)
    except Exception:
        if gold_temp_path.exists():
            gold_temp_path.unlink()
        if restricted_gold_path.exists() and temp_dir.exists():
            restricted_gold_path.unlink()
        if temp_dir.exists():
            for child in temp_dir.iterdir():
                if child.is_file():
                    child.unlink()
            temp_dir.rmdir()
        raise
    return PreparedSplitArtifacts(
        manifest_path=output_dir / manifest_name,
        validation_cohort_path=output_dir / validation_name,
        final_cohort_path=output_dir / final_name,
        restricted_gold_path=restricted_gold_path,
    )


def load_runner_cohort(
    path: Path, *, split_manifest_path: Path | None = None
) -> list[dict[str, str]]:
    """Load an evidence-only cohort and fail if a label-like field is present."""

    path = Path(path)
    manifest_path = split_manifest_path
    if manifest_path is None and (path.parent / "split_manifest.json").exists():
        manifest_path = path.parent / "split_manifest.json"
    if manifest_path is not None:
        load_split_manifest_for_runner(manifest_path, cohort_path=path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        forbidden = fieldnames & GOLD_FIELDS
        if forbidden:
            raise ValueError(
                f"runner cohort contains forbidden label fields: {sorted(forbidden)}"
            )
        return list(reader)
