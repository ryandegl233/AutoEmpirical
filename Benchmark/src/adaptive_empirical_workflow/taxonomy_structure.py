"""Validate, hash, and render record-independent taxonomy structure artifacts."""

from __future__ import annotations

import hashlib
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .contracts import (
    BoundaryCard,
    BoundaryCriterion,
    EvidenceDimension,
    TaxonomySemanticOrigin,
    TaxonomyStructure,
)


_RECORD_ID_PATTERN = re.compile(r"\b[a-z0-9][a-z0-9_-]*:[0-9a-f]{16}\b", re.IGNORECASE)
_RECORD_SPECIFIC_TERM_PATTERN = re.compile(
    r"\b(?:gt|gold|ground[ _-]+truth|expected[ _-]+answer|"
    r"prototype(?:[ _-]+case)?|development[ _-]+case|case[ _-]+specific)\b",
    re.IGNORECASE,
)
_ISSUE_URL_PATTERN = re.compile(r"https?://|\b(?:github|gitlab)\.com/", re.IGNORECASE)
OFFICIAL_BOUNDARY_CARD_GENERATOR_VERSION = "official-definition-pair-v1"
_BUNDLED_ASE2022_ARTIFACT = "ase2022_taxonomy_structure_v1.json"
_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "domain",
        "artifact_sha256",
        "corpus_path",
        "corpus_sha256",
        "source_paths",
        "source_sha256",
        "record_count",
        "record_ids_sha256",
        "fields",
        "forbidden_text_digest",
        "ngram_words",
        "min_ngram_chars",
        "approved_operational_text_sha256",
    }
)
_TRUSTED_ASE2022_SOURCE_PATHS = {
    "stage1": (
        "../../Dataset/by_paper/ase2022_towards_understanding_the_faults_of/"
        "stage1.csv"
    ),
    "stage2": (
        "../../Dataset/by_paper/ase2022_towards_understanding_the_faults_of/"
        "stage2.csv"
    ),
    "stage3": (
        "../../Dataset/by_paper/ase2022_towards_understanding_the_faults_of/"
        "stage3.csv"
    ),
    "excluded_cohort": (
        "../results/ase2022_camel_mas_baseline/ase2022_camel_mas_cohort.csv"
    ),
}
_TRUSTED_ASE2022_SOURCE_SHA256 = {
    "stage1": "6f3de07df4d81972c5b17bc0a0f46b56b599552414d1a3d34c7d63eb65b65663",
    "stage2": "1f3d3fa75b29e3285c5ca752fd385348c05d6cf9d4a55975b8e9a53f3dc1924d",
    "stage3": "89ae81045656805f4fb2087a2d23429549b51e945f55fedaac534c2a9daa92bc",
    "excluded_cohort": (
        "ea4d59d1a05b497068d07c9a4c95cf59bcc75211eb8ca2fb5256163b1a349032"
    ),
}
_TRUSTED_ASE2022_BINDING_SHA256 = (
    "ef828b12633f0355526f6dfaa9e2cd112b10ff63a7bcc90cbbf157f774846004"
)
_FORBIDDEN_CORPUS_FIELDS = frozenset(
    {"decision", "symptom", "root_cause", "gt", "ground_truth", "gold"}
)


@dataclass(frozen=True)
class _FrozenCorpusBinding:
    forbidden_ngram_hashes: frozenset[str]
    approved_text_hashes: frozenset[str]
    ngram_words: int
    min_ngram_chars: int


def _string_values(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            string for item in value.values() for string in _string_values(item)
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(string for item in value for string in _string_values(item))
    return ()


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_file_sha256(value: bytes) -> str:
    return _sha256_bytes(value.replace(b"\r\n", b"\n"))


def _canonical_json_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _text_hash(value: str) -> str:
    return _sha256_bytes(_normalize_text(value).encode("utf-8"))


def _ngram_hashes(value: str, *, words: int, min_chars: int) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9_]+", _normalize_text(value))
    return frozenset(
        _sha256_bytes(" ".join(tokens[index : index + words]).encode("utf-8"))
        for index in range(len(tokens) - words + 1)
        if len(" ".join(tokens[index : index + words])) >= min_chars
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid frozen corpus binding {field}")
    return value


def _load_frozen_corpus_binding(
    binding_path: Path,
    *,
    artifact_path: Path,
    trusted_source_paths: Mapping[str, str],
    trusted_source_sha256: Mapping[str, str],
    trusted_binding_sha256: str,
) -> _FrozenCorpusBinding:
    try:
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen corpus binding") from error
    if _canonical_json_sha256(payload) != trusted_binding_sha256:
        raise ValueError("trusted bundled binding hash mismatch")
    if not isinstance(payload, dict):
        raise ValueError("invalid frozen corpus binding schema")
    if "source_paths" not in payload:
        raise ValueError("frozen corpus binding does not match trusted source paths")
    if set(payload) != _BINDING_KEYS:
        raise ValueError("invalid frozen corpus binding schema")
    if payload["schema_version"] != 1 or payload["domain"] != "ase2022":
        raise ValueError("invalid frozen corpus binding identity")
    source_paths = payload.get("source_paths")
    if not isinstance(source_paths, dict) or source_paths != trusted_source_paths:
        raise ValueError("frozen corpus binding does not match trusted source paths")
    try:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("frozen corpus binding artifact is unavailable") from error
    if _require_sha256(
        payload["artifact_sha256"], field="artifact_sha256"
    ) != _canonical_json_sha256(artifact_payload):
        raise ValueError("frozen corpus binding artifact hash mismatch")
    source_hashes = payload["source_sha256"]
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or set(source_hashes) != set(source_paths)
    ):
        raise ValueError("frozen corpus binding does not match trusted sources")
    if source_hashes != trusted_source_sha256:
        raise ValueError("frozen corpus binding does not match trusted source hashes")
    for name, digest in source_hashes.items():
        source_name = source_paths[name]
        if not isinstance(name, str) or not name or not isinstance(source_name, str):
            raise ValueError("invalid frozen corpus binding source name")
        expected_digest = _require_sha256(digest, field=f"source_sha256.{name}")
        source_path = (binding_path.parent / source_name).resolve()
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise ValueError("trusted frozen corpus source is unavailable") from error
        if _normalized_file_sha256(source_bytes) != expected_digest:
            raise ValueError("trusted frozen corpus source hash mismatch")
    fields = payload["fields"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or len(fields) != len(set(fields))
        or _FORBIDDEN_CORPUS_FIELDS.intersection(fields)
    ):
        raise ValueError("invalid frozen corpus binding fields")
    corpus_name = payload["corpus_path"]
    if not isinstance(corpus_name, str) or not corpus_name:
        raise ValueError("invalid frozen corpus binding corpus_path")
    corpus_path = (binding_path.parent / corpus_name).resolve()
    try:
        corpus_bytes = corpus_path.read_bytes()
    except OSError as error:
        raise ValueError("frozen corpus binding source is unavailable") from error
    if _require_sha256(
        payload["corpus_sha256"], field="corpus_sha256"
    ) != _normalized_file_sha256(corpus_bytes):
        raise ValueError("frozen corpus binding source hash mismatch")
    record_count = payload["record_count"]
    ngram_words = payload["ngram_words"]
    min_ngram_chars = payload["min_ngram_chars"]
    if not isinstance(record_count, int) or record_count < 1:
        raise ValueError("invalid frozen corpus binding record_count")
    if not isinstance(ngram_words, int) or ngram_words < 4:
        raise ValueError("invalid frozen corpus binding ngram_words")
    if not isinstance(min_ngram_chars, int) or min_ngram_chars < 24:
        raise ValueError("invalid frozen corpus binding min_ngram_chars")
    try:
        with corpus_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise ValueError("invalid frozen corpus binding source") from error
    if len(rows) != record_count or any(
        not row.get("record_id") or any(field not in row for field in fields)
        for row in rows
    ):
        raise ValueError("frozen corpus binding record set mismatch")
    record_ids = "\n".join(row["record_id"] for row in rows)
    if _require_sha256(
        payload["record_ids_sha256"], field="record_ids_sha256"
    ) != _sha256_bytes(record_ids.encode("utf-8")):
        raise ValueError("frozen corpus binding record IDs mismatch")
    entries = [
        f"{row['record_id']}\0{field}\0{_normalize_text(row[field])}"
        for row in rows
        for field in fields
        if _normalize_text(row[field])
    ]
    if _require_sha256(
        payload["forbidden_text_digest"], field="forbidden_text_digest"
    ) != _sha256_bytes("\n".join(entries).encode("utf-8")):
        raise ValueError("frozen corpus binding text digest mismatch")
    approved = payload["approved_operational_text_sha256"]
    if not isinstance(approved, list) or len(approved) != len(set(approved)):
        raise ValueError("invalid frozen corpus binding approved text hashes")
    approved_hashes = frozenset(
        _require_sha256(value, field="approved_operational_text_sha256")
        for value in approved
    )
    forbidden_hashes = frozenset(
        digest
        for row in rows
        for field in fields
        for digest in _ngram_hashes(
            row[field], words=ngram_words, min_chars=min_ngram_chars
        )
    )
    return _FrozenCorpusBinding(
        forbidden_ngram_hashes=forbidden_hashes,
        approved_text_hashes=approved_hashes,
        ngram_words=ngram_words,
        min_ngram_chars=min_ngram_chars,
    )


def _validate_record_independence(
    payload: object,
    *,
    structure: TaxonomyStructure,
    corpus_binding: _FrozenCorpusBinding | None,
) -> None:
    for value in _string_values(payload):
        if (
            _RECORD_ID_PATTERN.search(value)
            or _RECORD_SPECIFIC_TERM_PATTERN.search(value)
            or _ISSUE_URL_PATTERN.search(value)
        ):
            raise ValueError("taxonomy structure contains record-specific content")
    if corpus_binding is None:
        return
    approved_hashes = corpus_binding.approved_text_hashes.union(
        _text_hash(value)
        for node in structure.nodes
        for value in (node.label, node.definition)
    )
    for value in _string_values(payload):
        if _text_hash(value) in approved_hashes:
            continue
        if _ngram_hashes(
            value,
            words=corpus_binding.ngram_words,
            min_chars=corpus_binding.min_ngram_chars,
        ).intersection(corpus_binding.forbidden_ngram_hashes):
            raise ValueError("taxonomy structure contains record-specific content")


def _official_labels(
    taxonomy: Mapping[str, Sequence[str]], dimension: str
) -> tuple[str, ...]:
    labels = taxonomy.get(dimension)
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or not labels
    ):
        raise ValueError(f"supplied taxonomy has no {dimension} labels")
    normalized = tuple(str(label) for label in labels)
    if any(not label for label in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise ValueError(
            f"supplied {dimension} taxonomy labels must be unique and non-empty"
        )
    return normalized


def _label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
    if not slug:
        raise ValueError("official taxonomy label cannot produce a stable slug")
    return slug


def materialize_official_boundary_cards(
    structure: TaxonomyStructure,
) -> tuple[BoundaryCard, ...]:
    """Materialize the complete pair graph from paper definitions only."""

    cards: list[BoundaryCard] = []
    for dimension, prefix in (("symptom", "symptom"), ("root_cause", "root")):
        nodes = tuple(node for node in structure.nodes if node.dimension == dimension)
        if any(
            node.definition_semantic_origin
            is not TaxonomySemanticOrigin.PAPER_DEFINITION
            for node in nodes
        ):
            raise ValueError(
                "official Boundary Cards require paper_definition node definitions"
            )
        for left_index, left in enumerate(nodes):
            for right_index in range(left_index + 1, len(nodes)):
                right = nodes[right_index]
                cards.append(
                    BoundaryCard(
                        card_id=(
                            f"{prefix}-p{left_index + 1:02d}-"
                            f"p{right_index + 1:02d}-official-definition-v1"
                        ),
                        dimension=dimension,
                        labels=(left.label, right.label),
                        semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
                        decision_question=(
                            "Which official definition is directly supported by the "
                            f"frozen evidence: {left.label} — {left.definition} OR "
                            f"{right.label} — {right.definition}"
                        ),
                        observable_slots=(
                            f"direct-support-for-{_label_slug(left.label)}",
                            f"direct-support-for-{_label_slug(right.label)}",
                            "direct-contradiction-of-baseline",
                        ),
                        criteria=(
                            BoundaryCriterion(
                                label=left.label,
                                positive_conditions=(left.definition,),
                                exclusion_conditions=(
                                    "Frozen evidence directly supports the official "
                                    f"definition of {right.label}: {right.definition}",
                                ),
                            ),
                            BoundaryCriterion(
                                label=right.label,
                                positive_conditions=(right.definition,),
                                exclusion_conditions=(
                                    "Frozen evidence directly supports the official "
                                    f"definition of {left.label}: {left.definition}",
                                ),
                            ),
                        ),
                    )
                )
    return tuple(cards)


def official_boundary_card_materialized_digest(
    structure: TaxonomyStructure,
) -> str:
    """Return the digest of the fixed-version, fully materialized card graph."""

    return _canonical_json_sha256(
        {
            "generator_version": OFFICIAL_BOUNDARY_CARD_GENERATOR_VERSION,
            "boundary_cards": [
                card.model_dump(mode="json")
                for card in materialize_official_boundary_cards(structure)
            ],
        }
    )


def official_boundary_card_manifest(structure: TaxonomyStructure) -> Mapping[str, str]:
    """Expose immutable generator identity for run-manifest trust binding."""

    return {
        "generator_version": OFFICIAL_BOUNDARY_CARD_GENERATOR_VERSION,
        "materialized_boundary_cards_sha256": (
            official_boundary_card_materialized_digest(structure)
        ),
    }


def route_boundary_card(
    structure: TaxonomyStructure,
    dimension: EvidenceDimension,
    baseline_label: str,
    proposed_label: str,
) -> BoundaryCard:
    """Route either direction of one owned label pair to exactly one frozen card."""

    if dimension not in {EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE}:
        raise ValueError("Boundary Card routing requires an owned Stage 3 dimension")
    if baseline_label == proposed_label:
        raise ValueError("Boundary Card labels must be distinct")
    owned_labels = {
        node.label for node in structure.nodes if node.dimension == dimension.value
    }
    all_labels = {node.label for node in structure.nodes}
    if baseline_label not in all_labels or proposed_label not in all_labels:
        raise ValueError("Boundary Card routing requires official labels")
    if baseline_label not in owned_labels or proposed_label not in owned_labels:
        raise ValueError("Boundary Card labels must belong to the owned dimension")
    pair = frozenset((baseline_label, proposed_label))
    matches = tuple(
        card
        for card in structure.boundary_cards
        if card.dimension == dimension.value and frozenset(card.labels) == pair
    )
    if len(matches) != 1:
        raise ValueError("Boundary Card pair must route to exactly one frozen card")
    return matches[0]


def _validate_against_official_taxonomy(
    structure: TaxonomyStructure,
    taxonomy: Mapping[str, Sequence[str]],
    *,
    require_complete_official_cards: bool = False,
) -> None:
    nodes_by_label = {node.label: node for node in structure.nodes}
    official_by_dimension = {
        dimension: set(_official_labels(taxonomy, dimension))
        for dimension in ("symptom", "root_cause")
    }
    for dimension, official in official_by_dimension.items():
        structured = {
            node.label for node in structure.nodes if node.dimension == dimension
        }
        if structured != official:
            raise ValueError(
                f"{dimension} structure labels must exactly match supplied taxonomy"
            )

    for node in structure.nodes:
        related = {
            "parent": node.parents,
            "child": node.children,
            "nearest neighbor": node.nearest_neighbors,
        }
        for relation, labels in related.items():
            for label in labels:
                target = nodes_by_label.get(label)
                if target is None:
                    raise ValueError(f"unknown {relation} label: {label}")
                if target.dimension != node.dimension:
                    raise ValueError(f"taxonomy {relation} must use the same dimension")
        for parent in node.parents:
            if node.label not in nodes_by_label[parent].children:
                raise ValueError(
                    "taxonomy parent/child relationships must be reciprocal"
                )
        for child in node.children:
            if node.label not in nodes_by_label[child].parents:
                raise ValueError(
                    "taxonomy parent/child relationships must be reciprocal"
                )

    for card in structure.boundary_cards:
        card_nodes = []
        for label in card.labels:
            node = nodes_by_label.get(label)
            if node is None:
                raise ValueError(f"boundary card references unknown label: {label}")
            card_nodes.append(node)
        if any(node.dimension != card.dimension for node in card_nodes):
            raise ValueError("boundary card labels must use the same dimension")
        if not require_complete_official_cards and (
            card.labels[1] not in card_nodes[0].nearest_neighbors
            or card.labels[0] not in card_nodes[1].nearest_neighbors
        ):
            raise ValueError(
                "boundary card labels must be reciprocal nearest neighbors"
            )
    if require_complete_official_cards:
        expected_cards = materialize_official_boundary_cards(structure)
        if structure.boundary_cards != expected_cards:
            raise ValueError(
                "boundary cards must exactly match the complete fixed-version "
                "official-definition materialization"
            )


def _load_structure_payload(source: Path) -> tuple[object, TaxonomyStructure]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid taxonomy structure artifact: {source}") from error
    try:
        structure = TaxonomyStructure.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid taxonomy structure: {error}") from None
    return payload, structure


def load_taxonomy_structure(
    path: str | Path,
    taxonomy: Mapping[str, Sequence[str]],
    *,
    corpus_binding_path: str | Path | None = None,
) -> TaxonomyStructure:
    """Load a structure; ASE2022 always uses its pre-registered trust roots."""

    source = Path(path)
    payload, structure = _load_structure_payload(source)
    corpus_binding: _FrozenCorpusBinding | None = None
    if structure.domain == "ase2022":
        if corpus_binding_path is not None:
            raise ValueError("bundled ASE2022 must use the trusted bundled binding")
        if source.name != _BUNDLED_ASE2022_ARTIFACT:
            raise ValueError(
                "ASE2022 taxonomy structure requires frozen corpus binding"
            )
        binding_source = source.with_name(
            "ase2022_taxonomy_structure_v1_corpus_binding.json"
        )
        corpus_binding = _load_frozen_corpus_binding(
            binding_source,
            artifact_path=source,
            trusted_source_paths=_TRUSTED_ASE2022_SOURCE_PATHS,
            trusted_source_sha256=_TRUSTED_ASE2022_SOURCE_SHA256,
            trusted_binding_sha256=_TRUSTED_ASE2022_BINDING_SHA256,
        )
    elif corpus_binding_path is not None:
        raise ValueError("corpus bindings are only supported by trusted ASE2022")
    _validate_record_independence(
        payload,
        structure=structure,
        corpus_binding=corpus_binding,
    )
    _validate_against_official_taxonomy(
        structure,
        taxonomy,
        require_complete_official_cards=structure.domain == "ase2022",
    )
    return structure


def taxonomy_structure_hash(structure: TaxonomyStructure) -> str:
    """Return a deterministic hash suitable for run manifests."""

    legacy_canonical = json.dumps(
        structure.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if structure.domain != "ase2022":
        return hashlib.sha256(legacy_canonical).hexdigest()
    try:
        materialized_cards = materialize_official_boundary_cards(structure)
    except ValueError:
        materialized_cards = ()
    if structure.boundary_cards != materialized_cards:
        return hashlib.sha256(legacy_canonical).hexdigest()
    boundary_card_manifest = official_boundary_card_manifest(structure)
    canonical = json.dumps(
        {
            "boundary_card_manifest": boundary_card_manifest,
            "structure": structure.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def render_taxonomy_structure(
    structure: TaxonomyStructure, dimension: EvidenceDimension
) -> str:
    """Render only the owned Stage 3 dimension as structured model guidance."""

    if dimension not in {EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE}:
        raise ValueError("taxonomy structure guidance is only defined for Stage 3")
    owned = dimension.value
    payload = {
        "domain": structure.domain,
        "dimension": owned,
        "nodes": [
            node.model_dump(mode="json")
            for node in structure.nodes
            if node.dimension == owned
        ],
        "boundary_cards": [
            card.model_dump(mode="json")
            for card in structure.boundary_cards
            if card.dimension == owned
        ],
    }
    return "TAXONOMY STRUCTURE (ROLE-SCOPED):\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )
