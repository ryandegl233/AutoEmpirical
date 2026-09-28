from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow import taxonomy_structure
from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceDimension
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    load_taxonomy_structure,
    render_taxonomy_structure,
    taxonomy_structure_hash,
)
from Benchmark.src.ase2022_llm_baseline import (
    ROOT_CAUSE_DEFINITIONS,
    SYMPTOM_DEFINITIONS,
)


WORKTREE = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = WORKTREE / "Benchmark/configs/ase2022_taxonomy_structure_v1.json"
BINDING_PATH = (
    WORKTREE / "Benchmark/configs/ase2022_taxonomy_structure_v1_corpus_binding.json"
)
OFFICIAL_TAXONOMY_PATH = (
    WORKTREE / "Benchmark/results/ase2022_llm_baseline/ase2022_stage3_taxonomy.json"
)
FORBIDDEN_RECORD_SPECIFIC_KEYS = {
    "record_id",
    "gt",
    "ground_truth",
    "prototype",
    "prototype_case",
    "development_case",
}


def _keys(payload: object) -> Iterator[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key.lower()
            yield from _keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _keys(value)


def _string_values(payload: object) -> Iterator[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _string_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _string_values(value)


def test_record_specific_key_scan_recurses_through_nested_lists() -> None:
    payload = {"nodes": [{"criteria": [{"prototype_case": "canary"}]}]}

    assert "prototype_case" in set(_keys(payload))


def test_ase2022_taxonomy_artifact_binds_all_paper_labels_and_definitions() -> None:
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    structure = load_taxonomy_structure(ARTIFACT_PATH, taxonomy)
    by_dimension = {
        dimension: {
            node.label: node for node in structure.nodes if node.dimension == dimension
        }
        for dimension in ("symptom", "root_cause")
    }

    assert set(by_dimension["symptom"]) == set(taxonomy["symptom"])
    assert set(by_dimension["root_cause"]) == set(taxonomy["root_cause"])
    assert {
        label: node.definition for label, node in by_dimension["symptom"].items()
    } == SYMPTOM_DEFINITIONS
    assert {
        label: node.definition for label, node in by_dimension["root_cause"].items()
    } == ROOT_CAUSE_DEFINITIONS
    assert {
        node.definition_semantic_origin.value
        for dimension_nodes in by_dimension.values()
        for node in dimension_nodes.values()
    } == {"paper_definition"}
    assert {
        node.structure_semantic_origin.value
        for dimension_nodes in by_dimension.values()
        for node in dimension_nodes.values()
    } == {"operational_definition"}
    assert all("semantic_origin" not in node for node in payload["nodes"])
    assert all(
        node["definition_semantic_origin"] == "paper_definition"
        and node["structure_semantic_origin"] == "operational_definition"
        for node in payload["nodes"]
    )
    for dimension_nodes in by_dimension.values():
        for node in dimension_nodes.values():
            for neighbor in node.nearest_neighbors:
                assert node.label in dimension_nodes[neighbor].nearest_neighbors


def test_ase2022_artifact_has_complete_unique_official_definition_boundary_graph() -> (
    None
):
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))
    structure = load_taxonomy_structure(ARTIFACT_PATH, taxonomy)
    cards = taxonomy_structure.materialize_official_boundary_cards(structure)

    symptom_cards = [card for card in cards if card.dimension == "symptom"]
    root_cards = [card for card in cards if card.dimension == "root_cause"]
    unordered_pairs = [
        (card.dimension, frozenset(card.labels)) for card in structure.boundary_cards
    ]

    assert len(symptom_cards) == 10
    assert len(root_cards) == 153
    assert len(cards) == len(structure.boundary_cards) == 163
    assert len(unordered_pairs) == len(set(unordered_pairs))
    assert cards == structure.boundary_cards
    assert {card.semantic_origin.value for card in cards} == {"paper_definition"}


@pytest.mark.parametrize(
    ("dimension", "left", "right"),
    (
        (
            EvidenceDimension.ROOT_CAUSE,
            "Improper Exception Handling",
            "API Misuse",
        ),
        (
            EvidenceDimension.ROOT_CAUSE,
            "WebGL Limits",
            "Incorrect Code Logic",
        ),
        (
            EvidenceDimension.ROOT_CAUSE,
            "Untimely Update",
            "Incorrect Code Logic",
        ),
    ),
)
def test_general_boundary_pairs_route_to_one_symmetric_card(
    dimension: EvidenceDimension, left: str, right: str
) -> None:
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))
    structure = load_taxonomy_structure(ARTIFACT_PATH, taxonomy)

    forward = taxonomy_structure.route_boundary_card(structure, dimension, left, right)
    reverse = taxonomy_structure.route_boundary_card(structure, dimension, right, left)

    assert forward == reverse
    assert frozenset(forward.labels) == frozenset((left, right))


def test_ase2022_taxonomy_artifact_is_role_isolated_and_hash_stable() -> None:
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))
    first = load_taxonomy_structure(ARTIFACT_PATH, taxonomy)
    second = load_taxonomy_structure(ARTIFACT_PATH, taxonomy)

    symptom_guidance = render_taxonomy_structure(first, EvidenceDimension.SYMPTOM)
    root_cause_guidance = render_taxonomy_structure(first, EvidenceDimension.ROOT_CAUSE)

    assert "Crash" in symptom_guidance
    assert "Incorrect Code Logic" not in symptom_guidance
    assert "Incorrect Code Logic" in root_cause_guidance
    assert "Crash" not in root_cause_guidance
    assert "symptom-p01-p02-official-definition-v1" in symptom_guidance
    assert "root-p01-p02-official-definition-v1" in root_cause_guidance
    assert taxonomy_structure_hash(first) == taxonomy_structure_hash(second)
    assert taxonomy_structure.official_boundary_card_manifest(first) == {
        "generator_version": "official-definition-pair-v1",
        "materialized_boundary_cards_sha256": (
            taxonomy_structure.official_boundary_card_materialized_digest(first)
        ),
    }


def test_ase2022_taxonomy_artifact_contains_only_record_independent_data() -> None:
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    forbidden = FORBIDDEN_RECORD_SPECIFIC_KEYS.intersection(_keys(payload))
    values = tuple(_string_values(payload))
    record_id = re.compile(r"\b[a-z0-9][a-z0-9_-]*:[0-9a-f]{16}\b", re.IGNORECASE)
    record_specific_term = re.compile(
        r"\b(?:gt|gold|ground[ _-]+truth|expected[ _-]+answer|"
        r"prototype(?:[ _-]+case)?|development[ _-]+case|case[ _-]+specific)\b",
        re.IGNORECASE,
    )
    issue_url = re.compile(r"https?://|\b(?:github|gitlab)\.com/", re.IGNORECASE)

    assert not forbidden
    assert not [value for value in values if record_id.search(value)]
    assert not [value for value in values if record_specific_term.search(value)]
    assert not [value for value in values if issue_url.search(value)]


def test_ase2022_taxonomy_artifact_has_content_addressed_frozen_corpus_binding() -> (
    None
):
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    canonical = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    corpus = (BINDING_PATH.parent / binding["corpus_path"]).resolve()

    assert binding["artifact_sha256"] == hashlib.sha256(canonical).hexdigest()
    corpus_bytes = corpus.read_bytes().replace(b"\r\n", b"\n")
    assert binding["corpus_sha256"] == hashlib.sha256(corpus_bytes).hexdigest()
    assert binding["record_count"] == 100
    assert set(binding["source_sha256"]) == {
        "stage1",
        "stage2",
        "stage3",
        "excluded_cohort",
    }
    assert set(binding["source_paths"]) == set(binding["source_sha256"])
    for name, relative_path in binding["source_paths"].items():
        source_bytes = (BINDING_PATH.parent / relative_path).resolve().read_bytes()
        normalized = source_bytes.replace(b"\r\n", b"\n")
        assert binding["source_sha256"][name] == hashlib.sha256(normalized).hexdigest()
    assert not {"decision", "symptom", "root_cause"}.intersection(binding["fields"])
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in binding["approved_operational_text_sha256"]
    )


def test_bundled_ase2022_artifact_rejects_an_explicit_binding_override() -> None:
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="trusted bundled binding"):
        load_taxonomy_structure(
            ARTIFACT_PATH,
            taxonomy,
            corpus_binding_path=BINDING_PATH,
        )


def test_bundled_ase2022_artifact_rejects_a_replaced_default_sidecar(
    tmp_path: Path,
) -> None:
    taxonomy = json.loads(OFFICIAL_TAXONOMY_PATH.read_text(encoding="utf-8"))
    artifact = tmp_path / ARTIFACT_PATH.name
    binding_path = tmp_path / BINDING_PATH.name
    artifact.write_bytes(ARTIFACT_PATH.read_bytes())
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    binding["record_count"] = 1
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted bundled binding hash"):
        load_taxonomy_structure(artifact, taxonomy)
