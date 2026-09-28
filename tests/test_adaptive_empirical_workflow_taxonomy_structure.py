from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow import taxonomy_structure
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    EvidenceDimension,
    TaxonomyStructure,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    _FrozenCorpusBinding,
    _ngram_hashes,
    _validate_record_independence,
    load_taxonomy_structure,
    render_taxonomy_structure,
    taxonomy_structure_hash,
)


TAXONOMY = {
    "symptom": ["Crash", "Build & Initialization Failure"],
    "root_cause": ["Incorrect Code Logic", "Improper Model Attribute"],
}


def test_production_module_exposes_no_self_signed_test_binding_loader() -> None:
    assert not hasattr(taxonomy_structure, "_load_test_bound_taxonomy_structure")


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "domain": "test",
        "nodes": [
            {
                "label": "Crash",
                "dimension": "symptom",
                "definition": "The process terminates unexpectedly.",
                "semantic_origin": "paper_definition",
                "abstraction_level": "observable_outcome",
                "responsibility_scope": "runtime_process",
                "concept_kind": "outcome",
                "parents": [],
                "children": [],
                "nearest_neighbors": ["Build & Initialization Failure"],
                "exclusion_rules": ["Do not use for a failure before runtime starts."],
            },
            {
                "label": "Build & Initialization Failure",
                "dimension": "symptom",
                "definition": "The program cannot build or initialize.",
                "semantic_origin": "paper_definition",
                "abstraction_level": "observable_outcome",
                "responsibility_scope": "build_or_initialization",
                "concept_kind": "outcome",
                "parents": [],
                "children": [],
                "nearest_neighbors": ["Crash"],
                "exclusion_rules": ["Do not use for a post-start runtime termination."],
            },
            {
                "label": "Incorrect Code Logic",
                "dimension": "root_cause",
                "definition": "The implementation logic is incorrect.",
                "semantic_origin": "paper_definition",
                "abstraction_level": "root_mechanism",
                "responsibility_scope": "implementation_logic",
                "concept_kind": "mechanism",
                "parents": [],
                "children": [],
                "nearest_neighbors": ["Improper Model Attribute"],
                "exclusion_rules": [
                    "Do not use when a model attribute is the direct defect."
                ],
            },
            {
                "label": "Improper Model Attribute",
                "dimension": "root_cause",
                "definition": "A model attribute has an improper value or shape.",
                "semantic_origin": "paper_definition",
                "abstraction_level": "root_mechanism",
                "responsibility_scope": "model_attribute",
                "concept_kind": "mechanism",
                "parents": [],
                "children": [],
                "nearest_neighbors": ["Incorrect Code Logic"],
                "exclusion_rules": ["Do not use for generic implementation logic."],
            },
        ],
        "boundary_cards": [
            {
                "card_id": "symptom-p01-p02-official-definition-v1",
                "dimension": "symptom",
                "labels": ["Crash", "Build & Initialization Failure"],
                "semantic_origin": "paper_definition",
                "decision_question": (
                    "Which official definition is directly supported by the frozen "
                    "evidence: Crash — The process terminates unexpectedly. OR Build "
                    "& Initialization Failure — The program cannot build or initialize."
                ),
                "observable_slots": [
                    "direct-support-for-crash",
                    "direct-support-for-build-initialization-failure",
                    "direct-contradiction-of-baseline",
                ],
                "criteria": [
                    {
                        "label": "Crash",
                        "positive_conditions": ["The process terminates unexpectedly."],
                        "exclusion_conditions": [
                            "Frozen evidence directly supports the official definition "
                            "of Build & Initialization Failure: The program cannot "
                            "build or initialize."
                        ],
                    },
                    {
                        "label": "Build & Initialization Failure",
                        "positive_conditions": [
                            "The program cannot build or initialize."
                        ],
                        "exclusion_conditions": [
                            "Frozen evidence directly supports the official definition "
                            "of Crash: The process terminates unexpectedly."
                        ],
                    },
                ],
            },
            {
                "card_id": "root-p01-p02-official-definition-v1",
                "dimension": "root_cause",
                "labels": ["Incorrect Code Logic", "Improper Model Attribute"],
                "semantic_origin": "paper_definition",
                "decision_question": (
                    "Which official definition is directly supported by the frozen "
                    "evidence: Incorrect Code Logic — The implementation logic is "
                    "incorrect. OR Improper Model Attribute — A model attribute has "
                    "an improper value or shape."
                ),
                "observable_slots": [
                    "direct-support-for-incorrect-code-logic",
                    "direct-support-for-improper-model-attribute",
                    "direct-contradiction-of-baseline",
                ],
                "criteria": [
                    {
                        "label": "Incorrect Code Logic",
                        "positive_conditions": [
                            "The implementation logic is incorrect."
                        ],
                        "exclusion_conditions": [
                            "Frozen evidence directly supports the official definition "
                            "of Improper Model Attribute: A model attribute has an "
                            "improper value or shape."
                        ],
                    },
                    {
                        "label": "Improper Model Attribute",
                        "positive_conditions": [
                            "A model attribute has an improper value or shape."
                        ],
                        "exclusion_conditions": [
                            "Frozen evidence directly supports the official definition "
                            "of Incorrect Code Logic: The implementation logic is "
                            "incorrect."
                        ],
                    },
                ],
            },
        ],
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "taxonomy-structure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _operational_card_payload() -> dict[str, object]:
    payload = _payload()
    payload["boundary_cards"] = [
        {
            "card_id": "symptom-runtime-vs-initialization",
            "dimension": "symptom",
            "labels": ["Crash", "Build & Initialization Failure"],
            "semantic_origin": "operational_definition",
            "decision_question": "Did termination occur after runtime began?",
            "observable_slots": ["failure_phase", "runtime_started"],
            "criteria": [
                {
                    "label": "Crash",
                    "positive_conditions": ["runtime began before termination"],
                    "exclusion_conditions": ["failure occurred during build"],
                },
                {
                    "label": "Build & Initialization Failure",
                    "positive_conditions": ["failure occurred before runtime began"],
                    "exclusion_conditions": ["runtime terminated after startup"],
                },
            ],
        },
        {
            "card_id": "root-logic-vs-model-attribute",
            "dimension": "root_cause",
            "labels": ["Incorrect Code Logic", "Improper Model Attribute"],
            "semantic_origin": "operational_definition",
            "decision_question": "Is the direct defect a model attribute?",
            "observable_slots": ["direct_defect", "responsibility_scope"],
            "criteria": [
                {
                    "label": "Incorrect Code Logic",
                    "positive_conditions": ["generic implementation logic is wrong"],
                    "exclusion_conditions": ["a model attribute is directly wrong"],
                },
                {
                    "label": "Improper Model Attribute",
                    "positive_conditions": ["a model attribute is directly wrong"],
                    "exclusion_conditions": ["only generic logic is implicated"],
                },
            ],
        },
    ]
    return payload


def _canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _replace_first_card_second_label(payload: dict[str, object], label: str) -> None:
    card = payload["boundary_cards"][0]  # type: ignore[index]
    card["labels"][1] = label  # type: ignore[index]
    card["criteria"][1]["label"] = label  # type: ignore[index]


def test_loads_complete_structure_and_hashes_canonical_content(tmp_path: Path) -> None:
    first = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)
    first_hash = taxonomy_structure_hash(first)

    changed = _payload()
    changed["nodes"][0]["abstraction_level"] = "runtime_outcome"  # type: ignore[index]
    second = load_taxonomy_structure(_write(tmp_path, changed), TAXONOMY)

    assert first.domain == "test"
    assert len(first.nodes) == 4
    assert len(first.boundary_cards) == 2
    assert len(first_hash) == 64
    assert taxonomy_structure_hash(second) != first_hash


def test_non_ase_loader_preserves_operational_boundary_card_semantics_and_hash(
    tmp_path: Path,
) -> None:
    payload = _operational_card_payload()
    structure = load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)
    legacy_canonical = json.dumps(
        structure.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert {card.semantic_origin.value for card in structure.boundary_cards} == {
        "operational_definition"
    }
    assert (
        taxonomy_structure_hash(structure)
        == hashlib.sha256(legacy_canonical).hexdigest()
    )


def test_non_ase_complete_paper_cards_keep_legacy_raw_structure_hash(
    tmp_path: Path,
) -> None:
    structure = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)
    legacy_canonical = json.dumps(
        structure.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert (
        taxonomy_structure_hash(structure)
        == hashlib.sha256(legacy_canonical).hexdigest()
    )


def test_official_definition_cards_materialize_complete_symmetric_pair_graph(
    tmp_path: Path,
) -> None:
    structure = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)

    cards = taxonomy_structure.materialize_official_boundary_cards(structure)

    assert len(cards) == 2
    assert {card.semantic_origin.value for card in cards} == {"paper_definition"}
    for card in cards:
        forward = taxonomy_structure.route_boundary_card(
            structure,
            EvidenceDimension(card.dimension),
            card.labels[0],
            card.labels[1],
        )
        reverse = taxonomy_structure.route_boundary_card(
            structure,
            EvidenceDimension(card.dimension),
            card.labels[1],
            card.labels[0],
        )
        assert forward == reverse == card


def test_public_boundary_card_manifest_binds_fixed_generator_and_materialization(
    tmp_path: Path,
) -> None:
    structure = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)

    manifest = taxonomy_structure.official_boundary_card_manifest(structure)

    assert manifest == {
        "generator_version": "official-definition-pair-v1",
        "materialized_boundary_cards_sha256": (
            taxonomy_structure.official_boundary_card_materialized_digest(structure)
        ),
    }
    assert len(manifest["materialized_boundary_cards_sha256"]) == 64


@pytest.mark.parametrize(
    ("dimension", "baseline", "proposed", "message"),
    (
        (EvidenceDimension.SYMPTOM, "Crash", "Crash", "distinct"),
        (
            EvidenceDimension.SYMPTOM,
            "Crash",
            "Incorrect Code Logic",
            "owned dimension",
        ),
        (EvidenceDimension.SYMPTOM, "Unknown", "Crash", "official label"),
    ),
)
def test_boundary_card_router_fails_closed_for_invalid_pairs(
    tmp_path: Path,
    dimension: EvidenceDimension,
    baseline: str,
    proposed: str,
    message: str,
) -> None:
    structure = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)

    with pytest.raises(ValueError, match=message):
        taxonomy_structure.route_boundary_card(structure, dimension, baseline, proposed)


def test_renderer_is_dimension_scoped_and_exposes_provenance(tmp_path: Path) -> None:
    structure = load_taxonomy_structure(_write(tmp_path, _payload()), TAXONOMY)

    rendered = render_taxonomy_structure(structure, EvidenceDimension.SYMPTOM)

    assert "Crash" in rendered
    assert "Build & Initialization Failure" in rendered
    assert "symptom-p01-p02-official-definition-v1" in rendered
    assert "paper_definition" in rendered
    assert "operational_definition" in rendered
    assert "Incorrect Code Logic" not in rendered
    assert "root-p01-p02-official-definition-v1" not in rendered


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["nodes"].pop(),
            "exactly match supplied taxonomy",
        ),
        (
            lambda payload: payload["nodes"][0].update(label="Unknown Symptom"),
            "exactly match supplied taxonomy",
        ),
        (
            lambda payload: _replace_first_card_second_label(
                payload, "Incorrect Code Logic"
            ),
            "same dimension",
        ),
        (
            lambda payload: _replace_first_card_second_label(
                payload, "Unknown Symptom"
            ),
            "unknown label",
        ),
        (
            lambda payload: payload["boundary_cards"][0]["criteria"].pop(),
            "criteria",
        ),
        (
            lambda payload: payload["nodes"][0].update(
                nearest_neighbors=["Unknown Symptom"]
            ),
            "unknown nearest neighbor",
        ),
    ),
)
def test_loader_fails_closed_on_invalid_structure(
    tmp_path: Path, mutate, message: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)


def test_structure_rejects_hidden_record_specific_fields(tmp_path: Path) -> None:
    payload = _payload()
    payload["boundary_cards"][0]["record_id"] = "development-case"  # type: ignore[index]

    with pytest.raises(ValueError, match="record_id"):
        load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)


def test_node_provenance_distinguishes_locked_definition_from_operational_structure(
    tmp_path: Path,
) -> None:
    payload = _payload()
    for node in payload["nodes"]:  # type: ignore[index]
        node.pop("semantic_origin")
        node["definition_semantic_origin"] = "paper_definition"
        node["structure_semantic_origin"] = "operational_definition"

    structure = load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)
    rendered = render_taxonomy_structure(structure, EvidenceDimension.SYMPTOM)

    assert structure.nodes[0].definition_semantic_origin.value == "paper_definition"
    assert (
        structure.nodes[0].structure_semantic_origin.value == "operational_definition"
    )
    assert '"definition_semantic_origin": "paper_definition"' in rendered
    assert '"structure_semantic_origin": "operational_definition"' in rendered
    rendered_payload = json.loads(rendered.split("\n", 1)[1])
    assert all("semantic_origin" not in node for node in rendered_payload["nodes"])
    assert {card["semantic_origin"] for card in rendered_payload["boundary_cards"]} == {
        "paper_definition"
    }


@pytest.mark.parametrize(
    "leaked_value",
    (
        "synthetic_study:0123456789abcdef",
        "Ground truth: Crash",
        "Use the gold expected-answer prototype for this decision.",
        "This rule contains case-specific development case text.",
    ),
)
def test_loader_rejects_record_specific_string_values(
    tmp_path: Path, leaked_value: str
) -> None:
    payload = _payload()
    payload["boundary_cards"][0]["decision_question"] = leaked_value  # type: ignore[index]

    with pytest.raises(ValueError, match="record-specific"):
        load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)


def test_ase2022_loader_requires_a_frozen_corpus_binding(tmp_path: Path) -> None:
    payload = _payload()
    payload["domain"] = "ase2022"

    with pytest.raises(ValueError, match="frozen corpus binding"):
        load_taxonomy_structure(_write(tmp_path, payload), TAXONOMY)


def test_bundled_production_loader_rejects_self_signed_fake_sources(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["domain"] = "ase2022"
    artifact = tmp_path / "ase2022_taxonomy_structure_v1.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    corpus = tmp_path / "frozen.csv"
    corpus.write_text(
        "record_id,title,body\npaper:0123456789abcdef,Safe title,Safe body\n",
        encoding="utf-8",
    )
    normalized_entries = (
        "paper:0123456789abcdef\u0000title\u0000safe title\n"
        "paper:0123456789abcdef\u0000body\u0000safe body"
    )
    binding = {
        "schema_version": 1,
        "domain": "ase2022",
        "artifact_sha256": _canonical_json_sha256(payload),
        "corpus_path": "frozen.csv",
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "source_paths": {"not_the_frozen_sources": "frozen.csv"},
        "source_sha256": {"not_the_frozen_sources": "0" * 64},
        "record_count": 1,
        "record_ids_sha256": hashlib.sha256(b"paper:0123456789abcdef").hexdigest(),
        "fields": ["title", "body"],
        "forbidden_text_digest": hashlib.sha256(
            normalized_entries.encode("utf-8")
        ).hexdigest(),
        "ngram_words": 8,
        "min_ngram_chars": 40,
        "approved_operational_text_sha256": [],
    }
    binding_path = tmp_path / "ase2022_taxonomy_structure_v1_corpus_binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted (?:bundled binding|source)"):
        load_taxonomy_structure(artifact, TAXONOMY)


def test_frozen_binding_rejects_case_copy_but_allows_quoted_official_definition() -> (
    None
):
    official_definition = "The process terminates unexpectedly."
    copied_case = "Browser conversion endpoint returns persistently incorrect tensors"
    payload = _payload()
    payload["domain"] = "ase2022"
    structure = TaxonomyStructure.model_validate(payload)
    binding = _FrozenCorpusBinding(
        forbidden_ngram_hashes=_ngram_hashes(
            f"{official_definition} {copied_case}", words=7, min_chars=40
        ),
        approved_text_hashes=frozenset(),
        ngram_words=7,
        min_ngram_chars=40,
    )
    _validate_record_independence(
        payload,
        structure=structure,
        corpus_binding=binding,
    )

    payload["boundary_cards"][0]["decision_question"] = copied_case  # type: ignore[index]

    with pytest.raises(ValueError, match="record-specific"):
        _validate_record_independence(
            payload,
            structure=TaxonomyStructure.model_validate(payload),
            corpus_binding=binding,
        )
