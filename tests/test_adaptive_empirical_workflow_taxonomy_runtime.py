from __future__ import annotations

import json
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.taxonomy_runtime import (
    load_bound_taxonomy_structure_for_domain,
)


WORKTREE = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = (
    WORKTREE / "Benchmark/results/ase2022_llm_baseline/ase2022_stage3_taxonomy.json"
)
ARTIFACT_PATH = WORKTREE / "Benchmark/configs/ase2022_taxonomy_structure_v1.json"


def test_production_consumer_loads_the_registered_ase2022_structure() -> None:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))

    structure = load_bound_taxonomy_structure_for_domain(
        "ase2022", taxonomy, repository_root=WORKTREE
    )

    assert structure.domain == "ase2022"
    assert len(structure.nodes) == 23
    assert len(structure.boundary_cards) == 163


def test_production_consumer_fails_closed_when_registered_binding_is_absent(
    tmp_path: Path,
) -> None:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    artifact = tmp_path / "Benchmark/configs/ase2022_taxonomy_structure_v1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(ARTIFACT_PATH.read_bytes())

    with pytest.raises(ValueError, match="frozen corpus binding"):
        load_bound_taxonomy_structure_for_domain(
            "ase2022", taxonomy, repository_root=tmp_path
        )
