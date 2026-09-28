"""Trusted production entry point for record-independent taxonomy guidance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import TaxonomyStructure
from .taxonomy_structure import load_taxonomy_structure


_REGISTERED_STRUCTURE_PATHS = {
    "ase2022": "Benchmark/configs/ase2022_taxonomy_structure_v1.json",
}
_REGISTERED_BINDING_PATHS = {
    "ase2022": ("Benchmark/configs/ase2022_taxonomy_structure_v1_corpus_binding.json"),
}


def bound_taxonomy_artifact_manifest_for_domain(
    domain: str,
    *,
    repository_root: str | Path | None = None,
) -> Mapping[str, str]:
    """Return content hashes for the domain's pre-registered trust artifacts."""

    try:
        structure_relative = _REGISTERED_STRUCTURE_PATHS[domain]
        binding_relative = _REGISTERED_BINDING_PATHS[domain]
    except KeyError as error:
        raise ValueError(
            f"no bound taxonomy structure is registered for domain {domain!r}"
        ) from error
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )

    def digest(relative: str) -> str:
        try:
            return hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(
                f"cannot read bound taxonomy artifact: {relative}"
            ) from error

    return {
        "taxonomy_structure_sha256": digest(structure_relative),
        "bound_corpus_sidecar_sha256": digest(binding_relative),
    }


def load_bound_taxonomy_structure_for_domain(
    domain: str,
    taxonomy: Mapping[str, Sequence[str]],
    *,
    repository_root: str | Path | None = None,
) -> TaxonomyStructure:
    """Load one pre-registered structure and all of its frozen trust bindings."""

    try:
        relative_path = _REGISTERED_STRUCTURE_PATHS[domain]
    except KeyError as error:
        raise ValueError(
            f"no bound taxonomy structure is registered for domain {domain!r}"
        ) from error
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    structure = load_taxonomy_structure(root / relative_path, taxonomy)
    if structure.domain != domain:
        raise ValueError("bound taxonomy structure domain mismatch")
    return structure
