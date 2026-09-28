"""Dataset adapters and record-local evidence specialists."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from Benchmark.src.annotation_contracts import NEW_PAPER_DOMAINS, annotation_guidance, annotation_mode

from .contracts import (
    EvidenceDelta,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceRequest,
    EvidenceView,
    RetrievalStatus,
    SpecialistType,
)
from .ledger import EvidenceLedger
from .frozen_evidence_runtime import (
    FrozenEvidenceLedger,
    FrozenEvidenceProjection,
    verified_frozen_evidence_items,
)
from .splits import load_split_manifest_for_runner
from .specialists import (
    QueryPassage,
    Specialist,
    SpecialistRegistry,
    extract_query_passages,
)

GOLD_FIELDS = frozenset({"decision", "symptom", "root_cause"})
DECISION_TAXONOMY = ["accepted_fault", "rejected_candidate"]


@dataclass(frozen=True)
class DomainProfile:
    name: str
    default_cohort_path: str
    default_taxonomy_path: str


DOMAIN_PROFILES = {
    "ase2022": DomainProfile(
        name="ase2022",
        default_cohort_path=(
            "Benchmark/results/ase2022_camel_mas_baseline/"
            "ase2022_camel_mas_cohort.csv"
        ),
        default_taxonomy_path=(
            "Benchmark/results/ase2022_camel_mas_baseline/"
            "ase2022_camel_mas_taxonomy.json"
        ),
    ),
    "issta2024": DomainProfile(
        name="issta2024",
        default_cohort_path=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "issta2024_stage2_filter_sample.csv"
        ),
        default_taxonomy_path=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "issta2024_taxonomy.json"
        ),
    ),
}
def _native_domain_profile(domain: str) -> DomainProfile:
    from Benchmark.src.paper_benchmark import get_paper_profile

    profile = get_paper_profile(domain)
    return DomainProfile(domain, profile.default_cohort_path, profile.default_taxonomy_path)


DOMAIN_PROFILES.update({
    domain: _native_domain_profile(domain) for domain in sorted(NEW_PAPER_DOMAINS)
})

_NEUTRAL_TAXONOMY_COMPARISON = (
    "TAXONOMY COMPARISON: Compare the record-local evidence against every "
    "supplied label using only the official definitions above. For the owned "
    "dimension, identify the evidence-supported behavior or mechanism, name "
    "the nearest competing label, and state the discriminating fact. Do not "
    "apply a global label precedence based on label order, frequency, or "
    "examples from other records. If the available evidence cannot distinguish "
    "the nearest boundary, report that uncertainty in the structured fields."
)

_TAXONOMY_BOUNDARIES = {
    "ase2022": _NEUTRAL_TAXONOMY_COMPARISON,
    "issta2024": (
        "Boundary rule: classify the directly observed runtime outcome, not "
        "the patched component name. Use Others only after contrasting every "
        "specific leaf definition."
    ),
}


@dataclass(frozen=True)
class DomainInputs:
    profile: DomainProfile
    records: list[dict[str, str]]
    taxonomy: dict[str, list[str]]


@dataclass(frozen=True)
class RecordRuntime:
    ledger: EvidenceLedger
    specialists: SpecialistRegistry


def taxonomy_guidance(
    domain: str,
    taxonomy: dict[str, list[str]],
    *,
    include_stage3_boundaries: bool = True,
) -> str:
    """Render authoritative label meanings for a stable model prefix."""

    domain_profile(domain)
    if domain in NEW_PAPER_DOMAINS:
        from Benchmark.src.paper_benchmark import get_paper_profile
        profile = get_paper_profile(domain)
        sections = ["PAPER-SPECIFIC ANNOTATION CODEBOOK:",
                    "Labels preserve the source dataset. Descriptions are operational guidance, not verbatim official definitions."]
        for dimension in ("symptom", "root_cause"):
            sections.append(annotation_guidance(taxonomy, dimension))
            definitions = profile.definitions.get(dimension, {})
            sections.extend(f"- {label}: {definitions.get(label, 'Use the supplied label meaning; report uncertainty when evidence is insufficient.')}"
                            for label in taxonomy.get(dimension, []))
        if include_stage3_boundaries:
            sections.append("Compare evidence-supported alternatives without global label precedence. Separate observed behavior from hypothesized causes and retain evidence limitations.")
        return "\n".join(sections)
    if domain == "ase2022":
        from Benchmark.src.ase2022_llm_baseline import (
            ROOT_CAUSE_DEFINITIONS,
            SYMPTOM_DEFINITIONS,
        )
    else:
        from Benchmark.src.issta2024_bugs_in_pods_baseline import (
            ROOT_CAUSE_DEFINITIONS,
            SYMPTOM_DEFINITIONS,
        )

    sections: list[str] = ["TAXONOMY DEFINITIONS:"]
    for dimension, definitions in (
        ("symptom", SYMPTOM_DEFINITIONS),
        ("root_cause", ROOT_CAUSE_DEFINITIONS),
    ):
        sections.append(f"{dimension.upper()}:")
        for label in taxonomy.get(dimension, []):
            definition = definitions.get(label)
            if definition:
                sections.append(f"- {label}: {definition}")
            else:
                sections.append(f"- {label}: no domain definition available")
    if include_stage3_boundaries:
        sections.append(_TAXONOMY_BOUNDARIES[domain])
    return "\n".join(sections)


def _allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def domain_profile(domain: str) -> DomainProfile:
    if domain in NEW_PAPER_DOMAINS:
        return _native_domain_profile(domain)
    try:
        return DOMAIN_PROFILES[domain]
    except KeyError as error:
        raise ValueError(f"unsupported domain {domain!r}") from error


def load_domain_inputs(
    domain: str,
    *,
    cohort_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> DomainInputs:
    profile = domain_profile(domain)
    cohort_source = Path(cohort_path or profile.default_cohort_path)
    taxonomy_source = Path(taxonomy_path or profile.default_taxonomy_path)
    if not cohort_source.exists():
        raise FileNotFoundError(f"cohort not found: {cohort_source}")
    if not taxonomy_source.exists():
        raise FileNotFoundError(f"taxonomy not found: {taxonomy_source}")
    manifest_source = (
        Path(split_manifest_path) if split_manifest_path is not None else None
    )
    if (
        manifest_source is None
        and (cohort_source.parent / "split_manifest.json").exists()
    ):
        manifest_source = cohort_source.parent / "split_manifest.json"
    if manifest_source is not None:
        load_split_manifest_for_runner(
            manifest_source,
            cohort_path=cohort_source,
            domain=domain,
            taxonomy_path=taxonomy_source,
        )

    _allow_large_csv_fields()
    with cohort_source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "record_id" not in (reader.fieldnames or []):
            raise ValueError("cohort must contain record_id")
        records = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]
    record_ids = [record["record_id"] for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("cohort contains an empty record_id")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("cohort contains duplicate record_id values")

    raw_taxonomy = json.loads(taxonomy_source.read_text(encoding="utf-8"))
    if isinstance(raw_taxonomy, dict) and isinstance(raw_taxonomy.get("nodes"), list):
        label_lists = {
            dimension: [
                str(node["label"])
                for node in raw_taxonomy["nodes"]
                if isinstance(node, dict)
                and node.get("dimension") == dimension
                and node.get("label")
            ]
            for dimension in ("symptom", "root_cause")
        }
    elif isinstance(raw_taxonomy, dict):
        label_lists = {
            dimension: raw_taxonomy.get(dimension, [])
            for dimension in ("symptom", "root_cause")
        }
    else:
        label_lists = {}
    if not all(
        isinstance(label_lists.get(dimension), list) and
        (label_lists[dimension] or annotation_mode(raw_taxonomy, dimension) == "free_text")
        for dimension in ("symptom", "root_cause")
    ):
        raise ValueError("taxonomy must contain non-empty symptom and root_cause lists")
    taxonomy = {
        "decision": list(DECISION_TAXONOMY),
        "symptom": [str(label) for label in label_lists["symptom"]],
        "root_cause": [str(label) for label in label_lists["root_cause"]],
    }
    if "annotation_modes" in raw_taxonomy:
        taxonomy["annotation_modes"] = dict(raw_taxonomy["annotation_modes"])
        for dimension in ("symptom", "root_cause"):
            mode = annotation_mode(taxonomy, dimension)
            if mode == "constant" and len(taxonomy[dimension]) != 1:
                raise ValueError(f"{dimension} constant mode requires exactly one label")
    if domain in NEW_PAPER_DOMAINS:
        from Benchmark.src.paper_benchmark import get_paper_profile, build_taxonomy
        paper_profile = get_paper_profile(domain)
        expected_taxonomy = build_taxonomy(domain)
        if any(taxonomy.get(key) != expected_taxonomy.get(key)
               for key in ("symptom", "root_cause", "annotation_modes")):
            raise ValueError(f"taxonomy does not match versioned {domain} paper codebook")
        for record in records:
            record_paper = record.get("paper_id", "") or record["record_id"].split(":", 1)[0]
            if record_paper != paper_profile.paper_id:
                raise ValueError(f"cohort paper binding mismatch for {domain}: {record['record_id']}")
            if ":" in record["record_id"] and record["record_id"].split(":", 1)[0] != paper_profile.paper_id:
                raise ValueError(f"record_id paper binding mismatch for {domain}: {record['record_id']}")
    return DomainInputs(
        profile=profile,
        records=records,
        taxonomy=taxonomy,
    )


def _evidence_item(
    *,
    record_id: str,
    specialist: SpecialistType | None,
    source_type: str,
    source_uri: str,
    content: str,
    retrieved_at: str,
    identity: str | None = None,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    identity_digest = hashlib.sha256((identity or content).encode("utf-8")).hexdigest()
    prefix = (
        "source"
        if identity is not None
        else specialist.value if specialist is not None else "initial"
    )
    return EvidenceItem(
        evidence_id=f"{prefix}-{identity_digest[:16]}",
        record_id=record_id,
        source_type=source_type,
        source_uri=source_uri,
        retrieved_at=retrieved_at,
        content=content,
        content_sha256=digest,
        explicitness=EvidenceExplicitness.DIRECT,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class _FrozenSource:
    source_type: str
    content: str
    source_kind: str
    aliases: tuple[str, ...]
    captured: bool


_UNAVAILABLE_SOURCE_MARKERS = frozenset({
    "not_fetched", "not_available_in_source", "comments_unavailable_in_source",
})
_CAPTURED_EMPTY_SOURCE_MARKERS = {
    "comments": frozenset({"no_comments_in_source", "[]"}),
    "changed_files": frozenset({"[]"}),
    "commit_history": frozenset({"[]"}),
}
_GENERIC_SOURCE_TARGETS = frozenset(
    {
        "record local source material",
        "record local sources",
        "frozen source",
        "frozen sources",
        "offline source",
        "offline sources",
    }
)


def _source_name(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


def _captured_record_source(
    record: dict[str, str], field_name: str
) -> tuple[bool, str]:
    if field_name not in record:
        return False, ""
    content = record[field_name].strip()
    if not content:
        return False, ""
    if content.casefold() in _UNAVAILABLE_SOURCE_MARKERS:
        return False, ""
    if content.casefold() in _CAPTURED_EMPTY_SOURCE_MARKERS.get(
        field_name, frozenset()
    ):
        return True, ""
    return True, content


def _frozen_sources(
    specialist: SpecialistType,
    record: dict[str, str],
    taxonomy: dict[str, list[str]],
    domain: str,
) -> tuple[_FrozenSource, ...]:
    if specialist is SpecialistType.ISSUE_PR:
        body_captured, body = _captured_record_source(record, "body")
        comments_captured, comments = _captured_record_source(record, "comments")
        return (
            _FrozenSource(
                source_type="issue_body",
                content=body,
                source_kind="prose",
                aliases=(
                    "issue",
                    "linked issue",
                    "linked issue and pull request",
                    "pull request",
                    "issue body",
                    "body",
                    "commit message",
                ),
                captured=body_captured,
            ),
            _FrozenSource(
                source_type="issue_comments",
                content=comments,
                source_kind="comments",
                aliases=(
                    "issue",
                    "linked issue",
                    "linked issue and pull request",
                    "pull request",
                    "comments",
                    "issue comments",
                    "linked issue comments",
                    "discussion",
                    "linked issue discussion",
                    "pull request discussion",
                ),
                captured=comments_captured,
            ),
        )
    if specialist is SpecialistType.CODE_CONTEXT:
        files_captured, changed_files = _captured_record_source(record, "changed_files")
        diff_captured, code_diff = _captured_record_source(record, "code_diff")
        return (
            _FrozenSource(
                source_type="changed_files",
                content=changed_files,
                source_kind="lines",
                aliases=("changed files", "file list", "files"),
                captured=files_captured,
            ),
            _FrozenSource(
                source_type="code_diff",
                content=code_diff,
                source_kind="diff",
                aliases=("code", "diff", "code diff", "patch", "code context"),
                captured=diff_captured,
            ),
        )
    if specialist is SpecialistType.TEST_EVIDENCE:
        captured, code_diff = _captured_record_source(record, "code_diff")
        return (
            _FrozenSource(
                source_type="code_diff",
                content=code_diff,
                source_kind="test_diff",
                aliases=("test", "tests", "test evidence", "test diff", "code diff"),
                captured=captured,
            ),
        )
    if specialist is SpecialistType.COMMIT_HISTORY:
        captured, history = _captured_record_source(record, "commit_history")
        return (
            _FrozenSource(
                source_type="commit_history",
                content=history,
                source_kind="prose",
                aliases=("commit", "history", "commit history"),
                captured=captured,
            ),
        )
    if specialist is SpecialistType.TAXONOMY_KNOWLEDGE:
        codebook = taxonomy_guidance(
            domain,
            taxonomy,
            include_stage3_boundaries=True,
        )
        return (
            _FrozenSource(
                source_type="taxonomy",
                content=codebook,
                source_kind="lines",
                aliases=("taxonomy", "codebook", "labels", "supplied taxonomy"),
                captured=True,
            ),
        )
    return ()


def _select_sources(
    specialist_type: SpecialistType,
    target_source: str,
    sources: tuple[_FrozenSource, ...],
) -> tuple[_FrozenSource, ...]:
    normalized_target = _source_name(target_source)
    if normalized_target in _GENERIC_SOURCE_TARGETS:
        return sources
    selected = tuple(
        source
        for source in sources
        if any(_source_name(alias) == normalized_target for alias in source.aliases)
    )
    if not selected:
        raise ValueError(
            f"target_source {target_source!r} does not match "
            f"{specialist_type.value} frozen sources"
        )
    return selected


def allowed_frozen_evidence_targets(
    domain: str,
) -> dict[str, tuple[str, ...]]:
    """Return the prompt-safe target names accepted by each local specialist."""

    catalog: dict[str, tuple[str, ...]] = {}
    for specialist in SpecialistType:
        sources = _frozen_sources(specialist, {}, {}, domain)
        if not sources:
            continue
        aliases = {alias for source in sources for alias in source.aliases}
        aliases.update(_GENERIC_SOURCE_TARGETS)
        catalog[specialist.value] = tuple(sorted(aliases))
    return catalog


def validate_frozen_evidence_target(
    request: EvidenceRequest,
    *,
    domain: str,
) -> None:
    """Reject a request that cannot route to a frozen record-local source."""

    _select_sources(
        request.target_specialist,
        request.target_source,
        _frozen_sources(request.target_specialist, {}, {}, domain),
    )


def _source_uri(*, record: dict[str, str], record_id: str, source_type: str) -> str:
    _, issue_url = _captured_record_source(record, "issue_url")
    if source_type in {"issue_body", "issue_comments"} and issue_url:
        return issue_url
    return f"record:{record_id}#{source_type}"


def _passage_item(
    *,
    record: dict[str, str],
    record_id: str,
    specialist_type: SpecialistType,
    request: EvidenceRequest,
    source: _FrozenSource,
    match: QueryPassage,
    retrieved_at: str,
) -> EvidenceItem:
    source_uri = _source_uri(
        record=record,
        record_id=record_id,
        source_type=source.source_type,
    )
    frozen_digest = hashlib.sha256(source.content.encode("utf-8")).hexdigest()
    identity = "\n".join(
        (
            record_id,
            source.source_type,
            source_uri,
            frozen_digest,
            str(match.passage_index),
            match.content,
        )
    )
    return _evidence_item(
        record_id=record_id,
        specialist=specialist_type,
        source_type=source.source_type,
        source_uri=source_uri,
        content=match.content,
        retrieved_at=retrieved_at,
        identity=identity,
        metadata={
            "request_id": request.request_id,
            "target_source": request.target_source,
            "query": request.query,
            "frozen_source_type": source.source_type,
            "frozen_source_sha256": frozen_digest,
            "passage_index": match.passage_index,
            "query_overlap_score": match.score,
        },
    )


def _record_specialist(
    specialist_type: SpecialistType,
    *,
    record: dict[str, str],
    taxonomy: dict[str, list[str]],
    domain: str,
    retrieved_at: str,
) -> Specialist:
    def retrieve(
        request: EvidenceRequest,
        view: EvidenceView,
    ) -> EvidenceDelta:
        sources = _select_sources(
            specialist_type,
            request.target_source,
            _frozen_sources(specialist_type, record, taxonomy, domain),
        )
        captured_sources = tuple(source for source in sources if source.captured)
        if not captured_sources:
            return EvidenceDelta(
                request_id=request.request_id,
                specialist=specialist_type,
                status=RetrievalStatus.UNAVAILABLE,
                diagnostics={
                    "reason": "source_not_captured",
                    "target_source": request.target_source,
                },
            )

        record_id = record["record_id"]
        ranked: list[tuple[int, int, int, _FrozenSource, QueryPassage]] = []
        extraction_limit = request.max_items + len(view.items)
        for source_order, source in enumerate(captured_sources):
            matches = extract_query_passages(
                source.content,
                query=request.query,
                max_items=extraction_limit,
                source_kind=source.source_kind,
            )
            ranked.extend(
                (
                    -match.score,
                    source_order,
                    match.passage_index,
                    source,
                    match,
                )
                for match in matches
            )
        ranked.sort(key=lambda entry: entry[:3])
        if not ranked:
            return EvidenceDelta(
                request_id=request.request_id,
                specialist=specialist_type,
                status=RetrievalStatus.ABSENT,
                diagnostics={
                    "reason": "no_matching_passage",
                    "target_source": request.target_source,
                },
            )

        existing_ids = {item.evidence_id for item in view.items}
        existing_content = {
            (item.source_uri, item.content_sha256) for item in view.items
        }
        seen_content = set(existing_content)
        items: list[EvidenceItem] = []
        for _, _, _, source, match in ranked:
            item = _passage_item(
                record=record,
                record_id=record_id,
                specialist_type=specialist_type,
                request=request,
                source=source,
                match=match,
                retrieved_at=retrieved_at,
            )
            content_key = (item.source_uri, item.content_sha256)
            if item.evidence_id in existing_ids or content_key in seen_content:
                continue
            seen_content.add(content_key)
            items.append(item)
            if len(items) == request.max_items:
                break
        if not items:
            return EvidenceDelta(
                request_id=request.request_id,
                specialist=specialist_type,
                status=RetrievalStatus.ABSENT,
                diagnostics={"reason": "evidence_already_in_ledger"},
            )
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=specialist_type,
            status=RetrievalStatus.FOUND,
            items=items,
            diagnostics={
                "source_types": [source.source_type for source in captured_sources]
            },
        )

    return retrieve


def build_record_runtime(
    record: dict[str, str],
    *,
    taxonomy: dict[str, list[str]],
    domain: str,
    retrieved_at: str | None = None,
    frozen_evidence_projection: FrozenEvidenceProjection | None = None,
) -> RecordRuntime:
    domain_profile(domain)
    record_id = record.get("record_id", "").strip()
    if not record_id:
        raise ValueError("record must contain a non-empty record_id")
    frozen_items = verified_frozen_evidence_items(
        frozen_evidence_projection,
        domain=domain,
        record_id=record_id,
    )
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    supported = (
        SpecialistType.COMMIT_HISTORY,
        SpecialistType.ISSUE_PR,
        SpecialistType.CODE_CONTEXT,
        SpecialistType.TEST_EVIDENCE,
        SpecialistType.TAXONOMY_KNOWLEDGE,
    )
    frozen_sources = tuple(
        source
        for specialist in supported
        for source in _frozen_sources(specialist, record, taxonomy, domain)
    )
    captured_source_types = sorted(
        {source.source_type for source in frozen_sources if source.captured}
    )
    unavailable_source_types = sorted(
        {source.source_type for source in frozen_sources if not source.captured}
        - set(captured_source_types)
    )
    _, issue_url = _captured_record_source(record, "issue_url")
    issue_uri = issue_url or f"record:{record_id}#initial-summary"
    summary_fields = tuple(
        (field_name.upper(), _captured_record_source(record, field_name)[1])
        for field_name in ("issue_url", "title", "body", "state", "created_at", "source_project")
    )
    summary = "\n".join(f"{name}: {value}" for name, value in summary_fields if value)
    if not summary:
        summary = "No issue summary is available for this record."
    initial = _evidence_item(
        record_id=record_id,
        specialist=None,
        source_type="record_summary",
        source_uri=issue_uri,
        content=summary,
        retrieved_at=timestamp,
        metadata={
            "captured_frozen_source_types": captured_source_types,
            "unavailable_frozen_source_types": unavailable_source_types,
            **(
                {
                    "frozen_evidence_runtime": {
                        "enabled": True,
                        "available": bool(
                            frozen_evidence_projection.audit["available"]
                        ),
                        "graph_sha256": str(
                            frozen_evidence_projection.audit["graph_sha256"]
                        ),
                        "projection_policy_sha256": str(
                            frozen_evidence_projection.audit["projection_policy_sha256"]
                        ),
                    }
                }
                if frozen_evidence_projection is not None
                else {}
            ),
        },
    )
    initial_items = [initial]
    comments_captured, comments = _captured_record_source(record, "comments")
    if comments_captured and comments:
        initial_items.append(
            _evidence_item(
                record_id=record_id,
                specialist=SpecialistType.ISSUE_PR,
                source_type="issue_comments",
                source_uri=issue_uri,
                content=comments,
                retrieved_at=timestamp,
            )
        )
    existing_ids = {item.evidence_id for item in initial_items}
    frozen_ids = [item.evidence_id for item in frozen_items]
    if len(frozen_ids) != len(set(frozen_ids)) or existing_ids.intersection(frozen_ids):
        raise ValueError("runtime frozen evidence contains duplicate evidence IDs")
    initial_items.extend(frozen_items)
    ledger_arguments = {
        "record_id": record_id,
        "task": (
            "Apply the paper-specific fault inclusion policy, then annotate "
            "symptom and root cause according to the native annotation contract."
            if domain in NEW_PAPER_DOMAINS else
            "Verify the candidate as a fault repair, then classify symptom "
            "and root cause only when accepted."
        ),
        "taxonomy": taxonomy,
        "domain_profile": domain,
        "initial_items": initial_items,
    }
    ledger = (
        EvidenceLedger(**ledger_arguments)
        if frozen_evidence_projection is None
        else FrozenEvidenceLedger(
            **ledger_arguments,
            frozen_evidence_projection=frozen_evidence_projection,
        )
    )
    specialists: dict[SpecialistType, Callable] = {
        specialist: _record_specialist(
            specialist,
            record=record,
            taxonomy=taxonomy,
            domain=domain,
            retrieved_at=timestamp,
        )
        for specialist in supported
    }
    return RecordRuntime(
        ledger=ledger,
        specialists=SpecialistRegistry(specialists),
    )
