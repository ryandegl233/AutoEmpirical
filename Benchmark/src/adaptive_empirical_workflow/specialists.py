"""Bounded evidence retrieval without classification authority."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .contracts import (
    EvidenceDelta,
    EvidenceRequest,
    EvidenceView,
    RetrievalStatus,
    SpecialistType,
)
from .ledger import EvidenceLedger

Specialist = Callable[[EvidenceRequest, EvidenceView], EvidenceDelta]


class RecoverableSpecialistError(RuntimeError):
    """An expected operational retrieval failure safe to report as unavailable."""


@dataclass(frozen=True)
class MergedEvidenceRequest:
    """One canonical request and every analyst request it represents."""

    request: EvidenceRequest
    source_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryPassage:
    """A source passage ranked by query overlap without losing source order."""

    content: str
    passage_index: int
    score: int


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _query_terms(value: str) -> frozenset[str]:
    separated = _CAMEL_BOUNDARY.sub(" ", value)
    return frozenset(token.casefold() for token in _TOKEN.findall(separated))


def _prose_passages(content: str) -> tuple[str, ...]:
    return tuple(
        passage.strip()
        for passage in re.split(r"(?:\r?\n)[ \t]*(?:\r?\n)+", content.strip())
        if passage.strip()
    )


def _comment_passages(content: str) -> tuple[str, ...]:
    stripped = content.strip()
    if not stripped:
        return ()

    try:
        serialized = json.loads(stripped)
    except json.JSONDecodeError:
        serialized = None
    if isinstance(serialized, list) and all(
        isinstance(comment, str) for comment in serialized
    ):
        return tuple(
            paragraph
            for comment in serialized
            for paragraph in _prose_passages(comment)
        )

    if "=====" in stripped:
        return tuple(
            paragraph
            for comment in stripped.split("=====")
            for paragraph in _prose_passages(comment)
            if _query_terms(paragraph)
        )
    return _prose_passages(stripped)


def _diff_passages(content: str) -> tuple[str, ...]:
    lines = content.strip().splitlines()
    if not lines:
        return ()

    passages: list[str] = []
    file_header: list[str] = []
    hunk: list[str] = []

    def flush() -> None:
        if hunk:
            passages.append("\n".join((*file_header, *hunk)).strip())
        elif file_header:
            passages.append("\n".join(file_header).strip())

    for line in lines:
        if line.startswith("diff --git "):
            flush()
            file_header = [line]
            hunk = []
            continue
        if line.startswith("@@"):
            if hunk:
                passages.append("\n".join((*file_header, *hunk)).strip())
            hunk = [line]
            continue
        if hunk:
            hunk.append(line)
        else:
            file_header.append(line)
    flush()
    return tuple(passage for passage in passages if passage)


def extract_query_passages(
    content: str,
    *,
    query: str,
    max_items: int,
    source_kind: str = "prose",
) -> tuple[QueryPassage, ...]:
    """Return deterministic positive-overlap passages from one frozen source."""

    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    if source_kind not in {"prose", "comments", "diff", "test_diff", "lines"}:
        raise ValueError(f"unsupported source_kind {source_kind!r}")

    if source_kind == "prose":
        passages = _prose_passages(content)
    elif source_kind == "comments":
        passages = _comment_passages(content)
    elif source_kind == "lines":
        passages = tuple(line.strip() for line in content.splitlines() if line.strip())
    else:
        passages = _diff_passages(content)
    indexed_passages = tuple(enumerate(passages))
    if source_kind == "test_diff":
        indexed_passages = tuple(
            (index, passage)
            for index, passage in indexed_passages
            if "test" in _query_terms(passage)
        )

    query_terms = _query_terms(query)
    ranked = [
        QueryPassage(
            content=passage,
            passage_index=index,
            score=len(query_terms.intersection(_query_terms(passage))),
        )
        for index, passage in indexed_passages
    ]
    matching = [passage for passage in ranked if passage.score > 0]
    matching.sort(key=lambda passage: (-passage.score, passage.passage_index))
    return tuple(matching[:max_items])


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_query(value: str) -> str:
    return " ".join(sorted(_query_terms(value)))


def evidence_request_key(
    request: EvidenceRequest,
) -> tuple[SpecialistType, str, str, str, int]:
    """Return the stable semantic identity used across retrieval rounds."""

    return (
        request.target_specialist,
        _normalized(request.target_source),
        _normalized(request.missing_fact),
        _normalized_query(request.query),
        request.max_items,
    )


def merge_duplicate_requests(
    requests: Iterable[EvidenceRequest],
) -> tuple[MergedEvidenceRequest, ...]:
    """Merge semantically identical retrieval requests in stable input order."""

    merged: dict[
        tuple[SpecialistType, str, str, str, int],
        tuple[EvidenceRequest, list[str]],
    ] = {}
    for request in requests:
        key = evidence_request_key(request)
        if key not in merged:
            merged[key] = (request, [])
        merged[key][1].append(request.request_id)
    return tuple(
        MergedEvidenceRequest(
            request=request,
            source_request_ids=tuple(source_request_ids),
        )
        for request, source_request_ids in merged.values()
    )


class SpecialistRegistry:
    """Dispatches bounded requests to configured evidence specialists."""

    def __init__(
        self,
        specialists: Mapping[SpecialistType, Specialist] | None = None,
    ) -> None:
        self._specialists = dict(specialists or {})

    def run(self, request: EvidenceRequest, ledger: EvidenceLedger) -> EvidenceDelta:
        specialist = self._specialists.get(request.target_specialist)
        if specialist is None:
            return EvidenceDelta(
                request_id=request.request_id,
                specialist=request.target_specialist,
                status=RetrievalStatus.UNAVAILABLE,
                diagnostics={"reason": "specialist_not_registered"},
            )

        try:
            delta = specialist(request, ledger.view())
        except RecoverableSpecialistError as error:
            return EvidenceDelta(
                request_id=request.request_id,
                specialist=request.target_specialist,
                status=RetrievalStatus.UNAVAILABLE,
                diagnostics={
                    "reason": "specialist_runtime_failure",
                    "exception_type": type(error).__name__,
                },
            )
        if not isinstance(delta, EvidenceDelta):
            raise TypeError("specialist response must be an EvidenceDelta")
        if delta.request_id != request.request_id:
            raise ValueError("specialist response request_id does not match request")
        if delta.specialist is not request.target_specialist:
            raise ValueError("specialist response type does not match request")
        return delta


def apply_evidence_delta(
    ledger: EvidenceLedger, delta: EvidenceDelta
) -> tuple[str, ...]:
    """Append exact retrieved items to the immutable ledger."""

    appended: list[str] = []
    for item in delta.items:
        ledger.append(item)
        appended.append(item.evidence_id)
    return tuple(appended)
