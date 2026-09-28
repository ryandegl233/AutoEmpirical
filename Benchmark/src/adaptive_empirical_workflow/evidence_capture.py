"""Offline-only GitHub capture for frozen, replayable one-hop evidence graphs.

The runtime workflow never imports or constructs a network transport from this
module.  A caller must explicitly inject a transport into
``capture_frozen_evidence_bundle``; the production transport is owned by the
offline CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from pydantic import Field, field_validator, model_validator

from .contracts import FrozenStrictModel, _validate_json_value
from .frozen_evidence_graph import (
    CapturePolicy,
    EvidenceAuthority,
    FrozenBlobManifestEntry,
    FrozenEvidenceBundleManifest,
    FrozenGraphEdge,
    FrozenGraphNode,
    FrozenNodeType,
    FrozenRecordGraph,
    canonical_json_bytes,
    canonical_record_ids_sha256,
    canonical_string_set_sha256,
    frozen_bundle_merkle_root,
    frozen_graph_edge_id,
    frozen_graph_node_id,
    frozen_record_graph_hash,
    frozen_relation_proof_locator,
    _materialized_file_node_type,
    _maintainer_bare_references,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[a-z0-9_.-]+/[a-z0-9_.-]+$")
_TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_CACHE_INDEX_MAX_BYTES = 4096
_CACHE_ENVELOPE_OVERHEAD_BYTES = 4096
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "decision",
        "gold",
        "groundtruth",
        "gt",
        "label",
        "labels",
        "rootcause",
        "symptom",
        "targetlabel",
    }
)
_CAPABILITIES: Mapping[FrozenNodeType, tuple[str, ...]] = {
    FrozenNodeType.SEED_ISSUE: ("study_scope", "symptom_observation"),
    FrozenNodeType.MAINTAINER_RESOLUTION: ("defect_mechanism",),
    FrozenNodeType.MAINTAINER_POINTER: ("study_scope",),
    FrozenNodeType.LINKED_ISSUE: ("study_scope", "symptom_observation"),
    FrozenNodeType.LINKED_PULL_REQUEST: ("study_scope",),
    FrozenNodeType.LINKED_COMMIT: ("defect_mechanism",),
    FrozenNodeType.MERGE_COMMIT: ("defect_mechanism",),
    FrozenNodeType.CHANGED_CODE: ("defect_mechanism",),
    FrozenNodeType.REGRESSION_TEST: (
        "defect_mechanism",
        "symptom_observation",
    ),
    FrozenNodeType.REPAIR_SUMMARY: ("defect_mechanism",),
    FrozenNodeType.NEIGHBOR_CASE: (),
}


class EvidenceCaptureError(ValueError):
    """Base class for deterministic capture failures."""


class CaptureSecurityError(EvidenceCaptureError):
    """The request or remote response violated the fixed origin policy."""


class CaptureLimitError(EvidenceCaptureError):
    """A pre-registered resource bound would be exceeded."""


class CaptureTransportError(EvidenceCaptureError):
    """The injected transport returned an unusable response."""


def _normalized_key(value: object) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _reject_forbidden_fields(value: object, *, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(key) in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"capture request contains forbidden field {path}.{key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _without_forbidden_fields(value: object) -> object:
    """Drop remote answer-like metadata before it can enter the local cache."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_forbidden_fields(item)
            for key, item in value.items()
            if _normalized_key(key) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_without_forbidden_fields(item) for item in value]
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bounded_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as error:
        raise CaptureTransportError(f"cannot read {label}") from error
    if len(raw) > max_bytes:
        raise CaptureLimitError(f"{label} bytes exceed the configured bound")
    return raw


def _require_sha256(value: str, *, field: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _canonical_issue_url(value: str) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise CaptureSecurityError("seed issue URL is malformed") from error
    segments = tuple(part for part in parsed.path.split("/") if part)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(segments) != 4
        or segments[2] != "issues"
        or not segments[3].isdigit()
        or str(int(segments[3])) != segments[3]
        or segments[0] != segments[0].casefold()
        or segments[1] != segments[1].casefold()
        or segments[0] in {".", ".."}
        or segments[1] in {".", ".."}
        or parsed.path != "/" + "/".join(segments)
    ):
        raise CaptureSecurityError(
            "seed issue URL must be one canonical GitHub issue URL"
        )
    repository = f"{segments[0]}/{segments[1]}"
    if not _REPOSITORY.fullmatch(repository):
        raise CaptureSecurityError("seed issue repository is not canonical")
    return value, repository, int(segments[3])


def _canonical_entity_url(
    value: str,
) -> tuple[str, str, str, str]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise CaptureSecurityError("GitHub relation URL is malformed") from error
    segments = tuple(part for part in parsed.path.split("/") if part)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(segments) != 4
        or parsed.path != "/" + "/".join(segments)
        or segments[0] != segments[0].casefold()
        or segments[1] != segments[1].casefold()
        or segments[0] in {".", ".."}
        or segments[1] in {".", ".."}
    ):
        raise CaptureSecurityError(
            "relation target must be one canonical GitHub entity URL"
        )
    observed_repository = f"{segments[0]}/{segments[1]}"
    kind, identity = segments[2], segments[3]
    if kind in {"issues", "pull"}:
        if not identity.isdigit() or str(int(identity)) != identity:
            raise CaptureSecurityError(
                "issue and pull relation numbers must be canonical"
            )
    elif kind == "commit":
        if not _COMMIT_SHA.fullmatch(identity):
            raise CaptureSecurityError("commit relations require an immutable SHA")
    else:
        raise CaptureSecurityError("relation target type is outside the one-hop policy")
    return value, observed_repository, kind, identity


def _api_url(repository: str, suffix: str) -> str:
    repository_parts = repository.split("/")
    if (
        not _REPOSITORY.fullmatch(repository)
        or len(repository_parts) != 2
        or any(part in {".", ".."} for part in repository_parts)
        or (suffix and not suffix.startswith("/"))
    ):
        raise CaptureSecurityError("cannot construct a noncanonical GitHub API URL")
    return f"https://api.github.com/repos/{repository}{suffix}"


def _validate_api_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise CaptureSecurityError("GitHub API URL is malformed") from error
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or parsed.hostname != "api.github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
        or not parsed.path.startswith("/repos/")
    ):
        raise CaptureSecurityError("capture transport is restricted to api.github.com")
    if parsed.query:
        if not re.fullmatch(r"per_page=100&page=(?:[1-9][0-9]*)", parsed.query):
            raise CaptureSecurityError(
                "GitHub API query is outside the fixed pagination form"
            )


class GitHubResponse(FrozenStrictModel):
    status_code: int = Field(ge=100, le=599)
    final_url: str = Field(min_length=1)
    headers: Mapping[str, str]
    payload: Any
    raw_size_bytes: int = Field(ge=1)

    @field_validator("payload", mode="before")
    @classmethod
    def payload_is_json(cls, value: object) -> object:
        _validate_json_value(value, path="payload")
        return value

    @field_validator("headers", mode="before")
    @classmethod
    def headers_are_strings(cls, value: object) -> object:
        if not isinstance(value, Mapping) or any(
            type(key) is not str or type(item) is not str for key, item in value.items()
        ):
            raise ValueError("response headers must be a string mapping")
        return {str(key).casefold(): str(item) for key, item in value.items()}


class GitHubTransport(Protocol):
    def get_json(self, url: str, *, max_bytes: int) -> GitHubResponse:
        """Perform one redirect-disabled HTTPS GET and return parsed JSON."""


class CaptureLimits(FrozenStrictModel):
    max_pages_per_endpoint: int = Field(default=3, ge=1, le=100)
    max_nodes_per_record: int = Field(default=64, ge=1, le=1000)
    max_files_per_pull: int = Field(default=100, ge=1, le=3000)
    max_maintainer_references_per_record: int = Field(default=8, ge=0, le=64)
    max_response_bytes: int = Field(default=2_000_000, ge=1)
    max_total_response_bytes: int = Field(default=20_000_000, ge=1)


class CaptureSeed(FrozenStrictModel):
    record_id: str = Field(min_length=1)
    issue_url: str
    repository: str = ""
    issue_number: int = 0

    @model_validator(mode="after")
    def canonicalize_issue_identity(self) -> "CaptureSeed":
        canonical, repository, number = _canonical_issue_url(self.issue_url)
        if self.issue_url != canonical:
            raise CaptureSecurityError("seed issue URL is not canonical")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "issue_number", number)
        return self


class CaptureRequest(FrozenStrictModel):
    schema_version: str
    domain: str = Field(min_length=1)
    bundle_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    split_manifest_sha256: str
    runner_cohort_sha256: str
    reserved_entity_denylist_sha256: str
    capture_policy_id: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    retrieval_tool_name: str = Field(min_length=1)
    retrieval_tool_version: str = Field(min_length=1)
    retrieval_tool_module_sha256: str
    records: tuple[CaptureSeed, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def request_cannot_ingest_model_answers(cls, value: object) -> object:
        _reject_forbidden_fields(value)
        return value

    @model_validator(mode="after")
    def request_is_reproducible(self) -> "CaptureRequest":
        if self.schema_version != "ase-github-capture-request-v1":
            raise ValueError("unsupported capture request schema")
        for field in (
            "split_manifest_sha256",
            "runner_cohort_sha256",
            "reserved_entity_denylist_sha256",
            "retrieval_tool_module_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        try:
            from datetime import datetime

            timestamp = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("retrieved_at must be an RFC3339 timestamp") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        record_ids = tuple(seed.record_id for seed in self.records)
        issue_urls = tuple(seed.issue_url for seed in self.records)
        if (
            len(record_ids) != len(set(record_ids))
            or tuple(sorted(record_ids)) != record_ids
        ):
            raise ValueError("capture records must be uniquely sorted by record_id")
        if len(issue_urls) != len(set(issue_urls)):
            raise ValueError("capture seed issue URLs must be unique")
        return self


class FrozenCaptureResult(FrozenStrictModel):
    output_dir: str
    manifest_sha256: str
    record_count: int = Field(ge=1)
    graph_sha256s: Mapping[str, str]


class _FetchSession:
    def __init__(
        self,
        *,
        transport: GitHubTransport,
        cache_dir: Path,
        limits: CaptureLimits,
    ) -> None:
        self.transport = transport
        self.cache_dir = cache_dir
        self.limits = limits
        self.total_bytes = 0

    @staticmethod
    def _validate_next_url(url: str, candidate: str) -> None:
        _validate_api_url(candidate)
        parsed = urlsplit(url)
        page_match = re.fullmatch(r"per_page=100&page=([1-9][0-9]*)", parsed.query)
        if page_match is None:
            raise CaptureSecurityError("GitHub next link followed a non-page request")
        expected = (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            f"?per_page=100&page={int(page_match.group(1)) + 1}"
        )
        if candidate != expected:
            raise CaptureSecurityError(
                "GitHub next link is not the exact next canonical page"
            )

    @classmethod
    def _next_url(cls, url: str, headers: Mapping[str, str]) -> str | None:
        link = headers.get("link")
        if link is None:
            return None
        candidates = re.findall(r'<([^<>]+)>\s*;\s*rel="next"', link)
        if len(candidates) > 1:
            raise CaptureSecurityError("GitHub pagination declares multiple next links")
        if not candidates:
            return None
        candidate = candidates[0]
        cls._validate_next_url(url, candidate)
        return candidate

    def _get_envelope(self, url: str) -> tuple[object, str | None]:
        _validate_api_url(url)
        cache_key = _sha256(canonical_json_bytes({"url": url}))
        index_path = self.cache_dir / "index" / f"{cache_key}.json"
        if index_path.exists():
            try:
                index_raw = _read_bounded_file(
                    index_path,
                    max_bytes=_CACHE_INDEX_MAX_BYTES,
                    label="capture cache index",
                )
                index = json.loads(index_raw.decode("utf-8"))
                if index_raw != canonical_json_bytes(index):
                    raise CaptureTransportError(
                        "capture cache index is not canonical JSON"
                    )
                if (
                    not isinstance(index, dict)
                    or set(index) != {"blob_sha256", "url"}
                    or index.get("url") != url
                    or type(index.get("blob_sha256")) is not str
                    or not _SHA256.fullmatch(index["blob_sha256"])
                ):
                    raise CaptureSecurityError("capture cache URL binding mismatch")
                blob_path = self.cache_dir / "blobs" / f"{index['blob_sha256']}.json"
                remaining_total = (
                    self.limits.max_total_response_bytes - self.total_bytes
                )
                if remaining_total <= 0:
                    raise CaptureLimitError(
                        "total GitHub response bytes exceed the configured bound"
                    )
                envelope_raw = _read_bounded_file(
                    blob_path,
                    max_bytes=(
                        min(self.limits.max_response_bytes, remaining_total)
                        + _CACHE_ENVELOPE_OVERHEAD_BYTES
                    ),
                    label="capture cache blob",
                )
                if _sha256(envelope_raw) != index["blob_sha256"]:
                    raise CaptureTransportError("capture cache content hash mismatch")
                envelope = json.loads(envelope_raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureTransportError(
                    "capture cache contains invalid JSON"
                ) from error
            if envelope_raw != canonical_json_bytes(envelope):
                raise CaptureTransportError("capture cache is not canonical JSON")
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"next_url", "payload", "source_size_bytes", "url"}
                or envelope.get("url") != url
                or type(envelope.get("source_size_bytes")) is not int
                or envelope.get("source_size_bytes", 0) < 1
            ):
                raise CaptureSecurityError(
                    "capture cache URL binding or source size is invalid"
                )
            payload = envelope.get("payload")
            if _without_forbidden_fields(payload) != payload:
                raise CaptureSecurityError(
                    "capture cache contains a forbidden answer or label field"
                )
            canonical_payload_size = len(canonical_json_bytes(payload))
            if envelope["source_size_bytes"] < canonical_payload_size:
                raise CaptureSecurityError(
                    "capture cache source size is smaller than its canonical payload"
                )
            next_url = envelope.get("next_url")
            if next_url is not None:
                if type(next_url) is not str:
                    raise CaptureSecurityError("capture cache next URL is malformed")
                self._validate_next_url(url, next_url)
            source_size = int(envelope["source_size_bytes"])
        else:
            remaining_total = self.limits.max_total_response_bytes - self.total_bytes
            if remaining_total <= 0:
                raise CaptureLimitError(
                    "total GitHub response bytes exceed the configured bound"
                )
            response = self.transport.get_json(
                url,
                max_bytes=min(self.limits.max_response_bytes, remaining_total),
            )
            if not isinstance(response, GitHubResponse):
                response = GitHubResponse.model_validate(response)
            headers = {
                str(key).casefold(): value for key, value in response.headers.items()
            }
            if (
                300 <= response.status_code < 400
                or "location" in headers
                or response.final_url != url
            ):
                raise CaptureSecurityError("GitHub redirect responses are forbidden")
            _validate_api_url(response.final_url)
            if response.status_code != 200:
                raise CaptureTransportError(
                    f"GitHub API returned HTTP {response.status_code}"
                )
            source_size = response.raw_size_bytes
            payload = _without_forbidden_fields(response.payload)
            next_url = self._next_url(url, headers)
            envelope_raw = canonical_json_bytes(
                {
                    "next_url": next_url,
                    "payload": payload,
                    "source_size_bytes": source_size,
                    "url": url,
                }
            )
            envelope_sha = _sha256(envelope_raw)
            blob_path = self.cache_dir / "blobs" / f"{envelope_sha}.json"
            if not blob_path.exists():
                _atomic_write(blob_path, envelope_raw)
            _atomic_write(
                index_path,
                canonical_json_bytes({"blob_sha256": envelope_sha, "url": url}),
            )
        if source_size > self.limits.max_response_bytes:
            raise CaptureLimitError("GitHub response bytes exceed the configured bound")
        self.total_bytes += source_size
        if self.total_bytes > self.limits.max_total_response_bytes:
            raise CaptureLimitError(
                "total GitHub response bytes exceed the configured bound"
            )
        return payload, next_url

    def get(self, url: str) -> object:
        payload, _ = self._get_envelope(url)
        return payload

    def get_pages(self, base_url: str) -> list[object]:
        items: list[object] = []
        url = f"{base_url}?per_page=100&page=1"
        for _ in range(self.limits.max_pages_per_endpoint):
            payload, next_url = self._get_envelope(url)
            if not isinstance(payload, list):
                raise CaptureTransportError(
                    "paginated GitHub endpoint returned a non-list"
                )
            items.extend(payload)
            if next_url is None:
                return items
            url = next_url
        raise CaptureLimitError("GitHub page bound reached before endpoint exhaustion")


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if type(value) is not str or not value:
        raise CaptureTransportError(f"GitHub response requires non-empty {field}")
    return value


def _optional_timestamp(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not str or not value:
        raise CaptureTransportError(f"GitHub response has invalid {field}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaptureTransportError(f"GitHub {label} response must be an object")
    return value


def _canonical_node_raw(
    *,
    canonical_uri: str,
    repository: str,
    updated_at: str | None,
    fields: Mapping[str, object],
) -> dict[str, object]:
    return {
        "canonical_uri": canonical_uri,
        "repository": repository,
        "updated_at": updated_at,
        **fields,
    }


def _make_node(
    *,
    record_id: str,
    node_type: FrozenNodeType,
    canonical_uri: str,
    repository: str,
    relation_depth: int,
    retrieved_at: str,
    remote_updated_at: str | None,
    raw: Mapping[str, object],
    content: str,
    authority: EvidenceAuthority,
    intrinsic_parent_node_id: str | None = None,
    immutable_ref: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[FrozenGraphNode, bytes]:
    raw_bytes = canonical_json_bytes(raw)
    content_sha = _sha256(content.encode("utf-8"))
    node_id = frozen_graph_node_id(
        record_id=record_id,
        node_type=node_type,
        canonical_uri=canonical_uri,
        immutable_ref=immutable_ref,
        content_sha256=content_sha,
    )
    return (
        FrozenGraphNode(
            node_id=node_id,
            record_id=record_id,
            node_type=node_type,
            canonical_uri=canonical_uri,
            repository=repository,
            relation_depth=relation_depth,
            intrinsic_parent_node_id=intrinsic_parent_node_id,
            retrieved_at=retrieved_at,
            remote_updated_at=remote_updated_at,
            immutable_ref=immutable_ref,
            raw_blob_sha256=_sha256(raw_bytes),
            content_sha256=content_sha,
            content=content,
            evidence_capabilities=_CAPABILITIES[node_type],
            authority=authority,
            metadata={} if metadata is None else metadata,
        ),
        raw_bytes,
    )


def _make_edge(
    *,
    record_id: str,
    source: FrozenGraphNode,
    target: FrozenGraphNode,
    relation: str,
    proof: FrozenGraphNode,
    json_pointer: str,
    reference: str,
) -> FrozenGraphEdge:
    locator = frozen_relation_proof_locator(
        record_id=record_id,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        json_pointer=json_pointer,
        reference=reference,
    )
    edge_id = frozen_graph_edge_id(
        record_id=record_id,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        relation_proof_locator=locator,
        relation_proof_sha256=proof.content_sha256,
    )
    return FrozenGraphEdge(
        edge_id=edge_id,
        record_id=record_id,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        relation_proof_locator=locator,
        relation_proof_sha256=proof.content_sha256,
    )


def _timeline_relations(
    timeline: Sequence[object], *, repository: str
) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    for event in timeline:
        if not isinstance(event, Mapping):
            raise CaptureTransportError("GitHub timeline events must be objects")
        event_name = event.get("event")
        if event_name == "cross-referenced":
            source = event.get("source")
            issue = source.get("issue") if isinstance(source, Mapping) else None
            target = issue.get("html_url") if isinstance(issue, Mapping) else None
            if type(target) is not str:
                continue
            canonical, observed_repository, kind, identity = _canonical_entity_url(
                target
            )
            if observed_repository != repository:
                continue
            relations.append((canonical, kind, identity))
        elif event_name in {"committed", "referenced"}:
            sha = event.get("commit_id")
            if type(sha) is str and _COMMIT_SHA.fullmatch(sha):
                canonical = f"https://github.com/{repository}/commit/{sha}"
                relations.append((canonical, "commit", sha))
    unique: dict[str, tuple[str, str, str]] = {}
    for relation in relations:
        unique[relation[0]] = relation
    return [unique[key] for key in sorted(unique)]


def _capture_record(
    request: CaptureRequest,
    seed: CaptureSeed,
    session: _FetchSession,
    limits: CaptureLimits,
) -> tuple[FrozenRecordGraph, dict[str, bytes]]:
    repository = seed.repository
    base = _api_url(repository, "")
    issue_api = f"{base}/issues/{seed.issue_number}"
    issue_payload = _mapping(session.get(issue_api), label="seed issue")
    if issue_payload.get("number") != seed.issue_number:
        raise CaptureSecurityError("seed issue response number mismatch")
    if issue_payload.get("html_url") != seed.issue_url:
        raise CaptureSecurityError("seed issue response canonical URL mismatch")
    issue_body = _required_string(issue_payload, "body")
    issue_updated = _optional_timestamp(issue_payload, "updated_at")
    timeline = session.get_pages(f"{issue_api}/timeline")
    comments = session.get_pages(f"{issue_api}/comments")
    timeline_relations = _timeline_relations(timeline, repository=repository)
    trusted_comments: list[Mapping[str, object]] = []
    for comment in comments:
        item = _mapping(comment, label="issue comment")
        if (
            item.get("author_association") in _TRUSTED_ASSOCIATIONS
            and type(item.get("body")) is str
            and str(item["body"]).strip()
        ):
            trusted_comments.append(item)
    chosen: Mapping[str, object] | None = None
    pointer_relations: list[tuple[str, str, str, str]] = []
    preloaded_issues: dict[str, Mapping[str, object]] = {}
    if trusted_comments:
        chosen = sorted(
            trusted_comments,
            key=lambda item: (
                str(item.get("updated_at") or item.get("created_at") or ""),
                int(item.get("id")) if isinstance(item.get("id"), int) else -1,
            ),
        )[-1]
        resolution_body = str(chosen["body"])
        references = _maintainer_bare_references(resolution_body)
        if len(references) > limits.max_maintainer_references_per_record:
            raise CaptureLimitError(
                "maintainer reference count exceeds the configured bound"
            )
        unique_references: dict[int, str] = {}
        for number, source_token in references:
            unique_references.setdefault(number, source_token)
        for number, source_token in unique_references.items():
            linked_api = f"{base}/issues/{number}"
            payload = _mapping(session.get(linked_api), label="linked issue")
            if payload.get("number") != number:
                raise CaptureSecurityError("linked issue response number mismatch")
            is_pull = "pull_request" in payload
            if is_pull:
                marker = payload.get("pull_request")
                if (
                    not isinstance(marker, Mapping)
                    or marker.get("url") != f"{base}/pulls/{number}"
                ):
                    raise CaptureSecurityError(
                        "pull request marker does not match the same repository path"
                    )
            kind = "pull" if is_pull else "issues"
            target_uri = f"https://github.com/{repository}/{kind}/{number}"
            if payload.get("html_url") != target_uri:
                raise CaptureSecurityError("linked issue response identity mismatch")
            pointer_relations.append((target_uri, kind, str(number), source_token))
            preloaded_issues[target_uri] = payload

    seed_target_uris = {target for target, _, _ in timeline_relations}
    unique_pointer_relations: dict[str, tuple[str, str, str, str]] = {}
    for relation in pointer_relations:
        if relation[0] not in seed_target_uris:
            unique_pointer_relations.setdefault(relation[0], relation)
    pointer_relations = [
        unique_pointer_relations[key] for key in sorted(unique_pointer_relations)
    ]
    relations: list[tuple[str, str, str, str, int]] = [
        (target, kind, identity, "seed", index)
        for index, (target, kind, identity) in enumerate(timeline_relations)
    ]
    relations.extend(
        (target, kind, identity, "pointer", index)
        for index, (target, kind, identity, _) in enumerate(pointer_relations)
    )
    relation_events = [
        {"relation": "direct_reference", "target_uri": target}
        for target, _, _ in timeline_relations
    ]
    seed_raw = _canonical_node_raw(
        canonical_uri=seed.issue_url,
        repository=repository,
        updated_at=issue_updated,
        fields={"body": issue_body, "relation_events": relation_events},
    )
    seed_node, seed_blob = _make_node(
        record_id=seed.record_id,
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri=seed.issue_url,
        repository=repository,
        relation_depth=0,
        retrieved_at=request.retrieved_at,
        remote_updated_at=issue_updated,
        raw=seed_raw,
        content=issue_body,
        authority=EvidenceAuthority.CONTEXT_ONLY,
    )
    nodes: list[FrozenGraphNode] = [seed_node]
    edges: list[FrozenGraphEdge] = []
    blobs: dict[str, bytes] = {seed_node.raw_blob_sha256: seed_blob}

    pointer_node: FrozenGraphNode | None = None
    if chosen is not None:
        resolution_body = str(chosen["body"])
        resolution_updated = _optional_timestamp(chosen, "updated_at")
        association = str(chosen["author_association"])
        resolution_raw = _canonical_node_raw(
            canonical_uri=seed.issue_url,
            repository=repository,
            updated_at=resolution_updated,
            fields={
                "author_association": association,
                "body": resolution_body,
                "relation_events": [
                    {
                        "relation": "resolution_reference",
                        "source_token": source_token,
                        "target_number": int(identity),
                        "target_repository": repository,
                        "target_uri": target_uri,
                    }
                    for target_uri, _, identity, source_token in pointer_relations
                ],
            },
        )
        resolution, raw_blob = _make_node(
            record_id=seed.record_id,
            node_type=FrozenNodeType.MAINTAINER_POINTER,
            canonical_uri=seed.issue_url,
            repository=repository,
            relation_depth=0,
            retrieved_at=request.retrieved_at,
            remote_updated_at=resolution_updated,
            raw=resolution_raw,
            content=resolution_body,
            authority=EvidenceAuthority.CONTEXT_ONLY,
            metadata={"author_association": association},
        )
        nodes.append(resolution)
        pointer_node = resolution
        blobs[resolution.raw_blob_sha256] = raw_blob
        edges.append(
            _make_edge(
                record_id=seed.record_id,
                source=seed_node,
                target=resolution,
                relation="seed_resolution",
                proof=resolution,
                json_pointer="/body",
                reference=resolution.content,
            )
        )

    for target_uri, kind, identity, relation_origin, relation_index in relations:
        edge_source = seed_node if relation_origin == "seed" else pointer_node
        if edge_source is None:
            raise EvidenceCaptureError("maintainer relation requires its pointer node")
        edge_relation = (
            "direct_reference" if relation_origin == "seed" else "resolution_reference"
        )
        if kind == "issues":
            linked_api = f"{base}/issues/{identity}"
            payload = preloaded_issues.get(target_uri) or _mapping(
                session.get(linked_api), label="linked issue"
            )
            if payload.get("html_url") != target_uri or payload.get("number") != int(
                identity
            ):
                raise CaptureSecurityError("linked issue response identity mismatch")
            content = _required_string(payload, "body")
            updated = _optional_timestamp(payload, "updated_at")
            raw = _canonical_node_raw(
                canonical_uri=target_uri,
                repository=repository,
                updated_at=updated,
                fields={"body": content, "number": int(identity)},
            )
            node, raw_blob = _make_node(
                record_id=seed.record_id,
                node_type=FrozenNodeType.LINKED_ISSUE,
                canonical_uri=target_uri,
                repository=repository,
                relation_depth=1,
                retrieved_at=request.retrieved_at,
                remote_updated_at=updated,
                raw=raw,
                content=content,
                authority=EvidenceAuthority.CONTEXT_ONLY,
            )
            nodes.append(node)
            blobs[node.raw_blob_sha256] = raw_blob
            edges.append(
                _make_edge(
                    record_id=seed.record_id,
                    source=edge_source,
                    target=node,
                    relation=edge_relation,
                    proof=edge_source,
                    json_pointer=f"/relation_events/{relation_index}/target_uri",
                    reference=target_uri,
                )
            )
        elif kind == "pull":
            pull_api = f"{base}/pulls/{identity}"
            issue_pull_request_url: str | None = None
            if relation_origin == "pointer":
                issue_payload = preloaded_issues.get(target_uri)
                marker = (
                    issue_payload.get("pull_request")
                    if issue_payload is not None
                    else None
                )
                if not isinstance(marker, Mapping) or marker.get("url") != pull_api:
                    raise CaptureSecurityError(
                        "pointer pull lost its exact issue endpoint marker"
                    )
                issue_pull_request_url = pull_api
            payload = _mapping(session.get(pull_api), label="linked pull request")
            if payload.get("html_url") != target_uri or payload.get("number") != int(
                identity
            ):
                raise CaptureSecurityError("linked pull response identity mismatch")
            title = _required_string(payload, "title")
            updated = _optional_timestamp(payload, "updated_at")
            merged = payload.get("merged")
            if type(merged) is not bool:
                raise CaptureTransportError("pull response requires merged boolean")
            merge_sha = payload.get("merge_commit_sha")
            materialized_files: list[tuple[str, str, FrozenNodeType]] = []
            summary: str | None = None
            if merged:
                if type(merge_sha) is not str or not _COMMIT_SHA.fullmatch(merge_sha):
                    raise CaptureSecurityError(
                        "merged pull requires an immutable merge SHA"
                    )
                files = session.get_pages(f"{pull_api}/files")
                if len(files) > limits.max_files_per_pull:
                    raise CaptureLimitError(
                        "pull file count exceeds the configured bound"
                    )
                for file_payload in files:
                    file_item = _mapping(file_payload, label="pull file")
                    filename = _required_string(file_item, "filename")
                    filename_parts = filename.split("/")
                    if "\\" in filename or any(
                        part in {"", ".", ".."} for part in filename_parts
                    ):
                        raise CaptureSecurityError(
                            "pull file path must be one canonical relative path"
                        )
                    patch = file_item.get("patch")
                    if type(patch) is not str or not patch:
                        continue
                    encoded_path = "/".join(
                        quote(part, safe="-._~") for part in filename_parts
                    )
                    file_uri = f"https://github.com/{repository}/blob/{merge_sha}/{encoded_path}"
                    node_type = _materialized_file_node_type(filename)
                    materialized_files.append((file_uri, patch, node_type))
                materialized_files.sort(key=lambda item: item[0])
                materialized_uris = tuple(item[0] for item in materialized_files)
                if len(materialized_uris) != len(set(materialized_uris)):
                    raise CaptureSecurityError(
                        "pull materialized files must be unique canonical URIs"
                    )
                commit_api = f"{base}/commits/{merge_sha}"
                commit_payload = _mapping(session.get(commit_api), label="merge commit")
                if (
                    commit_payload.get("sha") != merge_sha
                    or commit_payload.get("html_url")
                    != f"https://github.com/{repository}/commit/{merge_sha}"
                ):
                    raise CaptureSecurityError(
                        "merge commit response identity mismatch"
                    )
                commit = _mapping(
                    commit_payload.get("commit"), label="merge commit detail"
                )
                summary = _required_string(commit, "message")
            elif merge_sha is not None and type(merge_sha) is not str:
                raise CaptureTransportError("pull merge_commit_sha has invalid type")
            pull_raw = _canonical_node_raw(
                canonical_uri=target_uri,
                repository=repository,
                updated_at=updated,
                fields={
                    "files": [
                        {"canonical_uri": file_uri}
                        for file_uri, _, _ in materialized_files
                    ],
                    "issue_pull_request_url": issue_pull_request_url,
                    "merge_commit_sha": merge_sha,
                    "merged": merged,
                    "number": int(identity),
                    "title": title,
                },
            )
            pull_node, pull_blob = _make_node(
                record_id=seed.record_id,
                node_type=FrozenNodeType.LINKED_PULL_REQUEST,
                canonical_uri=target_uri,
                repository=repository,
                relation_depth=1,
                retrieved_at=request.retrieved_at,
                remote_updated_at=updated,
                raw=pull_raw,
                content=title,
                authority=EvidenceAuthority.CONTEXT_ONLY,
                metadata={"merged": merged, "merge_commit_sha": merge_sha},
            )
            nodes.append(pull_node)
            blobs[pull_node.raw_blob_sha256] = pull_blob
            edges.append(
                _make_edge(
                    record_id=seed.record_id,
                    source=edge_source,
                    target=pull_node,
                    relation=edge_relation,
                    proof=edge_source,
                    json_pointer=f"/relation_events/{relation_index}/target_uri",
                    reference=target_uri,
                )
            )
            if merged and isinstance(merge_sha, str) and summary is not None:
                for file_index, (file_uri, patch, node_type) in enumerate(
                    materialized_files
                ):
                    file_raw = _canonical_node_raw(
                        canonical_uri=file_uri,
                        repository=repository,
                        updated_at=updated,
                        fields={"patch": patch},
                    )
                    file_node, file_blob = _make_node(
                        record_id=seed.record_id,
                        node_type=node_type,
                        canonical_uri=file_uri,
                        repository=repository,
                        relation_depth=1,
                        retrieved_at=request.retrieved_at,
                        remote_updated_at=updated,
                        raw=file_raw,
                        content=patch,
                        authority=EvidenceAuthority.DIRECT,
                        intrinsic_parent_node_id=pull_node.node_id,
                        immutable_ref=merge_sha,
                    )
                    nodes.append(file_node)
                    blobs[file_node.raw_blob_sha256] = file_blob
                    edges.append(
                        _make_edge(
                            record_id=seed.record_id,
                            source=pull_node,
                            target=file_node,
                            relation="intrinsic_materialization",
                            proof=pull_node,
                            json_pointer=f"/files/{file_index}/canonical_uri",
                            reference=file_uri,
                        )
                    )
                commit_uri = f"https://github.com/{repository}/commit/{merge_sha}"
                summary_raw = _canonical_node_raw(
                    canonical_uri=commit_uri,
                    repository=repository,
                    updated_at=updated,
                    fields={"summary": summary},
                )
                summary_node, summary_blob = _make_node(
                    record_id=seed.record_id,
                    node_type=FrozenNodeType.REPAIR_SUMMARY,
                    canonical_uri=commit_uri,
                    repository=repository,
                    relation_depth=1,
                    retrieved_at=request.retrieved_at,
                    remote_updated_at=updated,
                    raw=summary_raw,
                    content=summary,
                    authority=EvidenceAuthority.DIRECT,
                    intrinsic_parent_node_id=pull_node.node_id,
                    immutable_ref=merge_sha,
                )
                nodes.append(summary_node)
                blobs[summary_node.raw_blob_sha256] = summary_blob
                edges.append(
                    _make_edge(
                        record_id=seed.record_id,
                        source=pull_node,
                        target=summary_node,
                        relation="intrinsic_materialization",
                        proof=pull_node,
                        json_pointer="/merge_commit_sha",
                        reference=merge_sha,
                    )
                )
        else:
            commit_api = f"{base}/commits/{identity}"
            payload = _mapping(session.get(commit_api), label="linked commit")
            if payload.get("sha") != identity or payload.get("html_url") != target_uri:
                raise CaptureSecurityError("linked commit response identity mismatch")
            commit = _mapping(payload.get("commit"), label="linked commit detail")
            content = _required_string(commit, "message")
            committer = commit.get("committer")
            updated = (
                _optional_timestamp(committer, "date")
                if isinstance(committer, Mapping)
                else None
            )
            raw = _canonical_node_raw(
                canonical_uri=target_uri,
                repository=repository,
                updated_at=updated,
                fields={"commit": {"message": content}, "merge_commit_sha": identity},
            )
            node, raw_blob = _make_node(
                record_id=seed.record_id,
                node_type=FrozenNodeType.LINKED_COMMIT,
                canonical_uri=target_uri,
                repository=repository,
                relation_depth=1,
                retrieved_at=request.retrieved_at,
                remote_updated_at=updated,
                raw=raw,
                content=content,
                authority=EvidenceAuthority.DIRECT,
                immutable_ref=identity,
            )
            nodes.append(node)
            blobs[node.raw_blob_sha256] = raw_blob
            edges.append(
                _make_edge(
                    record_id=seed.record_id,
                    source=edge_source,
                    target=node,
                    relation=edge_relation,
                    proof=edge_source,
                    json_pointer=f"/relation_events/{relation_index}/target_uri",
                    reference=target_uri,
                )
            )

    if len(nodes) > limits.max_nodes_per_record:
        raise CaptureLimitError("record node count exceeds the configured bound")
    ordered_nodes = tuple(
        sorted(
            nodes,
            key=lambda node: (
                0 if node.node_id == seed_node.node_id else 1,
                node.relation_depth,
                node.node_type.value,
                node.canonical_uri,
                node.node_id,
            ),
        )
    )
    ordered_edges = tuple(sorted(edges, key=lambda edge: edge.edge_id))
    graph_payload: dict[str, object] = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": request.domain,
        "record_id": seed.record_id,
        "repository": repository,
        "seed_node_id": seed_node.node_id,
        "nodes": [node.model_dump(mode="json") for node in ordered_nodes],
        "edges": [edge.model_dump(mode="json") for edge in ordered_edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    return FrozenRecordGraph.model_validate(graph_payload), blobs


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    with temp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_write_no_clobber(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    with temp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    except FileExistsError as error:
        raise FileExistsError(
            "an active manifest appeared during capture and was not overwritten"
        ) from error
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                # os.link above is the commit point.  A stale temporary hardlink
                # is safe and must not reclassify a valid active manifest as a
                # failed capture.
                pass


@contextmanager
def _exclusive_output_lock(output: Path) -> Any:
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".capture.lock"
    token = secrets.token_hex(32).encode("ascii")
    try:
        with lock_path.open("xb") as stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            "an exclusive capture lock already protects this output bundle"
        ) from error

    def assert_owned() -> None:
        try:
            observed = lock_path.read_bytes()
        except OSError as error:
            raise CaptureSecurityError("capture output lock disappeared") from error
        if observed != token:
            raise CaptureSecurityError("capture output lock ownership changed")

    try:
        yield assert_owned
    finally:
        try:
            if lock_path.read_bytes() == token:
                lock_path.unlink()
        except OSError:
            pass


def _capture_frozen_evidence_bundle_locked(
    request: CaptureRequest,
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    transport: GitHubTransport,
    limits: CaptureLimits,
    assert_output_lock: Any,
) -> FrozenCaptureResult:
    """Capture and publish a validated bundle, writing its active manifest last."""

    request = CaptureRequest.model_validate(request)
    limits = CaptureLimits.model_validate(limits)
    if request.capture_policy_id != "github-one-hop-maintainer-pointer-v2":
        raise ValueError("new capture requires github-one-hop-maintainer-pointer-v2")
    output = Path(output_dir).resolve()
    cache = Path(cache_dir).resolve()
    manifest_path = output / "frozen_evidence_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "an active manifest already exists and will not be overwritten"
        )
    session = _FetchSession(transport=transport, cache_dir=cache, limits=limits)
    graphs: list[FrozenRecordGraph] = []
    blobs: dict[str, bytes] = {}
    for seed in request.records:
        graph, graph_blobs = _capture_record(request, seed, session, limits)
        graphs.append(graph)
        for digest, raw in graph_blobs.items():
            previous = blobs.get(digest)
            if previous is not None and previous != raw:
                raise EvidenceCaptureError("content-addressed blob digest collision")
            blobs[digest] = raw

    repositories = tuple(sorted({seed.repository for seed in request.records}))
    policy = CapturePolicy(
        schema_version="ase-frozen-capture-policy-v2",
        policy_id=request.capture_policy_id,
        domain=request.domain,
        allowed_repositories=repositories,
        endpoint_allowlist=("api.github.com",),
        max_relation_depth=1,
        allow_redirects=False,
        runtime_network_forbidden=True,
        intrinsic_node_types=tuple(
            sorted(
                (
                    FrozenNodeType.MERGE_COMMIT,
                    FrozenNodeType.CHANGED_CODE,
                    FrozenNodeType.REGRESSION_TEST,
                    FrozenNodeType.REPAIR_SUMMARY,
                ),
                key=lambda item: item.value,
            )
        ),
        max_maintainer_references_per_record=(
            limits.max_maintainer_references_per_record
        ),
    )
    policy_raw = canonical_json_bytes(policy.model_dump(mode="json"))
    graphs_raw = b"".join(
        canonical_json_bytes(graph.model_dump(mode="json")) + b"\n"
        for graph in sorted(graphs, key=lambda item: item.record_id)
    )
    blob_entries = tuple(
        FrozenBlobManifestEntry(
            relative_path=f"blobs/{digest}.blob",
            sha256=digest,
            size_bytes=len(blobs[digest]),
            media_type="application/json",
        )
        for digest in sorted(blobs)
    )
    record_ids = tuple(sorted(graph.record_id for graph in graphs))
    manifest_payload: dict[str, object] = {
        "schema_version": "ase-frozen-evidence-bundle-v1",
        "status": "active",
        "domain": request.domain,
        "bundle_id": request.bundle_id,
        "split_id": request.split_id,
        "split_manifest_sha256": request.split_manifest_sha256,
        "runner_cohort_sha256": request.runner_cohort_sha256,
        "record_ids_sha256": canonical_record_ids_sha256(record_ids),
        "record_count": len(record_ids),
        "capture_policy": {
            "relative_path": "capture_policy.json",
            "sha256": _sha256(policy_raw),
        },
        "record_graphs": {
            "relative_path": "record_graphs.jsonl",
            "sha256": _sha256(graphs_raw),
        },
        "repo_allowlist_sha256": canonical_string_set_sha256(repositories),
        "reserved_entity_denylist_sha256": request.reserved_entity_denylist_sha256,
        "capture_started_at": request.retrieved_at,
        "capture_finished_at": request.retrieved_at,
        "retrieval_tool": {
            "name": request.retrieval_tool_name,
            "version": request.retrieval_tool_version,
            "module_sha256": request.retrieval_tool_module_sha256,
        },
        "blobs": [entry.model_dump(mode="json") for entry in blob_entries],
        "bundle_merkle_root": frozen_bundle_merkle_root(
            capture_policy_sha256=_sha256(policy_raw),
            record_graphs_sha256=_sha256(graphs_raw),
            blobs=blob_entries,
        ),
        "network_policy": "offline_capture_only_runtime_network_forbidden",
        "gold_accessed": False,
    }
    manifest = FrozenEvidenceBundleManifest.model_validate(manifest_payload)
    manifest_raw = canonical_json_bytes(manifest.model_dump(mode="json"))
    result = FrozenCaptureResult(
        output_dir=str(output),
        manifest_sha256=_sha256(manifest_raw),
        record_count=len(graphs),
        graph_sha256s={graph.record_id: graph.graph_sha256 for graph in graphs},
    )
    audit_raw = canonical_json_bytes(
        {
            "schema_version": "ase-frozen-capture-audit-v1",
            "status": "captured",
            "bundle_id": request.bundle_id,
            "record_count": len(graphs),
            "graph_sha256s": {
                graph.record_id: graph.graph_sha256
                for graph in sorted(graphs, key=lambda item: item.record_id)
            },
            "runtime_network_forbidden": True,
        }
    )

    # All capture, normalization, and validation occurs before publication.
    # Content-addressed blobs and bound artifacts are written atomically; the
    # active manifest is the final commit marker.
    assert_output_lock()
    if manifest_path.exists():
        raise FileExistsError(
            "an active manifest appeared during capture and was not overwritten"
        )
    for entry in blob_entries:
        _atomic_write(output / entry.relative_path, blobs[entry.sha256])
    _atomic_write(output / "capture_policy.json", policy_raw)
    _atomic_write(output / "record_graphs.jsonl", graphs_raw)
    _atomic_write(output / "capture_audit.json", audit_raw)
    assert_output_lock()
    _atomic_write_no_clobber(manifest_path, manifest_raw)
    return result


def capture_frozen_evidence_bundle(
    request: CaptureRequest,
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    transport: GitHubTransport,
    limits: CaptureLimits,
) -> FrozenCaptureResult:
    """Capture and publish a validated bundle under an exclusive output lock."""

    output = Path(output_dir).resolve()
    with _exclusive_output_lock(output) as assert_owned:
        return _capture_frozen_evidence_bundle_locked(
            request,
            output_dir=output,
            cache_dir=cache_dir,
            transport=transport,
            limits=limits,
            assert_output_lock=assert_owned,
        )


__all__ = [
    "CaptureLimitError",
    "CaptureLimits",
    "CaptureRequest",
    "CaptureSecurityError",
    "CaptureTransportError",
    "EvidenceCaptureError",
    "FrozenCaptureResult",
    "GitHubResponse",
    "GitHubTransport",
    "capture_frozen_evidence_bundle",
]
