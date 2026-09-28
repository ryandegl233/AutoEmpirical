"""Content-addressed contracts for pre-registered frozen evidence graphs.

This module deliberately contains no network transport.  Production callers can
only load a manifest whose repository-relative path and byte digest are bound in
the module-owned registry below.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_serializer, field_validator, model_validator

from .contracts import (
    FrozenStrictModel,
    _deep_freeze,
    _deep_thaw,
    _validate_json_value,
)


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[a-z0-9_.-]+/[a-z0-9_.-]+$")
_ALLOWED_CAPABILITIES = frozenset(
    {"symptom_observation", "defect_mechanism", "study_scope"}
)
_INTRINSIC_NODE_TYPES = frozenset(
    {
        "merge_commit",
        "changed_code",
        "regression_test",
        "repair_summary",
    }
)
_NODE_FIXED_CAPABILITIES = {
    "seed_issue": ("study_scope", "symptom_observation"),
    "maintainer_resolution": ("defect_mechanism",),
    "maintainer_pointer": ("study_scope",),
    "linked_issue": ("study_scope", "symptom_observation"),
    "linked_pull_request": ("study_scope",),
    "linked_commit": ("defect_mechanism",),
    "merge_commit": ("defect_mechanism",),
    "changed_code": ("defect_mechanism",),
    "regression_test": ("defect_mechanism", "symptom_observation"),
    "repair_summary": ("defect_mechanism",),
    "neighbor_case": (),
}
_RAW_CONTENT_POINTERS = {
    "seed_issue": "/body",
    "maintainer_resolution": "/body",
    "maintainer_pointer": "/body",
    "linked_issue": "/body",
    "linked_pull_request": "/title",
    "linked_commit": "/commit/message",
    "merge_commit": "/commit/message",
    "changed_code": "/patch",
    "regression_test": "/patch",
    "repair_summary": "/summary",
    "neighbor_case": "/body",
}
_FORBIDDEN_METADATA_KEYS = frozenset(
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

# The production loader accepts only these module-owned, byte-exact roots.
# Callers cannot supply a path or digest.  The ASE bundle contains only frozen
# GitHub evidence for the targeted error cluster and its harm sentinel; it does
# not contain labels or expected answers.
_REGISTERED_TRUST_ROOTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "ase2022": (
            "Benchmark/configs/frozen_evidence/"
            "ase2022_dev50_r1_r3_harm_v2/bundle/"
            "frozen_evidence_manifest.json",
            "8eb9a502fc9257fd6dd7e5be0c0602e541a4fc2e176bf050a7c332ab9d4b7a20",
        )
    }
)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON-compatible value using the workflow hash profile."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_string_set_sha256(values: Sequence[str]) -> str:
    normalized = tuple(values)
    if any(type(value) is not str or not value for value in normalized):
        raise ValueError("canonical string set values must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("canonical string set values must be unique")
    return _canonical_sha256(sorted(normalized))


def canonical_record_ids_sha256(record_ids: Sequence[str]) -> str:
    return canonical_string_set_sha256(record_ids)


def _require_sha256(value: str, *, field: str) -> None:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _normalized_metadata_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _reject_forbidden_metadata(value: object, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_metadata_key(str(key)) in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(
                    f"frozen evidence contains forbidden field {path}.{key}"
                )
            _reject_forbidden_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_metadata(item, path=f"{path}[{index}]")


def _repository_from_uri(value: str) -> tuple[str, tuple[str, ...]]:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("canonical_uri must be a canonical HTTPS GitHub URI")
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    if len(segments) < 4:
        raise ValueError("canonical_uri must identify a GitHub entity")
    if (
        parsed.path != "/" + "/".join(segments)
        or segments[0] != segments[0].casefold()
        or segments[1] != segments[1].casefold()
    ):
        raise ValueError("canonical_uri must use one canonical URI spelling")
    repository = f"{segments[0].casefold()}/{segments[1].casefold()}"
    return repository, segments[2:]


def _maintainer_bare_references(body: str) -> tuple[tuple[int, str], ...]:
    visible = list(body)
    folded_body = body.casefold()

    def mask(start: int, end: int) -> None:
        visible[start:end] = " " * (end - start)

    def line_end(start: int) -> int:
        end = body.find("\n", start)
        return len(body) if end < 0 else end

    def html_tag(start: int) -> tuple[int, str, bool] | None:
        cursor = start + 1
        closing = cursor < len(body) and body[cursor] == "/"
        if closing:
            cursor += 1
        name_start = cursor
        while cursor < len(body) and (
            body[cursor].isalnum() or body[cursor] in {"-", ":"}
        ):
            cursor += 1
        if cursor == name_start or not body[name_start].isalpha():
            return None
        name = body[name_start:cursor].casefold()
        quote: str | None = None
        while cursor < len(body):
            character = body[cursor]
            if quote is not None:
                if character == quote:
                    quote = None
            elif character in {'"', "'"}:
                quote = character
            elif character == ">":
                return cursor + 1, name, closing
            cursor += 1
        return None

    index = 0
    while index < len(body):
        at_line_start = index == 0 or body[index - 1] == "\n"
        if at_line_start:
            end = line_end(index)
            line = body[index:end].removesuffix("\r")
            fence = re.match(r"[ \t]{0,3}(`{3,}|~{3,})(?:[^`~].*)?$", line)
            if fence is not None:
                marker = fence.group(1)
                cursor = end + (end < len(body))
                block_end = len(body)
                while cursor < len(body):
                    candidate_end = line_end(cursor)
                    candidate = body[cursor:candidate_end].removesuffix("\r")
                    if re.fullmatch(
                        rf"[ \t]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*",
                        candidate,
                    ):
                        block_end = candidate_end + (candidate_end < len(body))
                        break
                    cursor = candidate_end + (candidate_end < len(body))
                mask(index, block_end)
                index = block_end
                continue

            definition = re.match(r"[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*(.*)$", line)
            if definition is not None:
                mask(index, end)
                block_end = end
                has_destination = bool(definition.group(1))
                for _ in range(2):
                    if block_end >= len(body):
                        break
                    next_start = block_end + 1
                    next_end = line_end(next_start)
                    next_line = body[next_start:next_end].removesuffix("\r")
                    indented = next_line[:1] in {" ", "\t"}
                    content = next_line.lstrip()
                    is_title = bool(content) and content[0] in {'"', "'", "("}
                    if (
                        not indented
                        or not content
                        or (has_destination and not is_title)
                    ):
                        break
                    mask(next_start, next_end)
                    block_end = next_end
                    has_destination = True
                index = block_end
                continue

        if body.startswith("<!--", index):
            close = body.find("-->", index + 4)
            end = len(body) if close < 0 else close + 3
            mask(index, end)
            index = end
            continue

        if body[index] == "`":
            tick_end = index + 1
            while tick_end < len(body) and body[tick_end] == "`":
                tick_end += 1
            marker = body[index:tick_end]
            close = body.find(marker, tick_end)
            if close >= 0:
                end = close + len(marker)
                mask(index, end)
                index = end
                continue

        if body[index : index + 7].casefold() in {"http://", "https:/"}:
            scheme_length = 7 if body[index : index + 7].casefold() == "http://" else 8
            if body[index : index + scheme_length].casefold() in {
                "http://",
                "https://",
            }:
                end = index + scheme_length
                while end < len(body) and not (
                    body[end].isspace() or body[end] in "<>()"
                ):
                    end += 1
                mask(index, end)
                index = end
                continue

        if body.startswith("][", index):
            end = index + 2
            escaped = False
            while end < len(body) and body[end] not in "\r\n":
                character = body[end]
                if character == "]" and not escaped:
                    end += 1
                    mask(index, end)
                    index = end
                    break
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                end += 1
            if index == end:
                continue
            mask(index, end)
            index = end
            continue

        if body.startswith("](", index):
            end = index + 2
            depth = 1
            escaped = False
            while end < len(body) and body[end] not in "\r\n":
                character = body[end]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0:
                        end += 1
                        mask(index, end)
                        index = end
                        break
                end += 1
            if index == end:
                continue
            mask(index, end)
            index = end
            continue

        if body[index] == "<":
            tag = html_tag(index)
            if tag is not None:
                end, name, closing = tag
                if not closing and name in {"code", "pre"}:
                    closing_start = folded_body.find(f"</{name}", end)
                    closing_tag = None if closing_start < 0 else html_tag(closing_start)
                    if (
                        closing_tag is None
                        or closing_tag[1] != name
                        or not closing_tag[2]
                    ):
                        end = len(body)
                    else:
                        end = closing_tag[0]
                mask(index, end)
                index = end
                continue

        index += 1
    return tuple(
        (int(match.group(1)), match.group(0))
        for match in re.finditer(
            r"(?<![\w/#\\])#([1-9][0-9]*)(?![\w/])", "".join(visible)
        )
    )


class EvidenceAuthority(str, Enum):
    DIRECT = "direct"
    CONTEXT_ONLY = "context_only"
    COUNTER_ONLY = "counter_only"


class FrozenNodeType(str, Enum):
    SEED_ISSUE = "seed_issue"
    MAINTAINER_RESOLUTION = "maintainer_resolution"
    MAINTAINER_POINTER = "maintainer_pointer"
    LINKED_ISSUE = "linked_issue"
    LINKED_PULL_REQUEST = "linked_pull_request"
    LINKED_COMMIT = "linked_commit"
    MERGE_COMMIT = "merge_commit"
    CHANGED_CODE = "changed_code"
    REGRESSION_TEST = "regression_test"
    REPAIR_SUMMARY = "repair_summary"
    NEIGHBOR_CASE = "neighbor_case"


def _materialized_file_node_type(filename: str) -> FrozenNodeType:
    lowered = filename.casefold()
    parts = tuple(part for part in lowered.split("/") if part)
    leaf = parts[-1] if parts else lowered
    is_regression_test = (
        any(part in {"test", "tests", "spec", "specs"} for part in parts[:-1])
        or leaf.startswith("test_")
        or "_test." in leaf
        or ".test." in leaf
        or ".spec." in leaf
    )
    return (
        FrozenNodeType.REGRESSION_TEST
        if is_regression_test
        else FrozenNodeType.CHANGED_CODE
    )


class CapturePolicy(FrozenStrictModel):
    schema_version: Literal[
        "ase-frozen-capture-policy-v1", "ase-frozen-capture-policy-v2"
    ]
    policy_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    allowed_repositories: tuple[str, ...] = Field(min_length=1)
    endpoint_allowlist: tuple[str, ...] = Field(min_length=1)
    max_relation_depth: Literal[1]
    allow_redirects: Literal[False]
    runtime_network_forbidden: Literal[True]
    intrinsic_node_types: tuple[FrozenNodeType, ...]
    max_maintainer_references_per_record: int | None = Field(
        default=None, ge=0, le=64, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def fixed_policy_is_safe(self) -> "CapturePolicy":
        if self.schema_version == "ase-frozen-capture-policy-v1":
            if (
                "max_maintainer_references_per_record" in self.__pydantic_fields_set__
                or self.max_maintainer_references_per_record is not None
            ):
                raise ValueError(
                    "v1 capture policy cannot declare max_maintainer_references"
                )
        elif (
            self.policy_id != "github-one-hop-maintainer-pointer-v2"
            or self.max_maintainer_references_per_record is None
        ):
            raise ValueError(
                "v2 capture policy requires github-one-hop-maintainer-pointer-v2 "
                "and max_maintainer_references_per_record"
            )
        repositories = self.allowed_repositories
        if (
            len(repositories) != len(set(repositories))
            or any(not _REPOSITORY.fullmatch(value) for value in repositories)
            or tuple(sorted(repositories)) != repositories
        ):
            raise ValueError("allowed repositories must be unique canonical values")
        endpoints = self.endpoint_allowlist
        if (
            len(endpoints) != len(set(endpoints))
            or tuple(sorted(endpoints)) != endpoints
            or any(value != "api.github.com" for value in endpoints)
        ):
            raise ValueError("capture endpoints must use the fixed GitHub API host")
        if (
            set(item.value for item in self.intrinsic_node_types)
            != _INTRINSIC_NODE_TYPES
        ):
            raise ValueError("intrinsic node types must match the fixed one-hop policy")
        return self


class FrozenBlobManifestEntry(FrozenStrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=1)
    media_type: Literal["application/json"]

    @model_validator(mode="after")
    def path_is_content_addressed(self) -> "FrozenBlobManifestEntry":
        _require_sha256(self.sha256, field="blob sha256")
        if self.relative_path != f"blobs/{self.sha256}.blob":
            raise ValueError("blob path must be derived from its content digest")
        return self


def frozen_graph_node_id(
    *,
    record_id: str,
    node_type: FrozenNodeType | str,
    canonical_uri: str,
    immutable_ref: str | None,
    content_sha256: str,
) -> str:
    value = node_type.value if isinstance(node_type, FrozenNodeType) else node_type
    digest = _canonical_sha256(
        {
            "canonical_uri": canonical_uri,
            "content_sha256": content_sha256,
            "immutable_ref": immutable_ref,
            "node_type": value,
            "record_id": record_id,
        }
    )
    return f"feg-node-{digest}"


class FrozenGraphNode(FrozenStrictModel):
    node_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    node_type: FrozenNodeType
    canonical_uri: str = Field(min_length=1)
    repository: str
    relation_depth: int = Field(ge=0, le=1)
    intrinsic_parent_node_id: str | None = None
    retrieved_at: str
    remote_updated_at: str | None = None
    immutable_ref: str | None = None
    raw_blob_sha256: str
    content_sha256: str
    content: str = Field(min_length=1)
    evidence_capabilities: tuple[str, ...]
    authority: EvidenceAuthority
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_is_json(cls, value: object) -> object:
        _validate_json_value(value, path="metadata")
        _reject_forbidden_metadata(value)
        return value

    @model_validator(mode="after")
    def validate_node_identity_and_scope(self) -> "FrozenGraphNode":
        _require_sha256(self.raw_blob_sha256, field="raw_blob_sha256")
        _require_sha256(self.content_sha256, field="content_sha256")
        if (
            hashlib.sha256(self.content.encode("utf-8")).hexdigest()
            != self.content_sha256
        ):
            raise ValueError("content_sha256 does not match node content")
        retrieved = _parse_timestamp(self.retrieved_at, field="retrieved_at")
        if self.remote_updated_at is not None:
            remote_updated = _parse_timestamp(
                self.remote_updated_at, field="remote_updated_at"
            )
            if remote_updated > retrieved:
                raise ValueError("remote_updated_at cannot postdate retrieved_at")
        if not _REPOSITORY.fullmatch(self.repository):
            raise ValueError("repository must be a canonical owner/repo")
        uri_repository, entity = _repository_from_uri(self.canonical_uri)
        if uri_repository != self.repository:
            raise ValueError("node URI repository does not match node repository")
        expected_id = frozen_graph_node_id(
            record_id=self.record_id,
            node_type=self.node_type,
            canonical_uri=self.canonical_uri,
            immutable_ref=self.immutable_ref,
            content_sha256=self.content_sha256,
        )
        if self.node_id != expected_id:
            raise ValueError("node_id does not match canonical node identity")
        if self.evidence_capabilities != _NODE_FIXED_CAPABILITIES[self.node_type.value]:
            raise ValueError(
                "node evidence capabilities must be fixed by its source type"
            )

        intrinsic = self.node_type.value in _INTRINSIC_NODE_TYPES
        direct = self.node_type in {
            FrozenNodeType.LINKED_ISSUE,
            FrozenNodeType.LINKED_PULL_REQUEST,
            FrozenNodeType.LINKED_COMMIT,
            FrozenNodeType.NEIGHBOR_CASE,
        }
        if self.node_type in {
            FrozenNodeType.SEED_ISSUE,
            FrozenNodeType.MAINTAINER_RESOLUTION,
            FrozenNodeType.MAINTAINER_POINTER,
        }:
            if self.relation_depth != 0 or self.intrinsic_parent_node_id is not None:
                raise ValueError("seed-owned nodes must remain at relation depth zero")
        elif self.relation_depth != 1:
            raise ValueError("direct and intrinsic nodes must be at relation depth one")
        if intrinsic != (self.intrinsic_parent_node_id is not None):
            raise ValueError("intrinsic nodes require exactly one intrinsic parent")
        if direct and self.intrinsic_parent_node_id is not None:
            raise ValueError("direct nodes cannot have an intrinsic parent")

        entity_kind = entity[0]
        if self.node_type in {
            FrozenNodeType.SEED_ISSUE,
            FrozenNodeType.MAINTAINER_POINTER,
            FrozenNodeType.LINKED_ISSUE,
        }:
            if entity_kind != "issues" or len(entity) != 2 or not entity[1].isdigit():
                raise ValueError("issue nodes require an immutable issue-number URI")
            if str(int(entity[1])) != entity[1]:
                raise ValueError("issue nodes require one canonical URI spelling")
        elif self.node_type is FrozenNodeType.LINKED_PULL_REQUEST:
            if entity_kind != "pull" or len(entity) != 2 or not entity[1].isdigit():
                raise ValueError("pull request nodes require a pull-number URI")
            if str(int(entity[1])) != entity[1]:
                raise ValueError(
                    "pull request nodes require one canonical URI spelling"
                )
        elif self.node_type in {
            FrozenNodeType.LINKED_COMMIT,
            FrozenNodeType.MERGE_COMMIT,
            FrozenNodeType.REPAIR_SUMMARY,
        }:
            if entity_kind != "commit" or len(entity) != 2:
                raise ValueError("commit nodes require a commit URI")
        elif self.node_type in {
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
        }:
            if entity_kind != "blob" or len(entity) < 3:
                raise ValueError("code and test nodes require an immutable blob URI")

        requires_commit = self.node_type in {
            FrozenNodeType.LINKED_COMMIT,
            FrozenNodeType.MERGE_COMMIT,
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
            FrozenNodeType.REPAIR_SUMMARY,
        }
        if requires_commit:
            if self.immutable_ref is None or not _COMMIT_SHA.fullmatch(
                self.immutable_ref
            ):
                raise ValueError("repair nodes require an immutable commit SHA")
            if entity[1] != self.immutable_ref:
                raise ValueError("node URI does not match immutable_ref")
        elif self.immutable_ref is not None:
            raise ValueError("non-repair nodes cannot declare immutable_ref")

        direct_authority_types = {
            FrozenNodeType.MAINTAINER_RESOLUTION,
            FrozenNodeType.LINKED_COMMIT,
            FrozenNodeType.MERGE_COMMIT,
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
            FrozenNodeType.REPAIR_SUMMARY,
        }
        if self.node_type is FrozenNodeType.NEIGHBOR_CASE:
            if self.authority is not EvidenceAuthority.COUNTER_ONLY:
                raise ValueError("neighbor evidence must remain counter-only")
        elif self.node_type in direct_authority_types:
            if self.authority is not EvidenceAuthority.DIRECT:
                raise ValueError(
                    "mechanism and maintainer nodes require direct authority"
                )
        elif self.authority is not EvidenceAuthority.CONTEXT_ONLY:
            raise ValueError("issue and pull nodes are context-only")
        if self.node_type in {
            FrozenNodeType.MAINTAINER_RESOLUTION,
            FrozenNodeType.MAINTAINER_POINTER,
        }:
            association = self.metadata.get("author_association")
            if association not in {"OWNER", "MEMBER", "COLLABORATOR"}:
                raise ValueError(
                    "maintainer resolution requires a trusted author_association"
                )
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


def frozen_graph_edge_id(
    *,
    record_id: str,
    from_node_id: str,
    to_node_id: str,
    relation: str,
    relation_proof_node_id: str,
    relation_proof_locator: str,
    relation_proof_sha256: str,
) -> str:
    digest = _canonical_sha256(
        {
            "from_node_id": from_node_id,
            "record_id": record_id,
            "relation": relation,
            "relation_proof_locator": relation_proof_locator,
            "relation_proof_node_id": relation_proof_node_id,
            "relation_proof_sha256": relation_proof_sha256,
            "to_node_id": to_node_id,
        }
    )
    return f"feg-edge-{digest}"


def frozen_relation_proof_locator(
    *,
    record_id: str,
    from_node_id: str,
    to_node_id: str,
    relation: str,
    relation_proof_node_id: str,
    json_pointer: str,
    reference: str,
) -> str:
    """Return one canonical relation tuple replayable against a raw blob."""

    if not json_pointer.startswith("/") or not reference:
        raise ValueError("relation proof requires a JSON pointer and reference")
    return canonical_json_bytes(
        {
            "from_node_id": from_node_id,
            "json_pointer": json_pointer,
            "record_id": record_id,
            "reference": reference,
            "relation": relation,
            "relation_proof_node_id": relation_proof_node_id,
            "to_node_id": to_node_id,
        }
    ).decode("utf-8")


class FrozenGraphEdge(FrozenStrictModel):
    edge_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    from_node_id: str = Field(min_length=1)
    to_node_id: str = Field(min_length=1)
    relation: Literal[
        "seed_resolution",
        "direct_reference",
        "resolution_reference",
        "intrinsic_materialization",
    ]
    relation_proof_node_id: str = Field(min_length=1)
    relation_proof_locator: str = Field(min_length=1)
    relation_proof_sha256: str

    @model_validator(mode="after")
    def identity_is_canonical(self) -> "FrozenGraphEdge":
        _require_sha256(self.relation_proof_sha256, field="relation_proof_sha256")
        if self.from_node_id == self.to_node_id:
            raise ValueError("frozen graph edges cannot be self-referential")
        try:
            locator = json.loads(self.relation_proof_locator)
        except json.JSONDecodeError as error:
            raise ValueError(
                "relation proof locator must be a canonical relation tuple"
            ) from error
        if (
            not isinstance(locator, dict)
            or canonical_json_bytes(locator).decode("utf-8")
            != self.relation_proof_locator
            or set(locator)
            != {
                "record_id",
                "from_node_id",
                "to_node_id",
                "relation",
                "relation_proof_node_id",
                "json_pointer",
                "reference",
            }
            or locator["record_id"] != self.record_id
            or locator["from_node_id"] != self.from_node_id
            or locator["to_node_id"] != self.to_node_id
            or locator["relation"] != self.relation
            or locator["relation_proof_node_id"] != self.relation_proof_node_id
            or not isinstance(locator["json_pointer"], str)
            or not locator["json_pointer"].startswith("/")
            or not isinstance(locator["reference"], str)
            or not locator["reference"]
        ):
            raise ValueError(
                "relation proof locator must be a canonical relation tuple"
            )
        expected = frozen_graph_edge_id(
            record_id=self.record_id,
            from_node_id=self.from_node_id,
            to_node_id=self.to_node_id,
            relation=self.relation,
            relation_proof_node_id=self.relation_proof_node_id,
            relation_proof_locator=self.relation_proof_locator,
            relation_proof_sha256=self.relation_proof_sha256,
        )
        if self.edge_id != expected:
            raise ValueError("edge_id does not match canonical edge identity")
        return self


def frozen_record_graph_hash(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("graph_sha256", None)
    return _canonical_sha256(payload)


class FrozenRecordGraph(FrozenStrictModel):
    schema_version: Literal["ase-frozen-record-graph-v1"]
    domain: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    repository: str
    seed_node_id: str = Field(min_length=1)
    nodes: tuple[FrozenGraphNode, ...] = Field(min_length=1)
    edges: tuple[FrozenGraphEdge, ...]
    capture_status: Literal["captured", "partial", "unavailable"]
    unavailable_reasons: tuple[str, ...]
    graph_sha256: str

    @model_validator(mode="after")
    def graph_is_one_hop_and_closed(self) -> "FrozenRecordGraph":
        _require_sha256(self.graph_sha256, field="graph_sha256")
        if not _REPOSITORY.fullmatch(self.repository):
            raise ValueError("graph repository must be canonical")
        node_ids = tuple(node.node_id for node in self.nodes)
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("frozen graph nodes and edges must be unique")
        entities: dict[str, list[FrozenGraphNode]] = {}
        for node in self.nodes:
            entities.setdefault(node.canonical_uri, []).append(node)
        for duplicates in entities.values():
            if len(duplicates) == 1:
                continue
            duplicate_types = {node.node_type for node in duplicates}
            if len(duplicates) == 2 and duplicate_types == {
                FrozenNodeType.SEED_ISSUE,
                FrozenNodeType.MAINTAINER_RESOLUTION,
            }:
                continue
            if len(duplicates) == 2 and duplicate_types == {
                FrozenNodeType.SEED_ISSUE,
                FrozenNodeType.MAINTAINER_POINTER,
            }:
                continue
            if FrozenNodeType.SEED_ISSUE in duplicate_types:
                if any(
                    node.node_type
                    not in {
                        FrozenNodeType.SEED_ISSUE,
                        FrozenNodeType.MAINTAINER_RESOLUTION,
                        FrozenNodeType.MAINTAINER_POINTER,
                    }
                    for node in duplicates
                ):
                    raise ValueError("direct reference cannot repeat the seed entity")
            raise ValueError("frozen graph contains a duplicate canonical entity")
        nodes = {node.node_id: node for node in self.nodes}
        if (
            self.seed_node_id not in nodes
            or nodes[self.seed_node_id].node_type is not FrozenNodeType.SEED_ISSUE
        ):
            raise ValueError("seed_node_id must identify the graph seed issue")
        if sum(node.node_type is FrozenNodeType.SEED_ISSUE for node in self.nodes) != 1:
            raise ValueError("a record graph must contain exactly one seed issue")
        if any(
            node.record_id != self.record_id or node.repository != self.repository
            for node in self.nodes
        ):
            raise ValueError("graph nodes cannot cross records or repositories")
        incoming: dict[str, list[FrozenGraphEdge]] = {
            node_id: [] for node_id in node_ids
        }
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for edge in self.edges:
            if edge.record_id != self.record_id:
                raise ValueError("graph edges cannot cross records")
            if (
                edge.from_node_id not in nodes
                or edge.to_node_id not in nodes
                or edge.relation_proof_node_id not in nodes
            ):
                raise ValueError("graph edge references an unknown node")
            proof = nodes[edge.relation_proof_node_id]
            if edge.relation_proof_sha256 != proof.content_sha256:
                raise ValueError("relation proof digest does not match proof node")
            source, target = nodes[edge.from_node_id], nodes[edge.to_node_id]
            locator = json.loads(edge.relation_proof_locator)
            pointer = locator["json_pointer"]
            if edge.relation in {"direct_reference", "resolution_reference"}:
                expected_reference = target.canonical_uri
                pointer_is_semantic = bool(
                    re.fullmatch(
                        r"/relation_events/(?:0|[1-9][0-9]*)/target_uri", pointer
                    )
                )
            elif edge.relation == "seed_resolution":
                expected_reference = target.content
                pointer_is_semantic = pointer == "/body"
            elif target.node_type in {
                FrozenNodeType.CHANGED_CODE,
                FrozenNodeType.REGRESSION_TEST,
            }:
                expected_reference = target.canonical_uri
                pointer_is_semantic = bool(
                    re.fullmatch(r"/files/(?:0|[1-9][0-9]*)/canonical_uri", pointer)
                )
            else:
                expected_reference = target.immutable_ref or target.canonical_uri
                pointer_is_semantic = pointer == "/merge_commit_sha"
            if not pointer_is_semantic:
                raise ValueError(
                    "relation proof must use its fixed semantic JSON pointer"
                )
            if locator["reference"] != expected_reference:
                raise ValueError(
                    "relation proof reference does not identify the actual edge target"
                )
            if (
                edge.relation == "intrinsic_materialization"
                and proof.node_id != source.node_id
            ):
                raise ValueError(
                    "intrinsic relation proof must be the intrinsic parent node"
                )
            if edge.relation == "seed_resolution":
                valid = source.node_id == self.seed_node_id and target.node_type in {
                    FrozenNodeType.MAINTAINER_RESOLUTION,
                    FrozenNodeType.MAINTAINER_POINTER,
                }
                if proof.node_id != target.node_id:
                    raise ValueError(
                        "seed resolution proof must be the resolution node"
                    )
            elif edge.relation in {"direct_reference", "resolution_reference"}:
                if target.canonical_uri == nodes[self.seed_node_id].canonical_uri:
                    raise ValueError("direct reference cannot repeat the seed entity")
                valid = (
                    (
                        source.node_id == self.seed_node_id
                        if edge.relation == "direct_reference"
                        else source.node_type is FrozenNodeType.MAINTAINER_POINTER
                    )
                    and target.node_type
                    in (
                        {
                            FrozenNodeType.LINKED_ISSUE,
                            FrozenNodeType.LINKED_PULL_REQUEST,
                            FrozenNodeType.LINKED_COMMIT,
                            FrozenNodeType.NEIGHBOR_CASE,
                        }
                        if edge.relation == "direct_reference"
                        else {
                            FrozenNodeType.LINKED_ISSUE,
                            FrozenNodeType.LINKED_PULL_REQUEST,
                        }
                    )
                    and proof.relation_depth == 0
                )
                if (
                    edge.relation == "resolution_reference"
                    and proof.node_id != source.node_id
                ):
                    raise ValueError(
                        "resolution reference proof must be its source pointer"
                    )
            else:
                if source.node_type is FrozenNodeType.LINKED_PULL_REQUEST:
                    merge_commit_sha = source.metadata.get("merge_commit_sha")
                    if (
                        source.metadata.get("merged") is not True
                        or not isinstance(merge_commit_sha, str)
                        or not _COMMIT_SHA.fullmatch(merge_commit_sha)
                    ):
                        raise ValueError(
                            "intrinsic repair nodes require a merged pull request"
                        )
                    if target.immutable_ref != merge_commit_sha:
                        raise ValueError(
                            "intrinsic repair node must match the pull request merge commit"
                        )
                elif target.immutable_ref != source.immutable_ref:
                    raise ValueError(
                        "intrinsic repair node must match its direct commit parent"
                    )
                valid = (
                    source.node_type
                    in {
                        FrozenNodeType.LINKED_PULL_REQUEST,
                        FrozenNodeType.LINKED_COMMIT,
                    }
                    and target.node_type.value in _INTRINSIC_NODE_TYPES
                    and target.intrinsic_parent_node_id == source.node_id
                    and proof.node_id == source.node_id
                    and (
                        target.node_type is not FrozenNodeType.MERGE_COMMIT
                        or source.node_type is FrozenNodeType.LINKED_PULL_REQUEST
                    )
                )
            if not valid:
                raise ValueError("edge violates the fixed one-hop relation policy")
            incoming[target.node_id].append(edge)
            adjacency[source.node_id].add(target.node_id)
        if incoming[self.seed_node_id]:
            raise ValueError("seed issue cannot have an incoming edge")
        if any(
            len(incoming[node_id]) != 1
            for node_id in node_ids
            if node_id != self.seed_node_id
        ):
            raise ValueError("every non-seed node requires exactly one provenance edge")

        visited: set[str] = set()
        active: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active:
                raise ValueError("frozen evidence graph contains a cycle")
            if node_id in visited:
                return
            active.add(node_id)
            for child in adjacency[node_id]:
                visit(child)
            active.remove(node_id)
            visited.add(node_id)

        visit(self.seed_node_id)
        if visited != set(node_ids):
            raise ValueError("frozen evidence graph contains unreachable nodes")
        expected_hash = frozen_record_graph_hash(self.model_dump(mode="json"))
        if self.graph_sha256 != expected_hash:
            raise ValueError("graph_sha256 does not match canonical graph")
        if self.capture_status == "captured" and self.unavailable_reasons:
            raise ValueError("captured graphs cannot contain unavailable reasons")
        if self.capture_status != "captured" and not self.unavailable_reasons:
            raise ValueError("incomplete graphs must explain unavailable sources")
        if len(self.unavailable_reasons) != len(set(self.unavailable_reasons)):
            raise ValueError("unavailable reasons must be unique")
        return self


class _ArtifactBinding(FrozenStrictModel):
    relative_path: str
    sha256: str

    @model_validator(mode="after")
    def binding_is_safe(self) -> "_ArtifactBinding":
        _require_sha256(self.sha256, field="artifact sha256")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError("bundle artifact bindings must be local file names")
        return self


class _RetrievalTool(FrozenStrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    module_sha256: str

    @model_validator(mode="after")
    def module_is_bound(self) -> "_RetrievalTool":
        _require_sha256(self.module_sha256, field="retrieval tool module_sha256")
        return self


class FrozenEvidenceBundleManifest(FrozenStrictModel):
    schema_version: Literal["ase-frozen-evidence-bundle-v1"]
    status: Literal["active"]
    domain: str = Field(min_length=1)
    bundle_id: str = Field(min_length=1)
    split_id: str = Field(min_length=1)
    split_manifest_sha256: str
    runner_cohort_sha256: str
    record_ids_sha256: str
    record_count: int = Field(ge=1)
    capture_policy: _ArtifactBinding
    record_graphs: _ArtifactBinding
    repo_allowlist_sha256: str
    reserved_entity_denylist_sha256: str
    capture_started_at: str
    capture_finished_at: str
    retrieval_tool: _RetrievalTool
    blobs: tuple[FrozenBlobManifestEntry, ...] = Field(min_length=1)
    bundle_merkle_root: str
    network_policy: Literal["offline_capture_only_runtime_network_forbidden"]
    gold_accessed: Literal[False]

    @model_validator(mode="after")
    def manifest_is_canonical(self) -> "FrozenEvidenceBundleManifest":
        for field in (
            "split_manifest_sha256",
            "runner_cohort_sha256",
            "record_ids_sha256",
            "repo_allowlist_sha256",
            "reserved_entity_denylist_sha256",
            "bundle_merkle_root",
        ):
            _require_sha256(getattr(self, field), field=field)
        started = _parse_timestamp(self.capture_started_at, field="capture_started_at")
        finished = _parse_timestamp(
            self.capture_finished_at, field="capture_finished_at"
        )
        if finished < started:
            raise ValueError("capture_finished_at cannot precede capture_started_at")
        digests = tuple(item.sha256 for item in self.blobs)
        if len(digests) != len(set(digests)) or tuple(sorted(digests)) != digests:
            raise ValueError("blob manifest must be uniquely sorted by sha256")
        if self.capture_policy.relative_path != "capture_policy.json":
            raise ValueError("capture policy binding uses an unexpected file name")
        if self.record_graphs.relative_path != "record_graphs.jsonl":
            raise ValueError("record graph binding uses an unexpected file name")
        expected_merkle = frozen_bundle_merkle_root(
            capture_policy_sha256=self.capture_policy.sha256,
            record_graphs_sha256=self.record_graphs.sha256,
            blobs=self.blobs,
        )
        if self.bundle_merkle_root != expected_merkle:
            raise ValueError("bundle Merkle root does not match bound artifacts")
        return self


def frozen_bundle_merkle_root(
    *,
    capture_policy_sha256: str,
    record_graphs_sha256: str,
    blobs: Sequence[FrozenBlobManifestEntry],
) -> str:
    _require_sha256(capture_policy_sha256, field="capture_policy_sha256")
    _require_sha256(record_graphs_sha256, field="record_graphs_sha256")
    return _canonical_sha256(
        {
            "blobs": [item.model_dump(mode="json") for item in blobs],
            "capture_policy_sha256": capture_policy_sha256,
            "record_graphs_sha256": record_graphs_sha256,
        }
    )


class FrozenEvidenceGraphBundle(FrozenStrictModel):
    domain: str
    bundle_id: str
    split_id: str
    policy: CapturePolicy
    graphs: Mapping[str, FrozenRecordGraph]
    record_ids: tuple[str, ...]
    manifest: FrozenEvidenceBundleManifest
    trust_manifest_sha256: str
    trust_manifest_relative_path: str

    @model_validator(mode="after")
    def freeze_graph_index(self) -> "FrozenEvidenceGraphBundle":
        if tuple(sorted(self.graphs)) != self.record_ids:
            raise ValueError("bundle graph index must match ordered record_ids")
        object.__setattr__(self, "graphs", MappingProxyType(dict(self.graphs)))
        return self

    @field_serializer("graphs")
    def serialize_graphs(
        self, value: Mapping[str, FrozenRecordGraph]
    ) -> dict[str, object]:
        return {
            key: graph.model_dump(mode="json") for key, graph in sorted(value.items())
        }


def _repository_root(repository_root: str | Path | None) -> Path:
    return (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )


def _resolve_inside(parent: Path, relative: str) -> Path:
    candidate = (parent / relative).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError("frozen evidence artifact escapes its bundle") from error
    return candidate


def _read_bound_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read bound frozen evidence {label}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen evidence {label} hash mismatch")
    return raw


def _load_canonical_json(raw: bytes, *, label: str) -> object:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen evidence {label} JSON") from error
    if raw != canonical_json_bytes(value):
        raise ValueError(f"frozen evidence {label} is not canonical JSON")
    return value


def _resolve_json_pointer(value: object, pointer: str) -> object:
    current = value
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError("raw content extraction JSON pointer is unresolved")
    return current


def load_bound_frozen_evidence_graph_for_domain(
    domain: str,
    *,
    repository_root: str | Path | None = None,
) -> FrozenEvidenceGraphBundle:
    """Load only the graph bundle fixed by the module-owned domain registry."""

    try:
        manifest_relative, expected_manifest_sha256 = _REGISTERED_TRUST_ROOTS[domain]
    except KeyError as error:
        raise ValueError(
            f"no frozen evidence trust root is registered for domain {domain!r}"
        ) from error
    _require_sha256(expected_manifest_sha256, field="registered manifest sha256")
    root = _repository_root(repository_root)
    manifest_path = (root / manifest_relative).resolve()
    try:
        observed_relative = manifest_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            "registered frozen evidence manifest escapes repository"
        ) from error
    if observed_relative != manifest_relative:
        raise ValueError("registered frozen evidence manifest path is not canonical")
    manifest_raw = _read_bound_bytes(
        manifest_path, expected_manifest_sha256, label="manifest"
    )
    manifest = FrozenEvidenceBundleManifest.model_validate(
        _load_canonical_json(manifest_raw, label="manifest")
    )
    if manifest.domain != domain:
        raise ValueError("frozen evidence manifest domain mismatch")

    bundle_dir = manifest_path.parent.resolve()
    policy_path = _resolve_inside(bundle_dir, manifest.capture_policy.relative_path)
    policy_raw = _read_bound_bytes(
        policy_path, manifest.capture_policy.sha256, label="capture policy"
    )
    policy = CapturePolicy.model_validate(
        _load_canonical_json(policy_raw, label="capture policy")
    )
    if policy.domain != domain:
        raise ValueError("capture policy domain mismatch")
    if (
        canonical_string_set_sha256(policy.allowed_repositories)
        != manifest.repo_allowlist_sha256
    ):
        raise ValueError("capture policy repository allowlist hash mismatch")

    blob_entries = {entry.sha256: entry for entry in manifest.blobs}
    raw_blobs: dict[str, bytes] = {}
    for entry in manifest.blobs:
        blob_path = _resolve_inside(bundle_dir, entry.relative_path)
        raw = _read_bound_bytes(blob_path, entry.sha256, label="blob")
        if len(raw) != entry.size_bytes:
            raise ValueError("frozen evidence blob size mismatch")
        _load_canonical_json(raw, label="blob")
        raw_blobs[entry.sha256] = raw

    graphs_path = _resolve_inside(bundle_dir, manifest.record_graphs.relative_path)
    graphs_raw = _read_bound_bytes(
        graphs_path, manifest.record_graphs.sha256, label="record graphs"
    )
    try:
        decoded = graphs_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("record graphs must be UTF-8 JSONL") from error
    if not decoded or not decoded.endswith("\n") or "\n\n" in decoded:
        raise ValueError("record graphs must be non-empty canonical JSONL")
    graphs: dict[str, FrozenRecordGraph] = {}
    canonical_lines: list[bytes] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid record graph JSONL row {line_number}") from error
        canonical_lines.append(canonical_json_bytes(payload) + b"\n")
        graph = FrozenRecordGraph.model_validate(payload)
        if graph.record_id in graphs:
            raise ValueError("record graph bundle contains duplicate record_id")
        if graph.domain != domain:
            raise ValueError("record graph domain mismatch")
        if graph.repository not in policy.allowed_repositories:
            raise ValueError("record graph repository is outside capture policy")
        graphs[graph.record_id] = graph
    if b"".join(canonical_lines) != graphs_raw:
        raise ValueError("record graph artifact is not canonical JSONL")
    record_ids = tuple(sorted(graphs))
    if len(record_ids) != manifest.record_count:
        raise ValueError("record graph count does not match manifest")
    if canonical_record_ids_sha256(record_ids) != manifest.record_ids_sha256:
        raise ValueError("record graph identity set does not match manifest")
    capture_started = _parse_timestamp(
        manifest.capture_started_at, field="capture_started_at"
    )
    capture_finished = _parse_timestamp(
        manifest.capture_finished_at, field="capture_finished_at"
    )
    if any(
        not capture_started
        <= _parse_timestamp(node.retrieved_at, field="retrieved_at")
        <= capture_finished
        for graph in graphs.values()
        for node in graph.nodes
    ):
        raise ValueError("node retrieved_at falls outside the capture window")
    for graph in graphs.values():
        node_raw_values: dict[str, dict[str, object]] = {}
        for node in graph.nodes:
            try:
                raw_value = json.loads(raw_blobs[node.raw_blob_sha256].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "frozen graph node requires JSON raw provenance"
                ) from error
            extracted = _resolve_json_pointer(
                raw_value, _RAW_CONTENT_POINTERS[node.node_type.value]
            )
            if extracted != node.content:
                raise ValueError(
                    "raw content extraction does not match frozen node content"
                )
            if (
                not isinstance(raw_value, dict)
                or raw_value.get("canonical_uri") != node.canonical_uri
                or raw_value.get("repository") != node.repository
                or raw_value.get("updated_at") != node.remote_updated_at
            ):
                raise ValueError(
                    "raw node provenance does not match URI, repository, or timestamp"
                )
            node_raw_values[node.node_id] = raw_value
            if node.node_type in {
                FrozenNodeType.MAINTAINER_RESOLUTION,
                FrozenNodeType.MAINTAINER_POINTER,
            } and (
                raw_value.get("author_association")
                != node.metadata.get("author_association")
            ):
                raise ValueError(
                    "raw maintainer pointer authority does not match frozen metadata"
                )
            if (
                policy.schema_version == "ase-frozen-capture-policy-v2"
                and node.node_type is FrozenNodeType.LINKED_ISSUE
            ):
                _, issue_entity = _repository_from_uri(node.canonical_uri)
                if type(raw_value.get("number")) is not int or raw_value.get(
                    "number"
                ) != int(issue_entity[1]):
                    raise ValueError(
                        "linked issue raw number does not match canonical URI"
                    )
            if node.node_type is not FrozenNodeType.LINKED_PULL_REQUEST:
                continue
            try:
                raw_pull = json.loads(raw_blobs[node.raw_blob_sha256].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "linked pull request requires JSON raw provenance"
                ) from error
            if not isinstance(raw_pull, dict):
                raise ValueError("linked pull request raw provenance must be an object")
            _, entity = _repository_from_uri(node.canonical_uri)
            if (
                raw_pull.get("number") != int(entity[1])
                or raw_pull.get("merged") is not node.metadata.get("merged")
                or raw_pull.get("merge_commit_sha")
                != node.metadata.get("merge_commit_sha")
            ):
                raise ValueError(
                    "raw pull request provenance does not match frozen metadata"
                )
        node_index = {node.node_id: node for node in graph.nodes}
        pointer_nodes = tuple(
            node
            for node in graph.nodes
            if node.node_type is FrozenNodeType.MAINTAINER_POINTER
        )
        resolution_edges = tuple(
            edge for edge in graph.edges if edge.relation == "resolution_reference"
        )
        if policy.schema_version == "ase-frozen-capture-policy-v1":
            if pointer_nodes or resolution_edges:
                raise ValueError(
                    "v1 capture policy cannot contain maintainer pointer evidence"
                )
        else:
            if any(
                node.node_type is FrozenNodeType.MAINTAINER_RESOLUTION
                for node in graph.nodes
            ):
                raise ValueError(
                    "v2 capture policy cannot contain legacy maintainer resolution"
                )
            if len(pointer_nodes) > 1:
                raise ValueError("v2 graph contains multiple maintainer pointer nodes")
            cap = policy.max_maintainer_references_per_record
            if cap is None:
                raise ValueError("v2 maintainer pointer policy is missing its cap")
            for pointer_node in pointer_nodes:
                raw_pointer = node_raw_values[pointer_node.node_id]
                events = raw_pointer.get("relation_events")
                if not isinstance(events, list):
                    raise ValueError(
                        "maintainer pointer requires canonical relation_events"
                    )
                visible_references = _maintainer_bare_references(pointer_node.content)
                if len(events) > cap or len(visible_references) > cap:
                    raise ValueError("maintainer pointer exceeds the policy cap")
                canonical_edges = tuple(
                    sorted(
                        (
                            edge
                            for edge in resolution_edges
                            if edge.from_node_id == pointer_node.node_id
                        ),
                        key=lambda edge: node_index[edge.to_node_id].canonical_uri,
                    )
                )
                target_identities = tuple(
                    (
                        node_index[edge.to_node_id].repository,
                        int(
                            _repository_from_uri(
                                node_index[edge.to_node_id].canonical_uri
                            )[1][1]
                        ),
                    )
                    for edge in canonical_edges
                )
                if len(target_identities) != len(set(target_identities)):
                    raise ValueError(
                        "maintainer pointer contains a duplicate canonical target"
                    )
                expected_events: list[dict[str, object]] = []
                event_numbers: set[int] = set()
                for index, edge in enumerate(canonical_edges):
                    target = node_index[edge.to_node_id]
                    _, entity = _repository_from_uri(target.canonical_uri)
                    target_number = int(entity[1])
                    locator = json.loads(edge.relation_proof_locator)
                    if (
                        edge.relation_proof_node_id != pointer_node.node_id
                        or locator["relation_proof_node_id"] != pointer_node.node_id
                        or locator["from_node_id"] != pointer_node.node_id
                        or locator["json_pointer"]
                        != f"/relation_events/{index}/target_uri"
                    ):
                        raise ValueError(
                            "maintainer pointer proof or canonical index is invalid"
                        )
                    expected_events.append(
                        {
                            "relation": "resolution_reference",
                            "source_token": f"#{target_number}",
                            "target_number": target_number,
                            "target_repository": target.repository,
                            "target_uri": target.canonical_uri,
                        }
                    )
                    event_numbers.add(target_number)
                if events != expected_events:
                    raise ValueError(
                        "maintainer pointer relation_events are not canonical"
                    )
                seed_reference_numbers = {
                    int(entity[1])
                    for edge in graph.edges
                    if edge.relation == "direct_reference"
                    and edge.from_node_id == graph.seed_node_id
                    for target in (node_index[edge.to_node_id],)
                    for _, entity in (_repository_from_uri(target.canonical_uri),)
                    if target.node_type
                    in {
                        FrozenNodeType.LINKED_ISSUE,
                        FrozenNodeType.LINKED_PULL_REQUEST,
                    }
                }
                parsed_numbers = {number for number, _ in visible_references}
                if parsed_numbers != event_numbers | (
                    parsed_numbers & seed_reference_numbers
                ):
                    raise ValueError(
                        "maintainer pointer body has an unmaterialized reference"
                    )
            for pull_node in (
                node
                for node in graph.nodes
                if node.node_type is FrozenNodeType.LINKED_PULL_REQUEST
            ):
                raw_pull = node_raw_values[pull_node.node_id]
                incoming_edge = next(
                    edge for edge in graph.edges if edge.to_node_id == pull_node.node_id
                )
                _, pull_entity = _repository_from_uri(pull_node.canonical_uri)
                expected_issue_marker = (
                    "https://api.github.com/repos/"
                    f"{pull_node.repository}/pulls/{int(pull_entity[1])}"
                    if incoming_edge.relation == "resolution_reference"
                    else None
                )
                if (
                    "issue_pull_request_url" not in raw_pull
                    or raw_pull.get("issue_pull_request_url") != expected_issue_marker
                ):
                    raise ValueError(
                        "linked pull raw issue marker does not match provenance edge"
                    )
                file_children = tuple(
                    sorted(
                        (
                            node
                            for node in graph.nodes
                            if node.intrinsic_parent_node_id == pull_node.node_id
                            and node.node_type
                            in {
                                FrozenNodeType.CHANGED_CODE,
                                FrozenNodeType.REGRESSION_TEST,
                            }
                        ),
                        key=lambda node: node.canonical_uri,
                    )
                )
                expected_files = [
                    {"canonical_uri": node.canonical_uri} for node in file_children
                ]
                raw_files = raw_pull.get("files")
                if raw_files != expected_files:
                    raise ValueError(
                        "linked pull raw files are not canonical materialized children"
                    )
                file_edges = {
                    edge.to_node_id: edge
                    for edge in graph.edges
                    if edge.from_node_id == pull_node.node_id
                    and node_index[edge.to_node_id].node_type
                    in {
                        FrozenNodeType.CHANGED_CODE,
                        FrozenNodeType.REGRESSION_TEST,
                    }
                }
                if set(file_edges) != {node.node_id for node in file_children}:
                    raise ValueError(
                        "linked pull files and materialization edges are not one-to-one"
                    )
                for index, child in enumerate(file_children):
                    _, child_entity = _repository_from_uri(child.canonical_uri)
                    expected_node_type = _materialized_file_node_type(
                        "/".join(child_entity[2:])
                    )
                    if child.node_type is not expected_node_type:
                        raise ValueError(
                            "materialized file path does not match its node type"
                        )
                    edge = file_edges[child.node_id]
                    locator = json.loads(edge.relation_proof_locator)
                    if (
                        edge.relation != "intrinsic_materialization"
                        or edge.relation_proof_node_id != pull_node.node_id
                        or locator["json_pointer"] != f"/files/{index}/canonical_uri"
                        or locator["reference"] != child.canonical_uri
                    ):
                        raise ValueError(
                            "linked pull files use a noncanonical materialization proof"
                        )
        for edge in graph.edges:
            locator = json.loads(edge.relation_proof_locator)
            proof = node_index[edge.relation_proof_node_id]
            raw_proof = json.loads(raw_blobs[proof.raw_blob_sha256].decode("utf-8"))
            proof_value = _resolve_json_pointer(raw_proof, locator["json_pointer"])
            if proof_value != locator["reference"]:
                raise ValueError(
                    "raw relation event does not match canonical relation proof"
                )
            if edge.relation in {"direct_reference", "resolution_reference"}:
                event_pointer = locator["json_pointer"].rsplit("/", 1)[0]
                event = _resolve_json_pointer(raw_proof, event_pointer)
                if (
                    not isinstance(event, dict)
                    or event.get("relation") != edge.relation
                ):
                    raise ValueError(
                        "raw relation event does not declare the canonical relation"
                    )
    referenced_blobs = {
        node.raw_blob_sha256 for graph in graphs.values() for node in graph.nodes
    }
    if referenced_blobs != set(blob_entries):
        raise ValueError("blob manifest must exactly cover graph raw sources")
    return FrozenEvidenceGraphBundle(
        domain=domain,
        bundle_id=manifest.bundle_id,
        split_id=manifest.split_id,
        policy=policy,
        graphs=MappingProxyType(dict(graphs)),
        record_ids=record_ids,
        manifest=manifest,
        trust_manifest_sha256=expected_manifest_sha256,
        trust_manifest_relative_path=manifest_relative,
    )


__all__ = [
    "CapturePolicy",
    "EvidenceAuthority",
    "FrozenBlobManifestEntry",
    "FrozenEvidenceBundleManifest",
    "FrozenEvidenceGraphBundle",
    "FrozenGraphEdge",
    "FrozenGraphNode",
    "FrozenNodeType",
    "FrozenRecordGraph",
    "canonical_json_bytes",
    "canonical_record_ids_sha256",
    "canonical_string_set_sha256",
    "frozen_bundle_merkle_root",
    "frozen_graph_edge_id",
    "frozen_graph_node_id",
    "frozen_relation_proof_locator",
    "frozen_record_graph_hash",
    "load_bound_frozen_evidence_graph_for_domain",
]
