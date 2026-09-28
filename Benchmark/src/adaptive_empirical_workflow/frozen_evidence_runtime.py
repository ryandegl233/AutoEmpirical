"""Zero-network projection of registered frozen evidence into Stage 3 items."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence
from types import MappingProxyType
from typing import Any

from .contracts import EvidenceExplicitness, EvidenceItem, EvidenceView
from .ledger import EvidenceLedger
from .frozen_evidence_graph import (
    EvidenceAuthority,
    FrozenEvidenceGraphBundle,
    FrozenGraphNode,
    FrozenNodeType,
    load_bound_frozen_evidence_graph_for_domain,
)


FROZEN_EVIDENCE_PROJECTION_POLICY_VERSION = "frozen-stage3-projection-v2"

_SOURCE_TYPES = MappingProxyType(
    {
        FrozenNodeType.SEED_ISSUE: "issue_body",
        FrozenNodeType.MAINTAINER_RESOLUTION: "maintainer_confirmation",
        FrozenNodeType.MAINTAINER_POINTER: "maintainer_confirmation",
        FrozenNodeType.LINKED_ISSUE: "issue_body",
        FrozenNodeType.LINKED_PULL_REQUEST: "linked_pull_request",
        FrozenNodeType.LINKED_COMMIT: "repair_commit",
        FrozenNodeType.MERGE_COMMIT: "repair_commit",
        FrozenNodeType.CHANGED_CODE: "code_diff",
        FrozenNodeType.REGRESSION_TEST: "regression_test",
        FrozenNodeType.REPAIR_SUMMARY: "repair",
        FrozenNodeType.NEIGHBOR_CASE: "issue_body",
    }
)

_AUTHORITATIVE_REPAIR_TYPES = frozenset(
    {
        FrozenNodeType.LINKED_COMMIT.value,
        FrozenNodeType.MERGE_COMMIT.value,
        FrozenNodeType.CHANGED_CODE.value,
        FrozenNodeType.REGRESSION_TEST.value,
        FrozenNodeType.REPAIR_SUMMARY.value,
    }
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


FROZEN_EVIDENCE_PROJECTION_POLICY_HASH = _canonical_hash(
    {
        "authoritative_repair_node_types": sorted(_AUTHORITATIVE_REPAIR_TYPES),
        "context_only_node_types": sorted(
            {
                FrozenNodeType.SEED_ISSUE.value,
                FrozenNodeType.MAINTAINER_POINTER.value,
                FrozenNodeType.LINKED_ISSUE.value,
                FrozenNodeType.LINKED_PULL_REQUEST.value,
            }
        ),
        "non_authoritative_maintainer_node_types": sorted(
            {
                FrozenNodeType.MAINTAINER_RESOLUTION.value,
                FrozenNodeType.MAINTAINER_POINTER.value,
            }
        ),
        "neighbor_case_usage": "counter_only",
        "policy_version": FROZEN_EVIDENCE_PROJECTION_POLICY_VERSION,
        "source_types": {
            node_type.value: source_type
            for node_type, source_type in sorted(
                _SOURCE_TYPES.items(), key=lambda entry: entry[0].value
            )
        },
    }
)


@dataclass(frozen=True)
class FrozenEvidenceProjection:
    """Record-exact evidence items plus content-free audit metadata."""

    domain: str
    record_id: str
    items: tuple[EvidenceItem, ...]
    audit: MappingProxyType[str, Any]
    _seal: object = field(repr=False, compare=False)


_PROJECTION_SEAL = object()
_REGISTERED_RUNTIME_SEAL = object()


@dataclass(frozen=True)
class RegisteredFrozenEvidenceRuntime:
    """Opaque handle created only after the module-owned trust loader succeeds."""

    _bundle: FrozenEvidenceGraphBundle = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __getattr__(self, name: str) -> object:
        return getattr(self._bundle, name)


@dataclass(frozen=True)
class _ProjectionAuthorityBinding:
    domain: str
    record_id: str
    projection_audit_sha256: str
    item_payload_sha256: MappingProxyType[str, str]
    item_provenance_sha256: MappingProxyType[str, str]


@dataclass(frozen=True)
class _LedgerAuthorityBinding:
    projection: _ProjectionAuthorityBinding
    projection_instance: FrozenEvidenceProjection
    initial_projection_evidence_ids: frozenset[str]


@dataclass(frozen=True)
class _ViewAuthorityBinding:
    ledger_ref: weakref.ReferenceType[EvidenceLedger]
    projection: _ProjectionAuthorityBinding
    projection_ref: weakref.ReferenceType[FrozenEvidenceProjection]
    item_payload_sha256: MappingProxyType[str, str]


def _make_runtime_identity_state() -> tuple[object, ...]:
    class WeakIdentityRegistry:
        def __init__(self) -> None:
            self.entries: dict[int, tuple[weakref.ReferenceType[object], object]] = {}

        def set(self, instance: object, value: object) -> None:
            identity = id(instance)

            def discard(reference: weakref.ReferenceType[object]) -> None:
                current = self.entries.get(identity)
                if current is not None and current[0] is reference:
                    self.entries.pop(identity, None)

            reference = weakref.ref(instance, discard)
            self.entries[identity] = (reference, value)

        def get(self, instance: object) -> object | None:
            current = self.entries.get(id(instance))
            if current is None or current[0]() is not instance:
                return None
            return current[1]

    runtime_instances = WeakIdentityRegistry()
    projection_bindings = WeakIdentityRegistry()
    ledger_bindings = WeakIdentityRegistry()
    view_bindings = WeakIdentityRegistry()

    def remember_runtime(runtime: RegisteredFrozenEvidenceRuntime) -> None:
        runtime_instances.set(runtime, True)

    def runtime_is_registered(runtime: RegisteredFrozenEvidenceRuntime) -> bool:
        return runtime_instances.get(runtime) is True

    def remember_projection(
        projection: FrozenEvidenceProjection,
        binding: _ProjectionAuthorityBinding,
    ) -> None:
        projection_bindings.set(projection, binding)

    def projection_binding(
        projection: FrozenEvidenceProjection,
    ) -> _ProjectionAuthorityBinding:
        binding = projection_bindings.get(projection)
        if not isinstance(binding, _ProjectionAuthorityBinding):
            raise ValueError(
                "runtime frozen evidence requires a verified sealed projection"
            )
        return binding

    def view_binding(view: EvidenceView) -> _ViewAuthorityBinding | None:
        binding = view_bindings.get(view)
        if not isinstance(binding, _ViewAuthorityBinding):
            return None
        ledger = binding.ledger_ref()
        projection = binding.projection_ref()
        if ledger is None or projection is None:
            return None
        ledger_binding = ledger_bindings.get(ledger)
        registered_projection = projection_bindings.get(projection)
        try:
            current_audit_sha256 = _projection_audit_binding_sha256(projection)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not isinstance(ledger_binding, _LedgerAuthorityBinding)
            or ledger_binding.projection is not binding.projection
            or registered_projection is not binding.projection
            or ledger.record_id != view.record_id
            or binding.projection.domain != view.domain_profile
            or binding.projection.record_id != view.record_id
            or current_audit_sha256 != binding.projection.projection_audit_sha256
        ):
            return None
        return binding

    def build_ledger_view(
        ledger: EvidenceLedger,
        evidence_ids: Sequence[str] | None,
    ) -> EvidenceView:
        ledger_binding = ledger_bindings.get(ledger)
        if not isinstance(ledger_binding, _LedgerAuthorityBinding):
            raise ValueError("frozen evidence ledger is not module-registered")
        projection = ledger_binding.projection_instance
        if projection_bindings.get(projection) is not ledger_binding.projection:
            raise ValueError("frozen evidence ledger lost its projection binding")
        view = EvidenceLedger.view(ledger, evidence_ids)
        matched_payloads: dict[str, str] = {}
        for item in view.items:
            payload = _canonical_item_payload(item)
            evidence_id = payload["evidence_id"]
            payload_sha256 = _canonical_hash(payload)
            if (
                evidence_id in ledger_binding.initial_projection_evidence_ids
                and ledger_binding.projection.item_payload_sha256.get(evidence_id)
                == payload_sha256
            ):
                matched_payloads[evidence_id] = payload_sha256
        view_bindings.set(
            view,
            _ViewAuthorityBinding(
                ledger_ref=weakref.ref(ledger),
                projection=ledger_binding.projection,
                projection_ref=weakref.ref(projection),
                item_payload_sha256=MappingProxyType(matched_payloads),
            ),
        )
        return view

    class FrozenEvidenceLedger(EvidenceLedger):
        """Ledger whose views are built and registered as one atomic operation."""

        def __init__(
            self,
            *,
            record_id: str,
            task: str,
            taxonomy: dict[str, list[str]],
            domain_profile: str,
            initial_items: Iterable[EvidenceItem],
            frozen_evidence_projection: FrozenEvidenceProjection,
        ) -> None:
            projection = projection_binding(frozen_evidence_projection)
            if projection.domain != domain_profile:
                raise ValueError("frozen evidence ledger crossed a domain boundary")
            if projection.record_id != record_id:
                raise ValueError("frozen evidence ledger crossed a record boundary")
            initial_items = tuple(initial_items)
            projection_items_by_id = {
                item.evidence_id: item for item in frozen_evidence_projection.items
            }
            initial_items_by_id = {item.evidence_id: item for item in initial_items}
            if any(
                initial_items_by_id.get(evidence_id) is not projection_item
                for evidence_id, projection_item in projection_items_by_id.items()
            ):
                raise ValueError(
                    "frozen evidence ledger requires the exact sealed projection items"
                )
            super().__init__(
                record_id=record_id,
                task=task,
                taxonomy=taxonomy,
                domain_profile=domain_profile,
                initial_items=initial_items,
            )
            ledger_bindings.set(
                self,
                _LedgerAuthorityBinding(
                    projection=projection,
                    projection_instance=frozen_evidence_projection,
                    initial_projection_evidence_ids=frozenset(projection_items_by_id),
                ),
            )

        def view(self, evidence_ids: Sequence[str] | None = None) -> EvidenceView:
            return build_ledger_view(self, evidence_ids)

    return (
        FrozenEvidenceLedger,
        remember_runtime,
        runtime_is_registered,
        remember_projection,
        projection_binding,
        build_ledger_view,
        view_binding,
    )


(
    FrozenEvidenceLedger,
    _remember_registered_runtime,
    _runtime_is_registered,
    _remember_projection,
    _projection_binding,
    _build_ledger_view,
    _view_binding,
) = _make_runtime_identity_state()
del _make_runtime_identity_state


def _canonical_item_payload(item: EvidenceItem) -> dict[str, Any]:
    """Snapshot one item through validation before reading any predicates."""

    dumped = item.model_dump(mode="json")
    canonical = EvidenceItem.model_validate(dumped)
    return canonical.model_dump(mode="json")


def _canonical_payload_is_authoritative(payload: dict[str, Any]) -> bool:
    metadata = payload["metadata"]
    return bool(
        metadata.get("frozen_evidence") is True
        and metadata.get("frozen_authority") == EvidenceAuthority.DIRECT.value
        and metadata.get("frozen_node_type") in _AUTHORITATIVE_REPAIR_TYPES
        and "defect_mechanism" in tuple(metadata.get("evidence_capabilities", ()))
    )


def _register_projected_item_provenance(
    item: EvidenceItem,
    node: FrozenGraphNode,
    *,
    bundle: FrozenEvidenceGraphBundle,
    graph_sha256: str,
) -> tuple[str, str]:
    payload = _canonical_item_payload(item)
    payload_sha256 = _canonical_hash(payload)
    provenance_sha256 = _canonical_hash(
        {
            "bundle_merkle_root": bundle.manifest.bundle_merkle_root,
            "frozen_node": node.model_dump(mode="json"),
            "graph_sha256": graph_sha256,
            "item": payload,
            "projection_policy_sha256": FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
            "trust_manifest_relative_path": bundle.trust_manifest_relative_path,
            "trust_manifest_sha256": bundle.trust_manifest_sha256,
        }
    )
    return payload_sha256, provenance_sha256


def _canonical_bound_view_item_payload(
    item: EvidenceItem,
    *,
    view: EvidenceView,
) -> dict[str, Any] | None:
    binding = _view_binding(view)
    if binding is None or not any(candidate is item for candidate in view.items):
        return None
    payload = _canonical_item_payload(item)
    evidence_id = payload["evidence_id"]
    payload_sha256 = _canonical_hash(payload)
    if (
        binding.item_payload_sha256.get(evidence_id) != payload_sha256
        or binding.projection.item_payload_sha256.get(evidence_id) != payload_sha256
        or evidence_id not in binding.projection.item_provenance_sha256
    ):
        return None
    return payload


def authoritative_revision_support(
    item: EvidenceItem,
    *,
    view: EvidenceView | None = None,
) -> bool:
    """Return whether an item is a direct repair/mechanism revision source."""

    if view is None:
        return False
    payload = _canonical_bound_view_item_payload(item, view=view)
    return payload is not None and _canonical_payload_is_authoritative(payload)


def frozen_evidence_may_be_support(item: EvidenceItem) -> bool:
    """Counter-only neighbors may never be used as positive support."""

    payload = _canonical_item_payload(item)
    metadata = payload["metadata"]
    return not (
        metadata.get("frozen_evidence") is True
        and metadata.get("frozen_authority") == EvidenceAuthority.COUNTER_ONLY.value
    )


def frozen_evidence_runtime_enabled(view: EvidenceView) -> bool:
    """Detect the sealed runtime marker even when capture projected zero nodes."""

    if _view_binding(view) is not None:
        return True
    for item in view.items:
        metadata = _canonical_item_payload(item)["metadata"]
        runtime_marker = metadata.get("frozen_evidence_runtime")
        if metadata.get("frozen_evidence") is True or (
            isinstance(runtime_marker, dict) and runtime_marker.get("enabled") is True
        ):
            return True
    return False


def _projection_audit_binding_sha256(
    projection: FrozenEvidenceProjection,
) -> str:
    return _canonical_hash(
        {
            "available": bool(projection.audit["available"]),
            "bundle_id": str(projection.audit["bundle_id"]),
            "domain": projection.domain,
            "enabled": bool(projection.audit["enabled"]),
            "graph_sha256": str(projection.audit["graph_sha256"]),
            "node_count": int(projection.audit["node_count"]),
            "nodes": [dict(node) for node in projection.audit["nodes"]],
            "projection_policy_sha256": str(
                projection.audit["projection_policy_sha256"]
            ),
            "record_id": projection.record_id,
            "unavailable_reasons": list(projection.audit["unavailable_reasons"]),
        }
    )


def derive_frozen_evidence_view(
    source: EvidenceView,
    evidence_ids: Sequence[str] | None = None,
) -> EvidenceView:
    """Rebuild an exact subset and retain authority only from a registered source."""

    source_by_id = {item.evidence_id: item for item in source.items}
    selected_ids = tuple(source_by_id) if evidence_ids is None else tuple(evidence_ids)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("derived evidence view IDs must be unique")
    unknown = tuple(
        evidence_id for evidence_id in selected_ids if evidence_id not in source_by_id
    )
    if unknown:
        raise ValueError(f"derived evidence view cites unknown IDs: {list(unknown)!r}")
    source_binding = _view_binding(source)
    if source_binding is None:
        return EvidenceView(
            record_id=source.record_id,
            task=source.task,
            taxonomy=source.model_dump(mode="python")["taxonomy"],
            domain_profile=source.domain_profile,
            ledger_version=source.ledger_version,
            items=tuple(source_by_id[evidence_id] for evidence_id in selected_ids),
        )
    ledger = source_binding.ledger_ref()
    if ledger is None:
        raise ValueError("registered evidence view lost its ledger binding")
    return _build_ledger_view(ledger, selected_ids)


def _project_node(
    node: FrozenGraphNode,
    *,
    bundle: FrozenEvidenceGraphBundle,
    graph_sha256: str,
) -> EvidenceItem:
    capabilities = tuple(node.evidence_capabilities)
    citation_usage = (
        "counter_only"
        if node.authority is EvidenceAuthority.COUNTER_ONLY
        else (
            "context_only"
            if node.authority is EvidenceAuthority.CONTEXT_ONLY
            else "support_or_counter"
        )
    )
    metadata: dict[str, object] = {
        "frozen_evidence": True,
        "frozen_graph_sha256": graph_sha256,
        "frozen_node_type": node.node_type.value,
        "frozen_authority": node.authority.value,
        "citation_usage": citation_usage,
        "projection_policy_sha256": FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
    }
    if capabilities:
        metadata["evidence_capabilities"] = list(capabilities)
    return EvidenceItem(
        evidence_id=node.node_id,
        record_id=node.record_id,
        source_type=_SOURCE_TYPES[node.node_type],
        source_uri=node.canonical_uri,
        retrieved_at=node.retrieved_at,
        content=node.content,
        content_sha256=node.content_sha256,
        explicitness=EvidenceExplicitness.DIRECT,
        metadata=metadata,
    )


def project_frozen_evidence_for_record(
    registered: RegisteredFrozenEvidenceRuntime,
    record_id: str,
) -> FrozenEvidenceProjection:
    """Project exactly one bound graph; never load, capture, or contact a network."""

    if (
        not isinstance(registered, RegisteredFrozenEvidenceRuntime)
        or registered._seal is not _REGISTERED_RUNTIME_SEAL
        or not _runtime_is_registered(registered)
    ):
        raise ValueError(
            "frozen evidence projection requires a module-registered runtime handle"
        )
    bundle = registered._bundle

    try:
        graph = bundle.graphs[record_id]
    except KeyError as error:
        raise ValueError(
            f"frozen evidence bundle has no graph for record {record_id!r}"
        ) from error
    if graph.record_id != record_id or graph.domain != bundle.domain:
        raise ValueError("frozen evidence graph record/domain binding mismatch")
    projected_nodes = graph.nodes if graph.capture_status == "captured" else ()
    items = tuple(
        _project_node(node, bundle=bundle, graph_sha256=graph.graph_sha256)
        for node in projected_nodes
    )
    item_payload_sha256: dict[str, str] = {}
    item_provenance_sha256: dict[str, str] = {}
    for node, item in zip(projected_nodes, items, strict=True):
        payload_sha256, provenance_sha256 = _register_projected_item_provenance(
            item,
            node,
            bundle=bundle,
            graph_sha256=graph.graph_sha256,
        )
        item_payload_sha256[item.evidence_id] = payload_sha256
        item_provenance_sha256[item.evidence_id] = provenance_sha256
    if any(item.record_id != record_id for item in items):
        raise ValueError("frozen evidence projection crossed a record boundary")
    audit_nodes = tuple(
        MappingProxyType(
            {
                "evidence_id": payload["evidence_id"],
                "node_type": metadata["frozen_node_type"],
                "authority": metadata["frozen_authority"],
                "citation_usage": metadata["citation_usage"],
                "capabilities": tuple(metadata.get("evidence_capabilities", ())),
                "revision_authoritative": _canonical_payload_is_authoritative(payload),
                "source_uri": payload["source_uri"],
                "content_sha256": payload["content_sha256"],
            }
        )
        for item in items
        for payload in (_canonical_item_payload(item),)
        for metadata in (payload["metadata"],)
    )
    projection = FrozenEvidenceProjection(
        domain=bundle.domain,
        record_id=record_id,
        items=items,
        audit=MappingProxyType(
            {
                "enabled": True,
                "available": graph.capture_status == "captured",
                "bundle_id": bundle.bundle_id,
                "domain": bundle.domain,
                "graph_sha256": graph.graph_sha256,
                "projection_policy_sha256": (FROZEN_EVIDENCE_PROJECTION_POLICY_HASH),
                "node_count": len(items),
                "nodes": audit_nodes,
                "unavailable_reasons": tuple(graph.unavailable_reasons),
            }
        ),
        _seal=_PROJECTION_SEAL,
    )
    _remember_projection(
        projection,
        _ProjectionAuthorityBinding(
            domain=bundle.domain,
            record_id=record_id,
            projection_audit_sha256=_projection_audit_binding_sha256(projection),
            item_payload_sha256=MappingProxyType(item_payload_sha256),
            item_provenance_sha256=MappingProxyType(item_provenance_sha256),
        ),
    )
    return projection


def verified_frozen_evidence_items(
    projection: FrozenEvidenceProjection | None,
    *,
    domain: str,
    record_id: str,
) -> tuple[EvidenceItem, ...]:
    """Open only a module-sealed, record-exact runtime projection."""

    if projection is None:
        return ()
    if (
        not isinstance(projection, FrozenEvidenceProjection)
        or projection._seal is not _PROJECTION_SEAL
    ):
        raise ValueError(
            "runtime frozen evidence requires a verified sealed projection"
        )
    binding = _projection_binding(projection)
    if projection.domain != domain or binding.domain != domain:
        raise ValueError("frozen evidence projection crossed a domain boundary")
    if projection.record_id != record_id or any(
        item.record_id != record_id for item in projection.items
    ):
        raise ValueError("frozen evidence projection crossed a record boundary")
    return projection.items


def frozen_evidence_projection_audit(
    projection: FrozenEvidenceProjection,
    *,
    cited_evidence_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return a JSON-safe, content-free audit payload."""

    return {
        "enabled": bool(projection.audit["enabled"]),
        "available": bool(projection.audit["available"]),
        "bundle_id": str(projection.audit["bundle_id"]),
        "graph_sha256": str(projection.audit["graph_sha256"]),
        "projection_policy_sha256": str(projection.audit["projection_policy_sha256"]),
        "node_count": int(projection.audit["node_count"]),
        "nodes": [
            {**dict(node), "cited": node["evidence_id"] in cited_evidence_ids}
            for node in projection.audit["nodes"]
        ],
        "unavailable_reasons": list(projection.audit["unavailable_reasons"]),
    }


def load_registered_frozen_evidence_runtime(
    domain: str,
    *,
    repository_root: str | None = None,
) -> RegisteredFrozenEvidenceRuntime:
    """Load the module-owned trust root; callers cannot supply a path or digest."""

    registered = RegisteredFrozenEvidenceRuntime(
        _bundle=load_bound_frozen_evidence_graph_for_domain(
            domain,
            repository_root=repository_root,
        ),
        _seal=_REGISTERED_RUNTIME_SEAL,
    )
    _remember_registered_runtime(registered)
    return registered


__all__ = [
    "FROZEN_EVIDENCE_PROJECTION_POLICY_HASH",
    "FROZEN_EVIDENCE_PROJECTION_POLICY_VERSION",
    "FrozenEvidenceLedger",
    "FrozenEvidenceProjection",
    "RegisteredFrozenEvidenceRuntime",
    "authoritative_revision_support",
    "derive_frozen_evidence_view",
    "frozen_evidence_may_be_support",
    "frozen_evidence_runtime_enabled",
    "frozen_evidence_projection_audit",
    "load_registered_frozen_evidence_runtime",
    "project_frozen_evidence_for_record",
    "verified_frozen_evidence_items",
]
