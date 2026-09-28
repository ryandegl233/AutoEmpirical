from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from types import MappingProxyType
from types import SimpleNamespace

import pytest

from Benchmark.src.adaptive_empirical_workflow import frozen_evidence_runtime
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    EvidenceExplicitness,
    EvidenceItem,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph import (
    EvidenceAuthority,
    FrozenEvidenceGraphBundle,
    FrozenGraphNode,
    FrozenNodeType,
    FrozenRecordGraph,
    frozen_graph_node_id,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import (
    FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
    authoritative_revision_support,
    derive_frozen_evidence_view,
    frozen_evidence_runtime_enabled,
    load_registered_frozen_evidence_runtime,
    project_frozen_evidence_for_record,
)
from Benchmark.src.adaptive_empirical_workflow.domains import build_record_runtime
from Benchmark.src.adaptive_empirical_workflow.agents import _validate_citation_fields
from Benchmark.src.adaptive_empirical_workflow.controller import (
    _isolated_classification_view,
)
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (
    _revision_support_has_required_frozen_authority,
)


RECORD_ID = "ase2022:owner/repo:1"
COMMIT = "a" * 40


def _node(
    node_type: FrozenNodeType,
    *,
    uri: str,
    content: str,
    authority: EvidenceAuthority,
    capabilities: tuple[str, ...],
    immutable_ref: str | None = None,
) -> FrozenGraphNode:
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return FrozenGraphNode.model_construct(
        node_id=frozen_graph_node_id(
            record_id=RECORD_ID,
            node_type=node_type,
            canonical_uri=uri,
            immutable_ref=immutable_ref,
            content_sha256=content_sha256,
        ),
        record_id=RECORD_ID,
        node_type=node_type,
        canonical_uri=uri,
        repository="owner/repo",
        relation_depth=0 if node_type is FrozenNodeType.SEED_ISSUE else 1,
        intrinsic_parent_node_id=None,
        retrieved_at="2026-08-16T08:00:00+00:00",
        remote_updated_at="2026-08-15T08:00:00+00:00",
        immutable_ref=immutable_ref,
        raw_blob_sha256="b" * 64,
        content_sha256=content_sha256,
        content=content,
        evidence_capabilities=capabilities,
        authority=authority,
        metadata=MappingProxyType({}),
    )


def _bundle(
    *nodes: FrozenGraphNode, capture_status: str = "captured"
) -> FrozenEvidenceGraphBundle:
    graph = FrozenRecordGraph.model_construct(
        schema_version="ase-frozen-record-graph-v1",
        domain="ase2022",
        record_id=RECORD_ID,
        repository="owner/repo",
        seed_node_id=nodes[0].node_id,
        nodes=tuple(nodes),
        edges=(),
        capture_status=capture_status,
        unavailable_reasons=(
            ("repair_source_unavailable",) if capture_status != "captured" else ()
        ),
        graph_sha256="c" * 64,
    )
    manifest = type("Manifest", (), {})()
    manifest.bundle_merkle_root = "d" * 64
    return FrozenEvidenceGraphBundle.model_construct(
        domain="ase2022",
        bundle_id="ase-dev50-v1",
        split_id="ase-dev50",
        policy=type("Policy", (), {"policy_id": "one-hop-v1"})(),
        graphs=MappingProxyType({RECORD_ID: graph}),
        record_ids=(RECORD_ID,),
        manifest=manifest,
        trust_manifest_sha256="e" * 64,
        trust_manifest_relative_path="registered/manifest.json",
    )


def _registered_bundle(
    monkeypatch: pytest.MonkeyPatch,
    *nodes: FrozenGraphNode,
    capture_status: str = "captured",
):
    bundle = _bundle(*nodes, capture_status=capture_status)
    monkeypatch.setattr(
        frozen_evidence_runtime,
        "load_bound_frozen_evidence_graph_for_domain",
        lambda domain, repository_root=None: bundle,
    )
    return load_registered_frozen_evidence_runtime("ase2022")


def _runtime_with_changed_code(
    monkeypatch: pytest.MonkeyPatch,
):
    code = _node(
        FrozenNodeType.CHANGED_CODE,
        uri=f"https://github.com/owner/repo/blob/{COMMIT}/src/a.py",
        content="- wrong\n+ fixed",
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
        immutable_ref=COMMIT,
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, code), RECORD_ID
    )
    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    return runtime, code.node_id


def test_projector_rejects_raw_caller_constructed_bundle() -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )

    with pytest.raises(ValueError, match="module-registered runtime handle"):
        project_frozen_evidence_for_record(_bundle(seed), RECORD_ID)


def test_projector_rejects_replaced_registered_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    registered = _registered_bundle(monkeypatch, seed)
    forged = replace(registered, _bundle=_bundle(seed))

    with pytest.raises(ValueError, match="module-registered runtime handle"):
        project_frozen_evidence_for_record(forged, RECORD_ID)


def test_projection_preserves_stable_source_identity_and_revision_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    code = _node(
        FrozenNodeType.CHANGED_CODE,
        uri=f"https://github.com/owner/repo/blob/{COMMIT}/src/a.py",
        content="- wrong\n+ fixed",
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
        immutable_ref=COMMIT,
    )

    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, seed, code), RECORD_ID
    )

    assert tuple(item.evidence_id for item in projection.items) == (
        seed.node_id,
        code.node_id,
    )
    assert projection.items[1].source_uri == code.canonical_uri
    assert projection.items[1].content_sha256 == code.content_sha256
    assert projection.items[1].explicitness is EvidenceExplicitness.DIRECT
    assert projection.items[1].metadata["frozen_node_type"] == "changed_code"
    assert projection.items[1].metadata["frozen_authority"] == "direct"
    assert "frozen_bundle_id" not in projection.items[1].metadata
    assert projection.items[1].metadata["evidence_capabilities"] == (
        "defect_mechanism",
    )
    assert authoritative_revision_support(projection.items[1]) is False
    assert authoritative_revision_support(projection.items[0]) is False
    assert projection.audit["projection_policy_sha256"] == (
        FROZEN_EVIDENCE_PROJECTION_POLICY_HASH
    )
    assert "content" not in projection.audit
    assert "raw_blob_sha256" not in str(projection.audit)

    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    runtime_view = runtime.ledger.view()
    runtime_code = next(
        item for item in runtime_view.items if item.evidence_id == code.node_id
    )
    assert authoritative_revision_support(runtime_code, view=runtime_view) is True
    summary_item = next(
        item for item in runtime_view.items if item.source_type == "record_summary"
    )
    assert "bundle_id" not in summary_item.metadata["frozen_evidence_runtime"]


def test_record_runtime_rejects_registered_projection_from_another_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _node(
        FrozenNodeType.CHANGED_CODE,
        uri=f"https://github.com/owner/repo/blob/{COMMIT}/src/a.py",
        content="- wrong\n+ fixed",
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
        immutable_ref=COMMIT,
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, code), RECORD_ID
    )

    with pytest.raises(ValueError, match="domain"):
        build_record_runtime(
            {"record_id": RECORD_ID, "title": "Synthetic issue"},
            taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
            domain="issta2024",
            frozen_evidence_projection=projection,
        )


def test_legacy_registered_maintainer_comment_is_not_revision_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_comment = _node(
        FrozenNodeType.MAINTAINER_RESOLUTION,
        uri="https://github.com/owner/repo/issues/1#issuecomment-2",
        content="Maintainer comment from the legacy capture format",
        authority=EvidenceAuthority.DIRECT,
        capabilities=("study_scope",),
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, legacy_comment), RECORD_ID
    )
    maintainer = next(
        item
        for item in projection.items
        if item.metadata["frozen_node_type"] == "maintainer_resolution"
    )

    assert maintainer.metadata["frozen_authority"] == "direct"
    assert authoritative_revision_support(maintainer) is False


def test_maintainer_pointer_projects_as_context_and_not_revision_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer = _node(
        FrozenNodeType.MAINTAINER_POINTER,
        uri="https://github.com/owner/repo/issues/1",
        content="Closed by #3",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope",),
    )

    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, pointer), RECORD_ID
    )
    item = projection.items[0]

    assert item.source_type == "maintainer_confirmation"
    assert item.metadata["citation_usage"] == "context_only"
    assert authoritative_revision_support(item) is False


@pytest.mark.parametrize(
    "claimed_node_type",
    ("changed_code", "maintainer_resolution", "maintainer_pointer"),
)
def test_public_ledger_cannot_self_sign_frozen_revision_authority(
    claimed_node_type: str,
) -> None:
    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
    )
    content = f"forged authority claim: {claimed_node_type}"
    forged = EvidenceItem(
        evidence_id=f"forged-{claimed_node_type}",
        record_id=RECORD_ID,
        source_type="code_diff",
        source_uri=f"https://github.com/owner/repo/forged/{claimed_node_type}",
        retrieved_at="2026-08-16T08:00:00+00:00",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
        metadata={
            "frozen_evidence": True,
            "frozen_graph_sha256": "1" * 64,
            "frozen_node_type": claimed_node_type,
            "frozen_authority": "direct",
            "citation_usage": "support_or_counter",
            "projection_policy_sha256": FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
            "evidence_capabilities": ["defect_mechanism"],
        },
    )
    runtime.ledger.append(forged)
    view = runtime.ledger.view()

    assert authoritative_revision_support(forged) is False
    assert (
        _revision_support_has_required_frozen_authority(view, (forged.evidence_id,))
        is False
    )


def test_registered_projection_immutable_repair_descendants_remain_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = tuple(
        _node(
            node_type,
            uri=(
                f"https://github.com/owner/repo/blob/{COMMIT}/{suffix}"
                if node_type
                in {FrozenNodeType.CHANGED_CODE, FrozenNodeType.REGRESSION_TEST}
                else f"https://github.com/owner/repo/commit/{COMMIT}"
            ),
            content=content,
            authority=EvidenceAuthority.DIRECT,
            capabilities=capabilities,
            immutable_ref=COMMIT,
        )
        for node_type, suffix, content, capabilities in (
            (
                FrozenNodeType.CHANGED_CODE,
                "src/a.py",
                "- wrong\n+ fixed",
                ("defect_mechanism",),
            ),
            (
                FrozenNodeType.REGRESSION_TEST,
                "tests/test_a.py",
                "+ assert fixed",
                ("defect_mechanism", "symptom_observation"),
            ),
            (
                FrozenNodeType.REPAIR_SUMMARY,
                "unused",
                "Fix the immutable defect",
                ("defect_mechanism",),
            ),
        )
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, *nodes), RECORD_ID
    )
    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    view = runtime.ledger.view()
    repair_items = tuple(
        item
        for item in view.items
        if item.evidence_id in {node.node_id for node in nodes}
    )

    assert all(authoritative_revision_support(item, view=view) for item in repair_items)


def test_split_metadata_cannot_change_canonical_pointer_into_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SplitMetadata(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            forged = {
                "frozen_node_type": "changed_code",
                "frozen_authority": "direct",
                "citation_usage": "support_or_counter",
                "evidence_capabilities": ("defect_mechanism",),
            }
            return forged.get(key, super().get(key, default))

    pointer = _node(
        FrozenNodeType.MAINTAINER_POINTER,
        uri="https://github.com/owner/repo/issues/1",
        content="Closed by #3",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope",),
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, pointer), RECORD_ID
    )
    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    trusted_view = runtime.ledger.view()
    trusted_pointer = next(
        item for item in trusted_view.items if item.evidence_id == pointer.node_id
    )
    split_pointer = trusted_pointer.model_copy(
        update={"metadata": SplitMetadata(dict(trusted_pointer.metadata))}
    )
    forged_view = trusted_view.model_copy(
        update={
            "items": tuple(
                split_pointer if item.evidence_id == pointer.node_id else item
                for item in trusted_view.items
            )
        }
    )

    assert split_pointer.model_dump(mode="json") == trusted_pointer.model_dump(
        mode="json"
    )
    assert split_pointer.metadata.get("frozen_node_type") == "changed_code"
    assert (
        _revision_support_has_required_frozen_authority(forged_view, (pointer.node_id,))
        is False
    )


def test_public_view_model_copy_does_not_inherit_registered_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, code_id = _runtime_with_changed_code(monkeypatch)
    trusted_view = runtime.ledger.view()
    copied_view = trusted_view.model_copy()

    assert (
        _revision_support_has_required_frozen_authority(copied_view, (code_id,))
        is False
    )


@pytest.mark.parametrize(
    "view_update",
    (
        {"task": "caller replacement task"},
        {"taxonomy": {"symptom": ("Forged",), "root_cause": ("Forged",)}},
        {"ledger_version": 999},
    ),
)
def test_visible_helpers_cannot_register_caller_replacement_views(
    monkeypatch: pytest.MonkeyPatch,
    view_update: dict[str, object],
) -> None:
    runtime, code_id = _runtime_with_changed_code(monkeypatch)
    trusted_view = runtime.ledger.view()
    copied_items = tuple(item.model_copy() for item in trusted_view.items)
    replacement = trusted_view.model_copy(update={**view_update, "items": copied_items})
    visible_registrars = tuple(
        value
        for name, value in vars(frozen_evidence_runtime).items()
        if callable(value) and "register" in name and "view" in name
    )
    for registrar in visible_registrars:
        registrar(runtime.ledger, replacement)

    assert (
        _revision_support_has_required_frozen_authority(replacement, (code_id,))
        is False
    )


def test_module_exposes_no_view_registrar_or_authority_registry() -> None:
    visible_view_registrars = tuple(
        name
        for name, value in vars(frozen_evidence_runtime).items()
        if callable(value) and "register" in name and "view" in name
    )
    visible_authority_registries = tuple(
        name
        for name, value in vars(frozen_evidence_runtime).items()
        if isinstance(value, dict) and "AUTHORITY_BINDINGS" in name
    )

    assert "_register_ledger_view" not in visible_view_registrars
    assert visible_authority_registries == ()


def test_exact_registered_item_copy_cannot_authorize_a_plain_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_runtime, code_id = _runtime_with_changed_code(monkeypatch)
    trusted_view = trusted_runtime.ledger.view()
    copied_code = next(
        item for item in trusted_view.items if item.evidence_id == code_id
    ).model_copy()
    plain_runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
    )
    plain_runtime.ledger.append(copied_code)
    injected_view = plain_runtime.ledger.view()

    assert (
        _revision_support_has_required_frozen_authority(injected_view, (code_id,))
        is False
    )


def test_controller_owned_isolated_view_preserves_registered_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, code_id = _runtime_with_changed_code(monkeypatch)
    trusted_view = runtime.ledger.view()
    controller_view = _isolated_classification_view(trusted_view)
    controller_subset = derive_frozen_evidence_view(trusted_view, (code_id,))

    assert (
        _revision_support_has_required_frozen_authority(trusted_view, (code_id,))
        is True
    )
    assert (
        _revision_support_has_required_frozen_authority(controller_view, (code_id,))
        is True
    )
    assert (
        _revision_support_has_required_frozen_authority(controller_subset, (code_id,))
        is True
    )


def test_serialized_registered_view_is_fail_closed_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, code_id = _runtime_with_changed_code(monkeypatch)
    payload = json.dumps(runtime.ledger.view().model_dump(mode="json"))
    script = """
import json
import sys
from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceView
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (
    _revision_support_has_required_frozen_authority,
)
view = EvidenceView.model_validate(json.loads(sys.stdin.read()))
print(_revision_support_has_required_frozen_authority(view, (sys.argv[1],)))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, code_id],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_registered_ledger_append_never_registers_forged_repair_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = _runtime_with_changed_code(monkeypatch)
    content = "caller supplied changed code"
    forged = EvidenceItem(
        evidence_id="forged-appended-code",
        record_id=RECORD_ID,
        source_type="code_diff",
        source_uri="https://github.com/owner/repo/forged/appended.py",
        retrieved_at="2026-08-16T08:00:00+00:00",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
        metadata={
            "frozen_evidence": True,
            "frozen_node_type": "changed_code",
            "frozen_authority": "direct",
            "evidence_capabilities": ["defect_mechanism"],
        },
    )
    runtime.ledger.append(forged)
    view = runtime.ledger.view()
    derived = derive_frozen_evidence_view(view, (forged.evidence_id,))

    assert (
        _revision_support_has_required_frozen_authority(view, (forged.evidence_id,))
        is False
    )
    assert (
        _revision_support_has_required_frozen_authority(derived, (forged.evidence_id,))
        is False
    )


def test_cross_record_public_copy_cannot_become_a_trusted_derived_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, code_id = _runtime_with_changed_code(monkeypatch)
    trusted_view = runtime.ledger.view()
    cross_record = trusted_view.model_copy(update={"record_id": "other-record"})
    derived = derive_frozen_evidence_view(cross_record)

    assert derived.record_id == "other-record"
    assert _revision_support_has_required_frozen_authority(derived, (code_id,)) is False


def test_neighbor_case_is_counter_only_and_cannot_gain_support_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neighbor = _node(
        FrozenNodeType.NEIGHBOR_CASE,
        uri="https://github.com/owner/repo/issues/2",
        content="A superficially similar case",
        authority=EvidenceAuthority.COUNTER_ONLY,
        capabilities=(),
    )

    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, neighbor), RECORD_ID
    )
    item = projection.items[0]

    assert item.metadata["frozen_authority"] == "counter_only"
    assert item.metadata["citation_usage"] == "counter_only"
    assert "evidence_capabilities" not in item.metadata
    assert authoritative_revision_support(item) is False

    view = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    ).ledger.view()
    with pytest.raises(ValueError, match="counter-only"):
        _validate_citation_fields(
            SimpleNamespace(supporting_evidence_ids=(item.evidence_id,)),
            view,
            "supporting_evidence_ids",
        )
    with pytest.raises(ValueError, match="counter-only"):
        _validate_citation_fields(
            SimpleNamespace(boundary_evidence_ids=(item.evidence_id,)),
            view,
            "boundary_evidence_ids",
        )
    _validate_citation_fields(
        SimpleNamespace(counter_evidence_ids=(item.evidence_id,)),
        view,
        "counter_evidence_ids",
    )


def test_projection_rejects_unknown_record_without_falling_back_to_other_graph() -> (
    None
):
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        registered = _registered_bundle(monkeypatch, seed)
        with pytest.raises(ValueError, match="record"):
            project_frozen_evidence_for_record(registered, "other-record")


def test_incomplete_graph_projects_no_evidence_or_revision_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _node(
        FrozenNodeType.CHANGED_CODE,
        uri=f"https://github.com/owner/repo/blob/{COMMIT}/src/a.py",
        content="- wrong\n+ fixed",
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
        immutable_ref=COMMIT,
    )

    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, code, capture_status="partial"), RECORD_ID
    )

    assert projection.items == ()
    assert projection.audit["available"] is False
    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    assert frozen_evidence_runtime_enabled(runtime.ledger.view()) is True


def test_record_runtime_injects_only_explicit_record_bound_frozen_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, seed), RECORD_ID
    )

    runtime = build_record_runtime(
        {"record_id": RECORD_ID, "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        retrieved_at="2026-08-16T08:00:00+00:00",
        frozen_evidence_projection=projection,
    )

    assert seed.node_id in {item.evidence_id for item in runtime.ledger.view().items}


def test_record_runtime_rejects_unsealed_caller_supplied_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, seed), RECORD_ID
    )
    forged = SimpleNamespace(record_id=RECORD_ID, items=projection.items, audit={})

    with pytest.raises(ValueError, match="verified|sealed"):
        build_record_runtime(
            {"record_id": RECORD_ID, "title": "Synthetic issue"},
            taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
            domain="ase2022",
            frozen_evidence_projection=forged,
        )


def test_record_runtime_rejects_replaced_sealed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _node(
        FrozenNodeType.SEED_ISSUE,
        uri="https://github.com/owner/repo/issues/1",
        content="Observed failure",
        authority=EvidenceAuthority.CONTEXT_ONLY,
        capabilities=("study_scope", "symptom_observation"),
    )
    projection = project_frozen_evidence_for_record(
        _registered_bundle(monkeypatch, seed), RECORD_ID
    )
    mutated = projection.items[0].model_copy(update={"content": "forged content"})
    forged = replace(projection, items=(mutated,))

    with pytest.raises(ValueError, match="verified|sealed"):
        build_record_runtime(
            {"record_id": RECORD_ID, "title": "Synthetic issue"},
            taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
            domain="ase2022",
            frozen_evidence_projection=forged,
        )
