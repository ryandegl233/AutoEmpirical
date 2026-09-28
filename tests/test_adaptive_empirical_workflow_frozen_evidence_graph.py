from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

import Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph as graph_module
import Benchmark.src.adaptive_empirical_workflow as public_api
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph import (
    CapturePolicy,
    EvidenceAuthority,
    FrozenBlobManifestEntry,
    FrozenGraphEdge,
    FrozenGraphNode,
    FrozenNodeType,
    FrozenRecordGraph,
    canonical_json_bytes,
    frozen_bundle_merkle_root,
    frozen_graph_edge_id,
    frozen_graph_node_id,
    frozen_relation_proof_locator,
    frozen_record_graph_hash,
    load_bound_frozen_evidence_graph_for_domain,
)


_RECORD_ID = "synthetic:repo:issue-1"
_REPOSITORY = "owner/repo"
_RETRIEVED_AT = "2026-08-16T08:00:00+00:00"
_COMMIT = "a" * 40


def _sha(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _node(
    *,
    node_type: FrozenNodeType,
    canonical_uri: str,
    content: str,
    raw_blob_sha256: str,
    relation_depth: int,
    intrinsic_parent_node_id: str | None = None,
    immutable_ref: str | None = None,
    authority: EvidenceAuthority = EvidenceAuthority.CONTEXT_ONLY,
    capabilities: tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> FrozenGraphNode:
    content_sha256 = _sha(content)
    node_id = frozen_graph_node_id(
        record_id=_RECORD_ID,
        node_type=node_type,
        canonical_uri=canonical_uri,
        immutable_ref=immutable_ref,
        content_sha256=content_sha256,
    )
    fixed_capabilities = {
        FrozenNodeType.SEED_ISSUE: ("study_scope", "symptom_observation"),
        FrozenNodeType.MAINTAINER_RESOLUTION: ("defect_mechanism",),
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
    return FrozenGraphNode(
        node_id=node_id,
        record_id=_RECORD_ID,
        node_type=node_type,
        canonical_uri=canonical_uri,
        repository=_REPOSITORY,
        relation_depth=relation_depth,
        intrinsic_parent_node_id=intrinsic_parent_node_id,
        retrieved_at=_RETRIEVED_AT,
        remote_updated_at="2026-08-15T12:00:00Z",
        immutable_ref=immutable_ref,
        raw_blob_sha256=raw_blob_sha256,
        content_sha256=content_sha256,
        content=content,
        evidence_capabilities=capabilities or fixed_capabilities[node_type],
        authority=authority,
        metadata=metadata or {},
    )


def _edge(
    *,
    source: FrozenGraphNode,
    target: FrozenGraphNode,
    relation: str,
    proof: FrozenGraphNode,
) -> FrozenGraphEdge:
    if relation == "direct_reference":
        reference = target.canonical_uri
        json_pointer = "/relation_events/0/target_uri"
    elif relation == "intrinsic_materialization":
        if target.node_type in {
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
        }:
            reference = target.canonical_uri
            json_pointer = "/files/0/canonical_uri"
        else:
            reference = target.immutable_ref or target.canonical_uri
            json_pointer = "/merge_commit_sha"
    else:
        reference = target.content
        json_pointer = "/body"
    locator = frozen_relation_proof_locator(
        record_id=_RECORD_ID,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        json_pointer=json_pointer,
        reference=reference,
    )
    edge_id = frozen_graph_edge_id(
        record_id=_RECORD_ID,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        relation_proof_locator=locator,
        relation_proof_sha256=proof.content_sha256,
    )
    return FrozenGraphEdge(
        edge_id=edge_id,
        record_id=_RECORD_ID,
        from_node_id=source.node_id,
        to_node_id=target.node_id,
        relation=relation,
        relation_proof_node_id=proof.node_id,
        relation_proof_locator=locator,
        relation_proof_sha256=proof.content_sha256,
    )


def _write_valid_bundle(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    bundle = root / "Benchmark/configs/frozen_evidence/synthetic-v1"
    blobs = bundle / "blobs"
    blobs.mkdir(parents=True)

    raw_payloads = {
        "seed": canonical_json_bytes(
            {
                "body": "Fixed by #2",
                "canonical_uri": "https://github.com/owner/repo/issues/1",
                "repository": _REPOSITORY,
                "updated_at": "2026-08-15T12:00:00Z",
                "relation_events": [
                    {
                        "relation": "direct_reference",
                        "target_uri": "https://github.com/owner/repo/pull/2",
                    }
                ],
            }
        ),
        "pr": canonical_json_bytes(
            {
                "number": 2,
                "title": "Repair pull request",
                "merged": True,
                "merge_commit_sha": _COMMIT,
                "files": [
                    {
                        "canonical_uri": (
                            f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts"
                        )
                    }
                ],
                "canonical_uri": "https://github.com/owner/repo/pull/2",
                "repository": _REPOSITORY,
                "updated_at": "2026-08-15T12:00:00Z",
            }
        ),
        "code": canonical_json_bytes(
            {
                "patch": "- old\n+ new",
                "canonical_uri": (
                    f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts"
                ),
                "repository": _REPOSITORY,
                "updated_at": "2026-08-15T12:00:00Z",
            }
        ),
    }
    blob_entries: list[FrozenBlobManifestEntry] = []
    blob_hashes: dict[str, str] = {}
    for name, raw in raw_payloads.items():
        digest = _sha(raw)
        blob_hashes[name] = digest
        relative = f"blobs/{digest}.blob"
        (bundle / relative).write_bytes(raw)
        blob_entries.append(
            FrozenBlobManifestEntry(
                relative_path=relative,
                sha256=digest,
                size_bytes=len(raw),
                media_type="application/json",
            )
        )

    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Fixed by #2",
        raw_blob_sha256=blob_hashes["seed"],
        relation_depth=0,
        capabilities=("study_scope", "symptom_observation"),
    )
    pull = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Repair pull request",
        raw_blob_sha256=blob_hashes["pr"],
        relation_depth=1,
        metadata={"merged": True, "merge_commit_sha": _COMMIT},
    )
    code = _node(
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts",
        content="- old\n+ new",
        raw_blob_sha256=blob_hashes["code"],
        relation_depth=1,
        intrinsic_parent_node_id=pull.node_id,
        immutable_ref=_COMMIT,
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
    )
    edges = (
        _edge(source=seed, target=pull, relation="direct_reference", proof=seed),
        _edge(
            source=pull, target=code, relation="intrinsic_materialization", proof=pull
        ),
    )
    graph_payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, pull, code)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    graph = FrozenRecordGraph.model_validate(graph_payload)

    policy = CapturePolicy(
        schema_version="ase-frozen-capture-policy-v1",
        policy_id="synthetic-one-hop-v1",
        domain="synthetic",
        allowed_repositories=(_REPOSITORY,),
        endpoint_allowlist=("api.github.com",),
        max_relation_depth=1,
        allow_redirects=False,
        runtime_network_forbidden=True,
        intrinsic_node_types=(
            FrozenNodeType.MERGE_COMMIT,
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
            FrozenNodeType.REPAIR_SUMMARY,
        ),
    )
    policy_bytes = canonical_json_bytes(policy.model_dump(mode="json"))
    policy_path = bundle / "capture_policy.json"
    policy_path.write_bytes(policy_bytes)

    graphs_bytes = canonical_json_bytes(graph.model_dump(mode="json")) + b"\n"
    graphs_path = bundle / "record_graphs.jsonl"
    graphs_path.write_bytes(graphs_bytes)
    sorted_blobs = tuple(sorted(blob_entries, key=lambda item: item.sha256))
    manifest_payload = {
        "schema_version": "ase-frozen-evidence-bundle-v1",
        "status": "active",
        "domain": "synthetic",
        "bundle_id": "synthetic-v1",
        "split_id": "synthetic-split-v1",
        "split_manifest_sha256": "b" * 64,
        "runner_cohort_sha256": "c" * 64,
        "record_ids_sha256": graph_module.canonical_record_ids_sha256((_RECORD_ID,)),
        "record_count": 1,
        "capture_policy": {
            "relative_path": "capture_policy.json",
            "sha256": _sha(policy_bytes),
        },
        "record_graphs": {
            "relative_path": "record_graphs.jsonl",
            "sha256": _sha(graphs_bytes),
        },
        "repo_allowlist_sha256": graph_module.canonical_string_set_sha256(
            (_REPOSITORY,)
        ),
        "reserved_entity_denylist_sha256": "d" * 64,
        "capture_started_at": _RETRIEVED_AT,
        "capture_finished_at": "2026-08-16T08:01:00+00:00",
        "retrieval_tool": {
            "name": "synthetic-fake-capture",
            "version": "v1",
            "module_sha256": "e" * 64,
        },
        "blobs": [entry.model_dump(mode="json") for entry in sorted_blobs],
        "network_policy": "offline_capture_only_runtime_network_forbidden",
        "gold_accessed": False,
    }
    manifest_payload["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest_payload["capture_policy"]["sha256"],
        record_graphs_sha256=manifest_payload["record_graphs"]["sha256"],
        blobs=sorted_blobs,
    )
    manifest_path = bundle / "frozen_evidence_manifest.json"
    manifest_bytes = canonical_json_bytes(manifest_payload)
    manifest_path.write_bytes(manifest_bytes)
    return root, _sha(manifest_bytes)


def _register_synthetic(monkeypatch: pytest.MonkeyPatch, manifest_sha: str) -> None:
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType(
            {
                "synthetic": (
                    "Benchmark/configs/frozen_evidence/synthetic-v1/"
                    "frozen_evidence_manifest.json",
                    manifest_sha,
                )
            }
        ),
    )


def test_registered_ase2022_target_bundle_loads_from_fixed_trust_root() -> None:
    bundle = load_bound_frozen_evidence_graph_for_domain("ase2022")

    assert bundle.bundle_id == "ase2022-dev50-r1-r3-harm-v2"
    assert len(bundle.record_ids) == 14
    assert bundle.manifest.gold_accessed is False
    assert bundle.policy.schema_version == "ase-frozen-capture-policy-v2"
    assert bundle.policy.policy_id == "github-one-hop-maintainer-pointer-v2"
    assert bundle.policy.max_maintainer_references_per_record == 8
    assert bundle.policy.runtime_network_forbidden is True
    assert all(graph.capture_status == "captured" for graph in bundle.graphs.values())
    assert (
        sum(
            node.node_type is FrozenNodeType.MAINTAINER_POINTER
            for graph in bundle.graphs.values()
            for node in graph.nodes
        )
        == 7
    )
    assert any(
        node.node_type is FrozenNodeType.CHANGED_CODE
        for graph in bundle.graphs.values()
        for node in graph.nodes
    )


def test_capture_policy_v2_requires_explicit_maintainer_reference_cap() -> None:
    common = {
        "policy_id": "github-one-hop-maintainer-pointer-v2",
        "domain": "synthetic",
        "allowed_repositories": (_REPOSITORY,),
        "endpoint_allowlist": ("api.github.com",),
        "max_relation_depth": 1,
        "allow_redirects": False,
        "runtime_network_forbidden": True,
        "intrinsic_node_types": (
            FrozenNodeType.MERGE_COMMIT,
            FrozenNodeType.CHANGED_CODE,
            FrozenNodeType.REGRESSION_TEST,
            FrozenNodeType.REPAIR_SUMMARY,
        ),
    }

    policy = CapturePolicy(
        schema_version="ase-frozen-capture-policy-v2",
        max_maintainer_references_per_record=8,
        **common,
    )
    assert policy.max_maintainer_references_per_record == 8

    with pytest.raises(ValueError, match="max_maintainer_references"):
        CapturePolicy(
            schema_version="ase-frozen-capture-policy-v2",
            **common,
        )

    legacy = CapturePolicy(
        schema_version="ase-frozen-capture-policy-v1",
        **{**common, "policy_id": "synthetic-one-hop-v1"},
    )
    assert "max_maintainer_references_per_record" not in legacy.model_dump(mode="json")


def _rewrite_graph_and_reseal_manifest(root: Path, mutate: object) -> str:
    bundle = root / "Benchmark/configs/frozen_evidence/synthetic-v1"
    graph_path = bundle / "record_graphs.jsonl"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(graph_payload)
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    graph_bytes = canonical_json_bytes(graph_payload) + b"\n"
    graph_path.write_bytes(graph_bytes)

    manifest_path = bundle / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_graphs"]["sha256"] = _sha(graph_bytes)
    blob_entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=blob_entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return _sha(manifest_bytes)


def test_bound_loader_accepts_complete_content_addressed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_sha = _write_valid_bundle(tmp_path)
    _register_synthetic(monkeypatch, manifest_sha)

    loaded = load_bound_frozen_evidence_graph_for_domain(
        "synthetic", repository_root=root
    )

    assert loaded.domain == "synthetic"
    assert loaded.record_ids == (_RECORD_ID,)
    assert loaded.graphs[_RECORD_ID].repository == _REPOSITORY
    assert len(loaded.graphs[_RECORD_ID].nodes) == 3
    assert loaded.trust_manifest_sha256 == manifest_sha
    assert loaded.trust_manifest_relative_path.endswith("frozen_evidence_manifest.json")


def test_bound_loader_has_no_caller_manifest_or_hash_override() -> None:
    with pytest.raises(TypeError):
        load_bound_frozen_evidence_graph_for_domain(  # type: ignore[call-arg]
            "ase2022",
            manifest_path="caller.json",
            expected_manifest_sha256="0" * 64,
        )


def test_loaded_bundle_is_deep_frozen_and_json_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_sha = _write_valid_bundle(tmp_path)
    _register_synthetic(monkeypatch, manifest_sha)
    loaded = load_bound_frozen_evidence_graph_for_domain(
        "synthetic", repository_root=root
    )

    with pytest.raises(TypeError):
        loaded.graphs[_RECORD_ID] = loaded.graphs[_RECORD_ID]  # type: ignore[index]
    with pytest.raises(Exception, match="frozen"):
        loaded.graphs[_RECORD_ID].repository = "other/repo"
    serialized = loaded.model_dump(mode="json")
    assert json.loads(json.dumps(serialized, ensure_ascii=False))["record_ids"] == [
        _RECORD_ID
    ]


def test_maintainer_resolution_requires_loader_verifiable_authority_metadata() -> None:
    raw_sha = "f" * 64

    with pytest.raises(ValueError, match="author_association"):
        _node(
            node_type=FrozenNodeType.MAINTAINER_RESOLUTION,
            canonical_uri="https://github.com/owner/repo/issues/1",
            content="Maintainer identifies the defect mechanism.",
            raw_blob_sha256=raw_sha,
            relation_depth=0,
            authority=EvidenceAuthority.DIRECT,
            capabilities=("defect_mechanism",),
        )


def test_intrinsic_materialization_proof_must_be_its_parent_node() -> None:
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Fixed by #2",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    pull = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Repair pull request",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
    )
    code = _node(
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts",
        content="- old\n+ new",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
        intrinsic_parent_node_id=pull.node_id,
        immutable_ref=_COMMIT,
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
    )
    edges = (
        _edge(source=seed, target=pull, relation="direct_reference", proof=seed),
        _edge(
            source=pull,
            target=code,
            relation="intrinsic_materialization",
            proof=seed,
        ),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, pull, code)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="relation proof"):
        FrozenRecordGraph.model_validate(payload)


def test_source_type_capability_ceiling_cannot_be_expanded_by_artifact() -> None:
    with pytest.raises(ValueError, match="capabilit"):
        _node(
            node_type=FrozenNodeType.CHANGED_CODE,
            canonical_uri=f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts",
            content="- old\n+ new",
            raw_blob_sha256="f" * 64,
            relation_depth=1,
            intrinsic_parent_node_id="feg-node-parent",
            immutable_ref=_COMMIT,
            authority=EvidenceAuthority.DIRECT,
            capabilities=("defect_mechanism", "study_scope"),
        )


def test_frozen_evidence_contracts_are_publicly_exported() -> None:
    assert public_api.CapturePolicy is CapturePolicy
    assert public_api.FrozenGraphNode is FrozenGraphNode
    assert (
        public_api.load_bound_frozen_evidence_graph_for_domain
        is load_bound_frozen_evidence_graph_for_domain
    )


def test_pr_intrinsic_code_must_match_the_frozen_merge_commit() -> None:
    other_commit = "b" * 40
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Fixed by #2",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    pull = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Repair pull request",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
        metadata={"merged": True, "merge_commit_sha": _COMMIT},
    )
    code = _node(
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{other_commit}/src/model.ts",
        content="- old\n+ new",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
        intrinsic_parent_node_id=pull.node_id,
        immutable_ref=other_commit,
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
    )
    edges = (
        _edge(source=seed, target=pull, relation="direct_reference", proof=seed),
        _edge(
            source=pull, target=code, relation="intrinsic_materialization", proof=pull
        ),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, pull, code)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="merge commit"):
        FrozenRecordGraph.model_validate(payload)


def test_unmerged_pr_cannot_own_intrinsic_repair_nodes() -> None:
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Related to #2",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    pull = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Unmerged proposal",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
        metadata={"merged": False, "merge_commit_sha": None},
    )
    code = _node(
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts",
        content="- old\n+ new",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
        intrinsic_parent_node_id=pull.node_id,
        immutable_ref=_COMMIT,
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
    )
    edges = (
        _edge(source=seed, target=pull, relation="direct_reference", proof=seed),
        _edge(
            source=pull, target=code, relation="intrinsic_materialization", proof=pull
        ),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, pull, code)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="merged pull request"):
        FrozenRecordGraph.model_validate(payload)


def test_remote_update_cannot_postdate_retrieval() -> None:
    with pytest.raises(ValueError, match="postdate"):
        FrozenGraphNode(
            **{
                **_node(
                    node_type=FrozenNodeType.SEED_ISSUE,
                    canonical_uri="https://github.com/owner/repo/issues/1",
                    content="Issue body",
                    raw_blob_sha256="f" * 64,
                    relation_depth=0,
                ).model_dump(mode="python"),
                "remote_updated_at": "2026-08-17T00:00:00Z",
            }
        )


def test_node_retrieval_time_must_fall_inside_capture_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)

    def move_retrieval_after_capture(payload: dict[str, object]) -> None:
        nodes = payload["nodes"]
        assert isinstance(nodes, list)
        assert isinstance(nodes[0], dict)
        nodes[0]["retrieved_at"] = "2026-08-16T09:00:00+00:00"

    manifest_sha = _rewrite_graph_and_reseal_manifest(
        root, move_retrieval_after_capture
    )
    _register_synthetic(monkeypatch, manifest_sha)

    with pytest.raises(ValueError, match="capture window"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_pr_merge_metadata_must_match_the_bound_raw_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)
    bundle = root / "Benchmark/configs/frozen_evidence/synthetic-v1"
    graph_path = bundle / "record_graphs.jsonl"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    pull = next(
        node
        for node in graph_payload["nodes"]
        if node["node_type"] == FrozenNodeType.LINKED_PULL_REQUEST.value
    )
    old_blob_sha = pull["raw_blob_sha256"]
    forged_raw = canonical_json_bytes(
        {
            "number": 2,
            "title": "Repair pull request",
            "merged": False,
            "merge_commit_sha": None,
            "canonical_uri": "https://github.com/owner/repo/pull/2",
            "repository": _REPOSITORY,
            "updated_at": "2026-08-15T12:00:00Z",
        }
    )
    forged_sha = _sha(forged_raw)
    forged_relative = f"blobs/{forged_sha}.blob"
    (bundle / forged_relative).write_bytes(forged_raw)
    pull["raw_blob_sha256"] = forged_sha
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    graph_bytes = canonical_json_bytes(graph_payload) + b"\n"
    graph_path.write_bytes(graph_bytes)

    manifest_path = bundle / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["blobs"]:
        if entry["sha256"] == old_blob_sha:
            entry.update(
                {
                    "relative_path": forged_relative,
                    "sha256": forged_sha,
                    "size_bytes": len(forged_raw),
                }
            )
    manifest["blobs"].sort(key=lambda entry: entry["sha256"])
    manifest["record_graphs"]["sha256"] = _sha(graph_bytes)
    blob_entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=blob_entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    _register_synthetic(monkeypatch, _sha(manifest_bytes))

    with pytest.raises(ValueError, match="raw pull request provenance"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_seed_resolution_relation_proof_must_be_the_resolution() -> None:
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Maintainer replied below.",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    resolution = _node(
        node_type=FrozenNodeType.MAINTAINER_RESOLUTION,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="The fault comes from the converter implementation.",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
        authority=EvidenceAuthority.DIRECT,
        capabilities=("defect_mechanism",),
        metadata={"author_association": "MEMBER"},
    )
    edge = _edge(
        source=seed,
        target=resolution,
        relation="seed_resolution",
        proof=seed,
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, resolution)],
        "edges": [edge.model_dump(mode="json")],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="resolution proof"):
        FrozenRecordGraph.model_validate(payload)


def test_direct_reference_cannot_point_back_to_the_seed_entity() -> None:
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Self reference",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    duplicate_entity = _node(
        node_type=FrozenNodeType.LINKED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="A second rendering of the seed",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
    )
    edge = _edge(
        source=seed,
        target=duplicate_entity,
        relation="direct_reference",
        proof=seed,
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, duplicate_entity)],
        "edges": [edge.model_dump(mode="json")],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="seed entity"):
        FrozenRecordGraph.model_validate(payload)


def test_relation_proof_locator_must_be_a_canonical_relation_tuple() -> None:
    proof_sha = "f" * 64
    edge_id = frozen_graph_edge_id(
        record_id=_RECORD_ID,
        from_node_id="feg-node-source",
        to_node_id="feg-node-target",
        relation="direct_reference",
        relation_proof_node_id="feg-node-source",
        relation_proof_locator="arbitrary non-parsable locator",
        relation_proof_sha256=proof_sha,
    )

    with pytest.raises(ValueError, match="canonical relation tuple"):
        FrozenGraphEdge(
            edge_id=edge_id,
            record_id=_RECORD_ID,
            from_node_id="feg-node-source",
            to_node_id="feg-node-target",
            relation="direct_reference",
            relation_proof_node_id="feg-node-source",
            relation_proof_locator="arbitrary non-parsable locator",
            relation_proof_sha256=proof_sha,
        )


def test_loader_rejects_node_content_not_extracted_from_bound_raw_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)

    def forge_normalized_content(payload: dict[str, object]) -> None:
        nodes = payload["nodes"]
        edges = payload["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        code = next(
            node
            for node in nodes
            if node["node_type"] == FrozenNodeType.CHANGED_CODE.value
        )
        old_id = code["node_id"]
        code["content"] = "Invented mechanism absent from raw blob"
        code["content_sha256"] = _sha(code["content"])
        code["node_id"] = frozen_graph_node_id(
            record_id=code["record_id"],
            node_type=code["node_type"],
            canonical_uri=code["canonical_uri"],
            immutable_ref=code["immutable_ref"],
            content_sha256=code["content_sha256"],
        )
        edge = next(item for item in edges if item["to_node_id"] == old_id)
        edge["to_node_id"] = code["node_id"]
        locator = json.loads(edge["relation_proof_locator"])
        locator["to_node_id"] = code["node_id"]
        edge["relation_proof_locator"] = canonical_json_bytes(locator).decode("utf-8")
        edge["edge_id"] = frozen_graph_edge_id(
            record_id=edge["record_id"],
            from_node_id=edge["from_node_id"],
            to_node_id=edge["to_node_id"],
            relation=edge["relation"],
            relation_proof_node_id=edge["relation_proof_node_id"],
            relation_proof_locator=edge["relation_proof_locator"],
            relation_proof_sha256=edge["relation_proof_sha256"],
        )

    manifest_sha = _rewrite_graph_and_reseal_manifest(root, forge_normalized_content)
    _register_synthetic(monkeypatch, manifest_sha)

    with pytest.raises(ValueError, match="raw content extraction"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_graph_rejects_duplicate_canonical_entities_with_different_content() -> None:
    raw_sha = "f" * 64
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="References #2 twice",
        raw_blob_sha256=raw_sha,
        relation_depth=0,
    )
    first = _node(
        node_type=FrozenNodeType.LINKED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/2",
        content="First representation",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
    )
    second = _node(
        node_type=FrozenNodeType.LINKED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/2",
        content="Conflicting representation",
        raw_blob_sha256=raw_sha,
        relation_depth=1,
    )
    edges = (
        _edge(source=seed, target=first, relation="direct_reference", proof=seed),
        _edge(source=seed, target=second, relation="direct_reference", proof=seed),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, first, second)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="canonical entity"):
        FrozenRecordGraph.model_validate(payload)


def test_raw_relation_event_must_match_every_canonical_locator_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)

    def forge_relation_target(payload: dict[str, object]) -> None:
        edges = payload["edges"]
        assert isinstance(edges, list)
        edge = next(item for item in edges if item["relation"] == "direct_reference")
        locator = json.loads(edge["relation_proof_locator"])
        locator["reference"] = "https://github.com/owner/repo/pull/999"
        edge["relation_proof_locator"] = canonical_json_bytes(locator).decode("utf-8")
        edge["edge_id"] = frozen_graph_edge_id(
            record_id=edge["record_id"],
            from_node_id=edge["from_node_id"],
            to_node_id=edge["to_node_id"],
            relation=edge["relation"],
            relation_proof_node_id=edge["relation_proof_node_id"],
            relation_proof_locator=edge["relation_proof_locator"],
            relation_proof_sha256=edge["relation_proof_sha256"],
        )

    manifest_sha = _rewrite_graph_and_reseal_manifest(root, forge_relation_target)
    _register_synthetic(monkeypatch, manifest_sha)

    with pytest.raises(ValueError, match="actual edge target"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_relation_reference_must_identify_the_actual_edge_target() -> None:
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Issue body",
        raw_blob_sha256="a" * 64,
        relation_depth=0,
    )
    target = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Merged fix",
        raw_blob_sha256="b" * 64,
        relation_depth=1,
        metadata={"merged": True, "merge_commit_sha": _COMMIT},
    )
    edge_payload = _edge(
        source=seed, target=target, relation="direct_reference", proof=seed
    ).model_dump(mode="json")
    locator = json.loads(edge_payload["relation_proof_locator"])
    locator["reference"] = "https://github.com/owner/repo/pull/999"
    edge_payload["relation_proof_locator"] = canonical_json_bytes(locator).decode(
        "utf-8"
    )
    edge_payload["edge_id"] = frozen_graph_edge_id(
        record_id=edge_payload["record_id"],
        from_node_id=edge_payload["from_node_id"],
        to_node_id=edge_payload["to_node_id"],
        relation=edge_payload["relation"],
        relation_proof_node_id=edge_payload["relation_proof_node_id"],
        relation_proof_locator=edge_payload["relation_proof_locator"],
        relation_proof_sha256=edge_payload["relation_proof_sha256"],
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [seed.model_dump(mode="json"), target.model_dump(mode="json")],
        "edges": [edge_payload],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="actual edge target"):
        FrozenRecordGraph.model_validate(payload)


def test_same_canonical_uri_cannot_change_semantic_node_type() -> None:
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Issue body",
        raw_blob_sha256="a" * 64,
        relation_depth=0,
    )
    linked = _node(
        node_type=FrozenNodeType.LINKED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/2",
        content="Related issue",
        raw_blob_sha256="b" * 64,
        relation_depth=1,
    )
    neighbor = _node(
        node_type=FrozenNodeType.NEIGHBOR_CASE,
        canonical_uri="https://github.com/owner/repo/issues/2",
        content="Related issue",
        raw_blob_sha256="b" * 64,
        relation_depth=1,
        authority=EvidenceAuthority.COUNTER_ONLY,
    )
    edges = (
        _edge(source=seed, target=linked, relation="direct_reference", proof=seed),
        _edge(source=seed, target=neighbor, relation="direct_reference", proof=seed),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, linked, neighbor)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="duplicate canonical entity"):
        FrozenRecordGraph.model_validate(payload)


def test_seed_uri_allows_only_one_maintainer_resolution() -> None:
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Issue body",
        raw_blob_sha256="a" * 64,
        relation_depth=0,
    )
    first = _node(
        node_type=FrozenNodeType.MAINTAINER_RESOLUTION,
        canonical_uri=seed.canonical_uri,
        content="First maintainer diagnosis",
        raw_blob_sha256="b" * 64,
        relation_depth=0,
        authority=EvidenceAuthority.DIRECT,
        metadata={"author_association": "MEMBER"},
    )
    second = _node(
        node_type=FrozenNodeType.MAINTAINER_RESOLUTION,
        canonical_uri=seed.canonical_uri,
        content="Conflicting maintainer diagnosis",
        raw_blob_sha256="c" * 64,
        relation_depth=0,
        authority=EvidenceAuthority.DIRECT,
        metadata={"author_association": "OWNER"},
    )
    edges = (
        _edge(source=seed, target=first, relation="seed_resolution", proof=first),
        _edge(source=seed, target=second, relation="seed_resolution", proof=second),
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, first, second)],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="duplicate canonical entity"):
        FrozenRecordGraph.model_validate(payload)


def test_repair_summary_requires_the_bound_commit_uri() -> None:
    with pytest.raises(ValueError, match="commit URI"):
        _node(
            node_type=FrozenNodeType.REPAIR_SUMMARY,
            canonical_uri="https://github.com/owner/repo/issues/1",
            content="Repair summary",
            raw_blob_sha256="a" * 64,
            relation_depth=1,
            intrinsic_parent_node_id="feg-node-parent",
            immutable_ref=_COMMIT,
            authority=EvidenceAuthority.DIRECT,
        )


@pytest.mark.parametrize("media_type", ["text/plain", "text/x-diff"])
def test_blob_manifest_matches_json_only_loader_contract(media_type: str) -> None:
    with pytest.raises(ValueError):
        FrozenBlobManifestEntry(
            relative_path=f"blobs/{'a' * 64}.blob",
            sha256="a" * 64,
            size_bytes=1,
            media_type=media_type,
        )


@pytest.mark.parametrize(
    "canonical_uri",
    [
        "https://github.com/Owner/Repo/issues/1",
        "https://github.com/owner/repo/issues/01",
        "https://github.com/owner/repo/issues/1/",
    ],
)
def test_node_uri_requires_one_canonical_spelling(canonical_uri: str) -> None:
    with pytest.raises(ValueError, match="canonical URI spelling"):
        _node(
            node_type=FrozenNodeType.LINKED_ISSUE,
            canonical_uri=canonical_uri,
            content="Related issue",
            raw_blob_sha256="a" * 64,
            relation_depth=1,
        )


def test_intrinsic_file_edge_cannot_be_proved_only_by_commit_sha() -> None:
    seed = _node(
        node_type=FrozenNodeType.SEED_ISSUE,
        canonical_uri="https://github.com/owner/repo/issues/1",
        content="Fixed by #2",
        raw_blob_sha256="a" * 64,
        relation_depth=0,
    )
    pull = _node(
        node_type=FrozenNodeType.LINKED_PULL_REQUEST,
        canonical_uri="https://github.com/owner/repo/pull/2",
        content="Repair pull request",
        raw_blob_sha256="b" * 64,
        relation_depth=1,
        metadata={"merged": True, "merge_commit_sha": _COMMIT},
    )
    code = _node(
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{_COMMIT}/src/model.ts",
        content="- old\n+ new",
        raw_blob_sha256="c" * 64,
        relation_depth=1,
        intrinsic_parent_node_id=pull.node_id,
        immutable_ref=_COMMIT,
        authority=EvidenceAuthority.DIRECT,
    )
    edges = (
        _edge(source=seed, target=pull, relation="direct_reference", proof=seed),
        _edge(
            source=pull,
            target=code,
            relation="intrinsic_materialization",
            proof=pull,
        ),
    )
    edge_payloads = [edge.model_dump(mode="json") for edge in edges]
    intrinsic = edge_payloads[1]
    locator = json.loads(intrinsic["relation_proof_locator"])
    locator["json_pointer"] = "/merge_commit_sha"
    locator["reference"] = _COMMIT
    intrinsic["relation_proof_locator"] = canonical_json_bytes(locator).decode("utf-8")
    intrinsic["edge_id"] = frozen_graph_edge_id(
        record_id=intrinsic["record_id"],
        from_node_id=intrinsic["from_node_id"],
        to_node_id=intrinsic["to_node_id"],
        relation=intrinsic["relation"],
        relation_proof_node_id=intrinsic["relation_proof_node_id"],
        relation_proof_locator=intrinsic["relation_proof_locator"],
        relation_proof_sha256=intrinsic["relation_proof_sha256"],
    )
    payload = {
        "schema_version": "ase-frozen-record-graph-v1",
        "domain": "synthetic",
        "record_id": _RECORD_ID,
        "repository": _REPOSITORY,
        "seed_node_id": seed.node_id,
        "nodes": [node.model_dump(mode="json") for node in (seed, pull, code)],
        "edges": edge_payloads,
        "capture_status": "captured",
        "unavailable_reasons": [],
    }
    payload["graph_sha256"] = frozen_record_graph_hash(payload)

    with pytest.raises(ValueError, match="semantic JSON pointer"):
        FrozenRecordGraph.model_validate(payload)


def test_bound_loader_rejects_arbitrary_intrinsic_proof_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)
    bundle = root / "Benchmark/configs/frozen_evidence/synthetic-v1"
    graph_path = bundle / "record_graphs.jsonl"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    pull = next(
        node
        for node in graph_payload["nodes"]
        if node["node_type"] == FrozenNodeType.LINKED_PULL_REQUEST.value
    )
    code = next(
        node
        for node in graph_payload["nodes"]
        if node["node_type"] == FrozenNodeType.CHANGED_CODE.value
    )
    intrinsic = next(
        edge
        for edge in graph_payload["edges"]
        if edge["relation"] == "intrinsic_materialization"
    )

    manifest_path = bundle / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_sha = pull["raw_blob_sha256"]
    old_entry = next(item for item in manifest["blobs"] if item["sha256"] == old_sha)
    raw_pull = json.loads((bundle / old_entry["relative_path"]).read_bytes())
    raw_pull.pop("files")
    raw_pull["unrelated_field"] = code["canonical_uri"]
    forged_raw = canonical_json_bytes(raw_pull)
    forged_sha = _sha(forged_raw)
    forged_relative = f"blobs/{forged_sha}.blob"
    (bundle / forged_relative).write_bytes(forged_raw)
    pull["raw_blob_sha256"] = forged_sha

    locator = json.loads(intrinsic["relation_proof_locator"])
    locator["json_pointer"] = "/unrelated_field"
    intrinsic["relation_proof_locator"] = canonical_json_bytes(locator).decode("utf-8")
    intrinsic["edge_id"] = frozen_graph_edge_id(
        record_id=intrinsic["record_id"],
        from_node_id=intrinsic["from_node_id"],
        to_node_id=intrinsic["to_node_id"],
        relation=intrinsic["relation"],
        relation_proof_node_id=intrinsic["relation_proof_node_id"],
        relation_proof_locator=intrinsic["relation_proof_locator"],
        relation_proof_sha256=intrinsic["relation_proof_sha256"],
    )
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    graph_bytes = canonical_json_bytes(graph_payload) + b"\n"
    graph_path.write_bytes(graph_bytes)

    manifest["blobs"] = [
        (
            {
                **item,
                "sha256": forged_sha,
                "relative_path": forged_relative,
                "size_bytes": len(forged_raw),
            }
            if item["sha256"] == old_sha
            else item
        )
        for item in manifest["blobs"]
    ]
    manifest["blobs"] = sorted(manifest["blobs"], key=lambda item: item["sha256"])
    manifest["record_graphs"]["sha256"] = _sha(graph_bytes)
    blob_entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=blob_entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    _register_synthetic(monkeypatch, _sha(manifest_bytes))

    with pytest.raises(ValueError, match="semantic JSON pointer"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_bound_loader_rejects_noncanonical_raw_blob_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _write_valid_bundle(tmp_path)
    bundle = root / "Benchmark/configs/frozen_evidence/synthetic-v1"
    graph_path = bundle / "record_graphs.jsonl"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    seed = next(
        node
        for node in graph_payload["nodes"]
        if node["node_type"] == FrozenNodeType.SEED_ISSUE.value
    )
    old_sha = seed["raw_blob_sha256"]

    manifest_path = bundle / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_entry = next(item for item in manifest["blobs"] if item["sha256"] == old_sha)
    old_raw = (bundle / old_entry["relative_path"]).read_bytes()
    forged_raw = b" " + old_raw
    forged_sha = _sha(forged_raw)
    forged_relative = f"blobs/{forged_sha}.blob"
    (bundle / forged_relative).write_bytes(forged_raw)

    seed["raw_blob_sha256"] = forged_sha
    graph_payload["graph_sha256"] = frozen_record_graph_hash(graph_payload)
    graph_bytes = canonical_json_bytes(graph_payload) + b"\n"
    graph_path.write_bytes(graph_bytes)

    manifest["blobs"] = [
        (
            {
                **item,
                "sha256": forged_sha,
                "relative_path": forged_relative,
                "size_bytes": len(forged_raw),
            }
            if item["sha256"] == old_sha
            else item
        )
        for item in manifest["blobs"]
    ]
    manifest["blobs"] = sorted(manifest["blobs"], key=lambda item: item["sha256"])
    manifest["record_graphs"]["sha256"] = _sha(graph_bytes)
    blob_entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=blob_entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    _register_synthetic(monkeypatch, _sha(manifest_bytes))

    with pytest.raises(ValueError, match="blob is not canonical JSON"):
        load_bound_frozen_evidence_graph_for_domain("synthetic", repository_root=root)


def test_capability_tuple_cannot_be_narrowed_or_expanded_by_artifact() -> None:
    with pytest.raises(ValueError, match="fixed by its source type"):
        _node(
            node_type=FrozenNodeType.SEED_ISSUE,
            canonical_uri="https://github.com/owner/repo/issues/1",
            content="Issue body",
            raw_blob_sha256="f" * 64,
            relation_depth=0,
            capabilities=("study_scope",),
        )
