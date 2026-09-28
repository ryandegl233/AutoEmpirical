from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import MappingProxyType

import pytest

import Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph as graph_module
import Benchmark.src.adaptive_empirical_workflow.evidence_capture as capture_module
from Benchmark.src.adaptive_empirical_workflow.domains import build_record_runtime
from Benchmark.scripts.freeze_ase2022_github_evidence import (
    UrllibGitHubTransport,
    main as capture_cli_main,
)
from Benchmark.src.adaptive_empirical_workflow.evidence_capture import (
    CaptureLimitError,
    CaptureLimits,
    CaptureRequest,
    CaptureSecurityError,
    CaptureTransportError,
    GitHubResponse,
    capture_frozen_evidence_bundle,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph import (
    FrozenBlobManifestEntry,
    FrozenNodeType,
    canonical_json_bytes,
    frozen_bundle_merkle_root,
    frozen_graph_edge_id,
    frozen_graph_node_id,
    frozen_record_graph_hash,
    load_bound_frozen_evidence_graph_for_domain,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import (
    authoritative_revision_support,
    load_registered_frozen_evidence_runtime,
    project_frozen_evidence_for_record,
)


_NOW = "2026-08-17T02:00:00+00:00"
_MERGE_SHA = "a" * 40
_DIRECT_SHA = "b" * 40


class FakeTransport:
    """Specific fake for the only external boundary: GitHub HTTPS GET."""

    def __init__(self, responses: dict[str, GitHubResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get_json(self, url: str, *, max_bytes: int) -> GitHubResponse:
        self.calls.append(url)
        try:
            response = self._responses[url]
        except KeyError as error:
            raise AssertionError(f"unexpected transport URL: {url}") from error
        if response.raw_size_bytes > max_bytes:
            raise CaptureLimitError(
                "GitHub raw response bytes exceed the configured bound"
            )
        return response


class ForbiddenTransport:
    def get_json(self, url: str, *, max_bytes: int) -> GitHubResponse:
        raise AssertionError(f"warm capture attempted network access: {url}")


def _response(url: str, payload: object, **kwargs: object) -> GitHubResponse:
    raw_size_bytes = int(
        kwargs.pop("raw_size_bytes", len(canonical_json_bytes(payload)))
    )
    return GitHubResponse(
        status_code=int(kwargs.pop("status_code", 200)),
        final_url=str(kwargs.pop("final_url", url)),
        headers=kwargs.pop("headers", {}),
        payload=payload,
        raw_size_bytes=raw_size_bytes,
    )


def _request(**changes: object) -> CaptureRequest:
    payload: dict[str, object] = {
        "schema_version": "ase-github-capture-request-v1",
        "domain": "ase2022",
        "bundle_id": "fake-one-hop-v1",
        "split_id": "fake-dev",
        "split_manifest_sha256": "1" * 64,
        "runner_cohort_sha256": "2" * 64,
        "reserved_entity_denylist_sha256": "3" * 64,
        "capture_policy_id": "github-one-hop-maintainer-pointer-v2",
        "retrieved_at": _NOW,
        "retrieval_tool_name": "ase-github-evidence-capture",
        "retrieval_tool_version": "1.0.0-test",
        "retrieval_tool_module_sha256": "4" * 64,
        "records": [
            {
                "record_id": "ase2022:owner/repo:issue-1",
                "issue_url": "https://github.com/owner/repo/issues/1",
            }
        ],
    }
    payload.update(changes)
    return CaptureRequest.model_validate(payload)


def _responses() -> dict[str, GitHubResponse]:
    api = "https://api.github.com/repos/owner/repo"
    issue = f"{api}/issues/1"
    timeline = f"{issue}/timeline?per_page=100&page=1"
    comments = f"{issue}/comments?per_page=100&page=1"
    linked_issue = f"{api}/issues/2"
    pull = f"{api}/pulls/3"
    pull_files = f"{pull}/files?per_page=100&page=1"
    merge_commit = f"{api}/commits/{_MERGE_SHA}"
    direct_commit = f"{api}/commits/{_DIRECT_SHA}"
    return {
        issue: _response(
            issue,
            {
                "number": 1,
                "html_url": "https://github.com/owner/repo/issues/1",
                "body": "The service crashes. Related to #2 and PR #3.",
                "updated_at": "2026-08-16T01:00:00Z",
            },
        ),
        timeline: _response(
            timeline,
            [
                {
                    "event": "cross-referenced",
                    "source": {
                        "issue": {
                            "html_url": "https://github.com/owner/repo/issues/2",
                            "pull_request": None,
                        }
                    },
                },
                {
                    "event": "cross-referenced",
                    "source": {
                        "issue": {
                            "html_url": "https://github.com/owner/repo/pull/3",
                            "pull_request": {
                                "url": "https://api.github.com/repos/owner/repo/pulls/3"
                            },
                        }
                    },
                },
                {
                    "event": "committed",
                    "commit_id": _DIRECT_SHA,
                },
            ],
        ),
        comments: _response(
            comments,
            [
                {
                    "id": 9,
                    "html_url": "https://github.com/owner/repo/issues/1#issuecomment-9",
                    "body": "The cache key was not invalidated after mutation.",
                    "author_association": "MEMBER",
                    "created_at": "2026-08-16T02:00:00Z",
                    "updated_at": "2026-08-16T02:00:00Z",
                }
            ],
        ),
        linked_issue: _response(
            linked_issue,
            {
                "number": 2,
                "html_url": "https://github.com/owner/repo/issues/2",
                "body": "A prior report describes stale cache state.",
                "updated_at": "2026-08-15T01:00:00Z",
            },
        ),
        pull: _response(
            pull,
            {
                "number": 3,
                "html_url": "https://github.com/owner/repo/pull/3",
                "title": "Invalidate the cache after mutation",
                "body": "Invalidate stale entries and add a regression test.",
                "merged": True,
                "merge_commit_sha": _MERGE_SHA,
                "updated_at": "2026-08-16T03:00:00Z",
            },
        ),
        pull_files: _response(
            pull_files,
            [
                {
                    "filename": "src/cache.py",
                    "status": "modified",
                    "patch": "- return old\n+ return refreshed",
                },
                {
                    "filename": "tests/test_cache.py",
                    "status": "added",
                    "patch": "+def test_invalidates_after_mutation(): pass",
                },
            ],
        ),
        merge_commit: _response(
            merge_commit,
            {
                "sha": _MERGE_SHA,
                "html_url": f"https://github.com/owner/repo/commit/{_MERGE_SHA}",
                "commit": {
                    "message": "Invalidate stale cache entries\n\nAdd regression coverage.",
                    "committer": {"date": "2026-08-16T03:00:00Z"},
                },
            },
        ),
        direct_commit: _response(
            direct_commit,
            {
                "sha": _DIRECT_SHA,
                "html_url": f"https://github.com/owner/repo/commit/{_DIRECT_SHA}",
                "commit": {
                    "message": "Guard cache generation changes",
                    "committer": {"date": "2026-08-15T03:00:00Z"},
                },
            },
        ),
    }


def _add_linked_issue_responses(
    responses: dict[str, GitHubResponse], *numbers: int
) -> None:
    api = "https://api.github.com/repos/owner/repo"
    for number in numbers:
        url = f"{api}/issues/{number}"
        responses[url] = _response(
            url,
            {
                "number": number,
                "html_url": f"https://github.com/owner/repo/issues/{number}",
                "body": f"Linked issue {number}",
                "updated_at": "2026-08-15T01:00:00Z",
            },
        )


def _capture_at_root(
    root: Path,
    *,
    transport: object,
    cache_dir: Path,
    request: CaptureRequest | None = None,
    limits: CaptureLimits | None = None,
) -> tuple[Path, object]:
    output = root / "Benchmark/configs/frozen_evidence/fake-one-hop-v1"
    result = capture_frozen_evidence_bundle(
        request or _request(),
        output_dir=output,
        cache_dir=cache_dir,
        transport=transport,
        limits=limits or CaptureLimits(),
    )
    return output, result


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_cached_envelope(cache: Path, url: str, mutate: object) -> None:
    index_path = next(
        path
        for path in (cache / "index").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["url"] == url
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    blob_path = cache / "blobs" / f"{index['blob_sha256']}.json"
    envelope = json.loads(blob_path.read_text(encoding="utf-8"))
    mutate(envelope)
    raw = canonical_json_bytes(envelope)
    digest = hashlib.sha256(raw).hexdigest()
    (cache / "blobs" / f"{digest}.json").write_bytes(raw)
    index["blob_sha256"] = digest
    index_path.write_bytes(canonical_json_bytes(index))


def _forge_pointer_raw_and_reseal_bundle(
    output: Path, mutate: object, *, node_type: str = "maintainer_pointer"
) -> str:
    graph_path = output / "record_graphs.jsonl"
    graph = json.loads(graph_path.read_text("utf-8"))
    pointer = next(node for node in graph["nodes"] if node["node_type"] == node_type)
    old_sha = pointer["raw_blob_sha256"]
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    old_entry = next(item for item in manifest["blobs"] if item["sha256"] == old_sha)
    raw = json.loads((output / old_entry["relative_path"]).read_text("utf-8"))
    assert callable(mutate)
    mutate(raw)
    forged_bytes = canonical_json_bytes(raw)
    forged_sha = hashlib.sha256(forged_bytes).hexdigest()
    forged_relative = f"blobs/{forged_sha}.blob"
    (output / forged_relative).write_bytes(forged_bytes)
    pointer["raw_blob_sha256"] = forged_sha

    graph["graph_sha256"] = frozen_record_graph_hash(graph)
    graph_bytes = canonical_json_bytes(graph) + b"\n"
    graph_path.write_bytes(graph_bytes)
    manifest["blobs"] = sorted(
        [
            (
                {
                    **item,
                    "relative_path": forged_relative,
                    "sha256": forged_sha,
                    "size_bytes": len(forged_bytes),
                }
                if item["sha256"] == old_sha
                else item
            )
            for item in manifest["blobs"]
        ],
        key=lambda item: item["sha256"],
    )
    manifest["record_graphs"]["sha256"] = hashlib.sha256(graph_bytes).hexdigest()
    entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _downgrade_pointer_bundle_policy_to_v1(output: Path) -> str:
    policy_path = output / "capture_policy.json"
    policy = json.loads(policy_path.read_text("utf-8"))
    policy["schema_version"] = "ase-frozen-capture-policy-v1"
    policy["policy_id"] = "github-one-hop-v1"
    policy.pop("max_maintainer_references_per_record")
    policy_bytes = canonical_json_bytes(policy)
    policy_path.write_bytes(policy_bytes)

    manifest_path = output / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["capture_policy"]["sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    entries = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=entries,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _mutate_bundle_and_reseal(output: Path, mutate: object) -> str:
    graph_path = output / "record_graphs.jsonl"
    graph = json.loads(graph_path.read_text("utf-8"))
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    entries = {item["sha256"]: item for item in manifest["blobs"]}
    raw_by_node_id = {
        node["node_id"]: json.loads(
            (output / entries[node["raw_blob_sha256"]]["relative_path"]).read_text(
                "utf-8"
            )
        )
        for node in graph["nodes"]
    }
    assert callable(mutate)
    mutate(graph, raw_by_node_id)

    rebuilt_entries: dict[str, dict[str, object]] = {}
    for node in graph["nodes"]:
        raw_bytes = canonical_json_bytes(raw_by_node_id[node["node_id"]])
        digest = hashlib.sha256(raw_bytes).hexdigest()
        relative = f"blobs/{digest}.blob"
        (output / relative).write_bytes(raw_bytes)
        node["raw_blob_sha256"] = digest
        rebuilt_entries[digest] = {
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": len(raw_bytes),
            "media_type": "application/json",
        }
    graph["graph_sha256"] = frozen_record_graph_hash(graph)
    graph_bytes = canonical_json_bytes(graph) + b"\n"
    graph_path.write_bytes(graph_bytes)
    manifest["blobs"] = [rebuilt_entries[key] for key in sorted(rebuilt_entries)]
    manifest["record_graphs"]["sha256"] = hashlib.sha256(graph_bytes).hexdigest()
    blob_models = tuple(
        FrozenBlobManifestEntry.model_validate(item) for item in manifest["blobs"]
    )
    manifest["bundle_merkle_root"] = frozen_bundle_merkle_root(
        capture_policy_sha256=manifest["capture_policy"]["sha256"],
        record_graphs_sha256=manifest["record_graphs"]["sha256"],
        blobs=blob_models,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


def _reseal_edge(edge: dict[str, object]) -> None:
    edge["edge_id"] = frozen_graph_edge_id(
        record_id=str(edge["record_id"]),
        from_node_id=str(edge["from_node_id"]),
        to_node_id=str(edge["to_node_id"]),
        relation=str(edge["relation"]),
        relation_proof_node_id=str(edge["relation_proof_node_id"]),
        relation_proof_locator=str(edge["relation_proof_locator"]),
        relation_proof_sha256=str(edge["relation_proof_sha256"]),
    )


def test_capture_builds_a_loader_replayable_strict_one_hop_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo-root"
    output, result = _capture_at_root(
        root,
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    relative = manifest_path.relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    bundle = load_bound_frozen_evidence_graph_for_domain(
        "ase2022", repository_root=root
    )
    graph = bundle.graphs["ase2022:owner/repo:issue-1"]
    node_types = {node.node_type for node in graph.nodes}

    assert result.manifest_sha256 == manifest_sha
    assert graph.capture_status == "captured"
    assert node_types == {
        FrozenNodeType.SEED_ISSUE,
        FrozenNodeType.MAINTAINER_POINTER,
        FrozenNodeType.LINKED_ISSUE,
        FrozenNodeType.LINKED_PULL_REQUEST,
        FrozenNodeType.LINKED_COMMIT,
        FrozenNodeType.CHANGED_CODE,
        FrozenNodeType.REGRESSION_TEST,
        FrozenNodeType.REPAIR_SUMMARY,
    }
    assert all(node.relation_depth <= 1 for node in graph.nodes)
    assert sum(edge.relation == "direct_reference" for edge in graph.edges) == 3
    assert (
        sum(edge.relation == "intrinsic_materialization" for edge in graph.edges) == 3
    )
    assert not any(
        node.node_type is FrozenNodeType.MERGE_COMMIT for node in graph.nodes
    )


def test_registered_captured_immutable_descendants_pass_authority_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root,
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    relative = manifest_path.relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )
    registered = load_registered_frozen_evidence_runtime(
        "ase2022", repository_root=str(root)
    )
    projection = project_frozen_evidence_for_record(
        registered, "ase2022:owner/repo:issue-1"
    )
    repair_items = tuple(
        item
        for item in projection.items
        if item.metadata["frozen_node_type"]
        in {"changed_code", "regression_test", "repair_summary"}
    )

    assert {item.metadata["frozen_node_type"] for item in repair_items} == {
        "changed_code",
        "regression_test",
        "repair_summary",
    }
    runtime = build_record_runtime(
        {"record_id": "ase2022:owner/repo:issue-1", "title": "Synthetic issue"},
        taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
        domain="ase2022",
        frozen_evidence_projection=projection,
    )
    view = runtime.ledger.view()
    repair_ids = {item.evidence_id for item in repair_items}
    runtime_repairs = tuple(
        item for item in view.items if item.evidence_id in repair_ids
    )
    assert all(
        authoritative_revision_support(item, view=view) for item in runtime_repairs
    )


def test_warm_capture_is_byte_identical_and_never_calls_transport(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    first_root = tmp_path / "cold"
    second_root = tmp_path / "warm"
    _capture_at_root(first_root, transport=FakeTransport(_responses()), cache_dir=cache)

    _capture_at_root(second_root, transport=ForbiddenTransport(), cache_dir=cache)

    assert _artifact_bytes(first_root) == _artifact_bytes(second_root)


def test_trusted_maintainer_bare_issue_reference_captures_context_pointer(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": ("https://github.com/owner/repo/issues/1#issuecomment-10"),
                "body": "same issue as #2",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )

    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))
    pointer = next(
        node for node in graph["nodes"] if node["node_type"] == "maintainer_pointer"
    )
    linked = next(
        node for node in graph["nodes"] if node["node_type"] == "linked_issue"
    )
    relation = next(
        edge for edge in graph["edges"] if edge["relation"] == "resolution_reference"
    )

    assert pointer["authority"] == "context_only"
    assert "defect_mechanism" not in pointer["evidence_capabilities"]
    assert linked["canonical_uri"] == "https://github.com/owner/repo/issues/2"
    assert linked["authority"] == "context_only"
    assert relation["from_node_id"] == pointer["node_id"]
    assert relation["to_node_id"] == linked["node_id"]
    locator = json.loads(relation["relation_proof_locator"])
    assert locator["json_pointer"] == "/relation_events/0/target_uri"
    policy = json.loads((output / "capture_policy.json").read_text("utf-8"))
    assert policy["schema_version"] == "ase-frozen-capture-policy-v2"
    assert policy["policy_id"] == "github-one-hop-maintainer-pointer-v2"
    assert policy["max_maintainer_references_per_record"] == 8


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("relation", "direct_reference"),
        ("source_token", "#9"),
        ("target_number", 9),
        ("target_repository", "other/repo"),
        ("duplicate_event", None),
    ],
)
def test_loader_rejects_forged_maintainer_pointer_event_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: object,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "body": "same issue as #2",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )

    def mutate(raw: dict[str, object]) -> None:
        events = raw["relation_events"]
        assert isinstance(events, list) and isinstance(events[0], dict)
        if field == "duplicate_event":
            events.append(dict(events[0]))
        else:
            events[0][field] = forged_value

    manifest_sha = _forge_pointer_raw_and_reseal_bundle(output, mutate)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="maintainer pointer"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_reordered_pointer_events_with_synchronized_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_ten = f"{api}/issues/10"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "See #2 and #10", "author_association": "MEMBER"}],
    )
    responses[issue_ten] = _response(
        issue_ten,
        {
            "number": 10,
            "html_url": "https://github.com/owner/repo/issues/10",
            "body": "Issue ten",
            "updated_at": "2026-08-15T01:00:00Z",
        },
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )

    def reorder(graph: dict[str, object], raws: dict[str, dict[str, object]]) -> None:
        nodes = graph["nodes"]
        edges = graph["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        pointer = next(
            node for node in nodes if node["node_type"] == "maintainer_pointer"
        )
        events = raws[pointer["node_id"]]["relation_events"]
        assert isinstance(events, list)
        events.reverse()
        target_indices = {
            event["target_uri"]: index for index, event in enumerate(events)
        }
        for edge in edges:
            if edge["relation"] != "resolution_reference":
                continue
            locator = json.loads(edge["relation_proof_locator"])
            locator["json_pointer"] = (
                f"/relation_events/{target_indices[locator['reference']]}/target_uri"
            )
            edge["relation_proof_locator"] = canonical_json_bytes(locator).decode()
            _reseal_edge(edge)

    manifest_sha = _mutate_bundle_and_reseal(output, reorder)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="canonical.*maintainer|maintainer.*canonical"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_resolution_proof_rebound_to_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "same issue as #2", "author_association": "MEMBER"}],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )

    def rebind(graph: dict[str, object], raws: dict[str, dict[str, object]]) -> None:
        nodes = graph["nodes"]
        edges = graph["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        seed = next(node for node in nodes if node["node_type"] == "seed_issue")
        pointer = next(
            node for node in nodes if node["node_type"] == "maintainer_pointer"
        )
        pointer_events = raws[pointer["node_id"]]["relation_events"]
        raws[seed["node_id"]]["relation_events"] = pointer_events
        edge = next(
            edge for edge in edges if edge["relation"] == "resolution_reference"
        )
        locator = json.loads(edge["relation_proof_locator"])
        locator["relation_proof_node_id"] = seed["node_id"]
        edge["relation_proof_node_id"] = seed["node_id"]
        edge["relation_proof_sha256"] = seed["content_sha256"]
        edge["relation_proof_locator"] = canonical_json_bytes(locator).decode()
        _reseal_edge(edge)

    manifest_sha = _mutate_bundle_and_reseal(output, rebind)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="proof.*pointer|pointer.*proof"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_duplicate_pointer_target_with_synchronized_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "same issue as #2", "author_association": "MEMBER"}],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )

    def duplicate(graph: dict[str, object], raws: dict[str, dict[str, object]]) -> None:
        nodes = graph["nodes"]
        edges = graph["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        pointer = next(
            node for node in nodes if node["node_type"] == "maintainer_pointer"
        )
        events = raws[pointer["node_id"]]["relation_events"]
        assert isinstance(events, list)
        events.append(dict(events[0]))
        original = next(
            edge for edge in edges if edge["relation"] == "resolution_reference"
        )
        duplicate_edge = dict(original)
        locator = json.loads(duplicate_edge["relation_proof_locator"])
        locator["json_pointer"] = "/relation_events/1/target_uri"
        duplicate_edge["relation_proof_locator"] = canonical_json_bytes(
            locator
        ).decode()
        _reseal_edge(duplicate_edge)
        edges.append(duplicate_edge)

    manifest_sha = _mutate_bundle_and_reseal(output, duplicate)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="duplicate|exactly one provenance"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_forged_maintainer_pointer_author_association(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "same issue as #2", "author_association": "MEMBER"}],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )
    manifest_sha = _forge_pointer_raw_and_reseal_bundle(
        output, lambda raw: raw.update({"author_association": "OWNER"})
    )
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="maintainer pointer"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_forged_linked_issue_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "same issue as #2", "author_association": "MEMBER"}],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )
    manifest_sha = _forge_pointer_raw_and_reseal_bundle(
        output,
        lambda raw: raw.update({"number": 9}),
        node_type="linked_issue",
    )
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="linked issue.*number"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_v1_policy_rejects_new_pointer_graph_while_registered_v2_bundle_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "same issue as #2", "author_association": "MEMBER"}],
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )
    manifest_sha = _downgrade_pointer_bundle_policy_to_v1(output)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="v1 capture policy"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)

    monkeypatch.undo()
    registered = load_bound_frozen_evidence_graph_for_domain("ase2022")
    assert registered.policy.schema_version == "ase-frozen-capture-policy-v2"
    assert any(
        node.node_type is FrozenNodeType.MAINTAINER_POINTER
        for graph in registered.graphs.values()
        for node in graph.nodes
    )


def test_trusted_maintainer_merged_pull_reference_materializes_repair(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_three = f"{api}/issues/3"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "body": "Closed by #3",
                "author_association": "OWNER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    responses[issue_three] = _response(
        issue_three,
        {
            "number": 3,
            "html_url": "https://github.com/owner/repo/pull/3",
            "pull_request": {"url": f"{api}/pulls/3"},
            "body": "Repair pull",
            "updated_at": "2026-08-16T03:00:00Z",
        },
    )

    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))
    by_type = {node["node_type"]: node for node in graph["nodes"]}

    assert {
        "maintainer_pointer",
        "linked_pull_request",
        "changed_code",
        "regression_test",
        "repair_summary",
    }.issubset(by_type)
    assert by_type["maintainer_pointer"]["authority"] == "context_only"
    assert by_type["linked_pull_request"]["authority"] == "context_only"
    assert all(
        by_type[node_type]["authority"] == "direct"
        for node_type in ("changed_code", "regression_test", "repair_summary")
    )
    assert any(edge["relation"] == "resolution_reference" for edge in graph["edges"])


def test_pointer_pr_raw_preserves_exact_issue_endpoint_marker(tmp_path: Path) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_three = f"{api}/issues/3"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "Closed by #3", "author_association": "OWNER"}],
    )
    responses[issue_three] = _response(
        issue_three,
        {
            "number": 3,
            "html_url": "https://github.com/owner/repo/pull/3",
            "pull_request": {"url": f"{api}/pulls/3"},
        },
    )
    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))
    pull = next(
        node for node in graph["nodes"] if node["node_type"] == "linked_pull_request"
    )
    manifest = json.loads((output / "frozen_evidence_manifest.json").read_text("utf-8"))
    entry = next(
        item for item in manifest["blobs"] if item["sha256"] == pull["raw_blob_sha256"]
    )
    raw_pull = json.loads((output / entry["relative_path"]).read_text("utf-8"))

    assert raw_pull["issue_pull_request_url"] == f"{api}/pulls/3"


def test_loader_replays_pointer_pr_issue_marker_after_full_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_three = f"{api}/issues/3"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "Closed by #3", "author_association": "OWNER"}],
    )
    responses[issue_three] = _response(
        issue_three,
        {
            "number": 3,
            "html_url": "https://github.com/owner/repo/pull/3",
            "pull_request": {"url": f"{api}/pulls/3"},
        },
    )
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root, transport=FakeTransport(responses), cache_dir=tmp_path / "cache"
    )

    def forge_marker(
        graph: dict[str, object], raws: dict[str, dict[str, object]]
    ) -> None:
        nodes = graph["nodes"]
        assert isinstance(nodes, list)
        pull = next(
            node for node in nodes if node["node_type"] == "linked_pull_request"
        )
        raws[pull["node_id"]][
            "issue_pull_request_url"
        ] = "https://api.github.com/repos/other/repo/pulls/3"

    manifest_sha = _mutate_bundle_and_reseal(output, forge_marker)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="pull raw issue marker"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_unmaterialized_raw_pull_file_after_full_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root,
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )

    def add_file(graph: dict[str, object], raws: dict[str, dict[str, object]]) -> None:
        nodes = graph["nodes"]
        assert isinstance(nodes, list)
        pull = next(
            node for node in nodes if node["node_type"] == "linked_pull_request"
        )
        files = raws[pull["node_id"]]["files"]
        assert isinstance(files, list)
        files.append(
            {
                "canonical_uri": (
                    f"https://github.com/owner/repo/blob/{_MERGE_SHA}/"
                    "src/unmaterialized.py"
                )
            }
        )

    manifest_sha = _mutate_bundle_and_reseal(output, add_file)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="pull.*files|files.*materialized"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_reordered_pull_files_with_synchronized_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root,
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )

    def reorder_files(
        graph: dict[str, object], raws: dict[str, dict[str, object]]
    ) -> None:
        nodes = graph["nodes"]
        edges = graph["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        pull = next(
            node for node in nodes if node["node_type"] == "linked_pull_request"
        )
        files = raws[pull["node_id"]]["files"]
        assert isinstance(files, list)
        files.reverse()
        target_indices = {
            item["canonical_uri"]: index for index, item in enumerate(files)
        }
        for edge in edges:
            target = next(
                node for node in nodes if node["node_id"] == edge["to_node_id"]
            )
            if target["node_type"] not in {"changed_code", "regression_test"}:
                continue
            locator = json.loads(edge["relation_proof_locator"])
            locator["json_pointer"] = (
                f"/files/{target_indices[locator['reference']]}/canonical_uri"
            )
            edge["relation_proof_locator"] = canonical_json_bytes(locator).decode()
            _reseal_edge(edge)

    manifest_sha = _mutate_bundle_and_reseal(output, reorder_files)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="canonical.*files|files.*canonical"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_loader_rejects_source_path_retyped_as_regression_test_after_full_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    output, _ = _capture_at_root(
        root,
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )

    def retype_source_as_test(
        graph: dict[str, object], raws: dict[str, dict[str, object]]
    ) -> None:
        nodes = graph["nodes"]
        edges = graph["edges"]
        assert isinstance(nodes, list) and isinstance(edges, list)
        node = next(
            node for node in nodes if node["canonical_uri"].endswith("/src/cache.py")
        )
        old_node_id = node["node_id"]
        node["node_type"] = "regression_test"
        node["evidence_capabilities"] = [
            "defect_mechanism",
            "symptom_observation",
        ]
        node["node_id"] = frozen_graph_node_id(
            record_id=node["record_id"],
            node_type=FrozenNodeType.REGRESSION_TEST,
            canonical_uri=node["canonical_uri"],
            immutable_ref=node["immutable_ref"],
            content_sha256=node["content_sha256"],
        )
        raws[node["node_id"]] = raws.pop(old_node_id)
        for edge in edges:
            if edge["to_node_id"] != old_node_id:
                continue
            locator = json.loads(edge["relation_proof_locator"])
            locator["to_node_id"] = node["node_id"]
            edge["to_node_id"] = node["node_id"]
            edge["relation_proof_locator"] = canonical_json_bytes(locator).decode()
            _reseal_edge(edge)

    manifest_sha = _mutate_bundle_and_reseal(output, retype_source_as_test)
    relative = (output / "frozen_evidence_manifest.json").relative_to(root).as_posix()
    monkeypatch.setattr(
        graph_module,
        "_REGISTERED_TRUST_ROOTS",
        MappingProxyType({"ase2022": (relative, manifest_sha)}),
    )

    with pytest.raises(ValueError, match="path.*node type|node type.*path"):
        load_bound_frozen_evidence_graph_for_domain("ase2022", repository_root=root)


def test_maintainer_reference_parser_ignores_code_urls_and_qualified_refs(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "body": (
                    "same issue as #2; `ignore #3`;\n"
                    "```text\nignore #4\n```\n"
                    "https://github.com/owner/repo/issues/5#6 "
                    "other/repo#7 #8suffix \\#9 [jump](#10) #11/evil #0 #03"
                ),
                "author_association": "COLLABORATOR",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    transport = FakeTransport(responses)

    output, _ = _capture_at_root(
        tmp_path / "root", transport=transport, cache_dir=tmp_path / "cache"
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))

    assert f"{api}/issues/2" in transport.calls
    assert all(
        f"{api}/issues/{number}" not in transport.calls
        for number in (3, 4, 5, 6, 7, 8, 9, 10, 11)
    )
    assert (
        sum(edge["relation"] == "resolution_reference" for edge in graph["edges"]) == 1
    )
    assert (
        capture_module._maintainer_bare_references
        is graph_module._maintainer_bare_references
    )


def test_maintainer_reference_parser_ignores_reference_and_html_fragments(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": "https://github.com/owner/repo/issues/1#issuecomment-10",
                "body": (
                    "visible #2\n"
                    "[jump]: #6\n"
                    "[angle]: <#7>\n"
                    '<a href="#8">fragment</a>'
                ),
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    transport = FakeTransport(responses)

    _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    assert f"{api}/issues/2" in transport.calls
    assert all(f"{api}/issues/{number}" not in transport.calls for number in (6, 7, 8))


def test_reference_definition_destination_on_indented_next_line_is_not_fetched(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": "https://github.com/owner/repo/issues/1#issuecomment-10",
                "body": "[same]: #6\n[next]:\n  #7\nvisible #2",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    _add_linked_issue_responses(responses, 6, 7)
    transport = FakeTransport(responses)

    _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    assert f"{api}/issues/2" in transport.calls
    assert all(f"{api}/issues/{number}" not in transport.calls for number in (6, 7))


def test_multiline_html_tags_and_comments_hide_fragment_references(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": "https://github.com/owner/repo/issues/1#issuecomment-10",
                "body": (
                    "<a title=\">\n#6\" href='#7'>fragment</a>\n"
                    "<!-- multiline\n#8\ncomment -->\n"
                    "visible #2"
                ),
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    _add_linked_issue_responses(responses, 6, 7, 8)
    transport = FakeTransport(responses)

    _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    assert f"{api}/issues/2" in transport.calls
    assert all(f"{api}/issues/{number}" not in transport.calls for number in (6, 7, 8))


def test_angle_bracket_comparison_keeps_visible_bare_reference(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": "https://github.com/owner/repo/issues/1#issuecomment-10",
                "body": "compare x < issue #2 > y",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    transport = FakeTransport(responses)

    _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    assert f"{api}/issues/2" in transport.calls


@pytest.mark.parametrize(
    "body",
    (
        "[text][#3]",
        '[text]: /url\n  "title #3"',
        "[text](foo(bar)#3)",
        "<code>#3</code>",
        "<pre>#3</pre>",
    ),
)
def test_hidden_markdown_and_html_reference_spans_do_not_materialize_repair(
    tmp_path: Path,
    body: str,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_three = f"{api}/issues/3"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "html_url": "https://github.com/owner/repo/issues/1#issuecomment-10",
                "body": body,
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    responses[issue_three] = _response(
        issue_three,
        {
            "number": 3,
            "html_url": "https://github.com/owner/repo/pull/3",
            "pull_request": {"url": f"{api}/pulls/3"},
            "body": "Repair pull",
            "updated_at": "2026-08-16T03:00:00Z",
        },
    )
    transport = FakeTransport(responses)

    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))
    repair_types = {
        "linked_pull_request",
        "merge_commit",
        "changed_code",
        "regression_test",
        "repair_summary",
    }

    assert issue_three not in transport.calls
    assert not any(
        edge["relation"] == "resolution_reference" for edge in graph["edges"]
    )
    assert repair_types.isdisjoint(node["node_type"] for node in graph["nodes"])
    assert (
        capture_module._maintainer_bare_references
        is graph_module._maintainer_bare_references
    )


def test_maintainer_reference_cap_fails_closed_before_target_fetch(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "body": "See #2 and #3",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )
    transport = FakeTransport(responses)

    with pytest.raises(CaptureLimitError, match="maintainer reference"):
        _capture_at_root(
            tmp_path / "root",
            transport=transport,
            cache_dir=tmp_path / "cache",
            limits=CaptureLimits(max_maintainer_references_per_record=1),
        )

    assert f"{api}/issues/2" not in transport.calls
    assert f"{api}/issues/3" not in transport.calls


def test_nontrusted_comment_cannot_create_pointer_or_fetch_target(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "Closed by #99", "author_association": "NONE"}],
    )
    transport = FakeTransport(responses)

    output, _ = _capture_at_root(
        tmp_path / "root", transport=transport, cache_dir=tmp_path / "cache"
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))

    assert all(node["node_type"] != "maintainer_pointer" for node in graph["nodes"])
    assert f"{api}/issues/99" not in transport.calls


def test_duplicate_target_is_deduplicated_and_seed_timeline_edge_wins(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    responses[timeline] = _response(
        timeline,
        [
            {
                "event": "cross-referenced",
                "source": {
                    "issue": {"html_url": "https://github.com/owner/repo/issues/2"}
                },
            }
        ],
    )
    responses[comments] = _response(
        comments,
        [
            {
                "id": 10,
                "body": "same issue as #2 and again #2",
                "author_association": "MEMBER",
                "created_at": "2026-08-16T02:00:00Z",
                "updated_at": "2026-08-16T02:00:00Z",
            }
        ],
    )

    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text("utf-8"))

    assert (
        sum(
            node["canonical_uri"] == "https://github.com/owner/repo/issues/2"
            for node in graph["nodes"]
        )
        == 1
    )
    target = next(
        node
        for node in graph["nodes"]
        if node["canonical_uri"] == "https://github.com/owner/repo/issues/2"
    )
    incoming = [
        edge for edge in graph["edges"] if edge["to_node_id"] == target["node_id"]
    ]
    assert len(incoming) == 1
    assert incoming[0]["relation"] == "direct_reference"


def test_pointer_pull_marker_must_bind_exact_same_repository_api_path(
    tmp_path: Path,
) -> None:
    responses = _responses()
    api = "https://api.github.com/repos/owner/repo"
    timeline = f"{api}/issues/1/timeline?per_page=100&page=1"
    comments = f"{api}/issues/1/comments?per_page=100&page=1"
    issue_three = f"{api}/issues/3"
    responses[timeline] = _response(timeline, [])
    responses[comments] = _response(
        comments,
        [{"id": 10, "body": "Closed by #3", "author_association": "OWNER"}],
    )
    responses[issue_three] = _response(
        issue_three,
        {
            "number": 3,
            "html_url": "https://github.com/owner/repo/pull/3",
            "pull_request": {"url": "https://api.github.com/repos/other/repo/pulls/3"},
        },
    )

    with pytest.raises(CaptureSecurityError, match="pull request marker"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
        )


def test_merged_pull_commit_response_must_bind_exact_identity(tmp_path: Path) -> None:
    responses = _responses()
    commit_url = f"https://api.github.com/repos/owner/repo/commits/{_MERGE_SHA}"
    payload = dict(responses[commit_url].payload)
    payload["html_url"] = f"https://github.com/other/repo/commit/{_MERGE_SHA}"
    responses[commit_url] = _response(commit_url, payload)

    with pytest.raises(CaptureSecurityError, match="merge commit response identity"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
        )


@pytest.mark.parametrize(
    "issue_url",
    [
        "http://github.com/owner/repo/issues/1",
        "https://attacker@github.com/owner/repo/issues/1",
        "https://github.com:443/owner/repo/issues/1",
        "https://github.com/owner/repo/issues/1?next=https://evil.invalid",
        "https://github.com/owner/repo/issues/1#fragment",
        "https://api.github.com/repos/owner/repo/issues/1",
        "https://github.com/Owner/repo/issues/1",
        "https://github.com/owner/repo/issues/01",
        "https://github.com/../search/issues/1",
        "https://github.com/owner/../issues/1",
    ],
)
def test_capture_rejects_noncanonical_seed_urls(issue_url: str) -> None:
    # Pydantic deliberately wraps validator ValueErrors at the request boundary.
    with pytest.raises(ValueError, match="canonical GitHub issue URL"):
        _request(
            records=[
                {
                    "record_id": "ase2022:owner/repo:issue-1",
                    "issue_url": issue_url,
                }
            ]
        )


def test_capture_rejects_redirects_without_following_location(tmp_path: Path) -> None:
    responses = _responses()
    issue_api = "https://api.github.com/repos/owner/repo/issues/1"
    responses[issue_api] = _response(
        issue_api,
        {},
        status_code=302,
        headers={"location": "https://evil.invalid/steal"},
    )

    with pytest.raises(CaptureSecurityError, match="redirect"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
        )

    assert not (
        tmp_path
        / "root/Benchmark/configs/frozen_evidence/fake-one-hop-v1/frozen_evidence_manifest.json"
    ).exists()


def test_capture_skips_cross_repository_timeline_targets(tmp_path: Path) -> None:
    responses = _responses()
    timeline = (
        "https://api.github.com/repos/owner/repo/issues/1/"
        "timeline?per_page=100&page=1"
    )
    responses[timeline] = _response(
        timeline,
        [
            {
                "event": "cross-referenced",
                "source": {
                    "issue": {"html_url": "https://github.com/other/repo/issues/9"}
                },
            }
        ],
    )

    transport = FakeTransport(responses)
    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=transport,
        cache_dir=tmp_path / "cache",
    )

    graph_payload = json.loads(
        (output / "record_graphs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert all(
        "other/repo" not in node["canonical_uri"] for node in graph_payload["nodes"]
    )
    assert all("other/repo" not in url for url in transport.calls)


@pytest.mark.parametrize(
    "target",
    (
        "http://github.com/owner/repo/issues/2",
        "https://attacker@github.com/owner/repo/issues/2",
        "https://github.com/owner/repo/issues/02",
        "https://github.com/owner/repo/issues/2?answer=1",
    ),
)
def test_capture_rejects_malformed_timeline_targets(
    tmp_path: Path, target: str
) -> None:
    responses = _responses()
    timeline = (
        "https://api.github.com/repos/owner/repo/issues/1/"
        "timeline?per_page=100&page=1"
    )
    responses[timeline] = _response(
        timeline,
        [
            {
                "event": "cross-referenced",
                "source": {"issue": {"html_url": target}},
            }
        ],
    )

    with pytest.raises(CaptureSecurityError, match="relation target|canonical"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
        )


def test_capture_rejects_dot_segments_in_changed_file_paths(tmp_path: Path) -> None:
    responses = _responses()
    files_url = (
        "https://api.github.com/repos/owner/repo/pulls/3/" "files?per_page=100&page=1"
    )
    responses[files_url] = _response(
        files_url,
        [{"filename": "../secrets.py", "status": "modified", "patch": "+ leak"}],
    )

    with pytest.raises(CaptureSecurityError, match="file path"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
        )


def test_capture_never_expands_relations_from_linked_issues(tmp_path: Path) -> None:
    responses = _responses()
    linked = "https://api.github.com/repos/owner/repo/issues/2"
    responses[linked] = _response(
        linked,
        {
            "number": 2,
            "html_url": "https://github.com/owner/repo/issues/2",
            "body": "This linked issue references #99, which must remain depth two.",
            "updated_at": "2026-08-15T01:00:00Z",
        },
    )
    transport = FakeTransport(responses)

    _capture_at_root(
        tmp_path / "root", transport=transport, cache_dir=tmp_path / "cache"
    )

    assert all("/issues/99" not in url for url in transport.calls)
    assert all("/issues/2/timeline" not in url for url in transport.calls)


def test_unmerged_pull_request_never_materializes_repair_nodes(tmp_path: Path) -> None:
    responses = _responses()
    pull_url = "https://api.github.com/repos/owner/repo/pulls/3"
    pull = dict(responses[pull_url].payload)
    pull.update({"merged": False, "merge_commit_sha": None})
    responses[pull_url] = _response(pull_url, pull)
    transport = FakeTransport(responses)

    output, _ = _capture_at_root(
        tmp_path / "root", transport=transport, cache_dir=tmp_path / "cache"
    )
    graph = json.loads((output / "record_graphs.jsonl").read_text(encoding="utf-8"))
    node_types = {node["node_type"] for node in graph["nodes"]}

    assert node_types.isdisjoint(
        {"changed_code", "regression_test", "repair_summary", "merge_commit"}
    )
    assert all("/pulls/3/files" not in url for url in transport.calls)
    assert all(f"/commits/{_MERGE_SHA}" not in url for url in transport.calls)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (CaptureLimits(max_nodes_per_record=4), "node"),
        (CaptureLimits(max_files_per_pull=1), "file"),
        (CaptureLimits(max_response_bytes=32), "bytes"),
    ],
)
def test_capture_fails_closed_when_a_resource_bound_is_exceeded(
    tmp_path: Path, limits: CaptureLimits, message: str
) -> None:
    with pytest.raises(CaptureLimitError, match=message):
        _capture_at_root(
            tmp_path / message,
            transport=FakeTransport(_responses()),
            cache_dir=tmp_path / f"cache-{message}",
            limits=limits,
        )

    assert not (
        tmp_path / message / "Benchmark/configs/frozen_evidence/fake-one-hop-v1/"
        "frozen_evidence_manifest.json"
    ).exists()


def test_full_page_at_page_limit_fails_instead_of_silently_truncating(
    tmp_path: Path,
) -> None:
    responses = _responses()
    timeline = (
        "https://api.github.com/repos/owner/repo/issues/1/"
        "timeline?per_page=100&page=1"
    )
    responses[timeline] = _response(
        timeline,
        [{"event": "unlabeled"} for _ in range(100)],
        headers={
            "link": (
                "<https://api.github.com/repos/owner/repo/issues/1/timeline?"
                'per_page=100&page=2>; rel="next"'
            )
        },
    )

    with pytest.raises(CaptureLimitError, match="page"):
        _capture_at_root(
            tmp_path / "root",
            transport=FakeTransport(responses),
            cache_dir=tmp_path / "cache",
            limits=CaptureLimits(max_pages_per_endpoint=1),
        )


def test_exactly_full_final_page_without_next_link_is_complete(tmp_path: Path) -> None:
    responses = _responses()
    timeline = (
        "https://api.github.com/repos/owner/repo/issues/1/"
        "timeline?per_page=100&page=1"
    )
    responses[timeline] = _response(
        timeline,
        [{"event": "unlabeled"} for _ in range(100)],
        headers={},
    )

    _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
        limits=CaptureLimits(max_pages_per_endpoint=1),
    )


def test_cache_strips_remote_label_fields_and_still_replays(tmp_path: Path) -> None:
    responses = _responses()
    issue_api = "https://api.github.com/repos/owner/repo/issues/1"
    payload = dict(responses[issue_api].payload)
    payload["labels"] = [{"name": "root cause: cache"}]
    responses[issue_api] = _response(issue_api, payload)
    cache = tmp_path / "cache"

    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(responses), cache_dir=cache
    )
    _capture_at_root(tmp_path / "warm", transport=ForbiddenTransport(), cache_dir=cache)

    assert all(b'"labels"' not in path.read_bytes() for path in cache.rglob("*.json"))


def test_cached_response_blobs_are_addressed_and_verified_by_content(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(_responses()), cache_dir=cache
    )

    blobs = tuple((cache / "blobs").glob("*.json"))
    assert blobs
    assert all(
        path.stem == hashlib.sha256(path.read_bytes()).hexdigest() for path in blobs
    )

    tampered = blobs[0]
    tampered.write_bytes(tampered.read_bytes() + b" ")
    with pytest.raises(CaptureTransportError, match="content hash"):
        _capture_at_root(
            tmp_path / "warm",
            transport=ForbiddenTransport(),
            cache_dir=cache,
        )


def test_warm_cache_cannot_bypass_a_tighter_response_byte_bound(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(_responses()), cache_dir=cache
    )

    with pytest.raises(CaptureLimitError, match="bytes"):
        _capture_at_root(
            tmp_path / "warm",
            transport=ForbiddenTransport(),
            cache_dir=cache,
            limits=CaptureLimits(max_response_bytes=32),
        )


@pytest.mark.parametrize(
    ("url", "mutate", "message"),
    [
        (
            "https://api.github.com/repos/owner/repo/issues/1/"
            "timeline?per_page=100&page=1",
            lambda envelope: envelope.update(
                {
                    "next_url": (
                        "https://api.github.com/repos/other/repo/issues/1/"
                        "timeline?per_page=100&page=2"
                    )
                }
            ),
            "next canonical page",
        ),
        (
            "https://api.github.com/repos/owner/repo/issues/1",
            lambda envelope: envelope.update({"source_size_bytes": -1}),
            "source size",
        ),
        (
            "https://api.github.com/repos/owner/repo/issues/1",
            lambda envelope: envelope["payload"].update(
                {"labels": [{"name": "injected"}]}
            ),
            "forbidden",
        ),
    ],
)
def test_warm_cache_replays_all_cold_path_security_checks(
    tmp_path: Path, url: str, mutate: object, message: str
) -> None:
    cache = tmp_path / "cache"
    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(_responses()), cache_dir=cache
    )
    _rewrite_cached_envelope(cache, url, mutate)

    with pytest.raises((CaptureSecurityError, CaptureTransportError), match=message):
        _capture_at_root(
            tmp_path / "warm",
            transport=ForbiddenTransport(),
            cache_dir=cache,
        )


def test_warm_cache_rejects_a_small_forged_positive_source_size(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(_responses()), cache_dir=cache
    )
    issue_url = "https://api.github.com/repos/owner/repo/issues/1"
    _rewrite_cached_envelope(
        cache,
        issue_url,
        lambda envelope: envelope.update({"source_size_bytes": 1}),
    )

    with pytest.raises(CaptureSecurityError, match="source size"):
        _capture_at_root(
            tmp_path / "warm",
            transport=ForbiddenTransport(),
            cache_dir=cache,
        )


def test_warm_cache_rejects_oversized_blob_before_json_decode(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _capture_at_root(
        tmp_path / "cold", transport=FakeTransport(_responses()), cache_dir=cache
    )
    issue_url = "https://api.github.com/repos/owner/repo/issues/1"
    index_path = next(
        path
        for path in (cache / "index").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["url"] == issue_url
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    oversized = b"x" * 5000
    digest = hashlib.sha256(oversized).hexdigest()
    (cache / "blobs" / f"{digest}.json").write_bytes(oversized)
    index["blob_sha256"] = digest
    index_path.write_bytes(canonical_json_bytes(index))

    with pytest.raises(CaptureLimitError, match="cache blob bytes"):
        _capture_at_root(
            tmp_path / "warm",
            transport=ForbiddenTransport(),
            cache_dir=cache,
            limits=CaptureLimits(max_response_bytes=64),
        )


@pytest.mark.parametrize("forbidden", ["decision", "symptom", "root_cause", "GT"])
def test_capture_request_rejects_label_or_decision_fields(forbidden: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _request(**{forbidden: "do not ingest"})


def test_capture_audit_and_graphs_do_not_copy_model_labels(tmp_path: Path) -> None:
    output, _ = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )
    forbidden = {"answer", "decision", "gold", "gt", "label", "rootcause", "symptom"}
    values = [
        json.loads(line)
        for line in (output / "record_graphs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    values.append(json.loads((output / "capture_audit.json").read_text("utf-8")))
    for value in values:
        stack = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                normalized = {
                    "".join(char for char in str(key).casefold() if char.isalnum())
                    for key in current
                }
                assert normalized.isdisjoint(forbidden)
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)


def test_multi_record_bundle_writes_canonical_jsonl_and_audits_every_graph(
    tmp_path: Path,
) -> None:
    responses = _responses()
    second_issue = "https://api.github.com/repos/owner/repo/issues/5"
    second_timeline = f"{second_issue}/timeline?per_page=100&page=1"
    second_comments = f"{second_issue}/comments?per_page=100&page=1"
    responses.update(
        {
            second_issue: _response(
                second_issue,
                {
                    "number": 5,
                    "html_url": "https://github.com/owner/repo/issues/5",
                    "body": "A second isolated issue.",
                    "updated_at": "2026-08-15T01:00:00Z",
                },
            ),
            second_timeline: _response(second_timeline, []),
            second_comments: _response(second_comments, []),
        }
    )
    request = _request(
        records=[
            {
                "record_id": "ase2022:owner/repo:issue-1",
                "issue_url": "https://github.com/owner/repo/issues/1",
            },
            {
                "record_id": "ase2022:owner/repo:issue-5",
                "issue_url": "https://github.com/owner/repo/issues/5",
            },
        ]
    )

    output, result = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(responses),
        cache_dir=tmp_path / "cache",
        request=request,
    )
    graph_lines = (output / "record_graphs.jsonl").read_bytes().splitlines()
    audit = json.loads((output / "capture_audit.json").read_text("utf-8"))

    assert result.record_count == 2
    assert len(graph_lines) == 2
    assert all(line == canonical_json_bytes(json.loads(line)) for line in graph_lines)
    assert set(audit["graph_sha256s"]) == {
        "ase2022:owner/repo:issue-1",
        "ase2022:owner/repo:issue-5",
    }


def test_existing_active_manifest_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir(parents=True)
    manifest = output / "frozen_evidence_manifest.json"
    manifest.write_bytes(canonical_json_bytes({"existing": "reviewed"}))

    with pytest.raises(FileExistsError, match="active manifest"):
        capture_frozen_evidence_bundle(
            _request(),
            output_dir=output,
            cache_dir=tmp_path / "cache",
            transport=FakeTransport(_responses()),
            limits=CaptureLimits(),
        )

    assert manifest.read_bytes() == b'{"existing":"reviewed"}'


def test_concurrent_capture_cannot_overwrite_an_inflight_bundle(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingTransport(FakeTransport):
        def get_json(self, url: str, *, max_bytes: int) -> GitHubResponse:
            if not started.is_set():
                started.set()
                assert release.wait(timeout=5)
            return super().get_json(url, max_bytes=max_bytes)

    root = tmp_path / "root"
    errors: list[BaseException] = []

    def first_capture() -> None:
        try:
            _capture_at_root(
                root,
                transport=BlockingTransport(_responses()),
                cache_dir=tmp_path / "cache-first",
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=first_capture)
    worker.start()
    assert started.wait(timeout=5)
    try:
        with pytest.raises(FileExistsError, match="capture lock"):
            _capture_at_root(
                root,
                transport=FakeTransport(_responses()),
                cache_dir=tmp_path / "cache-second",
            )
    finally:
        release.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert not errors
    output = root / "Benchmark/configs/frozen_evidence/fake-one-hop-v1"
    assert (output / "frozen_evidence_manifest.json").exists()
    assert not (output / ".capture.lock").exists()


def test_manifest_commit_success_is_not_reclassified_by_temp_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = Path.unlink

    def fail_manifest_temp_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(
            ".frozen_evidence_manifest.json."
        ) and path.name.endswith(".tmp"):
            raise OSError("synthetic post-commit cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_manifest_temp_cleanup)

    output, result = _capture_at_root(
        tmp_path / "root",
        transport=FakeTransport(_responses()),
        cache_dir=tmp_path / "cache",
    )

    manifest = output / "frozen_evidence_manifest.json"
    assert result.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_cli_requires_explicit_network_authorization_before_transport_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"unused": True}), encoding="utf-8")

    def forbidden_factory(token: str | None) -> object:
        raise AssertionError(f"transport constructed without authorization: {token}")

    exit_code = capture_cli_main(
        [
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        transport_factory=forbidden_factory,
    )

    assert exit_code == 2
    assert "--allow-network" in capsys.readouterr().err
    assert not (tmp_path / "bundle/frozen_evidence_manifest.json").exists()


def test_production_transport_bounds_raw_read_before_json_parsing() -> None:
    class OversizedResponse:
        status = 200
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.read_sizes: list[int] = []

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            return b"x" * size

        def geturl(self) -> str:
            return "https://api.github.com/repos/owner/repo/issues/1"

        def __enter__(self) -> "OversizedResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    response = OversizedResponse()

    class FakeOpener:
        def open(self, request: object, *, timeout: float) -> OversizedResponse:
            return response

    transport = UrllibGitHubTransport(None, opener=FakeOpener())

    with pytest.raises(CaptureLimitError, match="raw response bytes"):
        transport.get_json(
            "https://api.github.com/repos/owner/repo/issues/1", max_bytes=16
        )

    assert response.read_sizes == [17]
