from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.splits import (
    GOLD_FIELDS,
    SplitPreparationConfig,
    collect_contaminated_record_ids,
    family_group_key,
    load_runner_cohort,
    load_restricted_gold_for_evaluation,
    near_duplicate_clusters,
    load_split_manifest_for_runner,
    prepare_frozen_stage3_splits,
    strict_text_contamination_scan,
    verify_restricted_gold_security,
    validate_windows_acl_entries,
)


def test_contaminated_dev50_runner_is_active_evidence_only_and_frozen() -> None:
    root = Path("Benchmark/configs/splits/ase2022_stage3_contaminated_dev50")
    cohort = root / "development_runner_cohort.csv"
    taxonomy = Path(
        "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
        "ase2022_issue_only_holdout_taxonomy.json"
    )

    manifest = load_split_manifest_for_runner(
        root / "split_manifest.json",
        cohort_path=cohort,
        domain="ase2022",
        taxonomy_path=taxonomy,
    )
    with cohort.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert manifest["split_kind"] == "contaminated_development"
    assert manifest["splits"]["development"]["count"] == 50
    assert len(rows) == 50
    assert set(reader.fieldnames or ()).isdisjoint(GOLD_FIELDS)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record(
    index: int, *, title: str | None = None, body: str | None = None
) -> dict[str, str]:
    return {
        "record_id": f"ase2022:test:{index:03d}",
        "paper_id": "ase2022_towards_understanding_the_faults_of",
        "source_project": "tensorflow/tfjs",
        "issue_url": f"https://github.com/tensorflow/tfjs/issues/{1000 + index}",
        "title": title or f"Distinct fault report {index}",
        "body": body or f"A unique failure description and reproduction token-{index}.",
        "comments": f"Maintainer diagnosis for case {index}.",
        "created_at": "2021-01-01T00:00:00Z",
        "updated_at": "2021-01-02T00:00:00Z",
        "state": "closed",
        "symptom": "Crash" if index % 2 else "Incorrect Functionality",
        "root_cause": "API Misuse" if index % 3 else "Incorrect Code Logic",
        "decision": "accepted_fault",
        "original_label_json": json.dumps({"secret": index}),
    }


def test_contamination_collection_reads_nested_record_ids_and_hashes_only_sources_with_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    _write_csv(root / "old" / "cohort.csv", [_record(1)])
    (root / "predictions.jsonl").write_text(
        json.dumps({"record_id": "ase2022:test:002", "prediction": "Crash"}) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"record_ids": ["ase2022:test:003", "other-domain:1"]}),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")

    audit = collect_contaminated_record_ids([root], record_id_prefix="ase2022:")

    assert audit.record_ids == (
        "ase2022:test:001",
        "ase2022:test:002",
        "ase2022:test:003",
    )
    assert {Path(item.path).name for item in audit.sources} == {
        "cohort.csv",
        "predictions.jsonl",
        "manifest.json",
    }
    assert all(
        len(item.sha256) == 64 and item.record_id_count >= 1 for item in audit.sources
    )


def test_contamination_scanner_finds_lineage_and_nested_semantic_id_fields(
    tmp_path: Path,
) -> None:
    first = "ase2022_towards_understanding_the_faults_of:0000000000000001"
    second = "ase2022_towards_understanding_the_faults_of:0000000000000002"
    lineage = tmp_path / "stage2_stage3_lineage_repairs.csv"
    lineage.write_text(
        "old_stage3_record_id,new_record_id,reason\n"
        f"{first},{second},lineage repair\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps({"payload": {"recordId": first, "stage3_record_id": second}}),
        encoding="utf-8",
    )

    audit = collect_contaminated_record_ids([tmp_path])
    independent = strict_text_contamination_scan([tmp_path])

    assert set(audit.record_ids) == {first, second} == set(independent.record_ids)
    assert audit.candidate_file_count == 2
    assert audit.matched_source_count == 2
    assert audit.unmatched_candidate_count == 0
    assert audit.parse_failure_count == 0
    assert audit.candidate_files_sha256 == independent.candidate_files_sha256


def test_contamination_provenance_preserves_repository_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_csv(tmp_path / "results" / "cohort.csv", [_record(1)])
    monkeypatch.chdir(tmp_path)

    audit = collect_contaminated_record_ids(
        [Path("results")], record_id_prefix="ase2022:"
    )

    assert audit.sources[0].path == "results/cohort.csv"


def test_text_provenance_hash_is_independent_of_checkout_line_endings(
    tmp_path: Path,
) -> None:
    payload = json.dumps({"record_id": "ase2022:test:001"})
    (tmp_path / "lf.jsonl").write_bytes((payload + "\n").encode("utf-8"))
    (tmp_path / "crlf.jsonl").write_bytes((payload + "\r\n").encode("utf-8"))

    audit = collect_contaminated_record_ids([tmp_path], record_id_prefix="ase2022:")

    assert len({source.sha256 for source in audit.sources}) == 1


def test_family_key_canonicalizes_github_issue_pull_and_commit_urls() -> None:
    assert (
        family_group_key(
            {
                "source_project": "TensorFlow/TFJS",
                "issue_url": "https://github.com/tensorflow/tfjs/issues/42#x",
            }
        )
        == "github:tensorflow/tfjs:issue-or-pull:42"
    )
    assert (
        family_group_key(
            {
                "source_project": "tensorflow/tfjs",
                "issue_url": "https://github.com/tensorflow/tfjs/pull/42",
            }
        )
        == "github:tensorflow/tfjs:issue-or-pull:42"
    )
    assert (
        family_group_key(
            {
                "source_project": "tensorflow/tfjs",
                "issue_url": "https://github.com/tensorflow/tfjs/commit/ABCDEF",
            }
        )
        == "github:tensorflow/tfjs:commit:abcdef"
    )


def test_near_duplicate_cluster_joins_formatting_variants_but_not_distinct_reports() -> (
    None
):
    records = [
        _record(
            1,
            title="Model.load() CRASHES!",
            body="See https://github.com/a/b/issues/9\n```js\nload(x)\n```",
        ),
        _record(
            2,
            title="model load crashes",
            body="See http://github.com/a/b/issues/9/ load(x)",
        ),
        _record(
            3,
            title="Training is unexpectedly slow",
            body="GPU memory grows every epoch.",
        ),
    ]

    clusters = near_duplicate_clusters(records, threshold=0.82)

    assert clusters[records[0]["record_id"]] == clusters[records[1]["record_id"]]
    assert clusters[records[0]["record_id"]] != clusters[records[2]["record_id"]]


def test_prepare_splits_is_deterministic_leakage_free_and_hides_gold(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    contaminated_root = tmp_path / "old-results"
    output_a = tmp_path / "frozen-a_revision1"
    output_b = tmp_path / "frozen-b_revision1"
    rows = [_record(index) for index in range(1, 25)]
    rows[1]["issue_url"] = rows[0]["issue_url"].replace("/issues/", "/pull/")
    rows[3]["title"] = rows[2]["title"].upper()
    rows[3]["body"] = rows[2]["body"]
    _write_csv(source, rows)
    _write_csv(contaminated_root / "old.csv", [rows[4]])
    taxonomy.write_text(
        json.dumps({"symptom": ["Crash"], "root_cause": ["API Misuse"]}),
        encoding="utf-8",
    )

    def prepare(output: Path):
        return prepare_frozen_stage3_splits(
            SplitPreparationConfig(
                source_csv=source,
                taxonomy_path=taxonomy,
                contamination_roots=(contaminated_root,),
                output_dir=output,
                restricted_gold_path=tmp_path / "private-gold" / f"{output.name}.csv",
                split_revision=1,
                seed=20260816,
                validation_min_size=8,
                final_min_size=8,
                near_duplicate_threshold=0.82,
            )
        )

    first = prepare(output_a)
    second = prepare(output_b)

    manifest_a = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest_b = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest_a["splits"] == manifest_b["splits"]
    validation_ids = set(manifest_a["splits"]["validation"]["record_ids"])
    final_ids = set(manifest_a["splits"]["final"]["record_ids"])
    assert len(validation_ids) >= 8
    assert len(final_ids) >= 8
    assert not validation_ids & final_ids
    assert rows[4]["record_id"] not in validation_ids | final_ids
    assert manifest_a["leakage_audit"] == {
        "contaminated_id_intersection_count": 0,
        "contaminated_family_intersection_count": 0,
        "contaminated_near_duplicate_cluster_intersection_count": 0,
        "cross_split_family_intersection_count": 0,
        "cross_split_near_duplicate_cluster_intersection_count": 0,
    }
    assert manifest_a["label_access_policy"]["runner_may_load_gold"] is False
    assert (
        manifest_a["label_access_policy"]["restricted_gold_sha256"]
        == hashlib.sha256(first.restricted_gold_path.read_bytes()).hexdigest()
    )
    serialized_manifest = json.dumps(manifest_a)
    assert '"symptom":' not in serialized_manifest
    assert '"root_cause":' not in serialized_manifest
    assert "gold_path" not in serialized_manifest

    validation_rows = load_runner_cohort(first.validation_cohort_path)
    assert validation_rows
    assert all(not (GOLD_FIELDS & row.keys()) for row in validation_rows)
    assert all("original_label_json" not in row for row in validation_rows)
    with first.restricted_gold_path.open(encoding="utf-8", newline="") as handle:
        gold_rows = list(csv.DictReader(handle))
    assert {"record_id", "split", "symptom", "root_cause"} == set(gold_rows[0])


def test_prepare_splits_fails_closed_if_any_target_exists(tmp_path: Path) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    output = tmp_path / "frozen"
    _write_csv(source, [_record(index) for index in range(1, 7)])
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["API Misuse", "Incorrect Code Logic"],
            }
        ),
        encoding="utf-8",
    )
    output.mkdir()
    sentinel = output / "split_manifest.json"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_frozen_stage3_splits(
            SplitPreparationConfig(
                source_csv=source,
                taxonomy_path=taxonomy,
                contamination_roots=(),
                output_dir=output,
                restricted_gold_path=tmp_path / "private-gold" / "gold.csv",
                split_revision=1,
                validation_min_size=2,
                final_min_size=2,
            )
        )

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


def test_loader_rejects_revoked_split_sidecar(tmp_path: Path) -> None:
    cohort = tmp_path / "validation_runner_cohort.csv"
    _write_csv(cohort, [_record(1)])
    (tmp_path / "split_manifest.json").write_text(
        json.dumps(
            {
                "status": "revoked",
                "revocation": {"reason": "historical lineage IDs were missed"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revoked"):
        load_runner_cohort(cohort)


def test_restricted_gold_must_be_outside_repository(tmp_path: Path) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    _write_csv(source, [_record(index) for index in range(1, 7)])
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["API Misuse", "Incorrect Code Logic"],
            }
        ),
        encoding="utf-8",
    )
    in_repository_gold = Path("tmp") / "forbidden-split-gold.csv"

    with pytest.raises(ValueError, match="outside the repository"):
        prepare_frozen_stage3_splits(
            SplitPreparationConfig(
                source_csv=source,
                taxonomy_path=taxonomy,
                contamination_roots=(),
                output_dir=tmp_path / "frozen",
                restricted_gold_path=in_repository_gold,
                split_revision=1,
                validation_min_size=2,
                final_min_size=2,
            )
        )

    assert not in_repository_gold.exists()


def test_generated_gold_passes_platform_security_check(tmp_path: Path) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    _write_csv(source, [_record(index) for index in range(1, 7)])
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["API Misuse", "Incorrect Code Logic"],
            }
        ),
        encoding="utf-8",
    )
    artifacts = prepare_frozen_stage3_splits(
        SplitPreparationConfig(
            source_csv=source,
            taxonomy_path=taxonomy,
            contamination_roots=(),
            output_dir=tmp_path / "frozen",
            restricted_gold_path=tmp_path.parent / f"{tmp_path.name}-sealed-gold.csv",
            split_revision=1,
            validation_min_size=2,
            final_min_size=2,
        )
    )

    security = verify_restricted_gold_security(artifacts.restricted_gold_path)

    assert security["verified"] is True
    assert security["policy_version"] == "owner_only_acl_v2"
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    evaluator_rows = load_restricted_gold_for_evaluation(
        artifacts.restricted_gold_path,
        expected_sha256=manifest["label_access_policy"]["restricted_gold_sha256"],
    )
    assert len(evaluator_rows) == 4
    assert set(evaluator_rows[0]) == {"record_id", "split", "symptom", "root_cause"}


def test_revision4_uses_all_strictly_clean_records_without_splitting_units(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    rows = [_record(index) for index in range(1, 69)]
    rows[1]["title"] = rows[0]["title"]
    rows[1]["body"] = rows[0]["body"]
    _write_csv(source, rows)
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": sorted({row["symptom"] for row in rows}),
                "root_cause": sorted({row["root_cause"] for row in rows}),
            }
        ),
        encoding="utf-8",
    )

    artifacts = prepare_frozen_stage3_splits(
        SplitPreparationConfig(
            source_csv=source,
            taxonomy_path=taxonomy,
            contamination_roots=(),
            output_dir=tmp_path / "revision4",
            restricted_gold_path=tmp_path.parent / f"{tmp_path.name}-r4-gold.csv",
            validation_min_size=8,
            final_min_size=60,
            split_revision=4,
        )
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["split_revision"] == 4
    assert manifest["splits"]["validation"]["count"] == 8
    assert manifest["splits"]["final"]["count"] == 60
    assert manifest["pool_audit"]["pool_exhausted"] is True
    assert manifest["pool_audit"]["clean_record_count"] == 68
    assert manifest["pool_audit"]["clean_unit_count"] == 67
    assert manifest["pool_audit"]["unused_record_count"] == 0
    assert manifest["pool_audit"]["selection_uses_gold_labels"] is False
    assert manifest["usage_policy"]["validation_mode"] == "aggregate_pipeline_only"
    assert manifest["usage_policy"]["final_mode"] == "single_use_locked"


def test_windows_acl_allow_list_rejects_unknown_narrow_principal() -> None:
    owner = "S-1-5-21-1000"
    valid = [
        {"sid": owner, "type": "Allow", "inherited": False},
        {"sid": "S-1-5-18", "type": "Allow", "inherited": False},
        {"sid": "S-1-5-32-544", "type": "Allow", "inherited": False},
    ]
    assert validate_windows_acl_entries(owner_sid=owner, entries=valid)["verified"]

    with pytest.raises(PermissionError, match="unknown allow principal"):
        validate_windows_acl_entries(
            owner_sid=owner,
            entries=[
                *valid,
                {"sid": "S-1-5-21-9999", "type": "Allow", "inherited": False},
            ],
        )
