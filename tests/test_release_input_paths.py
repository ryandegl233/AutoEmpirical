import hashlib
import json

import pytest

from Benchmark.scripts.prepare_paper_benchmarks import verify_preparation


def test_relocated_sources_verify_without_original_workstation(tmp_path):
    root = tmp_path / "clone"
    folder = root / "Benchmark/inputs/example"
    folder.mkdir(parents=True)
    source = root / "source.csv"
    source.write_bytes(b"title\nexample\n")
    original_bytes = b"title\r\nexample\r\n"
    manifest = {"domain": "example", "record_ids": [], "annotation_record_ids": [],
                "artifact_sha256": {}, "source_paths": {"stage1": "Z:/missing/source.csv"},
                "source_sha256": {"stage1": hashlib.sha256(original_bytes).hexdigest()}}
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binding = {"schema_version": 1,
               "original_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "sources": {"stage1": {"path": "source.csv", "sha256": manifest["source_sha256"]["stage1"],
                           "normalized_lf_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}
    (folder / "release_paths.json").write_text(json.dumps(binding), encoding="utf-8")
    assert verify_preparation(folder, repository_root=root)["status"] == "PASS"
    source.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_preparation(folder, repository_root=root)
    binding["sources"]["stage1"]["path"] = "../outside.csv"
    (folder / "release_paths.json").write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes repository"):
        verify_preparation(folder, repository_root=root)


def test_release_binding_cannot_silently_replace_original_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "release_paths.json").write_text(json.dumps({
        "schema_version": 1, "original_manifest_sha256": "0" * 64, "sources": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        verify_preparation(tmp_path, repository_root=tmp_path)
