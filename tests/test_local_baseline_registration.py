import json

import pytest

from Benchmark.scripts import prepare_baseline_trust_root as prepare
from Benchmark.src.adaptive_empirical_workflow import baseline_anchor as anchors


def test_local_registration_binds_private_baseline_without_publishing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(anchors, "_repository_root", lambda: tmp_path)
    local = tmp_path / "Benchmark/cache"
    local.mkdir(parents=True)
    artifact = local / "synthetic.jsonl"
    artifact.write_text(json.dumps({"record_id": "synthetic-1", "config_hash": "a" * 64,
        "invalid": False, "final_prediction": {"symptom": "Crash", "root_cause": "Logic"}}) + "\n", encoding="utf-8")
    trust = local / "trust.json"
    prepare.register_baseline_trust_root(artifact, output=trust, artifact_id="synthetic-local",
        domain="ase2022", baseline_code_sha256="b" * 64)
    kwargs = dict(trust_manifest_path=trust, expected_domain="ase2022",
                  expected_record_ids=["synthetic-1"], taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]})
    with pytest.raises(ValueError, match="pre-registered"):
        anchors.load_baseline_anchor_bundle(artifact, **kwargs)
    prepare.activate_local_trust_root(trust)
    assert anchors.load_baseline_anchor_bundle(artifact, **kwargs).trust_artifact_id == "synthetic-local"
    manifest = json.loads(trust.read_text(encoding="utf-8"))
    manifest["artifact_id"] = "changed"
    trust.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match registry"):
        anchors.load_baseline_anchor_bundle(artifact, **kwargs)


def test_registration_requires_ignored_local_manifest_location(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Benchmark/cache"):
        prepare.activate_local_trust_root(tmp_path / "public-config.json")
