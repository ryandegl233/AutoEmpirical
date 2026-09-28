import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from Benchmark.scripts import run_experiment_two as runner


def test_freeze_from_source_only_configuration_after_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "inputs"
    source.mkdir()
    ids = [f"synthetic-{i}" for i in range(48)]
    with (source / "cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "title"])
        writer.writeheader()
        writer.writerows({"record_id": rid, "title": "Synthetic issue"} for rid in ids)
    for name in ("split", "examples", "example_source", "gold"):
        (source / name).write_text("{}", encoding="utf-8")
    (source / "supplemental").write_text(json.dumps({"records": []}), encoding="utf-8")
    paths = {k: k for k in ("split", "examples", "example_source", "gold", "supplemental")}
    paths["cohort"] = "cohort.csv"
    config = {"schema_version": 1, "model": "test-model", "record_ids": ids,
              "inputs": {k: {"path": p, "sha256": runner.digest(source / p)} for k, p in paths.items()}}
    config_path = source / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = SimpleNamespace(input_config=config_path, cases=None, arms=list(runner.ARMS),
                           graph_serialization="graph", case_workers=1)
    protocol = runner.freeze(args)
    assert protocol["record_ids"] == ids
    assert "gold" not in protocol["inputs"]
    assert "gold_path" not in runner.command(protocol, tmp_path / "run", "E00", dry=True)
    assert protocol["gold_path"] not in runner.command(protocol, tmp_path / "run", "E00", dry=True)
    runner.verify_frozen(protocol)
    (source / "cohort.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen code/input changed"):
        runner.verify_frozen(protocol)
    with pytest.raises(ValueError, match="input hash mismatch"):
        runner.freeze(args)


def test_release_evidence_has_relative_images_and_disjoint_examples():
    from pathlib import Path
    from Benchmark.src.adaptive_empirical_workflow.supplemental_evidence import SupplementalEvidenceBundle
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalExplainableTools

    root = Path(__file__).resolve().parents[1] / "Benchmark/inputs/ase2022_dev48"
    config = json.loads((root / "experiment_two.json").read_text(encoding="utf-8"))
    evidence = root / config["inputs"]["supplemental"]["path"]
    bundle = json.loads(evidence.read_text(encoding="utf-8"))
    assert all(not Path(image["path"]).is_absolute() for row in bundle["records"] for image in row["images"])
    SupplementalEvidenceBundle.load(evidence, allowed_record_ids=config["record_ids"])
    examples = json.loads((root / "examples/examples_formal_v2.json").read_text(encoding="utf-8"))
    assert set(config["record_ids"]).isdisjoint(row["record_id"] for row in examples["examples"])
    FormalExplainableTools.from_files(root / "examples/examples_formal_v2.json", root / "examples/example_source.csv")
    from Benchmark.src.adaptive_empirical_workflow.domains import load_domain_inputs
    load_domain_inputs("ase2022", cohort_path=root / "runtime/evaluation48.csv",
                       split_manifest_path=root / "runtime/split_manifest.json")
