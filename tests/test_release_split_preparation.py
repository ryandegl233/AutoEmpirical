import csv
import json

import pytest

from Benchmark.scripts import prepare_ase2022_contaminated_dev_split as dev
from Benchmark.scripts import prepare_ase2022_uncontaminated_stage3_splits as holdout


def test_development_reconstruction_uses_only_published_inputs(tmp_path):
    output = tmp_path / "ase2022_stage3_contaminated_dev50"
    dev.main(["--output-dir", str(output)])
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output / "development_runner_cohort.csv").open(encoding="utf-8")))
    assert len(rows) == 50
    assert [row["record_id"] for row in rows] == manifest["splits"]["development"]["record_ids"]
    assert not {"symptom", "root_cause", "decision", "gold"} & set(rows[0])
    assert "baseline_error_categories" not in json.dumps(manifest)
    with pytest.raises(FileExistsError):
        dev.main(["--output-dir", str(output)])


def test_new_holdout_defaults_find_taxonomy_and_exclude_released_development(tmp_path, monkeypatch):
    captured = []
    def capture(config):
        captured.append(config)
        raise RuntimeError("captured before writing")
    monkeypatch.setattr(holdout, "prepare_frozen_stage3_splits", capture)
    with pytest.raises(RuntimeError, match="captured before writing"):
        holdout.main(["--output-dir", str(tmp_path / "split"),
                      "--restricted-gold-output", str(tmp_path / "private/gold.csv"),
                      "--split-revision", "1"])
    config = captured[0]
    assert config.label_taxonomy_path.is_file()
    roots = set(config.contamination_roots)
    assert holdout.REPO_ROOT / "Benchmark/inputs" in roots
    assert holdout.REPO_ROOT / "Benchmark/runs" in roots
    assert holdout.REPO_ROOT / "Benchmark/configs/splits" in roots
