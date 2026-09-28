from __future__ import annotations

import importlib
import importlib.util

import pytest


def api():
    name = "Benchmark.src.annotation_contracts"
    assert importlib.util.find_spec(name) is not None, "native annotation contracts are not implemented"
    return importlib.import_module(name)


def taxonomy(mode="multi_label"):
    return {"symptom": ["Crash", "Hang"], "root_cause": ["Logic", "API"],
            "annotation_modes": {"symptom": mode, "root_cause": "single_label"}}


def test_native_multilabel_is_a_set_of_atomic_labels():
    module = api()
    spec = taxonomy()
    assert module.canonical_label(spec, "symptom", ["Hang", "Crash"]) == "Crash || Hang"
    assert module.canonical_label(spec, "symptom", "Hang || Crash") == "Crash || Hang"
    assert module.label_valid(spec, "symptom", "Crash || Hang")
    assert not module.label_valid(spec, "symptom", "Crash || Invented")
    assert not module.label_valid(spec, "symptom", "Crash || Crash")
    assert not module.label_valid(spec, "root_cause", "Logic || API")


def test_free_text_is_not_a_taxonomy_of_gold_sentences():
    module = api()
    spec = {**taxonomy("free_text"), "symptom": []}
    assert module.label_valid(spec, "symptom", "Vehicle drifts away during landing.")
    assert not module.label_valid(spec, "symptom", "  ")
    assert not module.label_valid(spec, "symptom", ["Crash"])


def test_multilabel_metrics_are_order_invariant_and_measure_partial_overlap():
    module = api()
    gold = [{"record_id": "a", "symptom": "Crash || Hang", "root_cause": "Logic"},
            {"record_id": "b", "symptom": "Hang", "root_cause": "API"}]
    predictions = [{"record_id": "a", "symptom": "Hang || Crash", "root_cause": "Logic"},
                   {"record_id": "b", "symptom": "Crash || Hang", "root_cause": "API"}]
    result = module.annotation_metrics(gold, predictions, taxonomy())
    assert result["symptom_set_exact_match"] == .5
    assert result["symptom_micro_precision"] == .75
    assert result["symptom_micro_recall"] == 1
    assert result["root_cause_accuracy"] == 1
    assert result["joint_accuracy"] == .5


@pytest.mark.parametrize("mode", ["free_text", "constant"])
def test_nondiscriminative_symptoms_never_report_classification_accuracy(mode):
    module = api()
    spec = {**taxonomy(mode), "symptom": [] if mode == "free_text" else ["Poor Performance"]}
    symptom = "Vehicle drifts during landing." if mode == "free_text" else "Poor Performance"
    gold = [{"record_id": "a", "symptom": symptom, "root_cause": "Logic"}]
    result = module.annotation_metrics(gold, gold, spec)
    assert result["symptom_accuracy"] is None
    assert result["joint_accuracy"] is None
    assert result["root_cause_accuracy"] == 1


def test_metrics_reject_unknown_extra_prediction_and_duplicate_gold_ids():
    module = api()
    gold = [{"record_id": "a", "symptom": "Crash", "root_cause": "Logic"}]
    with pytest.raises(ValueError, match="duplicate"):
        module.annotation_metrics(gold + gold, [], taxonomy())
    with pytest.raises(ValueError, match="unexpected"):
        module.annotation_metrics(gold, [{"record_id": "b"}], taxonomy())
