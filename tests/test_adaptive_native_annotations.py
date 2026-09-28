from __future__ import annotations

import csv
import hashlib
import json
import subprocess

import pytest

from Benchmark.src.adaptive_empirical_workflow.agents import _validate_classification_labels
from Benchmark.src.adaptive_empirical_workflow.capabilities import required_readiness_dimensions
from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceDimension, EvidenceExplicitness, EvidenceItem, EvidenceView, SymptomReport
from Benchmark.src.adaptive_empirical_workflow.domains import load_domain_inputs
from Benchmark.src.adaptive_empirical_workflow.evidence_capabilities import supports_readiness_dimension
from Benchmark.src.adaptive_empirical_workflow.experiment import evaluate_experiment
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import derive_frozen_evidence_view
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger


def test_adaptive_run_identity_binds_new_untracked_paper_codebook(tmp_path):
    from Benchmark.scripts.run_adaptive_empirical_workflow import _code_state
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "--allow-empty", "-qm", "test"], cwd=tmp_path, check=True)
    codebook = tmp_path / "Benchmark/configs/paper_codebooks_v1.json"
    codebook.parent.mkdir(parents=True)
    codebook.write_text('{"definition":"first"}', encoding="utf-8")
    first = _code_state(tmp_path)
    codebook.write_text('{"definition":"second"}', encoding="utf-8")
    assert first["dirty_source_state_sha256"] != _code_state(tmp_path)["dirty_source_state_sha256"]


def spec(mode):
    return {"symptom": [] if mode == "free_text" else ["Crash", "Hang"], "root_cause": ["Logic", "API"],
            "annotation_modes": {"symptom": mode, "root_cause": "single_label"}}


def item():
    content = "The vehicle unexpectedly drifts away during landing."
    return EvidenceItem(evidence_id="issue", record_id="r", source_type="issue_body",
                        source_uri="https://example.test/issues/1", retrieved_at="2026-09-08T00:00:00Z",
                        content=content, content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                        explicitness=EvidenceExplicitness.DIRECT)


@pytest.mark.parametrize("domain,mode,label,alternative", [
    ("icse2021", "multi_label", "Hang || Crash", "Crash"),
    ("fse2021", "free_text", "The vehicle drifts away during landing.", "The vehicle remains stable during landing."),
])
def test_native_taxonomy_survives_view_and_classification_validation(domain, mode, label, alternative):
    view = EvidenceView(record_id="r", task="annotate", taxonomy=spec(mode), domain_profile=domain,
                        ledger_version=1, items=(item(),))
    report = SymptomReport(label=label, alternative_label=alternative,
                           behavior_claim="The supplied report identifies unexpected landing behavior.",
                           boundary_reason="The original report distinguishes the observed outcome from its alternative.",
                           supporting_evidence_ids=["issue"], confidence=.8, evidence_sufficiency="sufficient")
    _validate_classification_labels(report, view, EvidenceDimension.SYMPTOM)
    assert view.model_dump()["taxonomy"]["annotation_modes"]["symptom"] == mode


def test_issue_domains_allow_issue_root_context_without_forcing_commit_evidence():
    assert supports_readiness_dimension(item(), EvidenceDimension.ROOT_CAUSE, domain="fse2021")
    assert required_readiness_dimensions("fse2021", "stage2") == (EvidenceDimension.FAULT_EXISTENCE, EvidenceDimension.STUDY_SCOPE)


def test_native_modes_survive_isolated_role_views():
    ledger = EvidenceLedger(record_id="r", task="annotate", taxonomy=spec("multi_label"),
                            domain_profile="icse2021", initial_items=(item(),))
    copied = derive_frozen_evidence_view(ledger.view())
    assert copied.taxonomy["annotation_modes"]["symptom"] == "multi_label"


@pytest.mark.parametrize("role,valid_id,invalid_id", [
    ("fault_evidence_analyst", "issue", "patch"),
    ("scope_boundary_analyst", "issue", "patch"),
    ("repair_causality_analyst", "patch", "issue"),
])
@pytest.mark.parametrize("citation_field", ["supporting_evidence_ids", "counter_evidence_ids"])
def test_stage2_owned_citation_mismatch_uses_existing_bounded_semantic_retry(role, valid_id, invalid_id, citation_field):
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    patch_content = "- incorrect_condition()\n+ correct_condition()"
    patch = item().model_copy(update={"evidence_id": "patch", "source_type": "code_diff", "content": patch_content,
                                       "content_sha256": hashlib.sha256(patch_content.encode()).hexdigest()})
    view = EvidenceView(record_id="r", task="screen", taxonomy={"symptom": ["Crash"], "root_cause": ["Logic"]},
                        domain_profile="issta2024", ledger_version=2, items=(item(), patch))
    calls = []
    def transport(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return json.dumps({"outcome": "pass", "claim": "The supplied evidence supports this named screening criterion.",
                           citation_field: [invalid_id if len(calls) == 1 else valid_id]})
    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=1, retry_delay_seconds=0))
    result = getattr(agents, role)(view)
    assert len(calls) == 2
    assert getattr(result, citation_field) == [valid_id]
    context = json.loads(calls[0][1])["context"]
    assert context["allowed_citation_ids"] == [valid_id]
    assert "capability" in calls[1][1]
    assert "allowed_citation_ids" in calls[0][0]


def test_free_text_cohort_loader_keeps_empty_enum_and_metadata(tmp_path):
    from Benchmark.src.paper_benchmark import build_taxonomy, PAPER_IDS
    cohort = tmp_path / "cohort.csv"
    cohort.write_text(f"record_id,paper_id,title,body,decision,symptom,root_cause\nr,{PAPER_IDS['fse2021']},Landing fault,Drifting,accepted_fault,Drifting,Logic\n", encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(json.dumps(build_taxonomy("fse2021")), encoding="utf-8")
    loaded = load_domain_inputs("fse2021", cohort_path=cohort, taxonomy_path=taxonomy)
    assert loaded.taxonomy["symptom"] == []
    assert loaded.taxonomy["annotation_modes"]["symptom"] == "free_text"


@pytest.mark.parametrize("mismatch", ["paper", "taxonomy"])
def test_new_domain_preflight_rejects_cross_paper_inputs_and_unbound_codebook(tmp_path, mismatch):
    from Benchmark.src.paper_benchmark import build_taxonomy, PAPER_IDS
    cohort = tmp_path / "cohort.csv"
    paper = PAPER_IDS["ase2022"] if mismatch == "paper" else PAPER_IDS["fse2021"]
    cohort.write_text(f"record_id,paper_id,title,body\nr,{paper},Fault,Observed failure\n", encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.json"
    payload = build_taxonomy("fse2021")
    if mismatch == "taxonomy":
        payload["root_cause"] = ["Invented label"]
    taxonomy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paper|codebook"):
        load_domain_inputs("fse2021", cohort_path=cohort, taxonomy_path=taxonomy)


def test_adaptive_evaluation_uses_native_metrics_instead_of_string_equality():
    gold = [{"record_id": "r", "decision": "accepted_fault", "symptom": "Crash || Hang", "root_cause": "Logic"}]
    rows = [{"record_id": "r", "stage3_valid": True, "symptom_prediction": "Hang || Crash", "root_cause_prediction": "Logic"}]
    result = evaluate_experiment(gold, rows, stage="stage3", taxonomy=spec("multi_label"))
    assert result["stage3"]["symptom_set_exact_match"] == 1
    assert result["stage3"]["joint_accuracy"] == 1
    text_gold = [{**gold[0], "symptom": "Drifting"}]
    text_rows = [{**rows[0], "symptom_prediction": "Unexpected drift during landing"}]
    result = evaluate_experiment(text_gold, text_rows, stage="stage3", taxonomy=spec("free_text"))
    assert result["stage3"]["symptom_accuracy"] is None
    assert result["stage3"]["joint_accuracy"] is None
    assert result["stage3"]["root_cause_accuracy"] == 1


@pytest.mark.parametrize("domain", ["fse2021", "icse2021", "icse2022", "icse2023", "icse2024"])
def test_real_adaptive_controller_and_structured_roles_run_both_stages_for_native_papers(domain):
    from Benchmark.src.paper_benchmark import build_taxonomy
    from Benchmark.src.annotation_contracts import annotation_mode, canonical_label
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.experiment import run_adaptive_record

    taxonomy = build_taxonomy(domain)
    mode = annotation_mode(taxonomy, "symptom")
    if mode == "free_text":
        symptom, alternative = "Unexpected behavior during operation.", "The operation completes without the reported behavior."
    elif mode == "multi_label":
        symptom = " || ".join(reversed(taxonomy["symptom"][:2]))
        alternative = taxonomy["symptom"][0]
    else:
        symptom, alternative = taxonomy["symptom"][0], taxonomy["symptom"][-1]
    root, other_root = taxonomy["root_cause"][0], taxonomy["root_cause"][-1]
    prompts = []

    def transport(system_prompt, user_prompt, *, options):
        prompts.append((options.role, system_prompt, user_prompt))
        payload = json.loads(user_prompt)
        view = payload.get("evidence_view", {})
        evidence_id = view.get("items", [{}])[0].get("evidence_id", "")
        role = options.role
        if role == "evidence_readiness":
            task = payload["context"]["task"]
            dimensions = required_readiness_dimensions(domain, task)
            response = {"task": task, "dimensions": [{"dimension": d.value, "sufficient": True,
                        "confirmed_evidence_ids": [evidence_id], "missing_facts": [], "evidence_requests": []} for d in dimensions]}
        elif role in {"fault_evidence_analyst", "scope_boundary_analyst"}:
            response = {"outcome": "pass", "claim": "The supplied source supports this owned screening criterion.",
                        "supporting_evidence_ids": [evidence_id]}
        elif role == "joint_anchor":
            common = {"supporting_evidence_ids": [evidence_id], "boundary_evidence_ids": [evidence_id],
                      "boundary_reason": "The directly stated behavior distinguishes the proposed annotation from its alternative.",
                      "confidence": .95, "evidence_sufficiency": "sufficient"}
            response = {"symptom": {**common, "label": symptom, "alternative_label": alternative,
                        "behavior_claim": "The operation exhibits unexpected behavior described in the report."},
                        "root_cause": {**common, "label": root, "alternative_label": other_root,
                        "defect_mechanism": "The report identifies an incorrect internal condition causing the behavior.",
                        "causal_chain": ["The operation reaches the affected condition.", "The condition selects an incorrect branch.", "The result violates the expected behavior."]},
                        "causal_account": "The incorrect internal condition explains the unexpected result.",
                        "shared_supporting_evidence_ids": [evidence_id]}
        elif role in {"symptom_verifier", "root_cause_verifier"}:
            dimension = "symptom" if role == "symptom_verifier" else "root_cause"
            response = {"dimension": dimension, "verdict": "accept",
                        "anchor_label": payload["context"]["anchor"][dimension]["label"],
                        "rationale": "The supplied source supports the bounded annotation and its evidence references.",
                        "supporting_evidence_ids": [evidence_id], "confidence": .95}
        elif role == "causal_consistency_checker":
            response = {"status": "consistent", "rationale": "The identified internal condition explains the reported unexpected behavior.",
                        "supporting_evidence_ids": [evidence_id]}
        elif role == "boundary_challenger":
            response = {"action": "pass", "rationale": "The supplied evidence distinguishes the proposed native annotation from its alternatives.",
                        "cited_evidence_ids": [evidence_id]}
        elif role == "stage3_arbitrator":
            packet = payload.get("packet", payload)
            response = {"resolution_status": "resolved", "symptom_label": symptom,
                        "root_cause_label": root, "confidence": .9,
                        "rationale": "The shared bounded evidence confirms the proposed native annotation.",
                        "supporting_evidence_ids": [packet["evidence_snapshot"]["items"][0]["evidence_id"]],
                        "resolved_dimensions": list(packet["disagreement"]["dimensions"])}
        else:
            raise AssertionError(f"unexpected fixture role {role}")
        return json.dumps(response)

    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=0))
    record = {"record_id": "native-fixture", "title": "Observed technical fault", "body": item().content,
              "decision": "SECRET_GOLD_DECISION", "symptom": "SECRET_GOLD_SYMPTOM", "root_cause": "SECRET_GOLD_ROOT",
              "original_label_json": '{"fix":"SECRET_GOLD_FIX"}'}
    row = run_adaptive_record(record, domain=domain, taxonomy=taxonomy, agents=agents, stage="all")
    assert row["stage2_valid"] is True, row
    assert row["stage3_valid"] is True, row
    assert row["symptom_prediction"] == canonical_label(taxonomy, "symptom", symptom)
    assert len(row["audit"]["stage3"]["reports"]) == 2
    assert all(len(report["verifications"]) == 2 for report in row["audit"]["stage3"]["reports"])
    assert domain in row["audit"]["stage2"]["reports"][0]["repair_claim"]
    assert not any("SECRET_GOLD" in system + user for _, system, user in prompts)
    assert sum(role == "joint_anchor" for role, _, _ in prompts) == 2
