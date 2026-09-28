"""Independent, read-only audit of frozen cohorts and saved MAS/Self evidence.

Uses only the standard library, never current production projection/validators.
v1 means the original unmasked safe-field projection; v2 masks exact unavailable
markers. An invalid model result remains in the denominator. This checks saved
provenance, not whether a quoted passage semantically proves a model conclusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

SAFE_FIELDS = ("source_project", "issue_url", "title", "body", "comments", "state",
               "created_at", "changed_files", "code_diff")
MISSING = {"not_fetched", "not_available_in_source", "comments_unavailable_in_source"}
KNOWN_EMPTY = {"no_comments_in_source", "[]"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]


def projection(row: dict, version: str) -> dict[str, str]:
    fields = {key: str(row.get(key, "") or "").strip() for key in SAFE_FIELDS}
    if version == "v2":
        fields = {key: "" if value.casefold() in MISSING else value
                  for key, value in fields.items()}
    return fields


def parse_raw_json(raw: str) -> Any:
    """Allow surrounding CAMEL prose/fences but no fabricated normalization."""
    decoder = json.JSONDecoder()
    for pos, character in enumerate(raw):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(raw[pos:])
                return value
            except json.JSONDecodeError:
                pass
    return None


def walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}" if path else key
            yield here, key, child
            yield from walk(child, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def response_contents(response: Any) -> list[str]:
    if not isinstance(response, dict):
        return []
    return [str(choice.get("message", {}).get("content", ""))
            for choice in response.get("choices", [])
            if isinstance(choice, dict) and choice.get("message", {}).get("content")]


class Audit:
    def __init__(self, version: str):
        self.version = version
        self.checks = Counter()
        self.failures: list[dict] = []
        self.observations: list[dict] = []
        self.counts = Counter({key: 0 for key in (
            "mas_actual_marker_citations", "mas_model_visible_missing_fields",
            "self_direct_missing_marker_items", "self_marker_validity_inventory_references",
            "self_marker_model_or_final_references", "self_records_with_quarantined_items")})
        self.context: dict = {}

    def check(self, condition: bool, name: str, **details):
        self.checks[name] += 1
        if not condition:
            self.failures.append({**self.context, "check": name, **details})

    def observe(self, name: str, **details):
        self.observations.append({**self.context, "kind": name, **details})

    def mas(self, row: dict, gold: dict, stage: str):
        record_id = row["record_id"]
        society = row.get("society", {})
        audit = row.get("explanation_audit", {})
        fields = audit.get("evidence_fields", {})
        expected = projection(gold, self.version)
        task = society.get("task_prompt", "")
        valid = not row.get("invalid", True)
        self.check(fields == expected, "mas_input_fields_match_frozen_cohort")
        self.check(row.get("paper_id") == gold.get("paper_id"), "mas_paper_binding")
        self.check(row.get("issue_url") == gold.get("issue_url"), "mas_url_binding")
        self.check(row.get("stage") == stage, "mas_stage_binding")
        self.check(bool(task) and digest(task) == audit.get("task_prompt_sha256")
                   == society.get("immutable_evidence_sha256"), "mas_task_hash")
        self.check(audit.get("record_id") == record_id, "mas_audit_record_binding")
        for key, value in fields.items():
            self.check(not value or value in task, "mas_field_in_task", field=key)
            if value.strip().casefold() in MISSING:
                self.counts["mas_model_visible_missing_fields"] += 1
                self.observe("missing_marker_visible_to_mas", field=key, marker=value)
                if self.version == "v2":
                    self.check(False, "v2_missing_marker_excluded_from_model_fields", field=key)

        trace = audit.get("request_trace", [])
        self.counts["mas_request_trace_records"] += len(trace)
        self.check(bool(trace) and audit.get("request_trace_available") is True,
                   "mas_request_trace_available")
        seen_calls: set[tuple] = set()
        observed: dict[str, list[str]] = {}
        for call in trace:
            role = call.get("role", "")
            call_key = (role, call.get("call_id"), call.get("attempt"))
            self.check(call_key not in seen_calls, "mas_unique_local_call_identity")
            seen_calls.add(call_key)
            self.check(call.get("record_id") == record_id, "mas_request_record_binding")
            self.check(str(call.get("call_id", "")).startswith(record_id + ":"),
                       "mas_request_id_bound_to_record")
            request = call.get("request", {})
            self.check(bool(request.get("messages")), "mas_request_messages_saved")
            self.check(any(task in str(message.get("content", ""))
                           for message in request.get("messages", [])),
                       "mas_request_contains_bound_immutable_task")
            contents = response_contents(call.get("response"))
            for exchange in call.get("http_exchanges", []):
                contents.extend(response_contents(exchange.get("response")))
            observed.setdefault(role, []).extend(contents)
            self.counts["mas_visible_response_contents"] += len(contents)
        for turn in society.get("turns", []):
            for part, role in (("assistant", "ai_assistant"), ("user", "ai_user")):
                content = turn.get(part, {}).get("content", "")
                if content:
                    self.check(content in observed.get(role, []), "mas_turn_matches_observed_response",
                               turn=turn.get("turn"), role=role)

        if not valid:
            self.check(audit.get("status") == "invalid", "mas_invalid_status_honest")
            self.observe("invalid_prediction_retained", error=row.get("error"),
                         stop_reason=society.get("stop_reason"))
            return
        self.check(audit.get("status") == "schema_and_citations_valid", "mas_valid_status")
        parsed = society.get("parsed_output", {})
        raw = society.get("raw_final_output", "")
        self.check(parse_raw_json(raw) == parsed, "mas_final_matches_raw_response")
        source = row.get("output_source")
        self.check(source == audit.get("source"), "mas_selected_source_binding")
        if source == "society_assistant":
            number = society.get("final_answer_turn")
            turns = society.get("turns", [])
            selected = turns[number - 1].get("assistant", {}) if type(number) is int and 1 <= number <= len(turns) else {}
            self.check(selected.get("content") == raw and selected.get("parsed_output") == parsed,
                       "mas_selected_turn_matches_final")
            self.check(raw in observed.get("ai_assistant", []), "mas_final_response_observed")
        elif source == "forced_finalizer":
            finalizer = society.get("forced_finalizer", {})
            attempts = finalizer.get("attempt_trace", [])
            self.check(finalizer.get("raw_output") == raw and finalizer.get("parsed_output") == parsed,
                       "mas_finalizer_matches_final")
            self.check(bool(attempts) and not attempts[-1].get("error") and
                       attempts[-1].get("raw_output") == raw, "mas_successful_finalizer_attempt")
            self.check(any(raw in values for values in observed.values()), "mas_finalizer_response_observed")
        else:
            self.check(False, "mas_known_output_source", source=source)
        dimensions = ("decision",) if stage == "stage2" else ("symptom", "root_cause")
        self.check(row.get("final_prediction") == {key: parsed.get(key) for key in dimensions},
                   "mas_final_labels_match_parsed")
        citations = []
        for dimension in dimensions:
            rationale = parsed.get(dimension + "_rationale", {})
            self.check(rationale == audit.get("rationales", {}).get(dimension),
                       "mas_rationale_matches_parsed", dimension=dimension)
            self.check(bool(str(rationale.get("reason", "")).strip()), "mas_explicit_reason_present")
            refs = rationale.get("evidence_refs", [])
            self.check(bool(refs) or bool(str(rationale.get("evidence_limitations", "")).strip()),
                       "mas_missing_citations_explained")
            for ref in refs:
                field, quote = ref.get("field"), ref.get("quote", "")
                content = fields.get(field, "")
                start = content.find(quote) if quote else -1
                self.check(field in SAFE_FIELDS and start >= 0, "mas_exact_quote_in_source", field=field)
                prohibited = content.strip().casefold() in MISSING or (field == "comments" and content.strip().casefold() in KNOWN_EMPTY)
                self.check(not prohibited, "mas_no_status_marker_as_technical_citation", field=field, quote=quote)
                if prohibited:
                    self.counts["mas_actual_marker_citations"] += 1
                citations.append({"dimension": dimension, "field": field, "quote": quote,
                                  "start_char": start, "end_char": start + len(quote),
                                  "offset_convention": "zero-based Unicode code points; end exclusive; first exact occurrence",
                                  "source_sha256": digest(content)})
                self.counts["mas_final_citations"] += 1
        self.check(citations == audit.get("citations"), "mas_saved_citation_offsets_hashes_and_quotes")

    def self_designed(self, row: dict, gold: dict, stage: str):
        record_id = row["record_id"]
        audit = row.get("audit", {})
        fields = projection(gold, self.version)
        valid = bool(row.get(stage + "_valid"))
        self.check(audit.get("record_id") == record_id, "self_audit_record_binding")
        self.check(audit.get("domain_profile") == self.context["domain"], "self_domain_binding")
        items = audit.get("evidence_items", [])
        ids = [item.get("evidence_id") for item in items]
        self.check(bool(ids) and len(ids) == len(set(ids)), "self_local_ledger_unique_ids")
        local_ids = set(ids)
        marker_ids: set[str] = set()
        field_for_type = {"issue_body": "body", "issue_comments": "comments", "code_diff": "code_diff",
                          "changed_files": "changed_files", "commit_history": "commit_history"}
        # v1 Self already ignored two older missing-source markers, but not not_fetched.
        unavailable = MISSING if self.version == "v2" else MISSING - {"not_fetched"}
        source_text = {}
        for source_type, field in field_for_type.items():
            value = str(fields.get(field, gold.get(field, "")) or "").strip()
            if value.casefold() in unavailable or (field in {"comments", "changed_files", "commit_history"} and value.casefold() in KNOWN_EMPTY):
                value = ""
            source_text[source_type] = value
        body = source_text["issue_body"]
        summary_parts = (("ISSUE_URL", fields["issue_url"]), ("TITLE", fields["title"]),
                         ("BODY", body), ("STATE", fields["state"]),
                         ("CREATED_AT", fields["created_at"]), ("SOURCE_PROJECT", fields["source_project"]))
        expected_summary = "\n".join(f"{key}: {value}" for key, value in summary_parts if value) or "No issue summary is available for this record."
        for item in items:
            self.counts["self_ledger_items"] += 1
            content = item.get("content", "")
            source_type = item.get("source_type")
            metadata = item.get("metadata", {})
            self.check(item.get("record_id") == record_id, "self_evidence_record_binding")
            self.check(item.get("content_sha256") == digest(content), "self_evidence_content_hash")
            if source_type == "record_summary":
                self.check(content == expected_summary, "self_summary_matches_frozen_record")
            elif source_type in source_text:
                source = source_text[source_type]
                self.check(bool(content) and content in source, "self_passage_matches_frozen_source", source_type=source_type)
                if metadata.get("frozen_source_sha256"):
                    self.check(metadata["frozen_source_sha256"] == digest(source), "self_frozen_source_hash")
            elif source_type == "taxonomy":
                self.counts["self_taxonomy_context_items_not_primary_evidence"] += 1
                self.observe("taxonomy_context_source", evidence_id=item.get("evidence_id"),
                             note="Hash checked; semantic grounding is separate from primary issue evidence.")
            else:
                self.check(False, "self_known_record_local_source_type", source_type=source_type)
            if source_type in {"record_summary", "issue_body", "issue_comments"}:
                self.check(item.get("source_uri") == fields["issue_url"], "self_issue_source_uri")
            if content.strip().casefold() in MISSING:
                marker_ids.add(item["evidence_id"])
                self.counts["self_direct_missing_marker_items"] += 1
                self.observe("missing_marker_in_self_ledger", evidence_id=item["evidence_id"], marker=content)
                self.check(False, "self_no_missing_marker_as_direct_evidence", evidence_id=item["evidence_id"])
        block = audit.get(stage) or {}
        for path, key, value in walk(block, stage):
            if key == "quarantined" and value:
                self.counts["self_records_with_quarantined_items"] += 1
                self.observe("quarantined_items_retained", path=path, value=value)
            if key.endswith("evidence_ids") and isinstance(value, list):
                for evidence_id in value:
                    self.counts["self_evidence_id_references"] += 1
                    if evidence_id not in local_ids:
                        self.observe("unbound_model_evidence_reference", path=path, evidence_id=evidence_id,
                                     row_valid=valid)
                        if valid and ("final_decision" in path or "quarantin" not in path):
                            self.check(False, "self_valid_output_reference_bound_to_local_ledger", path=path, evidence_id=evidence_id)
                    else:
                        self.check(True, "self_reference_bound_to_local_ledger")
                    if evidence_id in marker_ids:
                        category = "self_marker_validity_inventory_references" if key == "valid_evidence_ids" else "self_marker_model_or_final_references"
                        self.counts[category] += 1
                        self.observe(category, path=path, evidence_id=evidence_id)
        final = block.get("final_decision") or {}
        if valid:
            self.check(bool(final), "self_valid_output_has_final_decision")
            if stage == "stage2":
                self.check(row.get("stage2_prediction") == final.get("decision"), "self_final_filter_label_binding")
            else:
                self.check(row.get("symptom_prediction") == final.get("symptom_label") and
                           row.get("root_cause_prediction") == final.get("root_cause_label"),
                           "self_final_annotation_label_binding")
        else:
            self.observe("invalid_prediction_retained", stop_reason=row.get("stop_reason"),
                         unresolved=block.get("unresolved"), verification=block.get("verification"))
        self.counts["self_rows_without_raw_provider_request_trace"] += 1


def run(args: argparse.Namespace) -> dict:
    audit = Audit(args.projection_version)
    run_root, cohort_root = args.run_root.resolve(), args.cohort_root.resolve()
    plan_path = run_root / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    jobs = {(j["domain"], j["engine"], j["stage"]) for j in plan.get("jobs", [])
            if j.get("engine") in {"mas", "self"}}
    if not jobs:
        jobs = {(path.parent.name, path.name.split("_")[0], path.name.split("_")[1])
                for path in run_root.glob("*/*") if path.is_dir()
                and path.name in {"mas_stage2", "mas_stage3", "self_stage2", "self_stage3"}}
    archive = run_root / "source_snapshot.zip"
    if archive.exists():
        if plan.get("source_archive_sha256"):
            audit.check(file_digest(archive) == plan["source_archive_sha256"], "archived_source_zip_hash")
        with zipfile.ZipFile(archive) as handle:
            audit.check(handle.testzip() is None, "archived_source_zip_integrity")
            source_hash_path = run_root / "source_sha256.json"
            if source_hash_path.exists():
                for relative, expected_hash in json.loads(source_hash_path.read_text(encoding="utf-8")).items():
                    name = relative.replace("\\", "/")
                    audit.check(name in handle.namelist() and
                                hashlib.sha256(handle.read(name)).hexdigest() == expected_hash,
                                "archived_source_file_hash", path=name)
    inventories = []
    for domain, engine, stage in sorted(jobs):
        audit.context = {"domain": domain, "engine": engine, "stage": stage}
        cohort_file = cohort_root / domain / ("cohort.csv" if stage == "stage2" else "stage3_sample.csv")
        expected = read_csv(cohort_file)
        expected_by_id = {row["record_id"]: row for row in expected}
        audit.check(len(expected) == len(expected_by_id), "frozen_cohort_ids_unique")
        if args.projection_version == "v2":
            for record in expected:
                for key in SAFE_FIELDS:
                    audit.check(str(record.get(key, "")).strip().casefold() not in MISSING,
                                "v2_cohort_fields_already_projected", record_id=record["record_id"], field=key)
        expected_count = args.expected_filter if stage == "stage2" else args.expected_annotation
        # Full five-paper plan commits to 1,500 records; smoke counts are instead
        # derived from its separately frozen input files unless explicitly set.
        if expected_count is None and plan.get("expected_predictions") == 1500:
            expected_count = 100 if stage == "stage2" else 50
        if expected_count is not None:
            audit.check(len(expected) == expected_count, "frozen_cohort_size", expected=expected_count, actual=len(expected))
        if stage == "stage2":
            balance = Counter(row.get("decision") for row in expected)
            audit.check(balance.get("accepted_fault", 0) == balance.get("rejected_candidate", 0)
                        and set(balance) <= {"accepted_fault", "rejected_candidate"},
                        "filter_frozen_class_balance", balance=dict(balance))
        else:
            audit.check(all(row.get("decision") == "accepted_fault" for row in expected),
                        "annotation_only_frozen_gold_positive")
        manifest_path = cohort_root / domain / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact_key = "cohort" if stage == "stage2" else "stage3_sample"
            audit.check(file_digest(cohort_file) == manifest.get("artifact_sha256", {}).get(artifact_key),
                        "cohort_bound_to_preparation_manifest")
            if args.projection_version == "v2":
                audit.check(manifest.get("evidence_projection_version") == "paper-primary-evidence-v2",
                            "preparation_projection_version_bound")
        folder = run_root / domain / (engine + "_" + stage)
        files = sorted(folder.rglob("*predictions*.jsonl")) if folder.exists() else []
        audit.check(len(files) <= 1, "one_prediction_file_per_job", files=[str(p) for p in files])
        rows = []
        malformed = []
        prediction_snapshots = []
        for path in files:
            # A live JSONL can append during the audit. Bind the exact byte prefix
            # consumed here, not a later second read containing additional rows.
            payload = path.read_bytes()
            prediction_snapshots.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                                         "byte_count": len(payload),
                                         "hash_scope": "exact bytes observed; verify this prefix if the file later grows"})
            try:
                decoded = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                malformed.append({"file": str(path), "byte_offset": exc.start, "error": str(exc)})
                decoded = payload[:exc.start].decode("utf-8")
            for number, line in enumerate(decoded.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    malformed.append({"file": str(path), "line": number, "error": str(exc)})
        if malformed:
            audit.observe("malformed_partial_lines_retained", lines=malformed)
            if not args.partial:
                audit.check(False, "complete_jsonl_records", lines=malformed)
        observed_ids = [row.get("record_id") for row in rows]
        audit.check(len(observed_ids) == len(set(observed_ids)), "prediction_ids_unique")
        audit.check(set(observed_ids).issubset(expected_by_id), "prediction_ids_within_frozen_cohort")
        missing = sorted(set(expected_by_id) - set(observed_ids))
        if not args.partial:
            audit.check(not missing, "complete_fixed_cohort_predictions", missing_count=len(missing))
        valid_count = 0
        for row in rows:
            record_id = row.get("record_id")
            audit.context = {"domain": domain, "engine": engine, "stage": stage, "record_id": record_id}
            gold = expected_by_id.get(record_id)
            if gold is None:
                continue
            is_valid = not row.get("invalid", True) if engine == "mas" else bool(row.get(stage + "_valid"))
            valid_count += int(is_valid)
            if engine == "mas":
                audit.mas(row, gold, stage)
            else:
                audit.self_designed(row, gold, stage)
        inventories.append({"domain": domain, "engine": engine, "stage": stage,
                            "expected": len(expected), "saved": len(rows), "valid": valid_count,
                            "invalid_or_unresolved": len(rows) - valid_count,
                            "missing": len(missing), "missing_record_ids": missing,
                            "malformed_lines": malformed,
                            "prediction_files": prediction_snapshots,
                            "cohort_path": str(cohort_file), "cohort_sha256": file_digest(cohort_file)})
    incomplete = any(job["missing"] or job["malformed_lines"] for job in inventories)
    return {"status": "FAIL" if audit.failures else "PARTIAL" if incomplete else "PASS",
            "projection_version": args.projection_version, "partial_mode": args.partial,
            "run_root": str(run_root), "cohort_root": str(cohort_root),
            "scope": "Saved provenance only; semantic support of classification is not verified.",
            "self_trace_limit": "Self saves explicit role/readiness/arbitration reports and ledger, but these prediction files do not contain raw provider request/response traces.",
            "aborted": json.loads((run_root / "aborted.json").read_text(encoding="utf-8")) if (run_root / "aborted.json").exists() else None,
            "total_expected": sum(job["expected"] for job in inventories),
            "total_saved": sum(job["saved"] for job in inventories),
            "total_valid": sum(job["valid"] for job in inventories),
            "total_invalid_or_unresolved": sum(job["invalid_or_unresolved"] for job in inventories),
            "total_missing": sum(job["missing"] for job in inventories),
            "check_count": sum(audit.checks.values()), "check_counts": dict(audit.checks),
            "counts": dict(audit.counts), "failures": audit.failures,
            "observations": audit.observations, "jobs": inventories}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--projection-version", choices=("v1", "v2"), required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--expected-filter", type=int)
    parser.add_argument("--expected-annotation", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    output = args.output or args.run_root / ("independent_evidence_audit_partial.json" if args.partial else "independent_evidence_audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "total_expected", "total_saved", "total_valid", "total_invalid_or_unresolved", "total_missing", "check_count", "counts")}, ensure_ascii=True))
    print(json.dumps({"failure_count": len(result["failures"]), "output": str(output.resolve())}))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
