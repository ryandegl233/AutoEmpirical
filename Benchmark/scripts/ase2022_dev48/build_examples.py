"""Re-express the two approved source-only examples; never read evaluation answers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalExplainableTools


def atom(key, kind, statement, fact):
    return dict(atom_id=key, kind=kind, statement=statement,
                citations=[dict(evidence_id=fact["evidence_id"], quote=fact["quote"], kind=kind)])


def candidate(dimension, label, premises, assessment, reason, missing=""):
    return dict(dimension=dimension, label=label, premise_ids=premises,
                relation="causal_link" if dimension == "root_cause" else "definition_match",
                assessment=assessment, reason=reason, missing_evidence=missing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'Benchmark/inputs/ase2022_dev48/examples')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new local output directory; do not overwrite a frozen version.')
    args = parser.parse_args(argv)
    old = args.source_dir
    prior = json.loads((old/'examples_v1.json').read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first, second = prior["examples"]
    facts = first["query"]["facts"]
    example1 = dict(record_id=first["record_id"], query=dict(
        atoms=[atom("O1", "observation", "A completed fullFaceDescriptions call has a recorded elapsed time.", facts[0]),
               atom("A1", "source_assertion", "The comment reports a longer first forward pass on some machines.", facts[1]),
               atom("A2", "source_assertion", "The commenter explicitly calls GPU texture allocation a wild guess and states uncertainty.", facts[2])],
        candidates=[candidate("symptom", "Poor Performance", ["O1", "A1"], "supported",
                              "A completed timed operation plus the reported unusually long first pass supports reported slowness; the numeric timing alone is not a learned threshold."),
                    candidate("root_cause", "WebGL Limits", ["A2"], "unknown",
                              "The source supplies a qualified GPU allocation hypothesis, not an established resource limit causing this delay.",
                              "Evidence identifying a concrete WebGL limit and connecting it to the delay; a patch is not inherently required."),
                    candidate("root_cause", "Incorrect Code Logic", ["O1", "A1", "A2"], "unknown",
                              "An initial delay is compatible with several mechanisms; these observations identify no erroneous code path.",
                              "Evidence distinguishing a code logic defect from a backend/resource constraint or expected initialization cost.")]))
    facts = second["query"]["facts"]
    example2 = dict(record_id=second["record_id"], query=dict(
        atoms=[atom("O1", "observation", "The command terminates with exit code 7.", facts[0]),
               atom("O2", "observation", "The trace records an abort and out-of-memory error during WebAssembly instantiation.", facts[1])],
        candidates=[candidate("symptom", "Crash", ["O1", "O2"], "supported",
                              "An abort and nonzero process termination conditionally support unexpected termination. This does not establish priority over initialization failure."),
                    candidate("symptom", "Build & Initialization Failure", ["O2"], "unknown",
                              "Instantiation appears in the trace, but the quoted error alone does not establish the application's initialization phase.",
                              "Execution context locating the failure in initialization versus an already running application."),
                    candidate("symptom", "Poor Performance", ["O1", "O2"], "unknown",
                              "Memory exhaustion is recorded together with termination. The applicable symptom boundary requires comparison with the execution failure.",
                              "Evidence of the execution phase and whether abnormal resource use or the resulting termination is the observed behavior being classified."),
                    candidate("root_cause", "Incorrect Code Logic", ["O2"], "unknown",
                              "Out of memory reports the failure condition; it does not identify the code mechanism that caused allocation failure.",
                              "Evidence linking the allocation failure to a specific erroneous implementation, rather than an environment/resource constraint.")]))
    bundle = dict(source_sha256=hashlib.sha256((old / "example_source.csv").read_bytes()).hexdigest(), examples=[example1, example2])
    target = args.output_dir / "examples_formal_v2.json"
    if target.exists():
        raise SystemExit("Existing example freeze retained; use a new version for changes")
    target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tool = FormalExplainableTools.from_files(target, old / "example_source.csv")
    (args.output_dir / "example_freeze.json").write_text(json.dumps({
        "provenance": "Same two user-approved source-only records as A v1; no GT, human answers or evaluation traces read.",
        "source_path": str(old / "example_source.csv"), **tool.manifest()
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(bundle["examples"]), "status": "validated_and_frozen"}))


if __name__ == '__main__':
    main()
