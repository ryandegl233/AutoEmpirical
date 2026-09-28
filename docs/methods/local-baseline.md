# Local baseline registration

Baseline-preservation mode needs your own saved baseline predictions. A self-signed
manifest passed directly to the runner is rejected. Registration is an explicit
local decision that pins the manifest digest; it does not certify prediction truth.
No real baseline predictions, counts or error analyses ship with this release.

1. Generate or supply a compatible local JSONL under `Benchmark/results/` or
   `Benchmark/runs/`. Each row needs `record_id`, one shared `config_hash`, `invalid`
   and `final_prediction` (empty for invalid rows); do not include gold labels.
2. Prepare and explicitly activate a local trust manifest. Supply the SHA256 of
   the code used for that baseline, not the current code if it differs:

```powershell
python Benchmark/scripts/prepare_baseline_trust_root.py --artifact Benchmark/results/my-baseline/predictions.jsonl --output Benchmark/cache/my-baseline-trust.json --artifact-id my-baseline-v1 --domain ase2022 --baseline-code-sha256 <64-hex-code-digest> --activate-local
```

3. Pass `--baseline-preservation --baseline-anchor-path <local-jsonl>
   --baseline-trust-manifest Benchmark/cache/my-baseline-trust.json` together with
   an explicitly bound cohort, taxonomy and active split to the adaptive runner.

Without `--activate-local`, preparation writes an untrusted candidate only.
The active manifest and registry live under ignored `Benchmark/cache/`. Manifest,
artifact path, record-set and content hashes are checked on load. Changing a file
requires deliberate re-registration and a new experimental batch. The historical
private registration is retained for local compatibility; its artifact is not a
public dependency. Tests exercise registration with synthetic temporary data.
