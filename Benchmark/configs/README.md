# Benchmark configs

This directory contains paper codebooks, taxonomy structure, active/revoked split
manifests, offline gold bindings and frozen source-evidence bundles. Frozen files
retain their bytes across checkouts. Input cohorts are in `../inputs/`.

Real-run baseline statistics and per-record model error analyses are excluded.
To use your own baseline, follow [local registration](../../docs/methods/local-baseline.md);
its manifest and registry stay in ignored `Benchmark/cache/`.

Some original manifests mention historical runs as contamination provenance.
Those references do not make the predictions public runtime dependencies.
Credentials belong in environment variables or an ignored `.env` file.
