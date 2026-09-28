# Frozen seven-paper inputs

Each paper has 100 filtering inputs and 50 annotation inputs. These are development
cohorts, with author-selection semantics and native annotation types preserved.
No model predictions or scores are included.

The nine original files per paper retain their original bytes and hashes.
`release_paths.json` binds each original manifest to source files in this checkout;
`_sampling_sources/` retains the source-only ASE cohort reused for sampling.
Historical absolute paths inside the original manifests are provenance, not runtime
requirements. Verification accepts only the declared LF/CRLF source serialization;
all frozen input artifacts require exact byte hashes.

```bash
python Benchmark/scripts/prepare_paper_benchmarks.py --verify-existing
python Benchmark/scripts/seven_papers/verify_prepared_cohorts.py
```

Install Git LFS and run `git lfs pull` before verification. Gold fields inside
cohorts, prompt exports and source snapshots are for scoring only. Runners select
explicit evidence fields when constructing model inputs. See the Benchmark guide.
