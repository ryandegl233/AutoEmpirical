# Reconstruction provenance

The final unified dataset is unchanged in this release. Collection, integration,
lineage repair and audit programs are now in [`../scripts/`](../scripts/README.md).

Existing published artifacts retain their paths to preserve provenance:

| Source | Published location |
| --- | --- |
| PyTorch candidates and integration audit | `reports/pytorch_stage1_reconstruction/` |
| Transaction Bugs candidates and integration audit | `reports/txbug_stage1_reconstruction/` |
| Source evidence sidecars | `Dataset/evidence/` |
| Final dataset hashes | `reports/SHA256SUMS.txt` |

Seven-paper preparation reads five Transaction Bugs candidate CSVs from the
published report directories. Git LFS is required. Input manifests bind their
hashes; do not move or rewrite these files without updating those bindings.

Ignored raw JSONL downloads and `fixcode/cache/` remain local. They are not all
known duplicates of the published sidecars. They are not needed for running the
released frozen benchmark inputs. Recollection can differ because upstream pages
change; record URLs, fetch times, revisions, original licenses and content hashes.
