import re
import unittest
from pathlib import Path

import pandas as pd
from dateutil import parser


ROOT = Path(__file__).resolve().parents[2]
DATASET_COLUMNS = [
    "record_id",
    "paper_id",
    "source_project",
    "issue_url",
    "title",
    "body",
    "comments",
    "created_at",
    "updated_at",
    "state",
    "symptom",
    "root_cause",
    "bug_type",
    "component",
    "sub_component",
    "trigger_condition",
    "consequence",
    "fix_type",
    "severity_or_impact",
    "original_label_json",
    "source_file",
    "source_sheet",
    "source_row_index",
    "code_diff",
]
MISSING_TIMESTAMP_TOKENS = {
    "",
    "not_available_in_source",
    "Not found",
    "-",
}


def load_stage(stage):
    return pd.read_csv(
        ROOT / "Dataset" / f"{stage}.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


class CurrentDatasetQualityTests(unittest.TestCase):
    def test_updated_at_has_no_numeric_only_values(self):
        for stage in ["stage1", "stage2", "stage3"]:
            with self.subTest(stage=stage):
                df = load_stage(stage)
                numeric_only = df["updated_at"].str.strip().str.match(r"^\d{1,4}$")
                self.assertFalse(
                    numeric_only.any(),
                    df.loc[
                        numeric_only,
                        ["record_id", "paper_id", "issue_url", "updated_at"],
                    ].head(10).to_dict("records"),
                )

    def test_parseable_updated_at_is_not_before_created_at(self):
        for stage in ["stage1", "stage2", "stage3"]:
            with self.subTest(stage=stage):
                df = load_stage(stage)
                bad_rows = []
                for row in df.itertuples(index=False):
                    created_at = str(row.created_at).strip()
                    updated_at = str(row.updated_at).strip()
                    if (
                        created_at in MISSING_TIMESTAMP_TOKENS
                        or updated_at in MISSING_TIMESTAMP_TOKENS
                    ):
                        continue
                    if re.fullmatch(r"\d{1,4}", updated_at):
                        bad_rows.append(row)
                        continue
                    if parser.parse(updated_at) < parser.parse(created_at):
                        bad_rows.append(row)

                self.assertEqual(
                    [],
                    [
                        {
                            "record_id": row.record_id,
                            "paper_id": row.paper_id,
                            "issue_url": row.issue_url,
                            "created_at": row.created_at,
                            "updated_at": row.updated_at,
                        }
                        for row in bad_rows[:10]
                    ],
                )

    def test_data_dictionary_matches_current_csv_schema(self):
        dictionary = ROOT / "metadata" / "data_dictionary.md"
        documented_fields = []
        for line in dictionary.read_text(encoding="utf-8").splitlines():
            if line.startswith("## `"):
                documented_fields.append(line.split("`")[1])

        self.assertEqual(DATASET_COLUMNS, documented_fields)


if __name__ == "__main__":
    unittest.main()
