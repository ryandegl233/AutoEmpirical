import json
import unittest

import pandas as pd

from Dataset.scripts.repair_stage1_stage2_lineage import (
    apply_stage1_lineage_repair,
    build_stage1_lineage_repair_plan,
    validate_stage1_lineage_repair_result,
)
from tests.data.test_repair_stage2_stage3_lineage import COLUMNS, row


class Stage1Stage2LineageRepairTests(unittest.TestCase):
    def test_plan_finds_stage2_rows_missing_from_stage1(self):
        stage1 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")], columns=COLUMNS
        )
        stage2 = pd.DataFrame(
            [
                row("paper:a", "paper", "https://example.test/a", "Crash"),
                row("paper:b", "paper", "https://example.test/b", "Hang"),
            ],
            columns=COLUMNS,
        )

        plan = build_stage1_lineage_repair_plan(stage1, stage2)

        self.assertEqual(plan.inserted_rows, 1)
        self.assertEqual(plan.inserted_keys, 1)

    def test_repair_preserves_stage1_and_clears_inserted_labels(self):
        stage1 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")], columns=COLUMNS
        )
        stage2 = pd.DataFrame(
            [row("paper:b", "paper", "https://example.test/b", "Crash")],
            columns=COLUMNS,
        )

        repaired_stage1, audit = apply_stage1_lineage_repair(stage1, stage2)

        self.assertEqual(len(repaired_stage1), 2)
        pd.testing.assert_frame_equal(repaired_stage1.iloc[[0]].reset_index(drop=True), stage1)
        inserted = repaired_stage1.iloc[1]
        self.assertEqual(inserted["issue_url"], "https://example.test/b")
        self.assertTrue(
            all(
                inserted[column] == ""
                for column in [
                    "symptom",
                    "root_cause",
                    "bug_type",
                    "component",
                    "sub_component",
                    "trigger_condition",
                    "consequence",
                    "fix_type",
                    "severity_or_impact",
                ]
            )
        )
        lineage = json.loads(inserted["original_label_json"])
        self.assertEqual(lineage["lineage_repair"]["source_stage"], "stage2")
        self.assertEqual(len(audit), 1)

    def test_repair_is_idempotent(self):
        stage1 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")], columns=COLUMNS
        )
        stage2 = pd.DataFrame(
            [row("paper:b", "paper", "https://example.test/b", "Crash")],
            columns=COLUMNS,
        )
        repaired_stage1, _ = apply_stage1_lineage_repair(stage1, stage2)

        second_stage1, second_audit = apply_stage1_lineage_repair(repaired_stage1, stage2)

        pd.testing.assert_frame_equal(second_stage1, repaired_stage1)
        self.assertTrue(second_audit.empty)

    def test_validation_accepts_complete_repair(self):
        stage1 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")], columns=COLUMNS
        )
        stage2 = pd.DataFrame(
            [row("paper:b", "paper", "https://example.test/b", "Crash")],
            columns=COLUMNS,
        )
        repaired_stage1, _ = apply_stage1_lineage_repair(stage1, stage2)

        validate_stage1_lineage_repair_result(stage1, stage2, repaired_stage1)

    def test_validation_rejects_remaining_missing_stage2_rows(self):
        stage1 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")], columns=COLUMNS
        )
        stage2 = pd.DataFrame(
            [row("paper:b", "paper", "https://example.test/b", "Crash")],
            columns=COLUMNS,
        )

        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_stage1_lineage_repair_result(stage1, stage2, stage1.copy())


if __name__ == "__main__":
    unittest.main()
