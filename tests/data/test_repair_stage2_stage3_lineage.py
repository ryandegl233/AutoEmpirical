import json
import unittest

import pandas as pd

from Dataset.scripts.repair_stage2_stage3_lineage import (
    FINAL_LABEL_COLUMNS,
    apply_lineage_repair,
    build_repair_plan,
    validate_repair_result,
)


COLUMNS = [
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
]


def row(record_id, paper_id, issue_url, symptom="", source_row_index="1"):
    values = {
        "record_id": record_id,
        "paper_id": paper_id,
        "source_project": "project",
        "issue_url": issue_url,
        "title": f"title {record_id}",
        "body": f"body {record_id}",
        "comments": "comments",
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-02T00:00:00+00:00",
        "state": "closed",
        "symptom": symptom,
        "root_cause": "cause" if symptom else "",
        "bug_type": "type" if symptom else "",
        "component": "component" if symptom else "",
        "sub_component": "sub" if symptom else "",
        "trigger_condition": "trigger" if symptom else "",
        "consequence": "consequence" if symptom else "",
        "fix_type": "fix" if symptom else "",
        "severity_or_impact": "major" if symptom else "",
        "original_label_json": json.dumps({"stage": "stage3" if symptom else "stage2"}),
        "source_file": "source.csv",
        "source_sheet": "sheet",
        "source_row_index": source_row_index,
    }
    return [values[column] for column in COLUMNS]


class LineageRepairTests(unittest.TestCase):
    def test_plan_aligns_ids_and_finds_missing_stage2_rows(self):
        stage2 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [
                row("paper:old-a", "paper", "https://example.test/a", "Crash"),
                row("paper:b", "paper", "https://example.test/b", "Hang"),
            ],
            columns=COLUMNS,
        )

        plan = build_repair_plan(stage2, stage3)

        self.assertEqual(plan.alignment_count, 1)
        self.assertEqual(plan.missing_count, 1)
        self.assertEqual(plan.missing_keys, [("paper", "https://example.test/b")])

    def test_repair_preserves_stage3_labels_and_clears_inserted_stage2_labels(self):
        stage2 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [
                row("paper:old-a", "paper", "https://example.test/a", "Crash"),
                row("paper:b", "paper", "https://example.test/b", "Hang"),
            ],
            columns=COLUMNS,
        )
        original_labels = stage3[FINAL_LABEL_COLUMNS].copy(deep=True)

        repaired_stage2, repaired_stage3, audit = apply_lineage_repair(stage2, stage3)

        self.assertEqual(len(repaired_stage2), 2)
        self.assertEqual(len(repaired_stage3), 2)
        pd.testing.assert_frame_equal(
            repaired_stage3[FINAL_LABEL_COLUMNS].reset_index(drop=True),
            original_labels.reset_index(drop=True),
        )
        inserted = repaired_stage2.loc[
            repaired_stage2["issue_url"] == "https://example.test/b"
        ].iloc[0]
        self.assertTrue(all(inserted[column] == "" for column in FINAL_LABEL_COLUMNS))
        self.assertEqual(
            repaired_stage3.loc[
                repaired_stage3["issue_url"] == "https://example.test/a", "record_id"
            ].iloc[0],
            "paper:a",
        )
        self.assertEqual(set(audit["repair_type"]), {"record_id_aligned", "stage2_row_inserted"})

    def test_repair_is_idempotent(self):
        stage2 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [row("paper:old-a", "paper", "https://example.test/a", "Crash")],
            columns=COLUMNS,
        )

        repaired_stage2, repaired_stage3, _ = apply_lineage_repair(stage2, stage3)
        second_stage2, second_stage3, second_audit = apply_lineage_repair(
            repaired_stage2, repaired_stage3
        )

        pd.testing.assert_frame_equal(second_stage2, repaired_stage2)
        pd.testing.assert_frame_equal(second_stage3, repaired_stage3)
        self.assertTrue(second_audit.empty)

    def test_plan_rejects_ambiguous_stage2_keys(self):
        stage2 = pd.DataFrame(
            [
                row("paper:a1", "paper", "https://example.test/a"),
                row("paper:a2", "paper", "https://example.test/a"),
            ],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [row("paper:a3", "paper", "https://example.test/a", "Crash")],
            columns=COLUMNS,
        )

        with self.assertRaisesRegex(ValueError, "ambiguous Stage 2"):
            build_repair_plan(stage2, stage3)

    def test_validation_accepts_complete_repair(self):
        stage2 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [
                row("paper:old-a", "paper", "https://example.test/a", "Crash"),
                row("paper:b", "paper", "https://example.test/b", "Hang"),
            ],
            columns=COLUMNS,
        )
        repaired_stage2, repaired_stage3, _ = apply_lineage_repair(stage2, stage3)

        validate_repair_result(stage2, stage3, repaired_stage2, repaired_stage3)

    def test_validation_rejects_stage3_label_changes(self):
        stage2 = pd.DataFrame(
            [row("paper:a", "paper", "https://example.test/a")],
            columns=COLUMNS,
        )
        stage3 = pd.DataFrame(
            [row("paper:old-a", "paper", "https://example.test/a", "Crash")],
            columns=COLUMNS,
        )
        repaired_stage2, repaired_stage3, _ = apply_lineage_repair(stage2, stage3)
        repaired_stage3.loc[0, "symptom"] = "Changed"

        with self.assertRaisesRegex(ValueError, "Stage 3 labels changed"):
            validate_repair_result(stage2, stage3, repaired_stage2, repaired_stage3)


if __name__ == "__main__":
    unittest.main()
