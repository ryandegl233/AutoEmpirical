import unittest

import pandas as pd

from Dataset.scripts.repair_stage1_stage2_record_ids import (
    apply_record_id_repair,
    build_record_id_repair_plan,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["paper_id", "issue_url", "record_id", "symptom"])


class Stage1Stage2RecordIdRepairTests(unittest.TestCase):
    def test_repairs_stage2_and_stage3_using_stage1_id(self):
        stage1 = frame([["paper", "url-a", "paper:stage1-a", ""]])
        stage2 = frame([["paper", "url-a", "paper:stage2-a", ""]])
        stage3 = frame([["paper", "url-a", "paper:stage2-a", "Crash"]])

        repaired_stage2, repaired_stage3, audit = apply_record_id_repair(
            stage1, stage2, stage3, paper_id="paper"
        )

        self.assertEqual(repaired_stage2.loc[0, "record_id"], "paper:stage1-a")
        self.assertEqual(repaired_stage3.loc[0, "record_id"], "paper:stage1-a")
        self.assertEqual(repaired_stage3.loc[0, "symptom"], "Crash")
        self.assertEqual(len(audit), 1)

    def test_plan_ignores_urls_missing_from_stage1(self):
        stage1 = frame([["paper", "url-a", "paper:a", ""]])
        stage2 = frame([["paper", "url-b", "paper:b", ""]])
        stage3 = frame([["paper", "url-b", "paper:b", "Crash"]])

        plan = build_record_id_repair_plan(stage1, stage2, paper_id="paper")

        self.assertEqual(plan.change_count, 0)

    def test_plan_rejects_ambiguous_stage1_url(self):
        stage1 = frame(
            [
                ["paper", "url-a", "paper:a1", ""],
                ["paper", "url-a", "paper:a2", ""],
            ]
        )
        stage2 = frame([["paper", "url-a", "paper:b", ""]])

        with self.assertRaisesRegex(ValueError, "ambiguous Stage 1"):
            build_record_id_repair_plan(stage1, stage2, paper_id="paper")

    def test_repair_is_idempotent(self):
        stage1 = frame([["paper", "url-a", "paper:a", ""]])
        stage2 = frame([["paper", "url-a", "paper:b", ""]])
        stage3 = frame([["paper", "url-a", "paper:b", "Crash"]])
        stage2, stage3, _ = apply_record_id_repair(
            stage1, stage2, stage3, paper_id="paper"
        )

        second_stage2, second_stage3, audit = apply_record_id_repair(
            stage1, stage2, stage3, paper_id="paper"
        )

        pd.testing.assert_frame_equal(second_stage2, stage2)
        pd.testing.assert_frame_equal(second_stage3, stage3)
        self.assertTrue(audit.empty)


if __name__ == "__main__":
    unittest.main()
