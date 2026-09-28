import copy
from pathlib import Path
import tempfile
import unittest

from Dataset.scripts.repair_txbug_stage1_evidence import (
    apply_comments_to_rows,
    build_github_comments_query,
    normalize_evidence_comment,
    parse_github_comments_response,
    read_jsonl,
    write_jsonl,
)


class NormalizeEvidenceCommentTests(unittest.TestCase):
    def test_distinguishes_verified_zero_from_unavailable(self):
        self.assertEqual(
            "no_comments_in_source",
            normalize_evidence_comment("ok_zero_comments", ""),
        )
        self.assertEqual(
            "comments_unavailable_in_source",
            normalize_evidence_comment("source_unavailable", ""),
        )
        self.assertEqual(
            "developer discussion",
            normalize_evidence_comment("ok", " developer discussion "),
        )

    def test_refuses_collection_failure(self):
        with self.assertRaisesRegex(ValueError, "collection_failed"):
            normalize_evidence_comment("collection_failed", "")


class ApplyCommentsTests(unittest.TestCase):
    def test_requires_complete_evidence_and_changes_only_comments(self):
        original_rows = [
            {
                "record_id": "paper:a",
                "title": "A",
                "body": "body A",
                "comments": "",
                "root_cause": "",
            },
            {
                "record_id": "paper:b",
                "title": "B",
                "body": "body B",
                "comments": "old",
                "root_cause": "gold",
            },
        ]
        before = copy.deepcopy(original_rows)
        evidence = {
            "paper:a": {"retrieval_status": "ok", "comments": "new A"},
            "paper:b": {
                "retrieval_status": "ok_zero_comments",
                "comments": "",
            },
        }

        result = apply_comments_to_rows(original_rows, evidence)

        self.assertEqual(2, result["target_rows"])
        self.assertEqual(2, result["updated_rows"])
        self.assertEqual("new A", original_rows[0]["comments"])
        self.assertEqual("no_comments_in_source", original_rows[1]["comments"])
        for index, row in enumerate(original_rows):
            for field in row:
                if field != "comments":
                    self.assertEqual(before[index][field], row[field])

        with self.assertRaisesRegex(ValueError, "missing evidence"):
            apply_comments_to_rows(copy.deepcopy(before), {"paper:a": evidence["paper:a"]})


class GitHubGraphQLTests(unittest.TestCase):
    def test_builds_bounded_alias_query_and_parses_comments(self):
        query = build_github_comments_query(
            "pingcap",
            "tidb",
            [10, 20],
            comments_first=2,
        )
        self.assertIn("issue_10: issue(number: 10)", query)
        self.assertIn("issue_20: issue(number: 20)", query)
        self.assertIn("comments(first: 2)", query)

        payload = {
            "data": {
                "repository": {
                    "issue_10": {
                        "url": "https://github.com/pingcap/tidb/issues/10",
                        "comments": {
                            "nodes": [
                                {
                                    "author": {"login": "alice"},
                                    "body": "first",
                                    "createdAt": "2022-01-01T00:00:00Z",
                                    "updatedAt": "2022-01-01T00:00:00Z",
                                    "url": "https://github.com/pingcap/tidb/issues/10#issuecomment-1",
                                }
                            ],
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": "cursor-10",
                            },
                            "totalCount": 1,
                        },
                    },
                    "issue_20": None,
                }
            }
        }

        parsed = parse_github_comments_response(payload, [10, 20])

        self.assertEqual("ok", parsed[10]["retrieval_status"])
        self.assertIn("alice", parsed[10]["comments"])
        self.assertEqual(1, parsed[10]["source_comment_count"])
        self.assertEqual("source_unavailable", parsed[20]["retrieval_status"])


class JsonlCacheTests(unittest.TestCase):
    def test_round_trips_unicode_line_separator_inside_json_string(self):
        rows = [{"record_id": "paper:a", "comments": "before\u2028after"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(rows, read_jsonl(path))


if __name__ == "__main__":
    unittest.main()
