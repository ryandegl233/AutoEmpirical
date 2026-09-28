"""Behavior tests for the ignored ICSE 2021 IoT discussion repair tool."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


try:
    repair = importlib.import_module("Dataset.scripts.repair_iot_discussion_evidence")
except ModuleNotFoundError:
    repair = None

try:
    audit = importlib.import_module("Dataset.scripts.audit_iot_information_patch")
except ModuleNotFoundError:
    audit = None


class RepairModuleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(
            repair,
            "Dataset.scripts.repair_iot_discussion_evidence is not implemented",
        )


class GitHubUrlTests(RepairModuleTestCase):
    def test_preserves_issue_and_pull_request_kinds(self) -> None:
        self.assertEqual(
            repair.parse_github_record_url(
                "https://github.com/256dpi/arduino-mqtt/issues/135"
            ),
            ("256dpi", "arduino-mqtt", 135, "issue"),
        )
        self.assertEqual(
            repair.parse_github_record_url(
                "https://github.com/eclipse/smarthome/pull/1504"
            ),
            ("eclipse", "smarthome", 1504, "pull_request"),
        )


class GraphQLCollectionTests(RepairModuleTestCase):
    def test_initial_query_requests_every_discussion_channel(self) -> None:
        builder = getattr(repair, "build_initial_discussion_query", None)
        self.assertIsNotNone(builder, "initial GraphQL query builder is missing")
        query = builder(
            "owner",
            "repository",
            [(7, "issue"), (8, "pull_request")],
        )

        self.assertIn("record_7: issueOrPullRequest(number: 7)", query)
        self.assertIn("record_8: issueOrPullRequest(number: 8)", query)
        self.assertIn("... on Issue", query)
        self.assertIn("... on PullRequest", query)
        self.assertIn("comments(first: 100)", query)
        self.assertIn("reviews(first: 100)", query)
        self.assertIn("reviewThreads(first: 100)", query)

    def test_finds_top_level_and_nested_pagination_tasks(self) -> None:
        task_finder = getattr(repair, "find_pagination_tasks", None)
        self.assertIsNotNone(
            task_finder,
            "pagination task detection is missing",
        )
        payload = {
            "__typename": "PullRequest",
            "id": "pull-1",
            "comments": {
                "pageInfo": {"hasNextPage": True, "endCursor": "comments-c"},
                "nodes": [],
            },
            "reviews": {
                "pageInfo": {"hasNextPage": True, "endCursor": "reviews-c"},
                "nodes": [],
            },
            "reviewThreads": {
                "pageInfo": {"hasNextPage": True, "endCursor": "threads-c"},
                "nodes": [
                    {
                        "id": "thread-1",
                        "comments": {
                            "pageInfo": {
                                "hasNextPage": True,
                                "endCursor": "thread-comments-c",
                            },
                            "nodes": [],
                        },
                    }
                ],
            },
        }

        self.assertEqual(
            task_finder(payload),
            [
                {
                    "node_id": "pull-1",
                    "connection": "comments",
                    "cursor": "comments-c",
                },
                {
                    "node_id": "pull-1",
                    "connection": "reviews",
                    "cursor": "reviews-c",
                },
                {
                    "node_id": "pull-1",
                    "connection": "reviewThreads",
                    "cursor": "threads-c",
                },
                {
                    "node_id": "thread-1",
                    "connection": "comments",
                    "cursor": "thread-comments-c",
                },
            ],
        )

    def test_merges_nested_comment_page_without_losing_prior_nodes(self) -> None:
        merger = getattr(repair, "merge_connection_page", None)
        self.assertIsNotNone(merger, "connection page merger is missing")
        payload = {
            "__typename": "PullRequest",
            "id": "pull-1",
            "reviewThreads": {
                "nodes": [
                    {
                        "id": "thread-1",
                        "comments": {
                            "nodes": [{"id": "comment-1", "body": "first"}],
                            "pageInfo": {
                                "hasNextPage": True,
                                "endCursor": "cursor-1",
                            },
                        },
                    }
                ]
            },
        }
        page = {
            "nodes": [{"id": "comment-2", "body": "second"}],
            "pageInfo": {"hasNextPage": False, "endCursor": "cursor-2"},
        }

        merger(
            payload,
            {
                "node_id": "thread-1",
                "connection": "comments",
                "cursor": "cursor-1",
            },
            page,
        )

        comments = payload["reviewThreads"]["nodes"][0]["comments"]
        self.assertEqual(
            [node["id"] for node in comments["nodes"]],
            ["comment-1", "comment-2"],
        )
        self.assertEqual(
            comments["pageInfo"],
            {"hasNextPage": False, "endCursor": "cursor-2"},
        )

    def test_resilient_batch_isolates_one_unavailable_record(self) -> None:
        collector = getattr(repair, "collect_resilient_batch", None)
        self.assertIsNotNone(
            collector,
            "resilient batch collection is missing",
        )
        rows = [
            {
                "record_id": "iot:1",
                "paper_id": "iot",
                "source_project": "iot_projects",
                "issue_url": "https://github.com/owner/repository/issues/1",
            },
            {
                "record_id": "iot:2",
                "paper_id": "iot",
                "source_project": "iot_projects",
                "issue_url": "https://github.com/owner/repository/issues/2",
            },
        ]

        def batch_collector(owner, repository, batch):
            if any(row["record_id"] == "iot:2" for row in batch):
                raise RuntimeError(
                    "Could not resolve to an issue or pull request"
                )
            return [
                {
                    "record_id": row["record_id"],
                    "retrieval_status": "ok",
                }
                for row in batch
            ]

        result = collector(
            "owner",
            "repository",
            rows,
            batch_collector=batch_collector,
        )

        self.assertEqual(
            {
                row["record_id"]: row["retrieval_status"] for row in result
            },
            {"iot:1": "ok", "iot:2": "source_unavailable"},
        )


class DiscussionCompositionTests(RepairModuleTestCase):
    def test_combines_and_sorts_all_pull_request_discussion_channels(self) -> None:
        payload = {
            "__typename": "PullRequest",
            "comments": {
                "nodes": [
                    {
                        "id": "conversation-1",
                        "createdAt": "2020-01-03T00:00:00Z",
                        "author": {"login": "carol"},
                        "body": "conversation",
                    }
                ]
            },
            "reviews": {
                "nodes": [
                    {
                        "id": "review-1",
                        "submittedAt": "2020-01-01T00:00:00Z",
                        "author": {"login": "alice"},
                        "body": "review body",
                        "state": "CHANGES_REQUESTED",
                    }
                ]
            },
            "reviewThreads": {
                "nodes": [
                    {
                        "comments": {
                            "nodes": [
                                {
                                    "id": "review-comment-1",
                                    "createdAt": "2020-01-02T00:00:00Z",
                                    "author": {"login": "bob"},
                                    "body": "inline comment",
                                },
                                {
                                    "id": "conversation-1",
                                    "createdAt": "2020-01-03T00:00:00Z",
                                    "author": {"login": "carol"},
                                    "body": "conversation",
                                },
                            ]
                        }
                    }
                ]
            },
        }

        comments, count = repair.compose_discussion(payload)

        self.assertEqual(count, 3)
        self.assertEqual(
            comments,
            "[2020-01-01T00:00:00Z | alice | review:CHANGES_REQUESTED]\n"
            "review body\n\n"
            "[2020-01-02T00:00:00Z | bob | review_comment]\n"
            "inline comment\n\n"
            "[2020-01-03T00:00:00Z | carol | comment]\n"
            "conversation",
        )

    def test_verified_zero_and_unavailable_have_different_sentinels(self) -> None:
        self.assertEqual(
            repair.normalize_evidence_comment(
                {"retrieval_status": "ok_zero_comments", "comments": ""}
            ),
            "no_comments_in_source",
        )
        self.assertEqual(
            repair.normalize_evidence_comment(
                {"retrieval_status": "source_unavailable", "comments": ""}
            ),
            "comments_unavailable_in_source",
        )
        with self.assertRaisesRegex(ValueError, "collection_failed"):
            repair.normalize_evidence_comment(
                {"retrieval_status": "collection_failed", "comments": ""}
            )

    def test_rest_pull_request_channels_map_to_unified_discussion(self) -> None:
        mapper = getattr(repair, "build_payload_from_rest", None)
        self.assertIsNotNone(mapper, "REST discussion mapper is missing")
        payload = mapper(
            {"pull_request": {"url": "api/pulls/8"}, "node_id": "PR_8"},
            [
                {
                    "node_id": "IC_1",
                    "created_at": "2020-01-03T00:00:00Z",
                    "user": {"login": "carol"},
                    "body": "conversation",
                }
            ],
            [
                {
                    "node_id": "R_1",
                    "submitted_at": "2020-01-01T00:00:00Z",
                    "user": {"login": "alice"},
                    "body": "review body",
                    "state": "CHANGES_REQUESTED",
                }
            ],
            [
                {
                    "node_id": "RC_1",
                    "created_at": "2020-01-02T00:00:00Z",
                    "user": {"login": "bob"},
                    "body": "inline comment",
                }
            ],
        )

        comments, count = repair.compose_discussion(payload)
        self.assertEqual(count, 3)
        self.assertIn("| review:CHANGES_REQUESTED]", comments)
        self.assertIn("| review_comment]", comments)
        self.assertIn("| comment]", comments)


class ApplyCommentsTests(RepairModuleTestCase):
    def test_requires_complete_evidence_and_changes_only_comments(self) -> None:
        rows = [
            {
                "record_id": "iot:1",
                "paper_id": "iot",
                "title": "historical title",
                "body": "historical body",
                "comments": "no_comments_in_source",
                "root_cause": "gold label",
            },
            {
                "record_id": "other:1",
                "paper_id": "other",
                "title": "other title",
                "body": "other body",
                "comments": "other comments",
                "root_cause": "other label",
            },
        ]
        evidence = {
            "iot:1": {
                "record_id": "iot:1",
                "retrieval_status": "ok",
                "comments": "developer discussion",
            }
        }

        updated, count = repair.apply_comments_to_rows(rows, evidence, "iot")

        self.assertEqual(count, 1)
        self.assertEqual(
            updated[0],
            {
                "record_id": "iot:1",
                "paper_id": "iot",
                "title": "historical title",
                "body": "historical body",
                "comments": "developer discussion",
                "root_cause": "gold label",
            },
        )
        self.assertEqual(updated[1], rows[1])

        with self.assertRaisesRegex(ValueError, "missing evidence"):
            repair.apply_comments_to_rows(rows, {}, "iot")


class JsonlCacheTests(RepairModuleTestCase):
    def test_round_trips_unicode_line_separator_inside_json_string(self) -> None:
        rows = [{"record_id": "iot:1", "comments": "left\u2028right"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            repair.write_jsonl(path, rows)
            self.assertEqual(repair.read_jsonl(path), rows)


class SecretRedactionTests(RepairModuleTestCase):
    def test_redacts_azure_iot_shared_access_key_value(self) -> None:
        cleaned, findings = repair.redact_high_confidence_secrets(
            "HostName=example.azure-devices.net;"
            "DeviceId=sensor-1;"
            "SharedAccessKey=c3ludGhldGljLWtleS1tYXRlcmlhbA=="
        )

        self.assertEqual(
            cleaned,
            "HostName=example.azure-devices.net;"
            "DeviceId=sensor-1;"
            "SharedAccessKey="
            "[REDACTED_HIGH_CONFIDENCE_SECRET:"
            "azure_iot_shared_access_key]",
        )
        self.assertEqual(
            findings,
            {"azure_iot_shared_access_key": 1},
        )
        self.assertEqual(
            repair.redact_high_confidence_secrets(cleaned),
            (cleaned, {}),
        )

    def test_preserves_repeated_character_azure_key_placeholder(self) -> None:
        text = (
            "HostName=example.azure-devices.net;"
            "SharedAccessKey=" + ("x" * 44)
        )

        self.assertEqual(
            repair.redact_high_confidence_secrets(text),
            (text, {}),
        )

    def test_redacts_secret_values_outside_comments_in_dataset_rows(
        self,
    ) -> None:
        sanitizer = getattr(repair, "redact_secrets_in_rows", None)
        self.assertIsNotNone(sanitizer, "dataset row sanitizer is missing")
        rows = [
            {
                "record_id": "iot:1",
                "paper_id": "iot-paper",
                "body": (
                    "SharedAccessKey="
                    "c3ludGhldGljLWtleS1tYXRlcmlhbA=="
                ),
                "comments": "unchanged",
            },
            {
                "record_id": "other:1",
                "paper_id": "other-paper",
                "body": (
                    "SharedAccessKey="
                    "c3ludGhldGljLWtleS1tYXRlcmlhbA=="
                ),
                "comments": "unchanged",
            },
        ]

        cleaned, redactions = sanitizer(rows, "iot-paper")

        self.assertEqual(
            cleaned[0]["body"],
            "SharedAccessKey="
            "[REDACTED_HIGH_CONFIDENCE_SECRET:"
            "azure_iot_shared_access_key]",
        )
        self.assertEqual(cleaned[1], rows[1])
        self.assertEqual(
            redactions,
            [
                {
                    "record_id": "iot:1",
                    "field": "body",
                    "patterns": {"azure_iot_shared_access_key": 1},
                }
            ],
        )

    def test_redacts_complete_private_key_block_and_aws_access_key_id(self) -> None:
        redactor = getattr(repair, "redact_high_confidence_secrets", None)
        self.assertIsNotNone(redactor, "secret redactor is missing")
        text = (
            "before\n"
            "-----BEGIN PRIVATE KEY-----\n"
            "synthetic-key-material\n"
            "-----END PRIVATE KEY-----\n"
            "id=AKIAABCDEFGHIJKLMNOP\nafter"
        )

        cleaned, findings = redactor(text)

        self.assertEqual(
            cleaned,
            "before\n"
            "[REDACTED_HIGH_CONFIDENCE_SECRET:private_key]\n"
            "id=[REDACTED_HIGH_CONFIDENCE_SECRET:aws_access_key]\nafter",
        )
        self.assertEqual(
            findings,
            {"private_key": 1, "aws_access_key": 1},
        )

    def test_redacts_unterminated_private_key_from_header_to_end(self) -> None:
        cleaned, findings = repair.redact_high_confidence_secrets(
            "before\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "synthetic-key-material-without-end"
        )

        self.assertEqual(
            cleaned,
            "before\n[REDACTED_HIGH_CONFIDENCE_SECRET:private_key]",
        )
        self.assertEqual(findings, {"private_key": 1})


class SecretAuditTests(unittest.TestCase):
    def test_detects_azure_iot_shared_access_key_value(self) -> None:
        self.assertIsNotNone(audit, "IoT patch audit is not implemented")
        detector = getattr(audit, "raw_secret_counts", None)
        self.assertIsNotNone(detector, "raw secret detector is missing")

        self.assertEqual(
            detector(
                "HostName=example.azure-devices.net;"
                "SharedAccessKeyName=service;"
                "SharedAccessKey=c3ludGhldGljLWtleS1tYXRlcmlhbA=="
            ),
            {
                "aws_access_key": 0,
                "azure_iot_shared_access_key": 1,
                "private_key_header": 0,
            },
        )

    def test_computes_expected_security_only_body_redaction(self) -> None:
        redactor = getattr(audit, "redact_publication_secrets", None)
        self.assertIsNotNone(
            redactor,
            "independent publication redactor is missing",
        )
        placeholder = "SharedAccessKey=" + ("x" * 44)
        secret = (
            "SharedAccessKey="
            "c3ludGhldGljLWtleS1tYXRlcmlhbA=="
        )

        self.assertEqual(redactor(placeholder), placeholder)
        self.assertEqual(
            redactor(secret),
            "SharedAccessKey="
            "[REDACTED_HIGH_CONFIDENCE_SECRET:"
            "azure_iot_shared_access_key]",
        )


if __name__ == "__main__":
    unittest.main()
