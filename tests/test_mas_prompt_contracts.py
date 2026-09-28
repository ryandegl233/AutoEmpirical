from __future__ import annotations

import hashlib

import pytest

from Benchmark.src import ase2022_llm_baseline as ase3
from Benchmark.src import ase2022_stage2_filter_baseline as ase2
from Benchmark.src import issta2024_bugs_in_pods_baseline as issta


CONTRACT = (
    "Return strict JSON containing the selected labels, a brief reason, "
    "and evidence_refs with exact source quotes."
)
ASE_TAXONOMY = {"symptom": ["Crash"], "root_cause": ["Incorrect Code Logic"]}


def _record() -> dict[str, str]:
    return {
        "title": "Failure at runtime",
        "state": "closed",
        "created_at": "2026-01-02",
        "body": "The call crashes.\nApply the fix.",
        "comments": "  Confirmed by reporter.  ",
        "source_project": "runc",
        "issue_url": "https://github.com/opencontainers/runc/commit/abc",
        "changed_files": '["libcontainer/init.go", "README.md"]',
        "code_diff": "@@ -1 +1 @@\n-old()\n+fixed()",
        "decision": "PRIVATE_DECISION",
        "symptom": "PRIVATE_SYMPTOM",
        "root_cause": "PRIVATE_ROOT_CAUSE",
        "gold_annotation": "PRIVATE_GOLD",
    }


def test_default_prompts_preserve_frozen_utf8_bytes() -> None:
    # Captured before the optional contract was implemented, using this fixture.
    expected = {
        "ase2_system": "49992f2a195425a74cf0199fb143c168598905882c8e4c9fbd5bef05f6e3e1fa",
        "ase3_system": "2c747346535a08e8f4133e33b1101dc6a69bb6e3ddbffebaba8855e5b1f41a11",
        "ase2_user": "47297085135cc054ddfe119a5790ae342b3d171a340fb82cdb13bd80ce4cd62b",
        "ase3_user": "3c1798e0d0938e6cbeb2d1fe4872ea1ae3aef0e733557a22e24cce7454c63346",
        "issta2_system": "0a5448d541b292ec0dc760e949481ee7c0b509e9ed7463bb8405e7130e58ca37",
        "issta3_system": "2c410c1b03161005d89a3f91689062bc980a9093e9535b0049b0a6bb58c4f017",
        "issta2_user": "8aea64b06b60b03cf9932aad9ee36125ad529da2363864279786bd6325e39ddf",
        "issta3_user": "63246fe88fd9765c496fb6168964df06a5c7203d874f7b58d7f743e61867ca5c",
        "issta2_task": "0500b04f8cfec77ef8c90675347ea7e2ba944de8a87afa2560b3b5b0037b208c",
        "issta3_task": "ca7757e3e54159442872877f8696c1748abaa8e532a675cde728b0e1989fc16f",
    }
    record = _record()
    taxonomy = issta.build_issta2024_taxonomy()
    actual = {
        "ase2_system": ase2.build_system_prompt(),
        "ase3_system": ase3.build_system_prompt(ASE_TAXONOMY),
        "ase2_user": ase2.build_user_prompt(record),
        "ase3_user": ase3.build_user_prompt(record),
        "issta2_system": issta.build_stage2_system_prompt(),
        "issta3_system": issta.build_stage3_system_prompt(taxonomy),
        "issta2_user": issta.build_stage2_user_prompt(record),
        "issta3_user": issta.build_stage3_user_prompt(record),
        "issta2_task": issta.build_society_task(record, "stage2", taxonomy),
        "issta3_task": issta.build_society_task(record, "stage3", taxonomy),
    }
    assert {
        name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for name, prompt in actual.items()
    } == expected


@pytest.mark.parametrize("domain", ["ase2", "ase3", "issta2", "issta3"])
def test_optional_contract_replaces_label_only_requirement(domain: str) -> None:
    builders = {
        "ase2": lambda: ase2.build_system_prompt(output_contract=CONTRACT),
        "ase3": lambda: ase3.build_system_prompt(
            ASE_TAXONOMY, output_contract=CONTRACT
        ),
        "issta2": lambda: issta.build_stage2_system_prompt(output_contract=CONTRACT),
        "issta3": lambda: issta.build_stage3_system_prompt(
            issta.build_issta2024_taxonomy(), output_contract=CONTRACT
        ),
    }
    prompt = builders[domain]()
    assert prompt.count(CONTRACT) == 1
    assert "Do not add explanations" not in prompt
    assert "with no explanation" not in prompt
    assert "exactly one key named decision" not in prompt
    assert "exactly the keys symptom and root_cause" not in prompt
    if domain == "ase3":
        assert '"symptom": "<one exact symptom label>"' not in prompt
        assert "The system runs but is slow" not in prompt
        assert "Functionality is terminated unexpectedly" in prompt
        assert "For symptom, focus on the observable behavior." in prompt
        assert "Incorrect Code Logic" in prompt
    if domain.startswith("issta"):
        assert "Use only the supplied commit evidence." in prompt
    if domain == "issta3":
        for labels in issta.build_issta2024_taxonomy().values():
            assert all(label in prompt for label in labels)
        assert "Definition:" in prompt


@pytest.mark.parametrize("stage", ["stage2", "stage3"])
def test_issta_society_contract_reaches_task_without_gold(stage: str) -> None:
    task = issta.build_society_task(
        _record(), stage, issta.build_issta2024_taxonomy(), output_contract=CONTRACT
    )
    assert task.count(CONTRACT) == 1
    assert "Do not add explanations" not in task
    assert "PRIVATE_" not in task
    assert "@@ -1 +1 @@\n-old()\n+fixed()" in task


def test_ase_evidence_fields_match_presented_values_without_gold() -> None:
    record = _record()
    fields = ase3.model_evidence_fields(record)
    assert fields == {
        "title": "Failure at runtime",
        "state": "closed",
        "created_at": "2026-01-02",
        "body": "The call crashes.\nApply the fix.",
        "comments": "  Confirmed by reporter.  ",
    }
    for prompt in (ase2.build_user_prompt(record), ase3.build_user_prompt(record)):
        assert all(value in prompt for value in fields.values())
        assert "PRIVATE_" not in prompt
    assert record == _record()


def test_issta_evidence_fields_match_rendered_comments_and_changed_files() -> None:
    record = _record()
    fields = issta.model_evidence_fields(record)
    assert fields == {
        "title": "Failure at runtime",
        "state": "closed",
        "created_at": "2026-01-02",
        "body": "The call crashes.\nApply the fix.",
        "comments": "Confirmed by reporter.",
        "source_project": "runc",
        "issue_url": "https://github.com/opencontainers/runc/commit/abc",
        "changed_files": "- libcontainer/init.go\n- README.md",
        "code_diff": "@@ -1 +1 @@\n-old()\n+fixed()",
    }
    assert all(value in issta.build_stage2_user_prompt(record) for value in fields.values())
    assert "PRIVATE_" not in str(fields)
    assert record == _record()


@pytest.mark.parametrize(
    ("comments", "changed_files"),
    [
        ("", "[]"),
        (" no_comments_in_source ", "invalid json"),
        ("not_available_in_source", '{"path":"not a file list"}'),
    ],
)
def test_issta_unavailable_evidence_matches_displayed_placeholders(
    comments: str, changed_files: str
) -> None:
    record = {**_record(), "comments": comments, "changed_files": changed_files}
    fields = issta.model_evidence_fields(record)
    assert fields["comments"] == "not available"
    assert fields["changed_files"] == "- not available"
    prompt = issta.build_stage3_user_prompt(record)
    assert "Available Discussion:\nnot available\n" in prompt
    assert "Changed Files:\n- not available\n" in prompt
