"""Paper-faithful Stage 2 screening policies."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse
from Benchmark.src.annotation_contracts import NEW_PAPER_DOMAINS

from .contracts import Stage2Decision, Stage2EvidenceTests


ASE_PULL_REQUEST_SCOPE_BOUNDARY = (
    "ASE2022 studies issue reports; pull requests are outside scope."
)


def _value(outcome: object) -> str:
    return str(getattr(outcome, "value", outcome))


def stage2_acceptance_errors(
    domain: str,
    evidence_tests: Mapping[str, object],
) -> list[str]:
    """Return deterministic policy violations for an accepted decision."""

    errors: list[str] = []
    if _value(evidence_tests.get("fault_existence")) != "pass":
        errors.append(f"{domain} acceptance requires fault_existence=pass")
    if _value(evidence_tests.get("scope_exclusion")) != "pass":
        errors.append(f"{domain} acceptance requires scope_exclusion=pass")
    if (
        domain == "issta2024"
        and _value(evidence_tests.get("repair_causality")) != "pass"
    ):
        errors.append("issta2024 acceptance requires repair_causality=pass")
    return errors


def compose_stage2_decision(
    domain: str,
    tests: Stage2EvidenceTests,
) -> Stage2Decision:
    """Map named role outcomes to the sole authoritative Stage 2 decision."""

    errors = stage2_acceptance_errors(domain, tests)
    return Stage2Decision.REJECTED if errors else Stage2Decision.ACCEPTED


def stage2_test_policy_guidance(domain: str, named_test: str) -> str:
    """Return only the paper policy needed by one role-owned named test."""

    if domain in NEW_PAPER_DOMAINS:
        from Benchmark.src.paper_benchmark import get_paper_profile
        profile = get_paper_profile(domain)
        if named_test == "fault_existence":
            return (f"FAULT-EXISTENCE TEST ({domain}): Determine whether the frozen evidence establishes an observed technical fault. "
                    "Do not assign a final decision or annotation. An issue phrased as a question may still describe a fault.")
        if named_test == "study_scope":
            return (f"STUDY-SCOPE TEST ({domain}): {profile.filter_policy} "
                    "Assess this inclusion boundary only; never infer membership from record IDs or labels.")
        if named_test == "repair_causality":
            return (f"OPTIONAL REPAIR CONTEXT ({domain}): Assess direct repair evidence when present. "
                    "Absence of a patch cannot veto an otherwise eligible report.")
    if named_test == "fault_existence":
        if domain == "ase2022":
            return (
                "FAULT-EXISTENCE TEST (ASE2022): Determine only whether the frozen "
                "evidence describes a concrete technical failure or taxonomy-covered "
                "limitation. Questions with no observed failure, feature proposals, "
                "and unanchored maintenance do not pass. Return only this owned "
                "assessment and never assign a final decision or Stage 3 label."
            )
        if domain == "issta2024":
            return (
                "FAULT-EXISTENCE TEST (ISSTA2024): Determine only whether the frozen "
                "evidence establishes pre-existing incorrect software behavior. Do "
                "not evaluate any other gate. Return only this owned assessment and "
                "never assign a final decision or Stage 3 label."
            )
    if named_test == "study_scope":
        if domain == "ase2022":
            return (
                "STUDY-SCOPE TEST (ASE2022): Apply only the supplied issue-report "
                "inclusion and exclusion policy. Pull requests are outside scope. "
                "Treat the reported technical behavior as fixed input; do not reassess "
                "it or evaluate changes. Never assign a final decision or Stage 3 label."
            )
        if domain == "issta2024":
            return (
                "STUDY-SCOPE TEST (ISSTA2024): Determine only whether the candidate "
                "belongs to the studied container-runtime corpus. Treat other role "
                "claims as fixed inputs. Never assign a final decision or Stage 3 label."
            )
    if named_test == "repair_causality":
        if domain == "issta2024":
            return (
                "REPAIR-CAUSALITY TEST (ISSTA2024): Determine only whether the frozen "
                "commit evidence directly repairs the established pre-existing fault. "
                "Feature, cleanup, merge, documentation, and keyword-only changes do "
                "not pass. Return only this owned assessment and never assign a final "
                "decision or Stage 3 label."
            )
        if domain == "ase2022":
            return (
                "REPAIR-CAUSALITY TEST (ASE2022 OPTIONAL CONTEXT): Assess only causal "
                "repair evidence. This outcome cannot veto acceptance. Return only this "
                "owned assessment and never assign a final decision or Stage 3 label."
            )
    raise ValueError(f"unsupported Stage 2 test {named_test!r} for {domain!r}")


def artifact_scope_exclusion(
    domain: str,
    candidate_source_uri: str,
) -> str | None:
    """Return a deterministic study-scope exclusion, when one applies."""

    if domain != "ase2022":
        return None
    locator = candidate_source_uri.strip()
    if locator.casefold().startswith(("github.com/", "www.github.com/")):
        locator = f"https://{locator}"
    parsed = urlparse(locator)
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path_segments = [
        segment.casefold() for segment in parsed.path.split("/") if segment
    ]
    browser_pr = (
        host == "github.com"
        and len(path_segments) >= 4
        and path_segments[2] == "pull"
        and path_segments[3].isdigit()
    )
    api_pr = (
        host == "api.github.com"
        and len(path_segments) >= 5
        and path_segments[0] == "repos"
        and path_segments[3] == "pulls"
        and path_segments[4].isdigit()
    )
    if browser_pr or api_pr:
        return ASE_PULL_REQUEST_SCOPE_BOUNDARY
    return None


def stage2_policy_guidance(domain: str) -> str:
    if domain in NEW_PAPER_DOMAINS:
        from Benchmark.src.paper_benchmark import get_paper_profile
        return (f"STAGE 2 DOMAIN POLICY ({domain}): {get_paper_profile(domain).filter_policy} "
                "Acceptance requires fault_existence=pass and scope_exclusion=pass. "
                "Missing repair evidence is an evidence gap, not an automatic rejection. "
                "Use only supplied source evidence; never infer author dataset membership.")
    if domain == "ase2022":
        return (
            "STAGE 2 DOMAIN POLICY (ASE2022): This is a high-recall screen for "
            "a study-eligible fault case under the supplied ASE taxonomy, not "
            "a screen for confirmed library-internal code defects. Accept "
            "concrete technical problems or limitations that can plausibly be "
            "classified by an allowed symptom and root-cause label. In "
            "particular, API Misuse, Misconfiguration, Cross-platform App "
            "Framework Incompatibility, Browser or Device Incompatibility, "
            "WebGL Limits, Unimplemented Operator, dependency problems, and "
            "confusing documentation are eligible root causes even when the "
            "proximate cause is user, environment, platform, or support "
            "status. A named unsupported platform or integration limitation "
            "is compatibility fault evidence even if posed as a support "
            "question. A concrete warning, error, crash, inability to run or "
            "deploy, incorrect behavior, or severe performance obstacle is "
            "observable fault evidence. A workaround, maintainer closure, "
            "failure to reproduce, or lack of maintainer confirmation does "
            "not negate the reported case. Accept when fault_existence and "
            "scope_exclusion pass. A missing "
            "commit, pull request, patch, or regression test must not by "
            "itself cause rejection. Record absent repair evidence as "
            "repair_causality=unknown and an evidence gap; for this domain it "
            "only lowers confidence. Reject an open-ended usage or capability "
            "question when it describes no observed failure or concrete "
            "limitation. Also reject feature proposals and internal CI, test, "
            "release, or dependency maintenance when they are not anchored to "
            "a user-observed fault. Do not apply these exclusions merely "
            "because a genuine fault report is phrased as a question or its "
            "eventual fix involves tests or dependencies. Reject other "
            "requests with no concrete technical problem or taxonomy-covered "
            "limitation, unrelated enhancements or discussions, and reports "
            "too unclear to identify a study-eligible case. When evidence is "
            "ambiguous but supports a "
            "plausible taxonomy-covered case, prefer inclusion for later "
            "Stage 3 classification. "
            "evidence_sufficiency concerns fault existence and study scope, "
            "not whether a repair artifact is available."
        )
    if domain == "issta2024":
        return (
            "STAGE 2 DOMAIN POLICY (ISSTA2024): Decide whether the supplied "
            "commit explicitly repairs a pre-existing software fault in the "
            "studied container runtime. accepted_fault requires "
            "fault_existence=pass, repair_causality=pass, and "
            "scope_exclusion=pass. Reject feature additions without repair "
            "evidence, refactoring, cleanup, merge/release/documentation "
            "commits, test-only maintenance, and keyword-only fixes."
        )
    raise ValueError(f"unsupported domain {domain!r}")
