"""Shared configuration for official OpenAI-compatible LLM providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit


GEMINI_DEFAULT_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai"
)
GEMINI_PROVIDER_ALIASES = frozenset({"gemini", "gemeni", "genemi"})


def canonical_provider(value: str) -> str:
    """Return the manifest-safe provider name for accepted CLI aliases."""

    normalized = value.strip().lower()
    if normalized in GEMINI_PROVIDER_ALIASES:
        return "gemini"
    return normalized


def normalize_gemini_base_url(value: str) -> str:
    """Bind Gemini credentials to Google's official compatibility endpoint."""

    normalized = value.strip().rstrip("/")
    if not normalized:
        raise SystemExit("Missing GEMINI_BASE_URL or --base-url")
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() != "https":
        raise ValueError("Gemini provider requires an HTTPS base URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Gemini provider base URL cannot contain userinfo")
    if (parsed.hostname or "").lower() != "generativelanguage.googleapis.com":
        raise ValueError("Gemini provider requires the official Gemini host")
    if parsed.port not in {None, 443}:
        raise ValueError("Gemini provider base URL cannot use a custom port")
    if parsed.query or parsed.fragment:
        raise ValueError("Gemini provider base URL cannot contain a query or fragment")
    if parsed.path != "/v1beta/openai":
        raise ValueError("Gemini provider base path must be /v1beta/openai")
    return GEMINI_DEFAULT_BASE_URL


def resolve_gemini_config(
    env: Mapping[str, str],
    *,
    base_url_override: str | None = None,
) -> dict[str, str]:
    """Resolve official Gemini credentials without consulting proxy secrets."""

    api_key = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment"
        )
    return {
        "base_url": normalize_gemini_base_url(
            base_url_override
            or env.get("GEMINI_BASE_URL")
            or GEMINI_DEFAULT_BASE_URL
        ),
        "api_key": api_key,
    }


def resolve_selected_models(
    provider: str,
    selected_models: Sequence[str] | None,
    default_models: Sequence[str],
) -> list[str]:
    """Require an explicit Gemini model while preserving legacy defaults."""

    if selected_models:
        return list(selected_models)
    if canonical_provider(provider) == "gemini":
        raise SystemExit("--provider gemini requires explicit --models")
    return list(default_models)
