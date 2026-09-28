"""Explicit offline-build CLI for frozen GitHub evidence bundles.

This command is deliberately separate from every runtime workflow entrypoint.
It performs network I/O only after the operator supplies ``--allow-network``;
the resulting manifest still remains unusable at runtime until separately
reviewed and registered as a trust root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Benchmark.src.adaptive_empirical_workflow.evidence_capture as capture_module  # noqa: E402
from Benchmark.src.adaptive_empirical_workflow.evidence_capture import (  # noqa: E402
    CaptureLimitError,
    CaptureLimits,
    CaptureRequest,
    GitHubResponse,
    GitHubTransport,
    capture_frozen_evidence_bundle,
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


class UrllibGitHubTransport:
    """Small redirect-disabled REST transport used only by this CLI."""

    def __init__(
        self,
        token: str | None,
        *,
        timeout_seconds: float = 30.0,
        opener: Any | None = None,
    ) -> None:
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._opener = (
            opener
            if opener is not None
            else urllib.request.build_opener(_NoRedirectHandler())
        )

    def get_json(self, url: str, *, max_bytes: int) -> GitHubResponse:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("raw response byte bound must be a positive integer")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "api.github.com"
            or parsed.hostname != "api.github.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            raise ValueError("offline capture transport accepts only api.github.com")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "adaptive-empirical-expert-workflow-evidence-freezer/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            response = self._opener.open(request, timeout=self._timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise CaptureLimitError(
                    "GitHub raw response bytes exceed the configured bound"
                )
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("GitHub API returned non-JSON content") from error
            response_headers = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            return GitHubResponse(
                status_code=int(response.status),
                final_url=str(response.geturl()),
                headers=response_headers,
                payload=payload,
                raw_size_bytes=len(raw),
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线冻结 ASE2022 GitHub 一跳证据（不注册运行时信任根）"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="显式授权本次离线构建访问 api.github.com",
    )
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-maintainer-references", type=int, default=8)
    parser.add_argument("--max-response-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-total-response-bytes", type=int, default=20_000_000)
    return parser


def _load_request(path: Path) -> CaptureRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("无法读取有效的 capture request JSON") from error
    request = CaptureRequest.model_validate(payload)
    observed_module_sha = hashlib.sha256(
        Path(capture_module.__file__).read_bytes()
    ).hexdigest()
    if request.retrieval_tool_module_sha256 != observed_module_sha:
        raise ValueError("capture request 的 module_sha256 与当前采集器不一致")
    return request


def main(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: Callable[[str | None], GitHubTransport] = UrllibGitHubTransport,
) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_network:
        print(
            "拒绝执行：离线证据构建必须显式提供 --allow-network；运行时不会联网。",
            file=sys.stderr,
        )
        return 2
    try:
        request = _load_request(args.request)
        limits = CaptureLimits(
            max_pages_per_endpoint=args.max_pages,
            max_nodes_per_record=args.max_nodes,
            max_files_per_pull=args.max_files,
            max_maintainer_references_per_record=args.max_maintainer_references,
            max_response_bytes=args.max_response_bytes,
            max_total_response_bytes=args.max_total_response_bytes,
        )
        token = os.environ.get(args.token_env) if args.token_env else None
        result = capture_frozen_evidence_bundle(
            request,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            transport=transport_factory(token),
            limits=limits,
        )
    except (OSError, ValueError) as error:
        print(f"冻结失败：{error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
