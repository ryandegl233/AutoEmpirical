from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def model_ids_from_payload(payload: object) -> list[str]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        ids = []
        for item in payload["data"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
            elif isinstance(item, str):
                ids.append(item)
        return sorted(set(ids))
    if isinstance(payload, list):
        return sorted({item for item in payload if isinstance(item, str)})
    return []


def fetch_json_with_powershell(url: str, api_key: str) -> object:
    env = os.environ.copy()
    env["AE_LLM_API_KEY"] = api_key
    env["AE_LLM_URL"] = url
    script = r"""
$headers = @{
  Authorization = "Bearer $env:AE_LLM_API_KEY"
  Accept = "application/json"
}
$response = Invoke-RestMethod -Method Get -Uri $env:AE_LLM_URL -Headers $headers
$response | ConvertTo-Json -Depth 30
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(error)
    return json.loads(completed.stdout)


def fetch_json_with_urllib(url: str, api_key: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_models(
    base_url: str,
    api_key: str,
    http_client: str = "urllib",
) -> tuple[str, list[str]]:
    errors = []
    for suffix in ("/v1/models", "/models"):
        url = base_url.rstrip("/") + suffix
        try:
            if http_client == "powershell":
                payload = fetch_json_with_powershell(url, api_key)
            elif http_client == "urllib":
                payload = fetch_json_with_urllib(url, api_key)
            else:
                raise ValueError(f"Unsupported HTTP_CLIENT: {http_client}")
            return url, model_ids_from_payload(payload)
        except (
            urllib.error.URLError,
            json.JSONDecodeError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
        ) as exc:
            errors.append(f"{url}: {exc}")
    raise SystemExit("Could not fetch model list:\n" + "\n".join(errors))


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    base_url = os.environ.get("BASE_URL") or os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    http_client = os.environ.get("HTTP_CLIENT", "urllib")
    if not base_url:
        raise SystemExit("Missing BASE_URL in .env")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY or API_KEY in .env")

    url, models = fetch_models(base_url, api_key, http_client=http_client)
    print(f"Model endpoint: {url}")
    print(f"HTTP client: {http_client}")
    if not models:
        print("No model ids found in response.")
        return
    print("Available models:")
    for model in models:
        print(f"- {model}")


if __name__ == "__main__":
    main()
