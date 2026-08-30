"""Reproducible download of the public FinQA splits."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

from finqa_rag.common.atomic import atomic_write_json


FINQA_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_BASE_URL = (
    "https://raw.githubusercontent.com/czyssrs/FinQA/"
    f"{FINQA_REVISION}/dataset"
)
FINQA_FILES = ("train.json", "dev.json", "test.json")


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download_finqa(
    output_dir: Path,
    *,
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> dict[str, object]:
    """Download pinned public splits and record content hashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for filename in FINQA_FILES:
        url = f"{FINQA_BASE_URL}/{filename}"
        content = fetch_bytes(url)
        target = output_dir / filename
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(content)
        partial.replace(target)
        files[filename] = {
            "url": url,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    manifest: dict[str, object] = {
        "revision": FINQA_REVISION,
        "files": files,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
