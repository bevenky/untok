"""Download immutable, hash-verified source files. No implicit upstream refresh."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical output contains no timestamp or machine-specific absolute paths.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError(f"Source path escapes cache: {relative}")
    return candidate


def verify_sources(lock_path: str | Path, cache: str | Path) -> list[dict]:
    lock = read_json(lock_path)
    results = []
    for item in lock["files"]:
        path = source_path(cache, item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or changed pinned source: {item['path']}")
        results.append({"path": item["path"], "sha256": item["sha256"]})
    return results


def fetch_sources(lock_path: str | Path, cache: str | Path, workers: int = 4) -> list[dict]:
    files = read_json(lock_path)["files"]
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("Duplicate source paths")

    def fetch(item):
        path = source_path(cache, item["path"])
        if path.exists():
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Cached source hash mismatch: {item['path']}; refusing to overwrite")
            return
        if not item["url"].startswith("https://"):
            raise ValueError("Source URL must use HTTPS")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(item["url"], timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if sha256(temporary) != item["sha256"]:
                raise ValueError(f"Downloaded source hash mismatch: {item['path']}")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, files))
    return verify_sources(lock_path, cache)
