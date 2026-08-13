from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    """Return a content fingerprint without loading a whole source file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(paths: Iterable[Path], root: Path | None = None) -> str:
    """Hash sorted paths and their contents so ordering is stable across machines."""
    digest = hashlib.sha256()
    root = root.resolve() if root is not None else None
    for path in sorted((item.resolve() for item in paths), key=lambda item: str(item).lower()):
        name = str(path.relative_to(root) if root and path.is_relative_to(root) else path).replace("\\", "/")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
