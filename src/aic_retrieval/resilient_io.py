from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import IO, Any


TRANSIENT_STORAGE_ERRNOS = {5, 22, 32}
TRANSIENT_STORAGE_WINERRORS = {21, 433}


def open_with_retry(
    path: Path,
    mode: str,
    *,
    attempts: int = 8,
    base_delay_seconds: float = 0.05,
    **kwargs: Any,
) -> IO[Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return path.open(mode, **kwargs)
        except OSError as exc:
            if not _is_transient_storage_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, base_delay_seconds * (2**attempt)))
    raise AssertionError("unreachable")


def read_text_with_retry(path: Path, *, encoding: str = "utf-8", attempts: int = 8) -> str:
    for attempt in range(attempts):
        try:
            with open_with_retry(path, "r", encoding=encoding, attempts=1) as stream:
                return stream.read()
        except OSError as exc:
            if not _is_transient_storage_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, 0.05 * (2**attempt)))
    raise AssertionError("unreachable")


def read_bytes_with_retry(path: Path, *, attempts: int = 8) -> bytes:
    for attempt in range(attempts):
        try:
            with open_with_retry(path, "rb", attempts=1) as stream:
                return stream.read()
        except OSError as exc:
            if not _is_transient_storage_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, 0.05 * (2**attempt)))
    raise AssertionError("unreachable")


def path_exists_with_retry(path: Path, *, attempts: int = 8) -> bool:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return path.exists()
        except OSError as exc:
            if not _is_transient_storage_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, 0.05 * (2**attempt)))
    raise AssertionError("unreachable")


def sha256_file_with_retry(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
    attempts: int = 8,
) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        digest = hashlib.sha256()
        try:
            with open_with_retry(path, "rb", attempts=1) as stream:
                while chunk := stream.read(chunk_size):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as exc:
            if not _is_transient_storage_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, 0.05 * (2**attempt)))
    raise AssertionError("unreachable")


def _is_transient_storage_error(exc: OSError) -> bool:
    return exc.errno in TRANSIENT_STORAGE_ERRNOS or getattr(exc, "winerror", None) in TRANSIENT_STORAGE_WINERRORS
