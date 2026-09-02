from __future__ import annotations

import hashlib
import io

import pytest

from aic_retrieval import resilient_io


class FlakyPath:
    def __init__(self, payload: str, failures: int, errno: int = 22) -> None:
        self.payload = payload
        self.failures = failures
        self.errno = errno
        self.calls = 0

    def open(self, mode: str, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(self.errno, "temporary storage error")
        return io.StringIO(self.payload) if "b" not in mode else io.BytesIO(self.payload.encode("utf-8"))


class FlakyExistsPath:
    def __init__(self, failures: int, result: bool = True) -> None:
        self.failures = failures
        self.result = result
        self.calls = 0

    def exists(self) -> bool:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(22, "temporary storage error")
        return self.result


def test_read_text_retries_transient_external_storage_error(monkeypatch) -> None:
    path = FlakyPath('{"ready": true}', failures=2)
    delays: list[float] = []
    monkeypatch.setattr(resilient_io.time, "sleep", delays.append)

    assert resilient_io.read_text_with_retry(path) == '{"ready": true}'
    assert path.calls == 3
    assert delays == [0.05, 0.1]


def test_read_bytes_does_not_retry_non_transient_error(monkeypatch) -> None:
    path = FlakyPath("data", failures=1, errno=2)
    monkeypatch.setattr(resilient_io.time, "sleep", lambda _delay: pytest.fail("must not retry"))

    with pytest.raises(OSError) as error:
        resilient_io.read_bytes_with_retry(path)

    assert error.value.errno == 2
    assert path.calls == 1


def test_path_exists_retries_transient_external_storage_error(monkeypatch) -> None:
    path = FlakyExistsPath(failures=2)
    delays: list[float] = []
    monkeypatch.setattr(resilient_io.time, "sleep", delays.append)

    assert resilient_io.path_exists_with_retry(path)
    assert path.calls == 3
    assert delays == [0.05, 0.1]


def test_retry_delay_is_capped_for_persistent_transient_errors(monkeypatch) -> None:
    path = FlakyPath("data", failures=7)
    delays: list[float] = []
    monkeypatch.setattr(resilient_io.time, "sleep", delays.append)

    assert resilient_io.read_text_with_retry(path) == "data"
    assert max(delays) == 0.5
    assert len(delays) == 7


def test_sha256_file_retries_transient_external_storage_error(monkeypatch) -> None:
    path = FlakyPath("checksum payload", failures=2)
    delays: list[float] = []
    monkeypatch.setattr(resilient_io.time, "sleep", delays.append)

    assert resilient_io.sha256_file_with_retry(path) == hashlib.sha256(b"checksum payload").hexdigest()
    assert path.calls == 3
    assert delays == [0.05, 0.1]
