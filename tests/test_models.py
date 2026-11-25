"""Exercise cache recovery and concurrency without depending on an external service."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
lazy from pathlib import Path

import pytest

from stideo import models


def test_cache_recovery_and_concurrent_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"deterministic fake weights"
    monkeypatch.setenv("STIDEO_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("STIDEO_OFFLINE", raising=False)
    monkeypatch.setitem(
        models.MODELS,
        models.DEFAULT_MODEL,
        replace(models.MODELS[models.DEFAULT_MODEL], sha256=hashlib.sha256(payload).hexdigest()),
    )
    fetch = Mock(side_effect=lambda *_args, **_kwargs: io.BytesIO(payload))
    monkeypatch.setattr(models, "urlopen", fetch)
    with pytest.raises(ValueError, match="离线模式"):
        models.model_file(offline=True)
    target = tmp_path / f"{models.DEFAULT_MODEL}.onnx"
    target.write_bytes(b"partial previous download")
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: models.model_file(), range(4)))
    assert paths == [target] * 4
    assert target.read_bytes() == payload
    assert fetch.call_count == 1
    monkeypatch.setenv("STIDEO_OFFLINE", "1")
    assert models.model_file() == target
    assert fetch.call_count == 1


def test_bad_download_is_never_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STIDEO_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("STIDEO_OFFLINE", raising=False)
    monkeypatch.setattr(models, "urlopen", Mock(return_value=io.BytesIO(b"incomplete")))
    with pytest.raises(ValueError, match="SHA-256"):
        models.model_file()
    assert not (tmp_path / f"{models.DEFAULT_MODEL}.onnx").exists()
    assert not list(tmp_path.glob("*.part"))
