"""Acquire and verify the pinned DINOv2 model shared between checkouts."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from filelock import FileLock

MODEL_URL = (
    "https://huggingface.co/Xenova/dinov2-small/resolve/"
    "c2bb04a51fab207c420665f1946016107bffc701/onnx/model_quantized.onnx"
)
MODEL_SHA256 = "3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917"
MODEL_NAME = "dinov2-small-int8.onnx"


def verified(path: Path) -> bool:
    """Check the bytes even when a previous process populated the cache."""
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256


def model_file(*, offline: bool = False) -> Path:
    """Locate the model or publish a checksum-verified download atomically."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = Path(os.environ.get("STIDEO_MODEL_DIR") or cache / "stitch-videos" / "models").expanduser()
    target = directory / MODEL_NAME
    if verified(target):
        return target
    if offline or os.environ.get("STIDEO_OFFLINE") == "1":
        raise ValueError(f"离线模式下模型缺失或校验失败。请先运行 stideo --fetch-model，或将已校验的模型放在 {target}")
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=300):
        if verified(target):
            return target
        print("正在下载本地 DINOv2 视觉模型（约 24 MB）…", flush=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as temporary:
            download = Path(temporary.name)
        try:
            with urlopen(MODEL_URL, timeout=45) as source, download.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if not verified(download):
                raise ValueError("DINOv2 模型 SHA-256 校验失败，请重新运行 stideo --fetch-model")
            download.replace(target)
        finally:
            download.unlink(missing_ok=True)
    return target
