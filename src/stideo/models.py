"""Acquire and verify the pinned visual models shared between checkouts."""

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from filelock import FileLock


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str
    file: str
    sha256: str
    grid: int = 16
    side: int = 224
    family: str = "dino"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/onnx/{self.file}"


DEFAULT_MODEL = "dinov2-small-int8"
MODELS = {
    "dinov2-small-int8": Model(
        "Xenova/dinov2-small",
        "c2bb04a51fab207c420665f1946016107bffc701",
        "model_quantized.onnx",
        "3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917",
    ),
    "dinov2-small-fp32": Model(
        "Xenova/dinov2-small",
        "c2bb04a51fab207c420665f1946016107bffc701",
        "model.onnx",
        "83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea",
    ),
    "dinov2-base-int8": Model(
        "Xenova/dinov2-base",
        "d623d125a29e343851bcedff293fdc8b6bda292b",
        "model_quantized.onnx",
        "1a1210212e3d8c2924d2ad985e46207bb407ac9a9763a240cacdeb94cd0bbf9c",
    ),
    "dino-s16-int8": Model(
        "Xenova/dino-vits16",
        "517e3f80ee1d5d063ca5306afa66863e28848869",
        "model_quantized.onnx",
        "686df030e42d721ce72c8e89edc95cc9107068039298e3500d724727d69f172a",
        grid=14,
    ),
    "clip-b32-int8": Model(
        "Xenova/clip-vit-base-patch32",
        "6b7c523703ae26e39be4b03774966f8571c7f523",
        "vision_model_quantized.onnx",
        "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
        grid=7,
        family="clip",
    ),
    "siglip2-b32-int8": Model(
        "onnx-community/siglip2-base-patch32-256-ONNX",
        "7efccb86b5b6601bfe9e14326e1c11b40b68d44c",
        "vision_model_quantized.onnx",
        "82b3e1153198db893a275b0bf08c787e88b60e3213379b2cef2efc31f9e91b9a",
        grid=8,
        side=256,
        family="siglip",
    ),
    "mobilevit-xxs": Model(
        "Xenova/mobilevit-xx-small",
        "7dce9340b0bdd8d21f013ebbaa5206979c573ad1",
        "model.onnx",
        "70ff03c7139378ae036aa4ef2629b90a483da4b5697eb094d05a812e719fbd01",
        grid=8,
        side=256,
        family="mobilevit",
    ),
}


def verified(path: Path, sha256: str) -> bool:
    """Check the bytes even when a previous process populated the cache."""
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == sha256


def model_file(name: str = DEFAULT_MODEL, *, offline: bool = False) -> Path:
    """Locate the model or publish a checksum-verified download atomically."""
    spec = MODELS[name]
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = Path(os.environ.get("STIDEO_MODEL_DIR") or cache / "stitch-videos" / "models").expanduser()
    target = directory / f"{name}.onnx"
    if verified(target, spec.sha256):
        return target
    if offline or os.environ.get("STIDEO_OFFLINE") == "1":
        raise ValueError(
            f"离线模式下模型缺失或校验失败。请先运行 stideo --fetch-model --model {name}，"
            f"或将已校验的模型放在 {target}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=300):
        if verified(target, spec.sha256):
            return target
        print(f"正在下载视觉模型 {name}…", flush=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as temporary:
            download = Path(temporary.name)
        try:
            with (
                urlopen(spec.url, timeout=45) as source,  # noqa: S310 -- fixed HTTPS registry origin
                download.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)
            if not verified(download, spec.sha256):
                raise ValueError(
                    f"{name} 模型 SHA-256 校验失败，请重新运行 stideo --fetch-model --model {name}"
                )
            download.replace(target)
        finally:
            download.unlink(missing_ok=True)
    return target
