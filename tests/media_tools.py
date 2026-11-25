"""Independent checks using FFmpeg's public decoding and probing interfaces."""

import json
import subprocess
import sys
from typing import Any
lazy from pathlib import Path

import numpy as np

type Document = dict[str, Any]


def probe(path: Path, *options: str) -> Document:
    command = ["ffprobe", "-v", "error", *options, "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command))


def packets(path: Path) -> list[Document]:
    return probe(
        path,
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "packet=pts,dts,data_hash",
    )["packets"]


def stitch(source: Path, output: Path, *options: str, dry: bool = False) -> Document:
    report = output.with_suffix(".json")
    args = [sys.executable, "-m", "stideo", str(source), "-o", str(output), "--report", str(report), *options]
    if dry:
        args.append("--dry-run")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists() != dry
    return json.loads(report.read_text(encoding="utf-8"))


def decode(path: Path, options: list[str]) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            "1",
            "-i",
            str(path),
            *options,
            "pipe:1",
        ]
    )


def pictures(path: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    data = decode(
        path,
        [
            "-map",
            "0:v:0",
            "-vf",
            vf,
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-threads:v",
            "1",
            "-f",
            "rawvideo",
        ],
    )
    return np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)


def sound(path: Path) -> np.ndarray:
    data = decode(path, ["-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le"])
    return np.frombuffer(data, "<f4")
