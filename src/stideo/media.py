"""Probe tracks, presentation geometry and candidate frame rates."""

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
lazy from collections.abc import Callable
lazy from pathlib import Path

from stideo.commands import run

type Stream = dict[str, Any]


@dataclass
class Clip:
    path: Path
    video: int
    audio: int | None
    duration: float
    file_offset: float  # 所选音视频轨道的共同起点，相对容器起点
    video_offset: float  # 以下两个偏移量均相对共同起点
    audio_offset: float
    video_duration: float
    aspect: float  # 已计入旋转与非方形像素
    long_side: float
    fps: Fraction
    stream: Stream  # 原始视频轨道的编码、色彩与显示矩阵


def number(value: str | float | None, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except TypeError, ValueError:
        return default


def ratio(value: str | int | Fraction | None, default: Fraction = Fraction(0)) -> Fraction:
    try:
        result = Fraction(str(value).replace(":", "/"))
    except ValueError, ZeroDivisionError:
        return default
    return result if result > 0 else default


def probe(path: Path) -> Clip | None:
    raw = run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        f"无法读取 {path.name}",
    )
    info = json.loads(raw)
    streams = info.get("streams", [])

    # 容器可能带有封面图、代理轨道或多条音轨，优先采用标记为默认的轨道。
    def track(kind: str) -> Stream | None:
        choices = (
            s for s in streams if s["codec_type"] == kind and not s.get("disposition", {}).get("attached_pic")
        )
        return max(choices, key=lambda s: s.get("disposition", {}).get("default", 0), default=None)

    video, audio = track("video"), track("audio")
    if video is None:
        return None
    container = info.get("format", {})
    origin = number(container.get("start_time"))

    def track_duration(stream: Stream, start: float) -> float:
        duration = number(stream.get("duration"))
        if duration > 0:
            return duration
        # Matroska / WebM 常用 DURATION 标签记录轨道的结束时间。
        end = stream.get("tags", {}).get("DURATION", "")
        if re.fullmatch(r"\d+:\d+:\d+(?:\.\d+)?", end):
            hours, minutes, seconds = map(float, end.split(":"))
            end_seconds = 3600 * hours + 60 * minutes + seconds
            if end_seconds > start:
                return end_seconds - start
        return max(0.0, number(container.get("duration")) - (start - origin))

    start = number(video.get("start_time"), origin)
    video_duration = track_duration(video, start)
    if video_duration <= 0 or min(video["width"], video["height"]) <= 0:
        raise ValueError(f"无法确定 {path.name} 的画面尺寸或时长")
    audio_start = number(audio.get("start_time"), start) if audio else start
    audio_duration = track_duration(audio, audio_start) if audio else 0
    first = min(start, audio_start)
    duration = max(start + video_duration, audio_start + audio_duration) - first
    rotation = next(
        (s["rotation"] for s in video.get("side_data_list", []) if "rotation" in s),
        video.get("tags", {}).get("rotate", 0),
    )
    w = video["width"] * float(ratio(video.get("sample_aspect_ratio"), Fraction(1)))
    h = video["height"]
    if round(number(rotation) / 90) % 2:
        w, h = h, w
    nominal, average = ratio(video.get("r_frame_rate")), ratio(video.get("avg_frame_rate"))
    rate = nominal if nominal and (not average or abs(nominal / average - 1) < 0.02) else average
    return Clip(
        path,
        video["index"],
        audio["index"] if audio else None,
        duration,
        max(0.0, first - origin),
        start - first,
        audio_start - first,
        video_duration,
        w / h,
        max(w, h),
        rate or Fraction(30),
        video,
    )


def weighted_median[T](clips: list[Clip], key: Callable[[Clip], T]) -> T:
    middle = sum(c.duration for c in clips) / 2
    ordered = sorted(clips, key=key)
    for clip in ordered:
        middle -= clip.duration
        if middle <= 0:
            return key(clip)
    return key(ordered[-1])


def canvas(clips: list[Clip]) -> tuple[int, int]:
    aspect = weighted_median(clips, lambda c: c.aspect)
    length = min(1920, max(c.long_side for c in clips))

    def even(value: float) -> int:
        return max(2, 2 * round(value / 2))

    return even(length * min(1, aspect)), even(length / max(1, aspect))


def common_fps(clips: list[Clip]) -> Fraction:
    rate = min(Fraction(60), max(Fraction(1), weighted_median(clips, lambda c: c.fps)))
    common = [Fraction(i) for i in (12, 15, 24, 25, 30, 48, 50, 60)]
    common.extend(Fraction(i, 1001) for i in (24000, 30000, 48000, 60000))
    nearest = min(common, key=lambda f: abs(f - rate))
    return nearest if abs(nearest / rate - 1) < 0.02 else rate.limit_denominator(1001)
