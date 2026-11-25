"""按可复用的画面时长选择编码、画布与色彩规格，导出带连续 AAC 音轨的 MP4。"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from stideo.commands import FFMPEG, run
from stideo.media import Clip
from stideo.timing import Stage
from stideo.mp4 import Track, assemble

CODECS = ("h264", "hevc", "av1")
TAGS = {"h264": "avc1", "hevc": "hvc1", "av1": "av01"}
RATE = 48000


@dataclass(frozen=True)
class Color:
    primaries: str
    transfer: str
    matrix: str
    range: str
    depth: int

    @property
    def hdr(self):
        return self.transfer in ("arib-std-b67", "smpte2084")

    def label(self):
        kind = "传递函数未标记" if self.transfer == "unknown" else "HDR" if self.hdr else "SDR"
        return f"{kind} {self.primaries}/{self.transfer} {self.depth}-bit"


SDR = Color("bt709", "bt709", "bt709", "tv", 8)


def source_color(stream):
    def tag(key, fallback):
        value = stream.get(key)
        return fallback if value in (None, "unknown", "unspecified", "reserved") else value

    fmt = stream.get("pix_fmt", "")
    bits = re.search(r"(9|10|12|14|16)(?:le|be)$", fmt)
    raw_depth = str(stream.get("bits_per_raw_sample", ""))
    depth = int(raw_depth) if raw_depth.isdigit() and int(raw_depth) > 0 else 8
    if bits:
        depth = int(bits[1])
    wide = stream.get("color_primaries") == "bt2020" or stream.get("color_space") in ("bt2020nc", "bt2020c")
    transfer = tag("color_transfer", "unknown" if wide else "bt709")
    hdr = transfer in ("smpte2084", "arib-std-b67")
    sd = stream["width"] <= 720 and stream["height"] <= 576
    matrix = tag("color_space", "bt2020nc" if hdr else "smpte170m" if sd else "bt709")
    return Color(
        tag("color_primaries", "bt2020" if hdr or wide else "bt709"),
        transfer,
        matrix,
        tag("color_range", "pc" if fmt.startswith(("yuvj", "rgb", "gbr")) else "tv"),
        depth,
    )


@dataclass(frozen=True)
class Format:
    codec: str
    size: tuple
    rotation: int
    color: Color


def rotation_of(stream):
    side = next((s for s in stream.get("side_data_list", []) if "rotation" in s), {})
    r = float(side.get("rotation", stream.get("tags", {}).get("rotate", 0)))
    if abs(r - 90 * round(r / 90)) > 0.01:
        return None
    if "displaymatrix" in side:
        rows = [
            list(map(int, line.split(":")[1].split()))
            for line in side["displaymatrix"].splitlines()
            if ":" in line
        ]
        a, b, _ = rows[0]
        c, d, _ = rows[1]
        if a * d - b * c != 65536**2 or max(abs(v) for v in (a, b, c, d)) != 65536:
            return None
    return int(round(r / 90) * 90) % 360


def source_format(clip):
    s = clip.stream
    angle = rotation_of(s)
    if angle is None or s.get("sample_aspect_ratio") not in (None, "N/A", "0:1", "1:1"):
        return None
    w, h = s["width"], s["height"]
    return Format(s["codec_name"], (h, w) if angle % 180 else (w, h), angle, source_color(s))


def packet_check(clip, fps):
    s = clip.stream
    raw = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            str(clip.video),
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,flags:packet_side_data=side_data_type",
            "-of",
            "compact=p=0",
            str(clip.path),
        ],
        f"无法检查 {clip.path.name} 的接缝",
    )
    pts = []
    last = None
    for line in raw.splitlines():
        if b"New Extradata" in line or b"Param Change" in line:
            return "段内解码配置发生变化"
        if not line.startswith(b"pts="):
            continue
        fields = dict(x.split(b"=", 1) for x in line.split(b"|") if b"=" in x)
        flags = fields.get(b"flags", b"")
        if b"D" in flags or b"C" in flags:
            return "片头或片尾含隐藏参考帧或损坏包"
        if last is None and b"K" not in flags:
            return "片头不是独立关键帧"
        try:
            p, d = int(fields[b"pts"]), int(fields[b"dts"])
        except (KeyError, ValueError):
            return "时间戳不完整"
        if last is not None and d <= last:
            return "解码时间戳不连续"
        pts.append(p)
        last = d
    if not pts or pts[0] != min(pts) or len(set(pts)) != len(pts):
        return "片头或显示时间戳不适合直接复制"
    if fps:
        unit = Fraction(s["time_base"]) * fps
        if any((p - pts[0]) * unit.numerator != i * unit.denominator for i, p in enumerate(sorted(pts))):
            return "需要按指定帧率重采样"
        if abs(clip.duration - len(pts) / float(fps)) > 0.00001:
            return "需对齐指定帧率的结尾"
    return ""


def copy_reason(clip, fmt, size, fps):
    s = clip.stream
    if fmt is None or any(n % 2 for n in fmt.size):
        return "需归一化画幅、像素比例或显示变换"
    if size and fmt.size != size:
        return "需适配指定画布"
    if s.get("pix_fmt") not in ("yuv420p", "yuvj420p", "yuv420p10le"):
        return "需归一化像素格式"
    if any("DOVI" in x.get("side_data_type", "") for x in s.get("side_data_list", [])):
        return "需处理 Dolby Vision 色彩配置"
    if clip.video_offset > 0.0001 or clip.duration - clip.video_duration > 0.002:
        return "需延展画面以对齐音轨"
    return packet_check(clip, fps)


@dataclass
class Plan:
    target: Format
    reasons: dict
    scores: dict
    fps: Fraction
    scale: int
    encoders: set

    def describe(self, clips):
        t = self.target
        print("各编码可直接复制的最长素材时长　" + "　".join(f"{k} {v:.2f} 秒" for k, v in self.scores.items()))
        print(
            f"目标 {t.codec.upper()}，显示画布 {t.size[0]}×{t.size[1]}，{t.color.label()}，"
            f"转码帧率 {self.fps}。复制片段保留原始画面时间戳。"
        )
        for clip in clips:
            reason = self.reasons[clip.path]
            print(f"  {clip.path.name}　{'转码 · ' + reason if reason else '直接复制视频'}")


def build_plan(clips, codec, size, requested_fps, fallback_size, fps):
    encoders = set(
        re.findall(r"^ V\S{5}\s+(\S+)", run(FFMPEG[:3] + ["-encoders"], "无法读取编码器").decode(), re.M)
    )
    allowed = CODECS if codec == "auto" else (codec,)
    reasons, forms = {}, {}
    totals = defaultdict(float)
    for c in clips:
        f = source_format(c)
        forms[c.path] = f
        reason = (
            "原片编码与目标不同"
            if c.stream["codec_name"] not in allowed
            else copy_reason(c, f, size, requested_fps)
        )
        reasons[c.path] = reason
        if not reason:
            totals[f] += c.video_duration
    viable = {
        f: seconds
        for f, seconds in totals.items()
        if encoder_choices(f.codec, encoders) or all(forms[c.path] == f and not reasons[c.path] for c in clips)
    }
    scores = {k: max((v for f, v in viable.items() if f.codec == k), default=0.0) for k in allowed}
    if viable:
        target = min(
            viable, key=lambda f: (-viable[f], CODECS.index(f.codec), f.rotation, f.size, f.color.label())
        )
    else:
        available = [k for k in allowed if encoder_choices(k, encoders)]
        original_seconds = {
            k: sum(c.video_duration for c in clips if c.stream["codec_name"] == k) for k in available
        }
        chosen = max(available, key=original_seconds.get, default=None)
        if chosen is None:
            raise ValueError("当前 FFmpeg 缺少可用于目标格式的编码器：" + ", ".join(allowed))
        dominant = source_color(max(clips, key=lambda c: c.video_duration).stream)
        color = (
            Color(dominant.primaries, dominant.transfer, dominant.matrix, "tv", 10)
            if dominant.hdr and chosen != "h264"
            else SDR
        )
        target = Format(chosen, size or fallback_size, 0, color)
    for c in clips:
        f = forms[c.path]
        if reasons[c.path] or f == target:
            continue
        if f.codec != target.codec:
            reasons[c.path] = "视频编码不同"
        elif (f.size, f.rotation) != (target.size, target.rotation):
            reasons[c.path] = "需适配显示方向或画布"
        else:
            reasons[c.path] = "需转换色域、传递函数、范围或位深"
    scale = math.lcm(90000, fps.numerator)
    if scale > 2000000:
        scale = fps.numerator * max(1, round(90000 / fps.numerator))
    return Plan(target, reasons, scores, fps, scale, encoders)


def encoder_choices(codec, available):
    names = {
        "h264": ["libx264", "h264_videotoolbox"],
        "hevc": ["hevc_videotoolbox", "libx265"],
        "av1": ["libsvtav1", "libaom-av1", "librav1e"],
    }[codec]
    return [
        name for name in names if name in available and ("videotoolbox" not in name or sys.platform == "darwin")
    ]


def encoder_args(name, color, fps):
    if "videotoolbox" in name:
        quality = ["-allow_sw", "0", "-q:v", "65", "-bf", "0"]
    elif name == "libx264":
        quality = ["-preset", "veryfast", "-crf", "18", "-x264-params", "open-gop=0"]
    elif name == "libx265":
        quality = ["-preset", "veryfast", "-crf", "21", "-x265-params", "open-gop=0:log-level=error"]
    elif name == "libsvtav1":
        quality = ["-preset", "9", "-crf", "26"]
    elif name == "libaom-av1":
        quality = ["-cpu-used", "8", "-crf", "26", "-b:v", "0"]
    else:
        quality = ["-speed", "9", "-qp", "70"]
    return ["-c:v", name, *quality, "-g", str(max(1, round(2 * fps)))]


def color_args(color):
    return [
        "-color_primaries",
        color.primaries,
        "-color_trc",
        color.transfer,
        "-colorspace",
        color.matrix,
        "-color_range",
        color.range,
    ]


def video_filter(clip, plan, frames):
    t = plan.target
    w, h = t.size
    inc, out = source_color(clip.stream), t.color
    opts = f"out_color_matrix={out.matrix}:out_range={out.range}"
    different = (inc.primaries, inc.transfer) != (out.primaries, out.transfer)
    if different:
        if "unknown" in (inc.transfer, out.transfer):
            raise ValueError(
                f"{clip.path.name} 的宽色域素材缺少传递函数，需先明确 PQ、HLG 或 SDR 才能进行色彩转换"
            )
        opts += (
            f":in_primaries={inc.primaries}:in_transfer={inc.transfer}:in_color_matrix={inc.matrix}"
            f":in_range={inc.range}:out_primaries={out.primaries}:out_transfer={out.transfer}:intent=perceptual"
        )
    fit = f"scale=w='clip(round({h}*dar/2)*2,2,{w})':h='clip(round({w}/dar/2)*2,2,{h})':flags=lanczos:{opts}"
    fmt = "yuv420p10le" if out.depth > 8 else "yuv420p"
    filters = [
        f"setpts=PTS-STARTPTS+({clip.video_offset:.9f})/TB",
        f"fps={plan.fps}:start_time=0:eof_action=pass",
        fit,
        "setsar=1",
        f"format={fmt}",
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
    ]
    if different:
        filters.extend(
            f"sidedata=delete:type={tag}"
            for tag in ("MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL", "DYNAMIC_HDR_PLUS")
        )
    if any("DOVI" in side.get("side_data_type", "") for side in clip.stream.get("side_data_list", [])):
        filters.extend(["sidedata=delete:type=DOVI_METADATA", "sidedata=delete:type=DOVI_RPU_BUFFER"])
    filters.extend(
        {0: [], 90: ["transpose=clock"], 180: ["hflip", "vflip"], 270: ["transpose=cclock"]}[t.rotation]
    )
    filters += [
        f"setparams=color_primaries={out.primaries}:color_trc={out.transfer}:"
        f"colorspace={out.matrix}:range={'full' if out.range == 'pc' else 'limited'}",
        "tpad=stop_mode=clone:stop=-1",
        f"trim=end_frame={frames}",
    ]
    return ",".join(filters)


def prepare(clip, path, plan, encode):
    t = plan.target
    args = FFMPEG + [
        "-i",
        str(clip.path),
        "-map",
        f"0:{clip.video}",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]
    tail = ["-tag:v", TAGS[t.codec], "-video_track_timescale", str(plan.scale), *color_args(t.color), str(path)]
    if not encode:
        run(args + ["-c:v", "copy"] + tail, f"不能复制 {clip.path.name}")
        return round(clip.video_duration * plan.scale)
    for side in clip.stream.get("side_data_list", []):
        if "DOVI" in side.get("side_data_type", "") and not side.get("dv_bl_signal_compatibility_id", 0):
            raise ValueError(f"{clip.path.name} 的 Dolby Vision 没有可独立转换的兼容基础层")
    frames = max(1, round(clip.duration * plan.fps))
    vf = video_filter(clip, plan, frames)
    errors = []
    for name in encoder_choices(t.codec, plan.encoders):
        try:
            run(
                args + ["-vf", vf, "-fps_mode", "passthrough", *encoder_args(name, t.color, plan.fps)] + tail,
                f"{clip.path.name} 使用 {name} 转码失败",
            )
            print(f"  视频编码器 {name}", flush=True)
            return round(frames * plan.scale / plan.fps)
        except ValueError as exc:
            errors.append(str(exc))
            path.unlink(missing_ok=True)
    raise ValueError("\n".join(errors) or f"找不到 {t.codec} 编码器")


def make_audio(clips, parts, scale, work):
    pcm, audio = work / "sound.pcm", work / "sound.m4a"
    ticks = previous = 0
    with pcm.open("wb") as dst:
        for i, (clip, (_, span)) in enumerate(zip(clips, parts)):
            ticks += span
            end = (ticks * RATE + scale // 2) // scale
            n = end - previous
            previous = end
            if clip.audio is None:
                dst.seek(n * 4, 1)
                dst.truncate()
                continue
            af = [
                f"asetpts=PTS-STARTPTS+({clip.audio_offset:.9f})/TB",
                f"aresample={RATE}:async=1:first_pts=0",
                "aformat=channel_layouts=stereo",
                "apad",
                f"atrim=end_sample={n}",
            ]
            ramp = min(480, n // 2)
            if i and ramp:
                af.append(f"afade=t=in:ss=0:ns={ramp}")
            if i + 1 < len(clips) and ramp:
                af.append(f"afade=t=out:ss={n - ramp}:ns={ramp}")
            res = subprocess.run(
                FFMPEG
                + [
                    "-i",
                    str(clip.path),
                    "-map",
                    f"0:{clip.audio}",
                    "-af",
                    ",".join(af),
                    "-f",
                    "s16le",
                    "-ac",
                    "2",
                    "-ar",
                    str(RATE),
                    "pipe:1",
                ],
                stdout=dst,
                stderr=subprocess.PIPE,
            )
            if res.returncode or dst.tell() != end * 4:
                raise ValueError(f"{clip.path.name} 的音频处理失败\n{res.stderr.decode(errors='replace')}")
    run(
        FFMPEG
        + [
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            "2",
            "-i",
            str(pcm),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movie_timescale",
            str(RATE),
            str(audio),
        ],
        "连续音轨编码失败",
    )
    pcm.unlink()
    return audio


def thumbnails(path, start, duration):
    args = FFMPEG + [
        "-threads",
        "1",
        "-noautorotate",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-t",
        f"{duration:.9f}",
        "-vf",
        "scale=24:24:flags=area,setsar=1",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-threads:v",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode or p.stderr.strip():
        raise ValueError(p.stderr.decode(errors="replace") or "接缝解码失败")
    n = 24 * 24 * 3
    return [p.stdout[i : i + n] for i in range(0, len(p.stdout), n)]


def match_frames(reference, actual, offset=0):
    if not reference or not actual:
        raise ValueError("接缝缺少画面")
    for j in {0, len(reference) // 2}:
        near = actual[max(0, offset + j - 2) : offset + j + 3]
        if not near or min(sum(abs(a - b) for a, b in zip(reference[j], x)) / len(x) for x in near) > 3:
            raise ValueError("接缝显示画面与分段不一致")


def inspect_joins(output, parts, scale):
    bad = set()
    ticks = 0
    tail = []
    previous_window = 0.0
    for i, (path, span) in enumerate(parts):
        duration, start = span / scale, ticks / scale
        ticks += span
        try:
            # 从上一段内开始连续解码，另从本段开头随机定位，两种路径均须能读到正确画面。
            window = min(0.22, duration)
            head = thumbnails(path, 0, window)
            match_frames(head, thumbnails(output, start, window))
            if i and tail:
                across = thumbnails(output, start - previous_window, previous_window + window)
                match_frames(tail, across)
                match_frames(head, across, len(tail))
            previous_window = window
            at = max(0.0, duration - window)
            tail = thumbnails(path, at, window) if at else head
            if tail:
                match_frames(tail, thumbnails(output, start + at, window))
            # 低帧率素材的末帧可能已显示超过 0.22 秒，此窗口内没有新的压缩样本。
        except ValueError as exc:
            print(f"  第 {i + 1} 段接缝验证失败　{str(exc)[:220]}", flush=True)
            bad.add(i)
            tail = []
    return bad


def check_streams(output, plan, duration):
    data = json.loads(
        run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)], "无法检查成片轨道")
    )
    v, a = data["streams"]
    t = plan.target
    w, h = (v["height"], v["width"]) if t.rotation % 180 else (v["width"], v["height"])
    if (
        (w, h) != t.size
        or rotation_of(v) != t.rotation
        or v["codec_name"] != t.codec
        or source_color(v) != t.color
    ):
        raise ValueError("成片画幅、编码或色彩与目标不一致")
    if (
        a["codec_name"] != "aac"
        or int(a["sample_rate"]) != RATE
        or a["channels"] != 2
        or any(abs(float(s["duration"]) - duration) > 0.002 for s in (v, a))
    ):
        raise ValueError("成片音视频轨道或时长不一致")


def render(clips, output, plan, stage=nullcontext):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stitch-", dir=output.parent) as tmp:
        work = Path(tmp)
        parts, copied = [], set()

        def write_part(i, copy):
            c = clips[i]
            dest = work / f"{i:06}.mp4"
            dest.unlink(missing_ok=True)
            action = "直接复制视频" if copy else "视频转码"
            print(f"{action} [{i + 1}/{len(clips)}] {c.path.name}", flush=True)
            now = time.monotonic()
            with stage(action):
                span = prepare(c, dest, plan, not copy)
                track = Track(dest)
                if span <= sum(track.delta[:-1]):
                    raise ValueError(f"{c.path.name} 需要重新对齐片尾时间")
            print(f"  本段{action}耗时 {time.monotonic() - now:.2f} 秒", flush=True)
            return dest, span

        for i, c in enumerate(clips):
            if not plan.reasons[c.path]:
                try:
                    parts.append(write_part(i, True))
                    copied.add(i)
                    continue
                except ValueError as exc:
                    print(f"  改用转码　{exc}", flush=True)
            parts.append(write_part(i, False))
        staged = work / "joined.mp4"
        audio_spans = None
        for attempt in range(3):
            spans = [n for _, n in parts]
            if spans != audio_spans:
                with stage("音频对齐与 AAC 编码"):
                    (work / "sound.m4a").unlink(missing_ok=True)
                    audio = make_audio(clips, parts, plan.scale, work)
                audio_spans = spans
            with stage("复制压缩样本与 MP4 封装"):
                staged.unlink(missing_ok=True)
                assemble(parts, audio, staged, plan.target.rotation)
            with stage("成片接缝验证"):
                check_streams(staged, plan, sum(spans) / plan.scale)
                bad = inspect_joins(staged, parts, plan.scale)
            if not bad:
                break
            retry = copied & (bad | {i - 1 for i in bad} | {i + 1 for i in bad})
            if attempt or not retry:
                retry = set(copied)
            if not retry or attempt == 2:
                raise ValueError("成片接缝验证未通过")
            for i in sorted(retry):
                parts[i] = write_part(i, False)
                copied.remove(i)
        with output.open("xb"):
            pass
        try:
            os.replace(staged, output)
        except BaseException:
            output.unlink()
            raise
        seconds = sum(clips[i].video_duration for i in copied)
        print(
            f"最终直接复制 {len(copied)}/{len(clips)} 段（{seconds:.3f} 秒），其余 {len(clips) - len(copied)} 段转码。"
        )
        return {clips[i].path for i in copied}
