#!/usr/bin/env python3
"""默认按整段画面相似度（content）排序并拼接视频。也可按首尾衔接或文件名排序。

content 沿整段比较全画面与局部内容、色彩、构图、纹理和运动，辅以声音及首尾衔接。
请将 clip_content.py、smart_export.py 和 mp4_join.py 与本脚本放在同一目录。
content 首次运行会准备本地 DINOv2 模型和分析环境。
导出默认选择可直接复制时长最多的编码、画幅与色彩组合。转码使用本机 FFmpeg 编码器。
可指定 --codec h264、hevc 或 av1。--fps 指定固定帧率，--size 指定显示画布。
跨 HDR/SDR 转换使用 FFmpeg scale 的传递函数转换与感知色调映射（out_transfer、intent）。

    python3 stitch_videos.py ./素材 --dry-run
    python3 stitch_videos.py ./素材
    python3 stitch_videos.py ./素材 --codec hevc
    python3 stitch_videos.py ./素材 --order smooth
    python3 stitch_videos.py ./素材 --order natural
"""

from __future__ import annotations

import argparse
from array import array
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from smart_export import CODECS, FFMPEG, build_plan, render, run


VIDEO_EXTENSIONS = {
    ".3gp", ".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".mts", ".mxf", ".ogv", ".ts", ".webm", ".wmv",
}
OUTPUT_NAMES = {"content": "画面相似排序.mp4", "smooth": "连贯排序.mp4", "natural": "数字排序.mp4"}
THUMB_W, THUMB_H = 24, 16
FRAME_BYTES = THUMB_W * THUMB_H * 3
EDGE_SECONDS = 0.75
EXACT_LIMIT = 16


def shared_clock():
    # macOS 自带 Python 3.9 的 monotonic() 以进程启动为零点。POSIX 时钟可跨解释器读数。
    if hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.monotonic()


class Timings:
    def __init__(self):
        self.started = float(os.environ.pop("STITCH_TIMING_START", shared_clock()))
        self.seconds = {}

    def elapsed(self):
        return shared_clock() - self.started

    @contextmanager
    def stage(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.monotonic() - started)

    def report(self):
        total = self.elapsed()
        other = max(0.0, total - sum(self.seconds.values()))
        print("\n耗时统计（各阶段累计实际用时）")
        for name, seconds in [*self.seconds.items(), ("其他开销", other)]:
            print(f"  {name}  {seconds:.3f} 秒  {100 * seconds / total:.1f}%")
        minutes, seconds = divmod(total, 60)
        print(f"总耗时 {total:.3f} 秒（{int(minutes)} 分 {seconds:.2f} 秒）", flush=True)


@dataclass
class Clip:
    path: Path
    video: int
    audio: int | None
    duration: float
    file_offset: float      # 所选音视频轨道的共同起点，相对容器起点
    video_offset: float     # 以下两个偏移量均相对共同起点
    audio_offset: float
    video_duration: float
    aspect: float           # 已计入旋转与非方形像素
    long_side: float
    fps: Fraction
    stream: dict           # 原始视频轨道的编码、色彩与显示矩阵




def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def ratio(value, default=Fraction(0)):
    try:
        result = Fraction(str(value).replace(":", "/"))
        return result if result > 0 else default
    except (ValueError, ZeroDivisionError):
        return default


def probe(path):
    raw = run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
               "-of", "json", str(path)], f"无法读取 {path.name}")
    info = json.loads(raw)
    streams = info.get("streams", [])
    # 容器可能带有封面图、代理轨道或多条音轨，优先采用标记为默认的轨道。
    def track(kind):
        choices = (s for s in streams if s["codec_type"] == kind
                   and not s.get("disposition", {}).get("attached_pic"))
        return max(choices, key=lambda s: s.get("disposition", {}).get("default", 0),
                   default=None)

    video, audio = track("video"), track("audio")
    if video is None:
        return None
    container = info.get("format", {})
    origin = number(container.get("start_time"))
    def track_duration(stream, start):
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
    rotation = next((s["rotation"] for s in video.get("side_data_list", [])
                     if "rotation" in s), video.get("tags", {}).get("rotate", 0))
    w = video["width"] * float(ratio(video.get("sample_aspect_ratio"), Fraction(1)))
    h = video["height"]
    if round(number(rotation) / 90) % 2:
        w, h = h, w
    nominal, average = ratio(video.get("r_frame_rate")), ratio(video.get("avg_frame_rate"))
    rate = nominal if nominal and (not average or abs(nominal / average - 1) < .02) else average
    return Clip(path, video["index"], audio["index"] if audio else None, duration,
                max(0.0, first - origin), start - first, audio_start - first, video_duration,
                w / h, max(w, h), rate or Fraction(30), video)


def weighted_median(clips, key):
    middle = sum(c.duration for c in clips) / 2
    for clip in sorted(clips, key=key):
        middle -= clip.duration
        if middle <= 0:
            return key(clip)


def canvas(clips):
    aspect = weighted_median(clips, lambda c: c.aspect)
    length = min(1920, max(c.long_side for c in clips))
    def even(value):
        return max(2, 2 * round(value / 2))
    return even(length * min(1, aspect)), even(length / max(1, aspect))


def common_fps(clips):
    rate = min(Fraction(60), max(Fraction(1), weighted_median(clips, lambda c: c.fps)))
    common = [Fraction(i) for i in (12, 15, 24, 25, 30, 48, 50, 60)]
    common.extend(Fraction(i, 1001) for i in (24000, 30000, 48000, 60000))
    nearest = min(common, key=lambda f: abs(f - rate))
    return nearest if abs(nearest / rate - 1) < .02 else rate.limit_denominator(1001)


def read_window(clip, start, length, audio_file):
    # 一次定位同时解码音画。只读取边界附近的小图，开销不随整段时长增加。
    seek = clip.file_offset + start
    args = FFMPEG + ["-y"]
    window = ["-t", f"{length:.9f}"]
    if seek > 0:
        if start < clip.video_offset:
            # 视频首帧前可能已有声音。输入端跳转会直接跳到视频关键帧。
            window = ["-ss", f"{seek:.9f}", *window]
        else:
            args += ["-ss", f"{seek:.9f}"]
    args += ["-i", str(clip.path), "-map", f"0:{clip.video}", *window,
             "-vf", f"scale={THUMB_W}:{THUMB_H}:flags=area,setsar=1",
             "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with_sound = clip.audio is not None and audio_file is not None
    if with_sound:
        args += ["-map", f"0:{clip.audio}", *window, "-af",
                 "aresample=8000:async=1:first_pts=0", "-ac", "1", "-f", "s16le", str(audio_file)]
    raw = run(args, f"取样失败：{clip.path.name}")
    frames = [raw[i:i + FRAME_BYTES] for i in range(0, len(raw), FRAME_BYTES)]
    sound = array("h")
    if with_sound:
        sound.frombytes(audio_file.read_bytes())
        if sys.byteorder != "little":
            sound.byteswap()
        # 尾部没有声音时也要把静音计入衔接评分。
        sound.extend(array("h", [0]) * max(0, round(length * 8000) - len(sound)))
    return frames, sound


def picture(frame):
    # 4×4 区域的平均色描述构图，各通道的累积分布描述整体色调。
    tiles = [0] * 48
    hist = [[0] * 8 for _ in range(3)]
    for pixel in range(THUMB_W * THUMB_H):
        x, y = pixel % THUMB_W, pixel // THUMB_W
        tile = ((y * 4 // THUMB_H) * 4 + x * 4 // THUMB_W) * 3
        for channel in range(3):
            value = frame[pixel * 3 + channel]
            tiles[tile + channel] += value
            hist[channel][value // 32] += 1
    pixels = THUMB_W * THUMB_H
    layout = [value / (255 * pixels / 16) for value in tiles]
    colors = [sum(bins[:i]) / pixels for bins in hist for i in range(1, 8)]
    return layout, colors


def motion(first, second):
    # 在缩略图上匹配相邻帧的小幅平移，纹理不足时优先选择零位移。
    def luma(frame):
        return [(77*r + 150*g + 29*b) // 256 for r, g, b in zip(frame[::3], frame[1::3], frame[2::3])]
    a, b = luma(first), luma(second)
    points = [y * THUMB_W + x for y in range(2, THUMB_H - 2, 2) for x in range(2, THUMB_W - 2, 2)]
    def error(dx, dy):
        shift = dy * THUMB_W + dx
        return sum(abs(a[p] - b[p + shift]) for p in points) / len(points)
    dx, dy = min(((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)),
                 key=lambda s: error(*s) + 1.25 * (abs(s[0]) + abs(s[1])))
    return min(1, error(0, 0) / 64), dx / 2, dy / 2


def audio_features(samples):
    power = sum(x*x for x in samples)
    if not power:
        return 0.0, 0.0
    db = 10 * math.log10(power / (len(samples) * 32768**2))
    level = min(1, max(0, (db + 60) / 60))
    difference = sum((b-a)**2 for a, b in zip(samples, samples[1:]))
    return level, min(1, math.sqrt(difference / (4 * power)))


@dataclass
class Edge:
    layout: list[float]
    colors: list[float]
    movement: tuple
    sound: tuple
    aspect: float


def edge(clip, frames, sound, tail):
    if not frames:
        raise ValueError(f"{clip.path.name} 的{'结尾' if tail else '开头'}没有可解码画面")
    ordered = frames[::-1] if tail else frames
    samples = [ordered[round((len(ordered) - 1) * f)] for f in (0, 1/3, 2/3, 1)]
    layouts, colors = zip(*(picture(frame) for frame in samples))
    def blend(values):
        return [a*.55 + b*.25 + c*.15 + d*.05 for a, b, c, d in zip(*values)]
    first, second = (samples[1], samples[0]) if tail else (samples[0], samples[1])
    nearby = sound[-2400:] if tail else sound[:2400]
    return Edge(blend(layouts), blend(colors), motion(first, second), audio_features(nearby), clip.aspect)


def endpoints(clip, audio_file):
    length = min(clip.duration, EDGE_SECONDS)
    frames, sound = read_window(clip, 0, length, audio_file)
    if not frames and clip.video_offset > 0:
        frames, _ = read_window(clip, clip.video_offset, min(length, clip.video_duration), None)
        frames = frames[:1]
    head = edge(clip, frames, sound, False)
    if clip.duration > length:
        frames, sound = read_window(clip, clip.duration - length, length, audio_file)
        # 音轨更长或低帧率画面长时间停留时，找到实际显示的最后一帧。
        lookback = min(length, clip.video_duration)
        while not frames:
            start = clip.video_offset + clip.video_duration - lookback
            frames, _ = read_window(clip, start, lookback, None)
            frames = frames[-1:]
            if lookback >= clip.video_duration:
                break
            lookback = min(lookback * 2, clip.video_duration)
    return head, edge(clip, frames, sound, True)


def cut_cost(end, start):
    def distance(a, b):
        return sum(abs(x-y) for x, y in zip(a, b)) / len(a)
    speed, dx, dy = end.movement
    other_speed, other_dx, other_dy = start.movement
    volume, tone = end.sound
    other_volume, other_tone = start.sound
    framing = 1 - min(end.aspect, start.aspect) / max(end.aspect, start.aspect)
    return (.35 * distance(end.layout, start.layout) + .23 * distance(end.colors, start.colors)
            + .09 * abs(speed - other_speed) + .05 * (abs(dx-other_dx) + abs(dy-other_dy)) / 4
            + .15 * abs(volume-other_volume) + .03 * min(volume, other_volume) * abs(tone-other_tone)
            + .10 * framing)


def exact_path(cost):
    # Held–Karp 开放路径。表项是固定首段且恰好使用 mask 中所有片段的最低代价。
    n = len(cost)
    count = 1 << n
    best = array("d", [math.inf]) * (count * n)
    following = bytearray(count * n)
    for mask in range(1, count):
        members = mask
        while members:
            bit = members & -members
            members ^= bit
            first = bit.bit_length() - 1
            remaining = mask ^ bit
            index = mask * n + first
            if not remaining:
                best[index] = 0
                continue
            options = remaining
            base = remaining * n
            value = math.inf
            while options:
                bit = options & -options
                options ^= bit
                nxt = bit.bit_length() - 1
                candidate = cost[first][nxt] + best[base + nxt]
                if candidate < value:
                    value, following[index] = candidate, nxt
            best[index] = value
    mask = count - 1
    first = min(range(n), key=lambda i: best[mask * n + i])
    path = []
    while mask:
        path.append(first)
        nxt = following[mask * n + first]
        mask ^= 1 << first
        first = nxt
    return path


def approximate_path(cost):
    # 多起点贪心，再试单段移位。每个移位按有方向的接缝重新计价。
    n = len(cost)
    def total(path):
        return sum(cost[a][b] for a, b in zip(path, path[1:]))
    best = list(range(n))
    trials = min(n, 12)
    for trial in range(trials):
        path = [trial * (n - 1) // (trials - 1)]
        unused = set(best) - set(path)
        while unused:
            row = cost[path[-1]]
            nxt = min(unused, key=lambda j: (row[j], j))
            path.append(nxt)
            unused.remove(nxt)
        if total(path) < total(best):
            best = path
    def link(a, b):
        return 0 if a is None or b is None else cost[a][b]
    for _ in range(12):
        gain, move = 0.0, None
        for i, item in enumerate(best):
            rest = [None, *best[:i], *best[i+1:], None]
            a, b = rest[i:i+2]
            removed = link(a, item) + link(item, b) - link(a, b)
            for j, (a, b) in enumerate(zip(rest, rest[1:])):
                added = link(a, item) + link(item, b) - link(a, b)
                if removed - added > gain + 1e-12:
                    gain, move = removed - added, (i, j)
        if move is None:
            break
        i, j = move
        best.insert(j, best.pop(i))
    return best


def boundary_costs(clips):
    edges = []
    with tempfile.TemporaryDirectory(prefix="clip-edges-") as tmp:
        for i, clip in enumerate(clips, 1):
            print(f"分析首尾 [{i}/{len(clips)}] {clip.path.name}", flush=True)
            edges.append(endpoints(clip, Path(tmp) / "sound.pcm"))
    return [[cut_cost(end, head) for head, _ in edges] for _, end in edges]


def smooth_order(clips, stage=nullcontext):
    with stage("片段首尾分析"):
        cost = boundary_costs(clips)
    with stage("求解播放顺序"):
        if len(clips) <= EXACT_LIMIT:
            path = exact_path(cost)
            print("已求出当前衔接评分下的全局最优排列。")
        else:
            path = approximate_path(cost)
            print("已用多起点搜索与局部调整选出近似排列。")
    return [clips[i] for i in path]


def load_content_runtime(started):
    if not Path(__file__).resolve().with_name("clip_content.py").is_file():
        raise ValueError("请将 clip_content.py 与 stitch_videos.py 放在同一目录，再运行内容排序")
    try:
        import clip_content
        return clip_content
    except (ImportError, OSError) as error:
        uv = shutil.which("uv")
        runtime, opencv = "onnxruntime==1.23.2", "opencv-python-headless==4.14.0.94"
        if not uv or os.environ.get("STITCH_CONTENT_RUNTIME") == "1":
            raise ValueError(f"无法加载内容分析环境（{error}）。可安装 uv 后重试，"
                             f"或在 Python 3.12 环境运行 pip install {runtime} {opencv}") from error
        print("准备内容分析所用的独立 Python 环境…", flush=True)
        command = [uv, "run", "--no-project", "--python", "3.12", "--with", runtime,
                   "--with", opencv, "--", "python", "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
        os.execve(uv, command, {**os.environ, "STITCH_CONTENT_RUNTIME": "1",
                                "STITCH_TIMING_START": str(started)})


def content_order(clips, step, analysis=None, stage=nullcontext):
    if analysis is None:
        analysis = load_content_runtime(shared_clock())
    profiles = analysis.analyze(clips, step, FFMPEG, stage)
    if len(clips) == 1:
        return clips, {}
    with stage("片段首尾分析"):
        cuts = boundary_costs(clips)
    with stage("计算片段间相似度"):
        cost = analysis.content_costs(profiles, cuts)
    with stage("求解播放顺序"):
        path = exact_path(cost) if len(clips) <= EXACT_LIMIT else approximate_path(cost)
    method = "当前评分下的最优解" if len(clips) <= EXACT_LIMIT else "近似解"
    transitions = list(zip(path, path[1:]))
    total = sum(cost[a][b] for a, b in transitions)
    print(f"已求出{method}，相邻片段的综合差异总分为 {total:.3f}（越小越相近）。")
    notes = {clips[b].path: f"与前一段的综合差异 {cost[a][b]:.3f}" for a, b in transitions}
    return [clips[i] for i in path], notes




def parse_size(value):
    match = re.fullmatch(r"(\d+)[xX×](\d+)", value)
    if match:
        dims = tuple(map(int, match.groups()))
        if all(n >= 2 and n % 2 == 0 for n in dims):
            return dims
    raise argparse.ArgumentTypeError("尺寸应为两个正偶数，如 1920x1080")


def parse_fps(value):
    rate = ratio(value)
    if not 1 <= rate <= 240:
        raise argparse.ArgumentTypeError("帧率应在 1 到 240 之间，支持 30 或 60000/1001 等写法")
    return rate.limit_denominator(100000)


def main():
    timings = Timings()
    parser = argparse.ArgumentParser(description="按首尾衔接、整段内容或文件名，将文件夹内的视频剪成一个 MP4。",
                                     epilog=f"smooth 比较首尾 {EDGE_SECONDS} 秒。content 比较整段综合画面特征，兼顾声音及接缝。"
                                            f"{EXACT_LIMIT} 段以内求相应评分下的最优解。")
    parser.add_argument("folder", type=Path, nargs="?", default=Path("."), help="素材文件夹，默认当前目录")
    parser.add_argument("--order", choices=OUTPUT_NAMES, default="content",
                        help="content 综合画面相似度，smooth 首尾衔接，"
                             "natural 将文件名中的连续数字按数值比较。默认 %(default)s")
    parser.add_argument("--codec", "--video-codec", choices=("auto", *CODECS), default="auto",
                        help="最终视频编码，auto 按可直接复制的总时长选择（默认）")
    parser.add_argument("--content-step", type=float, default=1.0, metavar="秒",
                        help="content 模式的画面取样间隔，默认 1 秒，切镜处额外取样，0 表示逐帧提取视觉特征")
    parser.add_argument("-o", "--output", type=Path, help="输出 MP4 路径，默认保存到素材目录下的「合辑」子目录")
    parser.add_argument("--dry-run", action="store_true", help="预览顺序、目标编码、逐段复制/转码计划及耗时")
    parser.add_argument("--size", type=parse_size, metavar="宽x高", help="手动指定画布，如 1920x1080")
    parser.add_argument("--fps", type=parse_fps, metavar="帧率",
                        help="指定固定输出帧率。默认保留复制片段的时间戳，转码片段按素材选择，最高 60")
    args = parser.parse_args()
    if not math.isfinite(args.content_step) or args.content_step < 0:
        parser.error("--content-step 应为非负的有限秒数")
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"找不到素材文件夹：{folder}")
    for program in ("ffmpeg", "ffprobe"):
        if not shutil.which(program):
            parser.error(f"找不到 {program}，请安装 FFmpeg 并将其加入 PATH")
    output = (args.output or folder / "合辑" / OUTPUT_NAMES[args.order]).expanduser().absolute()
    if output.suffix.lower() != ".mp4":
        parser.error("输出文件请使用 .mp4 扩展名")
    if not args.dry_run and os.path.lexists(output):
        parser.error(f"输出文件已存在，请用 -o 指定新的文件名：{output}")
    paths = sorted((p for p in folder.iterdir() if not p.name.startswith(".") and p.is_file()
                    and p.suffix.lower() in VIDEO_EXTENSIONS and p.resolve() != output.resolve()),
                   key=lambda p: p.name)
    if args.order == "natural":
        paths.sort(key=lambda p: [int(s) if i % 2 else s.casefold()
                                  for i, s in enumerate(re.split(r"(\d+)", p.name))])
    if not paths:
        parser.error(f"目录中没有找到视频：{folder}")
    analysis = load_content_runtime(timings.started) if args.order == "content" else None
    timings.seconds["启动、扫描目录与环境准备"] = timings.elapsed()
    clips = []
    with timings.stage("读取媒体参数"):
        for path in paths:
            clip = probe(path)
            if clip is None:
                print(f"跳过没有视频轨道的文件 {path.name}")
            else:
                clips.append(clip)
    if not clips:
        parser.error(f"目录中没有找到视频：{folder}")
    fps = args.fps or common_fps(clips)
    notes = {}
    if args.order == "smooth" and len(clips) > 1:
        clips = smooth_order(clips, timings.stage)
    elif args.order == "content":
        clips, notes = content_order(clips, args.content_step, analysis, timings.stage)
    print(f"排序完成，自启动起 {timings.elapsed():.3f} 秒。", flush=True)
    print(f"共 {len(clips)} 段。播放顺序")
    for i, clip in enumerate(clips, 1):
        note = f"  [{notes[clip.path]}]" if clip.path in notes else ""
        print(f"  {i:>2}. {clip.path.name}{note}")
    with timings.stage("检查编码、画幅、色彩及接缝并选择导出规格"):
        plan = build_plan(clips, args.codec, args.size, args.fps, canvas(clips), fps)
    plan.describe(clips)
    if not args.dry_run:
        render(clips, output, plan, timings.stage)
        print(f"已生成 {output}")
    timings.report()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"错误：{exc}")
    except KeyboardInterrupt:
        sys.exit("\n已取消剪辑。")
