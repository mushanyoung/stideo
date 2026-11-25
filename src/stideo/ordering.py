"""Score clip boundaries and whole-clip similarity, then solve the playback path."""

import math
import sys
import tempfile
from array import array
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from stideo.commands import FFMPEG, run
from stideo.media import Clip
from stideo.timing import Stage

type Costs = list[list[float]]

THUMB_W, THUMB_H = 24, 16
FRAME_BYTES = THUMB_W * THUMB_H * 3
EDGE_SECONDS = 0.75
EXACT_LIMIT = 16


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


def content_order(clips, step, stage=nullcontext, *, offline=False):
    from stideo import content as analysis
    profiles = analysis.analyze(clips, step, FFMPEG, stage, offline=offline)
    samples = {c.path: p.samples for c, p in zip(clips, profiles, strict=True)}
    if len(clips) == 1:
        return clips, {}, samples
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
    return [clips[i] for i in path], notes, samples




