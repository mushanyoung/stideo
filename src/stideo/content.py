"""整段内容分析。由 stideo 的 content 模式调用。"""

import os
import subprocess
import tempfile
import time
from contextlib import closing, nullcontext
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from stideo.models import model_file
lazy from stideo.media import Clip
lazy from stideo.timing import Stage

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PHASES = 4
PHASE_MATCHES = np.array(list(permutations(range(PHASES))))
MOTION_FPS = 8
CUT_THRESHOLD = 0.28


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def frame_features(rgb: np.ndarray, model: cv2.dnn.Net) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    # 模型输入是 16×16 个图块。按原画幅放入完整图像，只汇总有效图块。
    rows, cols = max(1, round(16 * h / max(w, h))), max(1, round(16 * w / max(w, h)))
    y, x = (16 - rows) // 2, (16 - cols) // 2
    scaled = cv2.resize(rgb, (cols * 14, rows * 14), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.float32)
    normalized = (scaled.astype(np.float32) / 255 - MODEL_MEAN) / MODEL_STD
    canvas[y * 14 : (y + rows) * 14, x * 14 : (x + cols) * 14] = normalized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    model.setInput(tensor, "pixel_values")
    output = model.forward("last_hidden_state")
    tokens = output[0]
    patches = unit(tokens[1:].reshape(16, 16, -1)[y : y + rows, x : x + cols])
    regions = []
    for i in range(2):
        for j in range(2):
            region = patches[
                i * rows // 2 : ((i + 1) * rows + 1) // 2, j * cols // 2 : ((j + 1) * cols + 1) // 2
            ]
            regions.append(unit(region.mean(axis=(0, 1))))

    small = cv2.resize(rgb, (max(4, w // 3), max(4, h // 3)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    layout = cv2.resize(lab, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    colors = cv2.calcHist([lab], [0, 1, 2], None, [8, 6, 6], [0, 256] * 3)
    for axis, bins in enumerate(colors.shape):
        idx = np.arange(bins)
        lower = colors.take(idx - 1, axis, mode="clip")
        upper = colors.take(idx + 1, axis, mode="clip")
        colors = 0.6 * colors + 0.2 * lower + 0.2 * upper
    colors = np.sqrt(colors.ravel() / colors.sum())

    gray = lab[:, :, 0]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
    magnitude, angle = cv2.cartToPolar(gx, gy)
    strength = np.clip(magnitude / 32, 0, 1)
    orientation = (angle * (8 / np.pi)).astype(int) % 8
    edges = np.bincount(orientation.ravel(), weights=strength.ravel(), minlength=8)
    texture = np.sqrt(np.r_[(1 - strength).sum(), edges] / gray.size)

    # 整帧、全部图块、四个象限、颜色、纹理分别归一化，再按权重合并。
    visual = np.concatenate(
        [
            np.sqrt(0.46) * unit(tokens[0]),
            np.sqrt(0.17) * unit(patches.mean(axis=(0, 1))),
            np.sqrt(0.18 / 4) * np.asarray(regions).ravel(),
            np.sqrt(0.12) * colors,
            np.sqrt(0.07) * texture,
        ]
    )
    return visual, layout


def time_spans(boundaries: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.maximum(0, np.minimum(end, boundaries[1:]) - np.maximum(start, boundaries[:-1]))


def motion_profile(
    path: Path, shape: tuple[int, int], boundaries: np.ndarray, cuts: np.ndarray
) -> np.ndarray:
    """低分辨率连续画面上的光流，统计画面运动的速度、方向和切镜。"""
    h, w = shape
    still = np.zeros(15)
    still[0] = still[5] = 0.5  # 五档速度，静止及八个方向，另有一档切镜
    result = np.zeros((PHASES, len(still)))
    if not path.stat().st_size:
        return np.tile(np.sqrt(still), (PHASES, 1))
    frames = np.memmap(path, mode="r", dtype=np.uint8).reshape(-1, h, w)
    previous, last, last_time = frames[0], still, 0.0
    for i in range(1, len(frames)):
        current, now = frames[i], min(i / MOTION_FPS, boundaries[-1])
        last = still
        if cuts.searchsorted(now, side="right") > cuts.searchsorted(last_time, side="right"):
            last = np.zeros_like(still)
            last[-1] = 1
        elif cv2.norm(previous, current, cv2.NORM_L1) / previous.size > 0.65:
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 13, 2, 5, 1.1, 0)
            dx, dy = flow[1:-1:2, 1:-1:2].transpose(2, 0, 1)
            speed = np.hypot(dx, dy) * MOTION_FPS / max(h, w)
            speeds = np.digitize(speed, [0.018, 0.06, 0.18, 0.45])
            direction = (np.arctan2(dy, dx) * (4 / np.pi) + 0.5) % 8
            directions = np.where(speeds > 0, 1 + direction.astype(int), 0)
            last = np.r_[
                np.bincount(speeds.ravel(), minlength=5), np.bincount(directions.ravel(), minlength=9), 0
            ] / (2 * speed.size)
        result += time_spans(boundaries, last_time, now)[:, None] * last
        previous, last_time = current, now
    result += time_spans(boundaries, last_time, boundaries[-1])[:, None] * last
    return np.sqrt(result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12))


def audio_signature(path: Path) -> np.ndarray:
    # 按约 64 ms 的窗口覆盖整条音轨，统计响度分布和六段频带能量。
    levels = np.zeros(7)
    bands = np.zeros(6)
    if path.is_file() and path.stat().st_size:
        pcm = np.memmap(path, mode="r", dtype="<i2")
        width = 512
        taper = np.hanning(width)
        band_edges = [0, 8, 24, 48, 96, 160]
        for offset in range(0, len(pcm), width * 256):
            chunk = pcm[offset : offset + width * 256].astype(np.float32) / 32768
            last = len(chunk) % width
            if last:
                chunk = np.pad(chunk, (0, width - last))
            windows = chunk.reshape(-1, width)
            weights = np.ones(len(windows))
            if last:
                weights[-1] = last / width
            db = 10 * np.log10(np.maximum(1e-12, (windows**2).mean(axis=1)))
            levels += np.bincount(
                np.digitize(db, [-60, -48, -36, -26, -17, -9]), weights=weights, minlength=len(levels)
            )
            audible = db > -55
            if np.any(audible):
                power = abs(np.fft.rfft(windows[audible] * taper)) ** 2
                energy = np.add.reduceat(power, band_edges, axis=1)
                energy /= np.maximum(energy.sum(axis=1, keepdims=True), 1e-12)
                bands += weights[audible] @ energy
    if not levels.sum():
        levels[0] = 1
    return np.r_[0.65 * levels / levels.sum(), 0.35 * bands / max(bands.sum(), 1e-12)]


@dataclass
class Profile:
    visual: np.ndarray
    layout: np.ndarray
    motion: np.ndarray
    audio: np.ndarray
    samples: int


def sampling_command(
    clip: Clip, step: float, ffmpeg: list[str]
) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    w = max(2, round(320 * min(1, clip.aspect)))
    h = max(2, round(320 / max(1, clip.aspect)))
    mw, mh = max(8, round(96 * min(1, clip.aspect))), max(8, round(96 / max(1, clip.aspect)))
    select = f"isnan(prev_selected_t)+gte(t-prev_selected_t,{step:.9g})+gt(scene,{CUT_THRESHOLD})"
    vf = (
        f"scale={w}:{h}:flags=bilinear,setsar=1,select='{select}',"
        "metadata=mode=print:key=lavfi.scene_score:file=times.log:direct=1"
    )
    flow_vf = f"fps={MOTION_FPS}:start_time=0:eof_action=pass,scale={mw}:{mh}:flags=area,setsar=1"
    # 在同一个滤镜图中分流，避免稀疏的视觉取样阻塞连续的运动取样。
    graph = (
        f"[0:{clip.video}]setpts=PTS-STARTPTS,split[image][movement];"
        f"[image]{vf}[sample];[movement]{flow_vf}[gray]"
    )
    args = [
        *ffmpeg,
        "-i",
        str(clip.path.resolve()),
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        graph,
        "-map",
        "[sample]",
        "-fps_mode",
        "passthrough",
        "-threads:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
        "-map",
        "[gray]",
        "-threads:v",
        "1",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "motion.gray",
    ]
    if clip.audio is not None:
        af = (
            f"asetpts=PTS-STARTPTS+({clip.audio_offset:.9f})/TB,aresample=8000:async=1:first_pts=0,"
            f"apad,atrim=duration={clip.duration:.9f}"
        )
        args += ["-map", f"0:{clip.audio}", "-af", af, "-ac", "1", "-ar", "8000", "-f", "s16le", "audio.pcm"]
    return args, (h, w), (mh, mw)


def read_samples(work: Path, args: list[str], frame_size: int) -> Generator[tuple[float, float, bytes]]:
    with (work / "times.log").open("w+", encoding="ascii") as times, tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(args, cwd=work, stdout=subprocess.PIPE, stderr=errors)
        try:
            while raw := proc.stdout.read(frame_size):
                if len(raw) != frame_size:
                    raise ValueError("分析帧不完整")
                line = next((line for line in times if line.startswith("frame:")), "")
                key, _, score = times.readline().partition("=")
                if "pts_time:" not in line or key != "lavfi.scene_score":
                    raise ValueError("无法读取取样帧时间戳和镜头变化记录")
                yield float(line.split("pts_time:")[1]), float(score), raw
            if proc.wait():
                errors.seek(0)
                raise ValueError(errors.read().decode(errors="replace"))
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def summarize_samples(
    samples: Iterator[tuple[float, float, bytes]],
    shape: tuple[int, int],
    model: cv2.dnn.Net,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cuts = []
    count, last_time = 0, 0.0
    previous = features = visual_sum = None
    layout_sum = np.zeros((PHASES, 48))
    for timestamp, score, raw in samples:
        now = min(boundaries[-1], max(last_time, timestamp))
        if features is not None:
            if score > CUT_THRESHOLD:
                cuts.append(now)
            span = time_spans(boundaries, last_time, now)[:, None]
            visual_sum += span * features[0]
            layout_sum += span * features[1]
        if raw != previous:
            image = np.frombuffer(raw, dtype=np.uint8).reshape(*shape, 3)
            features = frame_features(image, model)
        if visual_sum is None:
            visual_sum = np.zeros((PHASES, len(features[0])))
        previous = raw
        last_time = now if count else 0.0
        count += 1
    if features is None:
        raise ValueError("没有可分析的画面")
    span = time_spans(boundaries, last_time, boundaries[-1])[:, None]
    visual_sum += span * features[0]
    layout_sum += span * features[1]
    return unit(visual_sum), layout_sum / np.diff(boundaries)[:, None], np.array(cuts), count


def profile(
    clip: Clip, step: float, model: cv2.dnn.Net, ffmpeg: list[str], stage: Stage = nullcontext
) -> Profile:
    boundaries = np.linspace(0, clip.video_duration, PHASES + 1)
    args, shape, motion_shape = sampling_command(clip, step, ffmpeg)
    with tempfile.TemporaryDirectory(prefix="clip-content-") as tmp:
        work = Path(tmp)
        with (
            stage("取样解码与画面特征提取"),
            closing(read_samples(work, args, shape[0] * shape[1] * 3)) as source,
        ):
            try:
                visual, layout, cuts, count = summarize_samples(source, shape, model, boundaries)
            except ValueError as exc:
                raise ValueError(f"无法分析 {clip.path.name}：{exc}") from exc
        with stage("运动特征提取"):
            motion = motion_profile(work / "motion.gray", motion_shape, boundaries, cuts)
        with stage("音频特征提取"):
            audio = audio_signature(work / "audio.pcm")
    return Profile(visual, layout, motion, audio, count)


def analyze(
    clips: list[Clip], step: float, ffmpeg: list[str], stage: Stage = nullcontext, *, offline: bool = False
) -> list[Profile]:
    with stage("加载视觉模型"):
        cv2.setNumThreads(min(4, os.cpu_count() or 1))
        model = cv2.dnn.readNetFromONNX(str(model_file(offline=offline)))
    result = []
    for i, clip in enumerate(clips, 1):
        print(f"分析整段 [{i}/{len(clips)}] {clip.path.name}", flush=True)
        started = time.monotonic()
        item = profile(clip, step, model, ffmpeg, stage)
        result.append(item)
        print(
            f"  已取样 {item.samples} 个画面，本段内容分析耗时 {time.monotonic() - started:.2f} 秒",
            flush=True,
        )
    return result


def content_costs(profiles: list[Profile], cuts: list[list[float]]) -> list[list[float]]:
    cost = [[0.0] * len(profiles) for _ in profiles]
    phases = np.arange(PHASES)
    for i, a in enumerate(profiles):
        for j in range(i):
            b = profiles[j]
            appearance = np.clip(1 - a.visual @ b.visual.T, 0, 1)
            layout = np.abs(a.layout[:, None, :] - b.layout[None, :, :]).mean(axis=2)
            movement = np.clip(1 - a.motion @ b.motion.T, 0, 1)
            differences = 0.81 * appearance + 0.08 * layout + 0.11 * movement
            # 每个时段只能匹配一次，避免两段素材仅因共用一个片头就被认为相似。
            whole = differences[phases, PHASE_MATCHES].mean(axis=1).min()
            sound = np.abs(a.audio - b.audio).sum() / 2
            shared = 0.78 * whole + 0.02 * sound
            cost[i][j] = float(shared + 0.13 * differences[-1, 0] + 0.07 * cuts[i][j])
            cost[j][i] = float(shared + 0.13 * differences[0, -1] + 0.07 * cuts[j][i])
    return cost
