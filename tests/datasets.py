"""Generate short deterministic video corpora. Run python -m tests.datasets DIRECTORY."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 160, 96
FPS = 12
SDR = ("bt709", "bt709", "bt709")
PQ = ("bt2020", "smpte2084", "bt2020nc")
ENCODERS = {
    "ffv1": ["-c:v", "ffv1"],
    "h264": ["-c:v", "libx264", "-preset", "fast", "-crf", "15"],
    "hevc": [
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-x265-params",
        "log-level=error:open-gop=0:pools=1:frame-threads=1",
        "-tag:v",
        "hvc1",
    ],
    "av1": ["-c:v", "libsvtav1", "-preset", "10", "-crf", "25"],
}


def ffmpeg(args: list[str], pixels: bytes | None = None) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", *args], input=pixels, check=True
    )


def encode(
    path: Path,
    frames: np.ndarray,
    *,
    codec: str = "h264",
    tone: int = 0,
    color: tuple[str, str, str] = SDR,
    options: tuple[str, ...] = (),
) -> None:
    n, height, width, _ = frames.shape
    audio = (
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={tone}:sample_rate=48000:duration={n / FPS:.9f}",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
        ]
        if tone
        else []
    )
    pix = "yuv420p10le" if color == PQ else "yuv420p"
    primaries, transfer, matrix = color
    ffmpeg(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            *audio,
            *ENCODERS[codec],
            "-threads",
            "1",
            "-g",
            str(2 * FPS),
            "-pix_fmt",
            pix,
            "-vf",
            f"format={pix},setparams=range=limited:color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}",
            "-color_range",
            "tv",
            "-color_primaries",
            primaries,
            "-color_trc",
            transfer,
            "-colorspace",
            matrix,
            *options,
            str(path),
        ],
        frames.tobytes(),
    )


def scene(subject: str, tick: int, variant: int) -> np.ndarray:
    """Draw three silhouettes against the same sky and ground, with crossed accent colors."""
    image = np.full((HEIGHT, WIDTH, 3), (126, 158, 172), dtype=np.uint8)
    image[63:] = (74, 89, 89)
    ink, light = (46, 52, 60), (209, 211, 191)
    accent = (188, 72, 54) if variant == 0 else (49, 101, 167)
    x = 4 * variant + (tick // 5) % 3
    if subject == "train":
        cv2.line(image, (0, 66), (159, 66), ink, 3)
        for col in (10, 54, 98):
            cv2.rectangle(image, (x + col, 37), (x + col + 39, 59), accent, -1)
            for window in (5, 18, 29):
                cv2.rectangle(image, (x + col + window, 40), (x + col + window + 6, 48), light, -1)
            for wheel in (8, 31):
                cv2.circle(image, (col + x + wheel, 60), 5, ink, -1)
    elif subject == "pines":
        for col, top, base in ((25, 22, 80), (70, 6, 72), (119, 19, 85)):
            cv2.rectangle(image, (col + x - 3, top + 25), (col + x + 3, base), ink, -1)
            for y in (top, top + 13, top + 23):
                triangle = np.array([(col + x, y), (col + x - 17, y + 26), (col + x + 17, y + 26)], np.int32)
                cv2.fillConvexPoly(image, triangle, accent if y == top + 13 else ink)
    elif subject == "lighthouse":
        cv2.fillConvexPoly(
            image, np.array([(74 + x, 24), (87 + x, 24), (99 + x, 82), (62 + x, 82)], np.int32), light
        )
        cv2.rectangle(image, (71 + x, 42), (90 + x, 50), accent, -1)
        cv2.rectangle(image, (72 + x, 17), (89 + x, 24), ink, -1)
        cv2.fillConvexPoly(image, np.array([(70 + x, 16), (80 + x, 7), (92 + x, 16)], np.int32), accent)
        for col in (15, 124):
            cv2.line(image, (col, 72), (col + 17, 70), light, 2)
    else:
        raise ValueError(subject)
    return image


def film(subject: str, count: int, variant: int = 0) -> np.ndarray:
    return np.stack([scene(subject, i, variant) for i in range(count)])


def scenes(folder: Path) -> None:
    # Identical introduction and credits outlast the complete smooth-order sampling window.
    specs = (
        (1, "train", 0),
        (2, "pines", 1),
        (3, "lighthouse", 0),
        (10, "pines", 0),
        (11, "train", 1),
        (20, "lighthouse", 1),
    )
    for number, subject, variant in specs:
        frames = film(subject, 6 * FPS, variant)
        frames[:FPS] = frames[-FPS:] = (88, 88, 88)
        encode(folder / f"take{number}.mp4", frames, options=("-bf", "0"))


def bridges(folder: Path) -> None:
    colors = [(150, 20, 32), (18, 164, 26), (22, 33, 174), (172, 158, 13), (155, 20, 158)]
    for i, number in enumerate((10, 1, 20, 2)):
        frames = np.empty((2 * FPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
        frames[:FPS] = colors[i]
        frames[FPS:] = colors[i + 1]
        encode(folder / f"cut{number}.mp4", frames, options=("-bf", "0"))


def codecs(folder: Path) -> None:
    # H.264 wins by duration despite HEVC having more files. Include changing SPS/B-frame depth.
    specs = (
        ("h264", 36, 0),
        ("hevc", 6, 440),
        ("h264", 24, 660),
        ("hevc", 6, 0),
        ("hevc", 6, 330),
        ("av1", 12, 0),
    )
    for i, (codec, frames, tone) in enumerate(specs, 1):
        options = ("-bf", "0" if i == 1 else "3") if codec == "h264" else ()
        encode(
            folder / f"shot{i}.mp4",
            film("train" if i % 2 else "pines", frames, i % 2),
            codec=codec,
            tone=tone,
            options=options,
        )


def geometry(folder: Path) -> None:
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        native = Path(tmp) / "landscape.mp4"
        encode(native, film("lighthouse", 36), options=("-bf", "0"))
        ffmpeg(
            [
                "-display_rotation",
                "90",
                "-i",
                str(native),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                str(folder / "rotated.mp4"),
            ]
        )
    encode(folder / "upright.mp4", np.rot90(film("train", 12), axes=(1, 2)))
    encode(folder / "wide.mp4", film("pines", 12)[:, ::2, ::2])


def ranges(folder: Path) -> None:
    for name, subject, length, codec, color in (
        ("pq", "lighthouse", 30, "hevc", PQ),
        ("sdr-hevc", "pines", 12, "hevc", SDR),
        ("sdr-h264", "train", 18, "h264", SDR),
    ):
        frames = film(subject, length)
        for i, code in enumerate((48, 96, 140, 180)):
            frames[:, :16, i * 40 : (i + 1) * 40] = code
        encode(folder / f"{name}.mp4", frames, codec=codec, color=color)


def timing(folder: Path) -> None:
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        base = Path(tmp) / "long.mp4"
        encode(base, film("train", 48), options=("-bf", "2"))
        ffmpeg(["-ss", "0.25", "-i", str(base), "-t", "1", "-c:v", "copy", str(folder / "trim.mp4")])
        ffmpeg(
            [
                "-i",
                str(base),
                "-an",
                "-vf",
                "select='lt(mod(n,4),2)'",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                str(folder / "vfr.mp4"),
            ]
        )
        ffmpeg(
            [
                "-i",
                str(base),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=660:sample_rate=48000:duration=1.5",
                "-filter_complex",
                "[0:v]trim=duration=1[v]",
                "-map",
                "[v]",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-c:a",
                "aac",
                str(folder / "long-audio.mp4"),
            ]
        )


def sounds(folder: Path) -> None:
    # All pictures match; the bins in the interior audio spectrum distinguish the pairs.
    for number, tone in ((1, 220), (2, 1800), (10, 1800), (20, 220)):
        frames = film("train", 4 * FPS)
        encode(
            folder / f"sound{number}.mp4",
            frames,
            tone=tone,
            options=("-af", "volume=0:enable='lt(t,1)+gte(t,3)'"),
        )


def motion(folder: Path) -> None:
    # Both directions visit precisely the same cyclic texture positions over time.
    texture = np.random.default_rng(724).integers(0, 256, (12, 10, 3), np.uint8)
    tile = cv2.resize(texture, (24, HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Moderate contrast keeps continuous panning below the scene-cut threshold.
    image = 96 + np.tile(tile, (1, 6, 1)) // 4
    for number, direction in ((1, 1), (2, -1), (10, -1), (20, 1)):
        frames = np.stack([np.roll(image, direction * tick * 2, axis=1) for tick in range(48)])
        still = np.repeat(image[None], FPS, axis=0)
        frames = np.concatenate((still, frames, still))
        encode(folder / f"pan{number}.mkv", frames, codec="ffv1")


def cuts(folder: Path) -> None:
    frames = np.zeros((3 * FPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
    frames[7:14] = 255
    frames[25:] = 255
    encode(folder / "flash.mp4", frames, options=("-bf", "0"))


DATASETS = {
    "sounds": sounds,
    "motion": motion,
    "cuts": cuts,
    "scenes": scenes,
    "bridges": bridges,
    "codecs": codecs,
    "geometry": geometry,
    "ranges": ranges,
    "timing": timing,
}


def generate(root: Path, names: list[str]) -> None:
    for name in names:
        target = root / name
        target.mkdir(parents=True, exist_ok=False)
        DATASETS[name](target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--only", choices=DATASETS, nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    generate(args.destination, args.only)


if __name__ == "__main__":
    main()
