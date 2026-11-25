"""Generate labelled development and holdout videos for visual-model comparisons."""

import argparse
import hashlib
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from filelock import FileLock

from tests.datasets import FPS, encode
from tests.model_scenes import (
    HEIGHT,
    LOCATIONS,
    SUBJECTS,
    WIDTH,
    location_backdrop,
    location_frame,
    object_frame,
)

FLOWERS = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
FLOWER_SHA = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
SETS = {"viewpoints": SUBJECTS, "locations": LOCATIONS, "cutaways": SUBJECTS, "flowers": FLOWERS}
SPLITS = {"dev": 173, "holdout": 3977}
VARIANTS = 3
FLOWER_VIEWS = 6
VERSION = 2
type Photos = dict[tuple[str, str, int], tuple[str, np.ndarray]]


def archive_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() == FLOWER_SHA:
                    return path
            raise ValueError(f"Unexpected SHA-256 for {path}")
        with tempfile.TemporaryDirectory(dir=path.parent) as work:
            pending = Path(work) / "download"
            with urlopen(FLOWER_URL, timeout=90) as src, pending.open("wb") as out:
                shutil.copyfileobj(src, out)
            with pending.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != FLOWER_SHA:
                    raise ValueError("Flower archive SHA-256 mismatch")
            pending.replace(path)
    return path


def flower_images(path: Path, out: Path) -> Photos:
    """Assign distinct source photos across both splits, then read the archive sequentially."""
    with tarfile.open(path, "r|gz") as archive:
        members = sorted(entry.name for entry in archive if entry.name.endswith(".jpg"))
    selected = {}
    for i, label in enumerate(FLOWERS):
        candidates = [name for name in members if name.split("/")[1] == label]
        development_and_pilot = np.random.default_rng(2026 + i).choice(
            candidates, 2 * FLOWER_VIEWS, replace=False
        )
        remaining = sorted(set(candidates) - set(development_and_pilot))
        holdout = np.random.default_rng(SPLITS["holdout"] + i).choice(remaining, FLOWER_VIEWS, replace=False)
        for split, picks in (("dev", development_and_pilot[:FLOWER_VIEWS]), ("holdout", holdout)):
            for view, name in enumerate(picks.tolist()):
                selected[name] = (split, label, view)
    photos = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in selected and member.name != "flower_photos/LICENSE.txt":
                continue
            with archive.extractfile(member) as source:
                data = source.read()
            if member.name in selected:
                bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                photos[selected[member.name]] = (member.name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            else:
                (out / "FLOWER-CREDITS.txt").write_bytes(data)
    return photos


def flower_film(photo: np.ndarray, variant: int) -> np.ndarray:
    height, width = photo.shape[:2]
    frames = []
    for tick in range(3 * FPS):
        fraction = tick / (3 * FPS - 1)
        scale = (0.90 - 0.12 * fraction) * min(width / WIDTH, height / HEIGHT)
        w, h = max(1, round(WIDTH * scale)), max(1, round(HEIGHT * scale))
        x = round((width - w) * (0.22 + 0.20 * variant + 0.10 * fraction))
        y = round((height - h) * (0.57 - 0.14 * fraction))
        frames.append(cv2.resize(photo[y : y + h, x : x + w], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
    return np.stack(frames)


def procedural(name: str, label: str, variant: int, seed: int, *, holdout: bool = False) -> np.ndarray:
    count = (4 if name == "cutaways" else 3) * FPS
    if name == "locations":
        background = location_backdrop(label, seed)
        return np.stack([location_frame(background, t, variant, seed) for t in range(count)])
    angle = (-0.95, 0.36, 1.12)[variant] + (seed % 21 - 10) / 45
    frames = np.stack(
        [
            object_frame(label, angle + 0.28 * math.sin(t / 15), variant, seed, holdout=holdout)
            for t in range(count)
        ]
    )
    if name == "cutaways":
        frames[:9] = frames[-9:] = (102, 106, 111)
        distractor = SUBJECTS[(SUBJECTS.index(label) + 1 + variant) % len(SUBJECTS)]
        frames[21:27] = object_frame(distractor, -angle, variant, seed, holdout=holdout)
    return frames


def speed_set(target: Path, seed: int) -> None:
    target.mkdir(parents=True)
    scene = procedural("locations", "platform", 1, seed)
    encode(target / "static.mp4", np.repeat(scene[:1], len(scene), axis=0), options=("-bf", "0"))
    moving = procedural("viewpoints", "biplane", 2, seed)
    encode(target / "portrait.mp4", moving[:, :, 48:-48], options=("-bf", "0"))
    mask = (np.arange(len(scene)) // 3) % 3
    scene[mask == 0] = 0
    scene[mask == 1] = 255
    encode(target / "cuts.mp4", scene, options=("-bf", "0"))


def write_case(target: Path, split: str, name: str, photos: Photos) -> None:
    target.mkdir()
    labels = SETS[name]
    rng = np.random.default_rng(SPLITS[split] + 101 * tuple(SETS).index(name))
    views = FLOWER_VIEWS if name == "flowers" else VARIANTS
    numbers = iter(rng.permutation(len(labels) * views).tolist())
    items = []
    for label in labels:
        for view in range(views):
            filename = f"clip{next(numbers):02}.mp4"
            item = {"file": filename, "label": label, "variant": view}
            if name == "flowers":
                source, photo = photos[split, label, view]
                item["source"] = source
                frames = flower_film(photo, view % VARIANTS)
            else:
                frames = procedural(name, label, view, int(rng.integers(100000)), holdout=split == "holdout")
            if view == 2 and name != "flowers":
                frames = frames[:, :, 32:-32]
            encode(target / filename, frames, options=("-bf", "0"))
            items.append(item)
    manifest = {"version": VERSION, "split": split, "dataset": name, "clips": items}
    (target / "truth.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def complete(target: Path, split: str, name: str) -> bool:
    if not target.exists():
        return False
    if name == "speed":
        files = ["static.mp4", "portrait.mp4", "cuts.mp4"]
    else:
        manifest = json.loads((target / "truth.json").read_text(encoding="utf-8"))
        if (manifest["version"], manifest["split"], manifest["dataset"]) != (VERSION, split, name):
            raise ValueError(f"Unexpected fixture version or labels in {target}. Use a new directory.")
        files = [item["file"] for item in manifest["clips"]]
    if not files or any(
        not (target / file).is_file() or not (target / file).stat().st_size for file in files
    ):
        raise ValueError(f"Incomplete fixtures in {target}. Use a new directory.")
    return True


def generate(root: Path, names: list[str], splits: list[str], *, flowers: Path | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".generation.lock")), tempfile.TemporaryDirectory(dir=root) as staging:
        photos = {}
        for split in splits:
            directory = root / split
            directory.mkdir(exist_ok=True)
            for name in ["speed", *names]:
                target = directory / name
                if complete(target, split, name):
                    continue
                pending = Path(staging) / name
                if name == "speed":
                    speed_set(pending, SPLITS[split])
                else:
                    if name == "flowers" and not photos:
                        if flowers is None:
                            raise ValueError("Specify a cache path for the flower archive.")
                        photos = flower_images(archive_file(flowers), root)
                    write_case(pending, split, name, photos)
                pending.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sets", choices=SETS, nargs="+", default=list(SETS)[:-1])
    parser.add_argument("--splits", choices=SPLITS, nargs="+", default=list(SPLITS))
    parser.add_argument(
        "--flower-archive",
        type=Path,
        default=Path.home() / ".cache" / "stideo-bench" / "flower_photos.tgz",
        help="Download and verify this cache file when flowers is requested.",
    )
    args = parser.parse_args()
    generate(args.destination, args.sets, args.splits, flowers=args.flower_archive)


if __name__ == "__main__":
    main()
