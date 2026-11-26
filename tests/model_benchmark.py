"""Compare vision models on identical labelled clips and isolated CPU workloads."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from stideo.media import Clip, probe
from stideo.models import DEFAULT_MODEL, MODELS, model_file
from stideo.ordering import DEFAULT_CONTENT_STEP, ContentRanking, content_ranking
from stideo.vision import VisualEncoder
from tests.model_datasets import SETS, SPLITS
from tests.model_metrics import quality, regressions

if sys.platform in {"darwin", "linux"}:
    import resource

BASELINE = "dinov2-small-int8"
METRICS = ("map", "top1", "adjacency")


def inputs(paths: list[Path]) -> list[Clip]:
    result = []
    for path in paths:
        clip = probe(path)
        if clip is None:
            raise ValueError(f"Missing video track in {path}")
        result.append(clip)
    return result


def rank(paths: list[Path], model: str, step: float = DEFAULT_CONTENT_STEP) -> tuple[ContentRanking, float]:
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = content_ranking(inputs(paths), step, offline=True, model_name=model)
    return result, time.perf_counter() - start


def timing_frames(root: Path, split: str) -> list[np.ndarray]:
    # Decode once outside the timer. Keep the aspect ratio and the same 320-pixel side as content profiling.
    frames = []
    for path in sorted((root / split / "speed").glob("*.mp4")):
        (clip,) = inputs([path])
        width, height = max(2, round(320 * min(1, clip.aspect))), max(2, round(320 / max(1, clip.aspect)))
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"fps=4,scale={width}:{height}:flags=bilinear",
                "-pix_fmt",
                "rgb24",
                "-threads:v",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        frames.extend(np.frombuffer(data, np.uint8).reshape(-1, height, width, 3))
    return frames


def timing(root: Path, split: str, model: str, rounds: int, step: float = DEFAULT_CONTENT_STEP) -> dict:
    frames = timing_frames(root, split)
    start = time.perf_counter()
    encoder = VisualEncoder.load(model, offline=True)
    load = time.perf_counter() - start
    for frame in frames[:3]:
        encoder.semantic(frame)
    durations = []
    pass_times = []
    for _ in range(rounds):
        start = time.perf_counter()
        for frame in frames:
            before = time.perf_counter()
            encoder.semantic(frame)
            durations.append((time.perf_counter() - before) * 1000)
        pass_times.append((time.perf_counter() - start) * 1000 / len(frames))
    del encoder
    paths = sorted((root / split / "speed").glob("*.mp4"))
    times = []
    for _ in range(rounds):
        result, elapsed = rank(paths, model, step)
        times.append(elapsed)
    return {
        "load_seconds": load,
        "frames_per_pass": len(frames),
        "rounds": rounds,
        "ms_per_frame": median(pass_times),
        "frame_p95_ms": float(np.percentile(durations, 95)),
        "sort_seconds": median(times),
        "sort_trials_seconds": times,
        "sampled_frames": sum(result.samples.values()),
    }


def scores(root: Path, split: str, names: list[str], model: str, step: float = DEFAULT_CONTENT_STEP) -> dict:
    cases = {}
    for name in names:
        directory = root / split / name
        manifest = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
        items = sorted(manifest["clips"], key=lambda x: x["file"])
        paths = [directory / clip["file"] for clip in items]
        ranking, elapsed = rank(paths, model, step)
        cases[name] = {
            **quality(ranking.costs, ranking.path, [x["label"] for x in items]),
            "order": [paths[i].name for i in ranking.path],
            "sort_seconds": elapsed,
            "samples": sum(ranking.samples.values()),
        }
    return {"cases": cases, "macro": {key: mean(case[key] for case in cases.values()) for key in METRICS}}


def peak_memory() -> float | None:
    if sys.platform not in {"darwin", "linux"}:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def evaluate(job: tuple[Path, str, list[str], list[str], int, float, bool]) -> tuple[str, dict]:
    root, model, names, splits, rounds, step, offline = job
    weights = model_file(model, offline=offline)
    result = {}
    for split in splits:
        sys.stderr.write(f"Scoring {model} on {split}\n")
        result[split] = {
            **scores(root, split, names, model, step),
            "timing": timing(root, split, model, rounds, step),
        }
    return model, {"weights_bytes": weights.stat().st_size, "peak_rss_mib": peak_memory(), "splits": result}


def eligible(candidate: dict, baseline: dict) -> bool:
    """Require similar absolute quality and a material win in quality or speed."""
    gains = {m: candidate["macro"][m] - baseline["macro"][m] for m in METRICS}
    worst = min(case["map"] - baseline["cases"][key]["map"] for key, case in candidate["cases"].items())
    speed = candidate["timing"]["ms_per_frame"] / baseline["timing"]["ms_per_frame"]
    wall = candidate["timing"]["sort_seconds"] / baseline["timing"]["sort_seconds"]
    return (
        gains["map"] >= -0.02
        and min(gains["adjacency"], gains["top1"]) >= -0.03
        and worst >= -0.08
        and wall <= 1.10
        and (speed <= 0.85 or (gains["map"] >= 0.03 and speed <= 1.10))
    )


def decision(models: dict) -> dict:
    if BASELINE not in models or any(set(m["splits"]) != set(SPLITS) for m in models.values()):
        return {"status": "incomplete"}
    if len(models) < 2 or any(
        set(data["cases"]) != set(SETS) or data["timing"]["rounds"] < 3
        for model in models.values()
        for data in model["splits"].values()
    ):
        return {"status": "incomplete"}
    original = models[BASELINE]["splits"]
    choices = {
        k: v["splits"]["dev"]
        for k, v in models.items()
        if k != BASELINE and eligible(v["splits"]["dev"], original["dev"])
    }
    if not choices:
        return {"status": "retain", "recommended": BASELINE, "dev_candidate": None}
    best = max(c["macro"]["map"] for c in choices.values())
    finalists = [k for k, c in choices.items() if c["macro"]["map"] >= best - 0.02]
    selection = min(finalists, key=lambda k: choices[k]["timing"]["ms_per_frame"])
    passes = eligible(models[selection]["splits"]["holdout"], original["holdout"])
    return {
        "status": "replace" if passes else "retain",
        "dev_candidate": selection,
        "recommended": selection if passes else BASELINE,
    }


def fixture_hash(root: Path, names: list[str], splits: list[str]) -> str:
    digest = hashlib.sha256()
    for split in splits:
        for name in [*names, "speed"]:
            for path in sorted((root / split / name).iterdir()):
                digest.update(str(path.relative_to(root)).encode())
                with path.open("rb") as stream:
                    digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def cpu_name() -> str:
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"model name", "Hardware", "Processor"}:
                return value.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--sets", nargs="+", choices=SETS)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--content-step",
        type=float,
        default=DEFAULT_CONTENT_STEP,
        help="Seconds between visual samples (default %(default)g, zero selects every frame).",
    )
    parser.add_argument(
        "--reference", type=Path, help="Fail on quality or timing regressions against a previous report."
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.content_step) or args.content_step < 0:
        parser.error("--content-step must be a finite non-negative number of seconds.")
    if args.rounds < 1 or args.output.exists():
        parser.error("Use a positive round count and a new output file.")
    if not args.sets:
        args.sets = [
            name
            for name in SETS
            if all((args.fixtures / s / name / "truth.json").is_file() for s in args.splits)
        ]
    if not args.sets:
        parser.error("Generate labelled fixtures first.")
    fingerprint = fixture_hash(args.fixtures, args.sets, args.splits)
    jobs = [
        (args.fixtures, m, args.sets, args.splits, args.rounds, args.content_step, args.offline)
        for m in dict.fromkeys(args.models)
    ]
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        models = dict(pool.map(evaluate, jobs))
    report = {
        "version": 1,
        "fixture_sha256": fingerprint,
        "platform": platform.system(),
        "cpu": cpu_name(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "threads": min(4, os.cpu_count() or 1),
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "default_model": DEFAULT_MODEL,
        "content_step": args.content_step,
        "models": models,
        "decision": decision(models),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if failures := regressions(reference, report):
            parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
