"""Benchmark warm end-to-end stitches and compare two versions on the same host."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from stideo.ordering import DEFAULT_CONTENT_STEP
from tests import datasets

CASES = {
    "scenes-content": ("scenes", "content", "auto"),
    "bridges-smooth": ("bridges", "smooth", "auto"),
    "bridges-natural": ("bridges", "natural", "auto"),
    **{f"mixed-{codec}": ("codecs", "natural", codec) for codec in ("auto", "h264", "hevc", "av1")},
}
# Expensive stages worth comparing in addition to process wall time.
VISUAL_STAGE = "取样解码与画面特征提取"
TIMED_STAGES = (VISUAL_STAGE, "片段首尾分析", "视频转码", "成片接缝验证")


def fingerprint() -> dict:
    executable = shutil.which("ffmpeg")
    version = subprocess.check_output([executable or "ffmpeg", "-version"], text=True).splitlines()[0]
    return {
        "system": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ffmpeg": version,
        "fixtures": hashlib.sha256(Path(datasets.__file__).read_bytes()).hexdigest(),
    }


def measure_case(
    executable: str, root: Path, directory: Path, case: str, repeats: int, step: float | None = None
) -> dict:
    dataset, order, codec = CASES[case]
    command = [executable, str(root / dataset), "--offline", "--order", order, "--codec", codec]
    prefix = case
    if step is not None:
        command.extend(["--content-step", str(step)])
        prefix = f"{case}-step-{step:g}"
    samples = defaultdict(list)
    for trial in range(repeats):
        output = directory / f"{prefix}-{trial}.mp4"
        report = output.with_suffix(".json")
        start = time.perf_counter()
        completed = subprocess.run(
            [*command, "-o", str(output), "--report", str(report)],
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - start
        if completed.returncode:
            raise ValueError(f"{case} failed\n{completed.stdout}\n{completed.stderr}")
        data = json.loads(report.read_text(encoding="utf-8"))
        samples["wall_seconds"].append(wall)
        samples["analyzed_frames"].append(sum(x["samples"] for x in data["clips"]))
        samples["copied_seconds"].append(sum(x["video_seconds"] for x in data["clips"] if x["copied"]))
        for name in TIMED_STAGES:
            samples[name].append(data["stage_seconds"].get(name, 0.0))
        if order == "content":
            # Releases predating this report field defaulted to one second.
            samples["content_step"].append(data.get("content_step", 1.0 if step is None else step))
        output.unlink()
    return {key: statistics.median(values) for key, values in samples.items()}


def measure(executable: str, root: Path, directory: Path, repeats: int) -> dict:
    results = {}
    for case in CASES:
        values = measure_case(executable, root, directory, case, repeats)
        if "content_step" in values and values["content_step"] != DEFAULT_CONTENT_STEP:
            values["matched_sampling"] = measure_case(
                executable, root, directory, case, repeats, step=DEFAULT_CONTENT_STEP
            )
        results[case] = values
    return results


def regressions(reference: dict, actual: dict, tolerance: float) -> list[str]:
    if (
        reference["environment"] != actual["environment"]
        or reference["cases"].keys() != actual["cases"].keys()
    ):
        raise ValueError("Benchmark environment or dataset changed. Record a new baseline on this host.")
    if min(reference["repeats"], actual["repeats"]) < 3:
        raise ValueError("Use at least three repetitions for a timing comparison.")
    failures = []
    for case, values in actual["cases"].items():
        earlier = reference["cases"][case]
        matched = earlier.get("matched_sampling", earlier)
        if matched.get("content_step") != values.get("content_step"):
            matched = earlier
        for metric in ("wall_seconds", *TIMED_STAGES):
            baseline = (matched if metric == VISUAL_STAGE else earlier)[metric]
            if values[metric] > baseline + max(0.35, baseline * tolerance):
                failures.append(f"{case} {metric}: {baseline:.3f} -> {values[metric]:.3f} s")
        if values["copied_seconds"] < earlier["copied_seconds"] - 0.001:
            failures.append(f"{case} copied fewer seconds of compatible video")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--executable", default=shutil.which("stideo") or "stideo")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.50)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.tolerance <= 5:
        parser.error("Expected a positive repeat count and a tolerance between 0 and 5.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with tempfile.TemporaryDirectory(prefix="stideo-bench-") as tmp:
        work = Path(tmp)
        datasets.generate(work, sorted({spec[0] for spec in CASES.values()}))
        subprocess.run([args.executable, "--fetch-model"], check=True, stdout=subprocess.DEVNULL)
        cases = measure(args.executable, work, work, args.repeats)
    report = {"version": 1, "environment": fingerprint(), "repeats": args.repeats, "cases": cases}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.baseline:
        previous = json.loads(args.baseline.read_text(encoding="utf-8"))
        if issues := regressions(previous, report, args.tolerance):
            parser.exit(1, "\n".join(issues) + "\n")


if __name__ == "__main__":
    main()
