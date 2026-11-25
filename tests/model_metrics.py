"""Behavioural retrieval and playback metrics for fully labelled clip sets."""

from itertools import groupby, pairwise
from statistics import mean


def query_scores(distances: list[float], labels: list[str], query: int) -> tuple[float, float]:
    """Compute expected AP and top-1 credit under a uniformly random ordering of exact ties."""
    ranked = sorted(
        (d, label == labels[query])
        for i, (d, label) in enumerate(zip(distances, labels, strict=True))
        if i != query
    )
    positives = labels.count(labels[query]) - 1
    if not positives:
        raise ValueError("Each retrieval query needs another clip with the same label.")
    ap = first = 0.0
    seen = hits = 0
    for _, ties in groupby(ranked, key=lambda x: x[0]):
        flags = [hit for _, hit in ties]
        count, relevant = len(flags), sum(flags)
        probability = relevant / count
        if not seen:
            first = probability
        for j in range(count):
            preceding = j * (relevant - 1) / max(1, count - 1)
            ap += probability * (hits + 1 + preceding) / (seen + j + 1)
        hits += relevant
        seen += count
    return ap / positives, first


def quality(costs: list[list[float]], path: list[int], labels: list[str]) -> dict:
    if sorted(path) != list(range(len(labels))):
        raise ValueError("Playback path must be a permutation of the inputs.")
    queries = [query_scores(row, labels, i) for i, row in enumerate(costs)]
    groups = sorted(set(labels))
    class_ap = {
        group: mean(ap for label, (ap, _) in zip(labels, queries, strict=True) if label == group)
        for group in groups
    }
    joined = sum(labels[a] == labels[b] for a, b in pairwise(path))
    return {
        "map": mean(class_ap.values()),
        "top1": mean(score for _, score in queries),
        "adjacency": joined / (len(labels) - len(groups)),
        "class_ap": class_ap,
    }


def regressions(reference: dict, current: dict) -> list[str]:
    """Compare a saved run on the same corpus and CPU, including each individual model."""
    identity = ("version", "fixture_sha256", "platform", "arch", "cpu", "threads")
    if any(reference[key] != current[key] for key in identity):
        raise ValueError(
            "Fixture bytes, CPU or protocol differ; run both revisions on the same host and clips."
        )
    issues = []
    if not (reference["models"].keys() & current["models"].keys()):
        raise ValueError("No shared model to compare")
    for name, candidate in current["models"].items():
        if name not in reference["models"]:
            continue  # A newly added preset still has its own absolute evaluation.
        for split, data in candidate["splits"].items():
            previous = reference["models"][name]["splits"][split]
            issues.extend(split_regressions(f"{name}/{split}", previous, data))
    return issues


def split_regressions(label: str, previous: dict, data: dict) -> list[str]:
    issues = []
    if min(data["timing"]["rounds"], previous["timing"]["rounds"]) < 3:
        raise ValueError("Use at least three timing rounds for a regression comparison.")
    for metric, allowance in (("map", 0.03), ("top1", 0.05), ("adjacency", 0.05)):
        if previous["macro"][metric] - data["macro"][metric] > allowance:
            issues.append(f"{label} {metric} declined")
    for case, stats in data["cases"].items():
        if previous["cases"][case]["map"] - stats["map"] > 0.08:
            issues.append(f"{label}/{case} mAP declined")
    for key, floor in (("ms_per_frame", 1.0), ("sort_seconds", 0.25)):
        old = previous["timing"][key]
        if data["timing"][key] > old + max(0.30 * old, floor):
            issues.append(f"{label} {key} increased")
    return issues
