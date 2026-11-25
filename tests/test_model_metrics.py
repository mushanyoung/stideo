"""Check model comparisons independently of OpenCV and timing variability."""

from copy import deepcopy
from itertools import permutations
from statistics import mean

import pytest

from tests.model_benchmark import BASELINE, decision
from tests.model_datasets import SETS
from tests.model_metrics import quality, query_scores, regressions


def test_tied_retrieval_matches_exhaustive_reference() -> None:
    labels = ["a", "b", "a", "a", "b"]
    ap, top = query_scores([0.0] * 5, labels, 0)
    runs = []
    for sequence in permutations(labels[1:]):
        hits = [i for i, value in enumerate(sequence, 1) if value == "a"]
        runs.append(mean(hit / rank for hit, rank in enumerate(hits, 1)))
    assert ap == pytest.approx(mean(runs))
    assert top == 0.5


def test_retrieval_and_playback_have_distinct_oracles() -> None:
    labels = ["a", "b", "a", "b"]
    costs = [[float(a != b) for b in labels] for a in labels]
    grouped = quality(costs, [3, 1, 2, 0], labels)
    alternating = quality(costs, [0, 1, 2, 3], labels)
    assert grouped["map"] == grouped["top1"] == grouped["adjacency"] == 1
    assert alternating["map"] == 1
    assert alternating["adjacency"] == 0
    with pytest.raises(ValueError, match="permutation"):
        quality(costs, [0, 1], labels)


def run(score: float, ms: float) -> dict:
    split = {
        "macro": dict.fromkeys(("map", "top1", "adjacency"), score),
        "cases": {name: {"map": score} for name in SETS},
        "timing": {"ms_per_frame": ms, "sort_seconds": 2.0, "rounds": 3},
    }
    return {"splits": {"dev": deepcopy(split), "holdout": deepcopy(split)}}


def test_holdout_veto_and_incomplete_protocol() -> None:
    models = {BASELINE: run(0.8, 30), "fast": run(0.8, 20), "third": run(0.8, 24)}
    assert decision(models)["recommended"] == "fast"
    models["fast"]["splits"]["holdout"]["cases"]["flowers"]["map"] = 0.6
    verdict = decision(models)
    assert verdict["status"] == "retain"
    assert verdict["dev_candidate"] == "fast"
    assert verdict["recommended"] == BASELINE
    del models["fast"]["splits"]["dev"]["cases"]["flowers"]
    assert decision(models)["status"] == "incomplete"


def test_reference_catches_quality_and_cpu_regressions() -> None:
    report = {
        "version": 1,
        "fixture_sha256": "same",
        "platform": "test",
        "cpu": "test",
        "arch": "test",
        "threads": 4,
        "models": {BASELINE: run(0.9, 20)},
    }
    changed = deepcopy(report)
    assert not regressions(report, changed)
    changed["models"][BASELINE] = run(0.6, 35)
    failures = regressions(report, changed)
    assert any("flowers" in f for f in failures)
    assert any("ms_per_frame" in f for f in failures)
    changed["fixture_sha256"] = "different"
    with pytest.raises(ValueError, match="Fixture"):
        regressions(report, changed)


def test_default_change_compares_the_models_users_actually_get() -> None:
    original = {
        "version": 1,
        "fixture_sha256": "same",
        "platform": "test",
        "cpu": "test",
        "arch": "test",
        "threads": 4,
        "models": {BASELINE: run(0.9, 20), "alternate": run(0.7, 10)},
        "default_model": BASELINE,
    }
    changed = {**original, "default_model": "alternate"}
    assert any("default" in issue and "mAP" in issue for issue in regressions(original, changed))
    changed["models"] = {"new": run(0.9, 12)}
    changed["default_model"] = "new"
    assert not regressions(original, changed)
