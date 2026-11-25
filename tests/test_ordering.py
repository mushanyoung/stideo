"""Behavioral ranking oracles independent of the embedding coordinates or score scale."""

from itertools import pairwise, permutations
lazy from pathlib import Path

import numpy as np
import pytest

from stideo.ordering import approximate_path, exact_path
from tests.media_tools import packets, stitch


def ids(report: dict) -> list[str]:
    return [clip["name"] for clip in report["clips"]]


def test_content_groups_subjects_across_accent_colors(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "subjects.mp4"
    result = stitch(corpus / "scenes", output)  # Default mode must exercise the real, pinned neural model.
    names = ids(result)
    assert result["order"] == "content"
    expected = {
        frozenset(pair)
        for pair in (("take1.mp4", "take11.mp4"), ("take2.mp4", "take10.mp4"), ("take3.mp4", "take20.mp4"))
    }
    assert expected <= {frozenset(pair) for pair in pairwise(names)}
    assert sorted(names) == sorted(p.name for p in (corpus / "scenes").iterdir())
    assert all(4 <= clip["samples"] <= 10 and clip["copied"] for clip in result["clips"])
    # Verify the physical playback order, independently of CLI reporting.
    expected_hashes = [p["data_hash"] for name in names for p in packets(corpus / "scenes" / name)]
    assert [p["data_hash"] for p in packets(output)] == expected_hashes


@pytest.mark.parametrize(("order", "sequence"), [("smooth", [10, 1, 20, 2]), ("natural", [1, 2, 10, 20])])
def test_order_without_model(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str, sequence: list[int]
) -> None:
    monkeypatch.setenv("STIDEO_MODEL_DIR", str(tmp_path / "empty-model-cache"))
    monkeypatch.setenv("STIDEO_OFFLINE", "1")
    result = stitch(corpus / "bridges", tmp_path / "order.mp4", "--order", order)
    assert ids(result) == [f"cut{i}.mp4" for i in sequence]
    assert not (tmp_path / "empty-model-cache").exists()
    assert all(x["samples"] == 0 and x["copied"] for x in result["clips"])
    assert ("片段首尾分析" in result["stage_seconds"]) == (order == "smooth")
    assert "加载视觉模型" not in result["stage_seconds"]
    assert "视频转码" not in result["stage_seconds"]


@pytest.mark.parametrize(("dataset", "prefix"), [("sounds", "sound"), ("motion", "pan")])
def test_similarity_covers_audio_and_movement(
    corpus: Path, tmp_path: Path, dataset: str, prefix: str
) -> None:
    names = ids(stitch(corpus / dataset, tmp_path / "order.mp4", dry=True))
    stems = [Path(name).stem for name in names]
    for first, second in ((1, 20), (2, 10)):
        assert abs(stems.index(f"{prefix}{first}") - stems.index(f"{prefix}{second}")) == 1


@pytest.mark.parametrize(("step", "minimum", "maximum"), [("60", 4, 6), ("0", 36, 36)])
def test_cuts_between_regular_samples(
    corpus: Path, tmp_path: Path, step: str, minimum: int, maximum: int
) -> None:
    report = stitch(corpus / "cuts", tmp_path / "cuts.mp4", "--content-step", step, dry=True)
    assert minimum <= report["clips"][0]["samples"] <= maximum
    assert report["clips"][0]["copied"] is None


def test_path_solvers_on_asymmetric_costs() -> None:
    rng = np.random.default_rng(41)
    costs = rng.integers(1, 70, size=(7, 7)).tolist()
    path = exact_path(costs)
    assert sorted(path) == list(range(7))
    assert sum(costs[a][b] for a, b in pairwise(path)) == min(
        sum(costs[a][b] for a, b in pairwise(order)) for order in permutations(range(7))
    )
    chain = rng.permutation(21).tolist()
    matrix = [[100.0] * len(chain) for _ in chain]
    for a, b in pairwise(chain):
        matrix[a][b] = 0.0
    assert approximate_path(matrix) == chain
