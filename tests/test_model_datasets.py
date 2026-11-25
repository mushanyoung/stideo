"""Exercise the labelled corpora, including real inference and shared model caches."""

import io
import json
import tarfile
lazy from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.media_tools import stitch
from tests.model_benchmark import BASELINE, scores
from tests.model_datasets import FLOWER_VIEWS, FLOWERS, flower_images, generate


@pytest.fixture(scope="module")
def model_clips(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("model-clips")
    generate(root, ["locations"], ["dev"])
    return root


@pytest.mark.parametrize("name", [BASELINE, "mobilevit-xxs"])
def test_retrieval_through_production_ranking(model_clips: Path, name: str) -> None:
    metrics = scores(model_clips, "dev", ["locations"], name)["macro"]
    assert metrics["map"] >= 0.80
    assert metrics["adjacency"] >= 0.75


def test_model_option_and_speed_workload(model_clips: Path, tmp_path: Path) -> None:
    report = stitch(
        model_clips / "dev" / "speed",
        tmp_path / "speed.mp4",
        "--model",
        "mobilevit-xxs",
        "--offline",
        dry=True,
    )
    samples = {c["name"]: c["samples"] for c in report["clips"]}
    assert report["model"] == "mobilevit-xxs"
    assert samples["cuts.mp4"] > samples["static.mp4"] >= 3
    assert samples["portrait.mp4"] >= 3


def test_generation_reuses_complete_cases(model_clips: Path) -> None:
    manifest = model_clips / "dev" / "locations" / "truth.json"
    known = manifest.stat().st_mtime_ns
    generate(model_clips, ["locations"], ["dev"])
    assert manifest.stat().st_mtime_ns == known
    rows = json.loads(manifest.read_bytes())["clips"]
    assert len({row["file"] for row in rows}) == len(rows) == 12


def test_photo_selection_is_disjoint_and_attributed(tmp_path: Path) -> None:
    path = tmp_path / "pictures.tgz"
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    assert ok
    with tarfile.open(path, "w:gz") as archive:
        entries = {
            f"flower_photos/{label}/{i}.jpg": encoded.tobytes()
            for label in FLOWERS
            for i in range(4 * FLOWER_VIEWS)
        }
        entries["flower_photos/LICENSE.txt"] = b"fixture attribution"
        for key, data in entries.items():
            member = tarfile.TarInfo(key)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    photos = flower_images(path, tmp_path)
    assert len({source for source, _ in photos.values()}) == 2 * len(FLOWERS) * FLOWER_VIEWS
    assert (tmp_path / "FLOWER-CREDITS.txt").read_bytes() == b"fixture attribution"
