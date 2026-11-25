"""Check that the comparison gate distinguishes jitter, copy loss and real slowdowns."""

import pytest

from tests.benchmark import TIMED_STAGES, regressions


def measurement(wall: float, kept: float, inference: float) -> dict:
    metrics = dict.fromkeys(TIMED_STAGES, 0.1)
    metrics.update(wall_seconds=wall, copied_seconds=kept)
    metrics[TIMED_STAGES[0]] = inference
    return {"environment": {"system": "fixture"}, "cases": {"example": metrics}, "repeats": 3}


def test_relative_time_and_copy_gate() -> None:
    baseline = measurement(2.0, 2.0, 0.5)
    assert not regressions(baseline, measurement(2.2, 2.0, 0.6), 0.5)
    assert len(regressions(baseline, measurement(3.5, 1.0, 1.0), 0.5)) == 3
    with pytest.raises(ValueError, match="environment"):
        regressions(baseline, {**baseline, "environment": {}}, 0.5)
