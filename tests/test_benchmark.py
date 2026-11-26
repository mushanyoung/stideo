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


def test_changed_sampling_keeps_wall_gate_and_compares_visual_time_at_equal_intervals() -> None:
    baseline = measurement(8.0, 36.0, 4.0)
    baseline["cases"]["example"].update(
        content_step=1.0, matched_sampling={TIMED_STAGES[0]: 7.0, "content_step": 0.5}
    )
    actual = measurement(10.0, 36.0, 7.1)
    case = actual["cases"]["example"]
    case["content_step"] = 0.5
    assert not regressions(baseline, actual, 0.5)
    case["wall_seconds"] = 14.0
    assert "wall_seconds" in regressions(baseline, actual, 0.5)[0]
    case["wall_seconds"] = 10.0
    case[TIMED_STAGES[0]] = 11.0
    assert TIMED_STAGES[0] in regressions(baseline, actual, 0.5)[0]
    case["content_step"] = 0.25
    case[TIMED_STAGES[0]] = 7.1
    assert TIMED_STAGES[0] in regressions(baseline, actual, 0.5)[0]
