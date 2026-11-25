"""Verify decoded media, packet identity, timing and work performed through the CLI."""

from fractions import Fraction
lazy from pathlib import Path

import numpy as np
import pytest

from tests.media_tools import packets, pictures, probe, sound, stitch


@pytest.mark.parametrize(
    ("requested", "codec", "kept"),
    [
        ("auto", "h264", {1, 3}),
        ("h264", "h264", {1, 3}),
        ("hevc", "hevc", {2, 4, 5}),
        ("av1", "av1", {6}),
    ],
)
def test_codec_quality_and_reuse(
    corpus: Path, tmp_path: Path, requested: str, codec: str, kept: set[int]
) -> None:
    folder = corpus / "codecs"
    output = tmp_path / "joined.mp4"
    report = stitch(folder, output, "--order", "natural", "--codec", requested)
    assert report["target"]["codec"] == codec
    if requested == "auto":
        assert report["copyable_seconds"] == {"h264": 5, "hevc": 1.5, "av1": 1}
    assert {i for i, x in enumerate(report["clips"], 1) if x["copied"]} == kept
    assert 0 < sum(report["stage_seconds"].values()) <= report["total_seconds"]
    video, audio = probe(output, "-show_streams")["streams"]
    assert video["codec_name"] == codec
    assert (audio["codec_name"], int(audio["sample_rate"]), audio["channels"]) == ("aac", 48000, 2)
    actual_packets = packets(output)
    counts = [36, 6, 24, 6, 6, 12]
    duration = Fraction(sum(counts), 12)
    assert abs(float(audio["duration"]) - float(duration)) <= 0.002
    assert sorted(p["pts"] * Fraction(video["time_base"]) for p in actual_packets) == [
        Fraction(i, 12) for i in range(sum(counts))
    ]
    dts = [p["dts"] for p in actual_packets]
    assert dts == sorted(set(dts))
    frames = pictures(output, (80, 48)).astype(float)
    pcm = sound(output)
    start = 0
    for i, (count, hz) in enumerate(zip(counts, [0, 440, 660, 0, 330, 0], strict=True), 1):
        source = folder / f"shot{i}.mp4"
        if i in kept:
            assert [p["data_hash"] for p in actual_packets[start : start + count]] == [
                p["data_hash"] for p in packets(source)
            ]
        # Every decoded frame including each seam must still resemble its source.
        difference = abs(frames[start : start + count] - pictures(source, (80, 48))).mean(axis=(1, 2, 3))
        assert difference.max() < 6, (source.name, difference.max())
        window = pcm[(start + 1) * 4000 : (start + count - 1) * 4000]
        if hz:
            power = abs(np.fft.rfft(window))
            assert abs(power.argmax() * 48000 / len(window) - hz) < 5
            assert np.max(abs(window)) > 0.02
        else:
            assert np.max(abs(window)) < 0.002
        start += count


def test_display_matrix_and_letterboxing(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "portrait.mp4"
    result = stitch(corpus / "geometry", output, "--order", "natural")
    assert (result["target"]["size"], result["target"]["rotation"]) == ([96, 160], 90)
    assert {x["name"] for x in result["clips"] if x["copied"]} == {"rotated.mp4"}
    decoded = pictures(output, (96, 160)).astype(float)
    offset = 0
    for clip in result["clips"]:
        source = pictures(corpus / "geometry" / clip["name"], (96, 160))
        length = len(source)
        assert abs(decoded[offset : offset + length] - source).mean(axis=(1, 2, 3)).max() < 5
        offset += length
    assert len(decoded) == offset


@pytest.mark.parametrize(
    ("codec", "transfer", "depth", "copy"),
    [
        ("auto", "smpte2084", 10, "pq.mp4"),
        ("h264", "bt709", 8, "sdr-h264.mp4"),
    ],
)
def test_hdr_sdr_conversion(
    corpus: Path, tmp_path: Path, codec: str, transfer: str, depth: int, copy: str
) -> None:
    output = tmp_path / "range.mp4"
    report = stitch(corpus / "ranges", output, "--order", "natural", "--codec", codec)
    assert {c["name"] for c in report["clips"] if c["copied"]} == {copy}
    v = probe(output, "-select_streams", "v:0", "-show_streams")["streams"][0]
    assert (v["color_transfer"], v["color_primaries"]) == (transfer, "bt2020" if depth == 10 else "bt709")
    assert v["pix_fmt"] == ("yuv420p10le" if depth == 10 else "yuv420p")
    decoded = pictures(output, (80, 48))
    assert len(decoded) == 60
    # Exercise the tone curve: every segment retains a visible subject on its background.
    offset = 0
    for clip in report["clips"]:
        count = round(clip["duration"] * 12)
        sample = decoded[offset + count // 2].astype(float)
        assert sample.std() > 12
        swatches = sample[3:6, 10::20].mean(axis=(0, 2))
        assert np.all(np.diff(swatches) > 4)
        reference = pictures(corpus / "ranges" / clip["name"], (80, 48))[0, 3:6, 10::20].mean(axis=(0, 2))
        change = np.max(abs(reference - swatches))
        same_transfer = (clip["name"] == "pq.mp4") == (transfer == "smpte2084")
        if same_transfer:
            assert change < 5
        else:
            assert change > 9
        offset += count


@pytest.mark.parametrize("force_fps", [False, True], ids=["preserve-vfr", "force-cfr"])
def test_trim_vfr_and_audio_extension(corpus: Path, tmp_path: Path, *, force_fps: bool) -> None:
    folder = corpus / "timing"
    output = tmp_path / "timeline.mp4"
    options = ["--fps", "12"] if force_fps else []
    report = stitch(folder, output, "--order", "natural", *options)
    clips = report["clips"]
    assert [x["name"] for x in clips] == ["long-audio.mp4", "trim.mp4", "vfr.mp4"]
    assert [c["copied"] for c in clips] == [False, False, not force_fps]
    assert "音轨" in clips[0]["reason"]
    assert "隐藏" in clips[1]["reason"] or "关键帧" in clips[1]["reason"]
    output_packets = packets(output)
    video, audio = probe(output, "-show_streams")["streams"]
    base = Fraction(video["time_base"])
    assert abs(float(video["duration"]) - float(audio["duration"])) < 0.002
    assert abs(float(video["duration"]) - sum(c["duration"] for c in clips)) <= 1 / 6
    assert len(pictures(output, (80, 48))) == len(output_packets)
    if force_fps:
        assert sorted(p["pts"] * base for p in output_packets) == [
            Fraction(i, 12) for i in range(len(output_packets))
        ]
    else:
        source = folder / "vfr.mp4"
        expected = packets(source)
        tail = output_packets[-len(expected) :]
        src_base = Fraction(probe(source, "-show_streams")["streams"][0]["time_base"])
        assert [p["data_hash"] for p in tail] == [p["data_hash"] for p in expected]
        assert [(p["pts"] - tail[0]["pts"]) * base for p in tail] == [
            (p["pts"] - expected[0]["pts"]) * src_base for p in expected
        ]


def test_explicit_canvas_and_frame_rate(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "square.mp4"
    report = stitch(corpus / "geometry", output, "--order", "natural", "--size", "128x128", "--fps", "24")
    assert report["target"]["size"] == [128, 128]
    assert all(not c["planned_copy"] and not c["copied"] for c in report["clips"])
    video = probe(output, "-select_streams", "v:0", "-show_streams")["streams"][0]
    assert (video["width"], video["height"], video["nb_frames"], video["avg_frame_rate"]) == (
        128,
        128,
        "120",
        "24/1",
    )
    assert len(pictures(output, (128, 128))) == 120
