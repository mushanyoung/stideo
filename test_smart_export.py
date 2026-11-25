"""验证混合编码及 B 帧接缝，可用 python3 -B -m unittest test_smart_export 运行。"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


def probe(path, *options):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", *options, "-of", "json", str(path)]))


def packets(path):
    return probe(
        path,
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "packet=pts,dts,data_hash",
    )["packets"]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 FFmpeg")
class SmartExportTest(unittest.TestCase):
    def test_copy_duration_and_decoder_changes(self):
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
        if "libx264 " not in encoders or "libx265 " not in encoders:
            self.skipTest("测试素材需要 libx264 和 libx265")
        script = Path(__file__).with_name("stitch_videos.py").resolve()
        # 两个 H.264 片段的总时长超过三个 HEVC 片段。两个 H.264 的解码配置也不同。
        sources = [("h264", 48, 0), ("hevc", 6, 0), ("h264", 24, 3), ("hevc", 6, 0), ("hevc", 6, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i, (codec, frames, bframes) in enumerate(sources):
                options = (
                    ["-c:v", "libx264", "-bf", str(bframes)]
                    if codec == "h264"
                    else ["-c:v", "libx265", "-x265-params", "log-level=error:open-gop=0", "-tag:v", "hvc1"]
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=s=160x96:r=24",
                        "-frames:v",
                        str(frames),
                        "-vf",
                        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                        *options,
                        str(folder / f"{i}.mp4"),
                    ],
                    check=True,
                )
            output = folder / "合辑" / "test.mp4"
            cmd = [sys.executable, "-B", str(script), str(folder), "--order", "natural"]
            log = subprocess.check_output([*cmd, "-o", str(output)], text=True)
            self.assertIn("最终直接复制 2/5 段", log)
            video, audio = probe(output, "-show_streams")["streams"]
            self.assertEqual((video["codec_name"], audio["codec_name"]), ("h264", "aac"))
            result = packets(output)
            start = 0
            for i, (codec, count, _) in enumerate(sources):
                if codec == "h264":
                    actual = [p["data_hash"] for p in result[start : start + count]]
                    self.assertEqual(actual, [p["data_hash"] for p in packets(folder / f"{i}.mp4")])
                start += count
            base = Fraction(video["time_base"])
            self.assertEqual(sorted(p["pts"] * base for p in result), [Fraction(i, 24) for i in range(start)])
            times = [p["dts"] for p in result]
            self.assertEqual(times, sorted(set(times)))
            self.assertAlmostEqual(float(audio["duration"]), start / 24, places=4)
            override = subprocess.check_output([*cmd, "--codec", "hevc", "--dry-run"], text=True)
            self.assertIn("目标 HEVC", override)


if __name__ == "__main__":
    unittest.main()
