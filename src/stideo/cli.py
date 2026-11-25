"""Stitch a folder of video clips into a continuous MP4."""

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
lazy from fractions import Fraction

from stideo.export import CODECS, Plan, build_plan, render
from stideo.media import Clip, canvas, common_fps, probe, ratio
from stideo.models import model_file
from stideo.ordering import EDGE_SECONDS, EXACT_LIMIT, content_order, smooth_order
from stideo.timing import Timings

VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".mxf",
    ".ogv",
    ".ts",
    ".webm",
    ".wmv",
}
OUTPUT_NAMES = {"content": "画面相似排序.mp4", "smooth": "连贯排序.mp4", "natural": "数字排序.mp4"}


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX×](\d+)", value)
    if match:
        dims = tuple(map(int, match.groups()))
        if all(n >= 2 and n % 2 == 0 for n in dims):
            return dims
    raise argparse.ArgumentTypeError("尺寸应为两个正偶数，如 1920x1080")


def parse_fps(value: str) -> Fraction:
    rate = ratio(value)
    if not 1 <= rate <= 240:
        raise argparse.ArgumentTypeError("帧率应在 1 到 240 之间，支持 30 或 60000/1001 等写法")
    return rate.limit_denominator(100000)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按首尾衔接、整段内容或文件名，将文件夹内的视频剪成一个 MP4。",
        epilog=f"smooth 比较首尾 {EDGE_SECONDS} 秒。content 比较整段综合画面特征，兼顾声音及接缝。"
        f"{EXACT_LIMIT} 段以内求相应评分下的最优解。",
    )
    parser.add_argument("folder", type=Path, nargs="?", default=Path(), help="素材文件夹，默认当前目录")
    parser.add_argument(
        "--order",
        choices=OUTPUT_NAMES,
        default="content",
        help="content 综合画面相似度，smooth 首尾衔接，"
        "natural 将文件名中的连续数字按数值比较。默认 %(default)s",
    )
    parser.add_argument(
        "--codec",
        "--video-codec",
        choices=("auto", *CODECS),
        default="auto",
        help="最终视频编码，auto 按可直接复制的总时长选择（默认）",
    )
    parser.add_argument(
        "--content-step",
        type=float,
        default=1.0,
        metavar="秒",
        help="content 模式的画面取样间隔，默认 1 秒，切镜处额外取样，0 表示逐帧提取视觉特征",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="输出 MP4 路径，默认保存到素材目录下的「合辑」子目录"
    )
    parser.add_argument("--dry-run", action="store_true", help="预览顺序、目标编码、逐段复制/转码计划及耗时")
    parser.add_argument("--size", type=parse_size, metavar="宽x高", help="手动指定画布，如 1920x1080")
    parser.add_argument(
        "--fps",
        type=parse_fps,
        metavar="帧率",
        help="指定固定输出帧率。默认保留复制片段的时间戳，转码片段按素材选择，最高 60",
    )
    parser.add_argument("--fetch-model", action="store_true", help="下载或校验视觉模型，输出路径并退出")
    parser.add_argument("--offline", action="store_true", help="仅使用已校验的本地模型")
    parser.add_argument(
        "--report", type=Path, metavar="JSON", help="将顺序、实际复制结果、取样数和耗时写入新 JSON 文件"
    )
    return parser


def inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, list[Path]]:
    if not math.isfinite(args.content_step) or args.content_step < 0:
        parser.error("--content-step 应为非负的有限秒数")
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"找不到素材文件夹：{folder}")
    for program in ("ffmpeg", "ffprobe"):
        if not shutil.which(program):
            parser.error(f"找不到 {program}，请安装 FFmpeg 并将其加入 PATH")
    output = (args.output or folder / "合辑" / OUTPUT_NAMES[args.order]).expanduser().absolute()
    if output.suffix.lower() != ".mp4":
        parser.error("输出文件请使用 .mp4 扩展名")
    if not args.dry_run and os.path.lexists(output):
        parser.error(f"输出文件已存在，请用 -o 指定新的文件名：{output}")
    paths = sorted(
        (
            p
            for p in folder.iterdir()
            if not p.name.startswith(".")
            and p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
            and p.resolve() != output.resolve()
        ),
        key=lambda p: p.name,
    )
    if args.order == "natural":
        paths.sort(
            key=lambda p: [
                int(s) if i % 2 else s.casefold() for i, s in enumerate(re.split(r"(\d+)", p.name))
            ]
        )
    if not paths:
        parser.error(f"目录中没有找到视频：{folder}")
    if args.report and (args.report.suffix.lower() != ".json" or os.path.lexists(args.report)):
        parser.error("--report 须指定新的 .json 文件")
    return output, paths


def main() -> None:
    timings = Timings()
    parser = argument_parser()
    args = parser.parse_args()
    if args.fetch_model:
        print(model_file(offline=args.offline))
        return
    output, paths = inputs(parser, args)
    timings.seconds["启动、扫描目录与环境准备"] = timings.elapsed()
    clips = []
    with timings.stage("读取媒体参数"):
        for path in paths:
            clip = probe(path)
            if clip is None:
                print(f"跳过没有视频轨道的文件 {path.name}")
            else:
                clips.append(clip)
    if not clips:
        parser.error(f"目录中没有找到视频轨道：{args.folder}")
    fps = args.fps or common_fps(clips)
    samples = {}
    notes = {}
    if args.order == "smooth" and len(clips) > 1:
        clips = smooth_order(clips, timings.stage)
    elif args.order == "content":
        clips, notes, samples = content_order(clips, args.content_step, timings.stage, offline=args.offline)
    print(f"排序完成，自启动起 {timings.elapsed():.3f} 秒。", flush=True)
    print(f"共 {len(clips)} 段。播放顺序")
    for i, clip in enumerate(clips, 1):
        note = f"  [{notes[clip.path]}]" if clip.path in notes else ""
        print(f"  {i:>2}. {clip.path.name}{note}")
    with timings.stage("检查编码、画幅、色彩及接缝并选择导出规格"):
        plan = build_plan(clips, args.codec, args.size, args.fps, fallback_size=canvas(clips), fps=fps)
    plan.describe(clips)
    copied = None
    if not args.dry_run:
        copied = render(clips, output, plan, timings.stage)
        print(f"已生成 {output}")
    timings.report()
    if args.report:
        write_report(args.report, args.order, clips, plan, copied=copied, samples=samples, timings=timings)


def write_report(
    path: Path,
    order: str,
    clips: list[Clip],
    plan: Plan,
    *,
    copied: set[Path] | None,
    samples: dict[Path, int],
    timings: Timings,
) -> None:
    target = plan.target
    payload = {
        "version": 1,
        "order": order,
        "target": {
            "codec": target.codec,
            "size": target.size,
            "rotation": target.rotation,
            "transfer": target.color.transfer,
            "depth": target.color.depth,
            "transcode_fps": str(plan.fps),
        },
        "copyable_seconds": plan.scores,
        "clips": [
            {
                "name": c.path.name,
                "duration": c.duration,
                "video_seconds": c.video_duration,
                "samples": samples.get(c.path, 0),
                "planned_copy": not plan.reasons[c.path],
                "reason": plan.reasons[c.path],
                "copied": None if copied is None else c.path in copied,
            }
            for c in clips
        ],
        "stage_seconds": timings.seconds,
        "total_seconds": timings.elapsed(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def entrypoint() -> None:
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"错误：{exc}")
    except KeyboardInterrupt:
        sys.exit("\n已取消剪辑。")
