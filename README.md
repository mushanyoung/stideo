# Stideo

Stideo 取自 *stitch video*。它把一个目录里的视频排成一部 MP4。默认比较整段画面的内容、运动和声音，把相近的片段放在一起。导出时优先复用原始视频压缩数据，保留可复制部分的画质和帧时间戳。

## 从克隆到第一部成片

需要 Git、uv，以及在 PATH 中可用的 FFmpeg 和 ffprobe。macOS 可以运行 `brew install uv ffmpeg`。Ubuntu 24.04 可以用 `sudo apt install ffmpeg` 安装媒体工具。uv 会获取 Python 3.15 并按仓库锁文件准备依赖。持续集成覆盖 Ubuntu 24.04 和 macOS 15。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```sh
git clone https://github.com/mushanyoung/stideo.git
cd stideo
uv sync --locked --no-dev
uv run --locked --no-dev stideo "/path/to/clips" --dry-run
uv run --locked --no-dev stideo "/path/to/clips" -o "/path/to/movie.mp4"
```

第一次使用 `content` 会下载所选的视觉模型。预览会执行相同的排序和导出规划，并列出每段复制或转码的原因。省略 `-o` 时，成片写入素材目录的 `合辑/` 子目录。输出须使用新的文件名。环境准备好后，在其他目录也可以直接调用本仓库的 `.venv/bin/stideo`。

## 怎样决定播放顺序

| 选项 | 适合的素材 | 排序依据 |
| --- | --- | --- |
| `--order content`（默认） | 同一批素材的拍摄顺序未知 | 整段取样，比较视觉模型特征、色彩、构图、纹理、光流、声音和接缝 |
| `--order smooth` | 希望上一段结尾自然接到下一段开头 | 比较相邻两段首尾各 0.75 秒的画面、运动和声音 |
| `--order natural` | 文件名已表达剪辑顺序 | 数字按数值比较，例如 `clip2` 在 `clip10` 前 |

`content` 默认每 0.5 秒取一个画面，约每秒 2 帧，遇到切镜会额外取样。短暂插入其他镜头时，半秒间隔能更好地抓到插镜前后的主体。长素材需要优先节省分析时间时可用 `--content-step 1`，`--content-step 0` 会覆盖每帧。相似度会考虑整段中不同时段的匹配。16 段以内求解当前评分下的最小总差异，更多片段使用多起点搜索和局部调整。`natural` 和 `smooth` 均可在空模型缓存、离线的环境中运行。

## 编码、画幅与耗时

默认 `--codec auto` 在 H.264、HEVC 和 AV1 中选择可直接复制总时长最多的编码、显示画幅、旋转和色彩组合。检查也涵盖像素比例、位深、HDR 传递函数、关键帧、隐藏参考帧和时间戳。需要转换的片段会适配画布并补边。HDR/SDR 转换使用 FFmpeg 的色彩变换，旧版 FFmpeg 使用 zscale 和 tonemap。转码需要本机 FFmpeg 有对应的视频编码器。

```sh
uv run stideo clips --order natural --codec h264 -o delivery.mp4
uv run stideo clips --codec hevc --size 1080x1920 --fps 30000/1001 -o portrait.mp4
uv run stideo clips --codec av1 --report run.json -o archive.mp4
```

音频会对齐成连续的 48 kHz 双声道 AAC，无声片段补静音。封装完成后会解码检查接缝和随机定位，必要时对相关片段转码重试。每次运行都输出各阶段耗时。`--report run.json` 额外记录播放顺序、取样间隔（`content_step`，单位为秒）、取样数、可复制秒数、计划和实际复制结果。预览的实际复制字段为 `null`。

## 模型缓存和离线使用

```sh
uv run stideo --fetch-model
uv run stideo --offline --fetch-model
uv run stideo clips --model mobilevit-xxs --dry-run
uv run stideo clips --offline --dry-run
```

模型权重来自固定修订版，以 SHA-256 校验。默认缓存在 `${XDG_CACHE_HOME:-~/.cache}/stitch-videos/models/`，每种模型使用单独的文件。原有的 `dinov2-small-int8.onnx` 可以直接复用。从原脚本迁移、重新克隆或新建虚拟环境都能复用同一文件。可以用 `STIDEO_MODEL_DIR` 指定模型目录，或把已经校验的同名文件拷贝到离线机器。并发首次运行只下载一次，损坏的缓存会在联网时重新获取。设置 `STIDEO_OFFLINE=1` 可以对所有运行启用离线模型检查。[固定模型版本](https://huggingface.co/Xenova/dinov2-small/tree/c2bb04a51fab207c420665f1946016107bffc701)

## 开发时启用 Ruff

运行一次 `git config --local core.hooksPath .githooks`。每次提交都会检查暂存快照，包括部分暂存的文件。Ruff 启用全部稳定规则，语言标点、命令行输出等例外列在 `pyproject.toml` 中。GitHub Actions 对推送和 PR 运行同样的 lint 与格式检查。

## 合成数据与回归测试

```sh
uv sync --locked
uv run stideo --fetch-model
uv run stideo --fetch-model --model dinov2-small-int8
uv run stideo --fetch-model --model mobilevit-xxs
uv run --locked ruff check .
uv run --locked ruff format --check .
STIDEO_OFFLINE=1 uv run --locked pytest -q
uv run python -m tests.datasets /tmp/stideo-fixtures
```

测试从图形、像素和正弦波生成真实短视频，调用正式 CLI、模型和 FFmpeg。生成器可以使用 `--only scenes bridges` 只生成指定数据集。完整套件需要带 libx264、libx265、libsvtav1 和 AAC 的 FFmpeg。生成的素材和成片都写在临时目录。

| 数据集 | 能检出的变化 |
| --- | --- |
| `scenes` | 六段共用片头片尾。火车、松树、灯塔各有两种配色。测试要求三对都相邻，允许互换组序和方向，并核对成片的实际播放次序 |
| `bridges` | 四个指定方向的颜色接缝，文件名故意打乱。检验首尾排序和数字排序的不同答案 |
| `sounds`、`motion` | 相同画面中的不同音调，或同一纹理的相反运动。共同片头片尾和交叉编号让排序依赖片段内部 |
| `cuts` | 常规采样之间的短暂切镜，检查稀疏模式的额外取样和逐帧模式的覆盖 |
| `codecs` | 少数长 H.264、多数短 HEVC、AV1、不同 B 帧和有声/静音交错。对四种编码选择核对复制秒数、逐包哈希、显示时间、逐帧误差和声音频率 |
| `geometry`、`ranges` | 元数据旋转、物理竖屏、分辨率、强制画布、PQ 10 位和 SDR。同一组灰阶色块检验转换后的像素值与色彩标签 |
| `timing` | 带隐藏参考帧的剪切、可变帧率、比画面长的音轨。检查直接复制时的时间戳及强制帧率后的时间网格 |

## 比较模型或算法的效率

```sh
uv run python -m tests.benchmark --output /tmp/stideo-before.json
# 在同一台机器上切换算法或模型，再运行
uv run python -m tests.benchmark --baseline /tmp/stideo-before.json --output /tmp/stideo-after.json
```

基准覆盖三种排序和四种编码策略。默认每项完整导出三次，取进程总耗时与关键阶段的中位数，另记录模型取样数和实际复制秒数。初次模型下载和素材生成在计时之外。默认耗时阈值取基线的 50% 与 0.35 秒中的较大值，复制时长减少也会返回失败。可以用 `--repeats` 和 `--tolerance` 调整。PR 的基线和候选版本在同一个 Linux runner 上处理同一批素材，JSON 可从 Actions 下载。合成场景用于发现可重复的退化，真实拍摄素材的主观连贯性仍适合单独评估。

若两个版本的默认取样间隔不同，基准会让旧版按新间隔补跑，比较相同取样设置下的视觉特征提取耗时。端到端总耗时始终比较各自的实际默认设置。

另有 138 段用于比较模型与取样间隔的短视频，涵盖跨视角物体、场景、插入镜头、花卉实拍照片和速度测试。在 M4 留出集上，默认模型 DINOv2 Small INT8 的同类接缝达成率从 1 秒取样时的 74.1% 升至半秒取样时的 83.5%。速度优先可用 `--model mobilevit-xxs`。在相同的 1 秒间隔下，它把三段素材的排序时间从 1.06 秒降到 0.52 秒，同类接缝达成率为 66.1%。[完整实测与复现方法](docs/model-evaluation.md)
