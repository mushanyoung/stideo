# 比较画面排序模型

更换 `--model` 会改变 `content` 的视觉特征。所有候选共用整段取样、色彩、构图、纹理、光流、声音、接缝评分和路径求解器。编码策略保持独立。

## 重现实验

```sh
uv sync --locked
uv run python -m tests.model_datasets build/model-datasets --sets viewpoints locations cutaways flowers
uv run python -m tests.model_benchmark build/model-datasets --output build/model-report.json
```

生成器默认只制作程序绘制的素材。加入 `flowers` 会下载并校验约 218 MiB 的 TensorFlow 花卉照片归档。可以用 `--flower-archive PATH` 复用已有归档，输出目录里的 `FLOWER-CREDITS.txt` 保留原作者和许可信息。重复对同一目录运行可补齐其他组或另一套划分。评测器默认使用两个划分中都已生成的数据集。[原始数据集](https://www.tensorflow.org/datasets/catalog/tf_flowers) · [归档校验和](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv)

| 每个划分 | 片段数 | 标注表达的相似性 |
| --- | ---: | --- |
| `viewpoints` | 12 | 椅子、桌子、双翼飞机、拖船各三个视角，交叉混用颜色、照明和画幅。留出集使用另一套三维造型 |
| `locations` | 12 | 书店、站台、温室、港口。不同场景共享显眼的前景，取景位置和活动方向有变化 |
| `cutaways` | 12 | 按主体分组。共同的首尾和中间插入的干扰镜头让接缝相似度和多数画面的相似度给出不同线索 |
| `flowers` | 30 | 五种花，每类六张独立的实拍照片，各制成一段运动镜头。两个划分使用不同原图 |
| `speed` | 3 | 静帧停留、竖幅运动、密集硬切，专用于计时 |

开发集和留出集一共 138 段。每组的 `truth.json` 记录编号与标签，文件名经过固定种子的置乱。开发集用来挑选候选模型，最终留出集排除了前期试验用过的花卉照片。

## 质量与性能指标

- **mAP** 衡量同类片段在完整候选列表中排得多靠前。每段轮流作为查询，剔除自身，按类别、数据集分别等权平均。
- **Top-1** 衡量最相近的一段也属于同类的比例。距离完全相同时按随机打破并列的期望计算 mAP 和 Top-1。
- **同类接缝比例** 衡量实际播放路径里形成了多少个同类接缝。分母是这组素材中最多可形成的同类接缝数。组的次序和整条路径的方向都可自由变化。
- **毫秒/帧** 对同样的 36 个预解码帧计时，包括模型预处理、前向推理和视觉描述向量的提取。先预热，再取三轮平均单帧耗时的中位数。报告还记录单帧 P95。
- **排序耗时** 取三次完整排序的中位数，含 FFprobe、从缓存加载模型、采样解码、声音与首尾分析、求解。素材生成、首次下载和成片导出各自独立。每种模型使用新的 Python 进程，macOS 和 Linux 的峰值 RSS 仅统计该进程。

只比较指定模型，或在同一台机器上对比两次实验，可以使用下面的命令。参考报告要求相同的素材字节、CPU 和线程数。质量超出容差或明显变慢时，命令会先保存完整报告，再以非零状态退出。

```sh
uv run python -m tests.model_benchmark build/model-datasets --models dinov2-small-int8 mobilevit-xxs --output build/pair-before.json
uv run python -m tests.model_benchmark build/model-datasets --models dinov2-small-int8 mobilevit-xxs --reference build/pair-before.json --output build/pair-after.json
```

默认切换的门槛以原来的 `dinov2-small-int8` 为基线。要求总体 mAP 降幅不超过 2 个百分点，Top-1 和同类接缝降幅各不超过 3 个百分点，单个数据集的 mAP 降幅不超过 8 个百分点，排序耗时增加不超过 10%。在此基础上，还要满足单帧耗时减少至少 15%，或 mAP 增加至少 3 个百分点且单帧耗时增加不超过 10%。开发集选出一个候选，再在留出集复核。基准的质量标签反映上述四种连续性偏好，实际剪辑还取决于素材和观看者。

## 模型和权重

| 命令行名称 | 输入与模型输出 |
| --- | --- |
| `dinov2-small-int8`、`dinov2-small-fp32` | 同一个 DINOv2 Small，224 像素。使用全局和区域 token。[权重](https://huggingface.co/Xenova/dinov2-small) |
| `dinov2-base-int8` | DINOv2 Base，224 像素。更宽的视觉特征。[权重](https://huggingface.co/Xenova/dinov2-base) |
| `dino-s16-int8` | 第一代 DINO Small，224 像素，16 像素图块。[权重](https://huggingface.co/Xenova/dino-vits16) |
| `clip-b32-int8` | CLIP ViT-B/32，224 像素。使用对比训练的图像投影，局部构图由排序器的区域色彩等特征提供。[权重](https://huggingface.co/Xenova/clip-vit-base-patch32) |
| `siglip2-b32-int8` | SigLIP 2 Base，256 像素，一次推理同时读取池化向量和局部图块。[权重](https://huggingface.co/onnx-community/siglip2-base-patch32-256-ONNX) |
| `mobilevit-xxs` | MobileViT XX-Small，256 像素。按模型约定输入 BGR，取分类头之前的空间特征图。[权重](https://huggingface.co/Xenova/mobilevit-xx-small) |

所有模型只下载视觉部分并在本地 CPU 运行。仓库按权重分别固定版本与 SHA-256。DINOv3 Small 和 ConvNeXt Tiny 的 ONNX Community 导出在本次 OpenCV 5.1 CPU 环境中遇到了不支持的序列算子，因此没有纳入计分。[DINOv3 Small 导出](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX)
