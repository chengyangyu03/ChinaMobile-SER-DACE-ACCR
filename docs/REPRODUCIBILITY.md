# 复现边界、冻结参数与哈希

## 数据边界

发布仓库不包含任何训练、验证、测试音频或 ASR 文本。数据按月份严格顺序切分：

| 划分 | 时间范围 | 样本数 | 对话数 | A：中性 | B：厌恶 | C：愤怒 |
|---|---|---:|---:|---:|---:|---:|
| Train | 2024-07 至 2025-05 | 22,362 | 11,607 | 10,994（49.1638%） | 8,576（38.3508%） | 2,792（12.4855%） |
| Validation | 2025-06 | 3,196 | 1,446 | 1,408（44.0551%） | 1,447（45.2753%） | 341（10.6696%） |
| Private test | 2025-07 | 3,025 | 1,442 | 1,302（43.0413%） | 1,340（44.2975%） | 383（12.6612%） |

私有测试集为 2025 年 7 月通话数据，共 3,025 条。公开准确率仅来自该集合；第三方只能在自己的数据上重新评估，不能仅凭仓库文件复算该准确率。各划分的首尾时间戳、对话数、类别分布和源表 SHA256 见 [`benchmarks/data_split_protocol.json`](../benchmarks/data_split_protocol.json)。

两个 adapter 的训练轨迹在访问私有测试集之前完成。DACE 的联合校准和平滑参数、ACCR 的 primary/joint 校准、路由阈值和平滑参数均来自按完整对话分组的外 4 折/内 3 折嵌套验证。

ACCR 的测试集成本–精度结果是在完整 DACE 双前向结果之后透明报告的 Pareto 派生结果；冻结路由规则本身仍只来自验证集。`models/accr/accr_frozen_parameters.json` 中保留了嵌套验证审计。

## 发布文件

```text
models/
├── shared/
│   ├── primary_adapter/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── classifier_head.pt
│   └── secondary_adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── classifier_head.pt
├── dace/dace_frozen_parameters.json
└── accr/accr_frozen_parameters.json
```

DACE 和 ACCR 使用完全相同的两个训练权重，仅冻结读出与路由方式不同，因而不重复存储 adapter。

## 完整性校验

GitHub 根目录的 `SHA256SUMS` 覆盖发布代码、文档、配置与冻结参数；下载的训练权重由 `MODEL_ARTIFACTS.json` 单独固定精确字节数和 SHA256。`BASE_MODEL_SHA256SUMS` 固定官方 Qwen2-Audio `master` 快照的五个权重分片、索引、配置、处理器与分词器。运行：

```bash
sha256sum -c SHA256SUMS
python scripts/download_project_models.py
python scripts/download_project_models.py --check-only
python scripts/verify_install.py --skip-base-model
```

下载基座后，`verify_install.py` 会检查五个分片的精确字节数、逐文件 SHA256 和 `model_type=qwen2_audio`。若官方 `master` 后续发生漂移，校验会明确失败而不会静默使用不同基座。每次推理 summary 会再次记录：

- 输入表与 manifest SHA256；
- 两个 adapter 与两个分类头 SHA256；
- 冻结参数 SHA256；
- 基座 config/index SHA256；
- 最终预测 CSV SHA256。

这使第三方能够区分“同一方法的不同新数据结果”和“权重/参数被更改后的结果”。

`verify_install.py` 还会检查无显式片段序号时的 prompt 回退契约必须为 `side=unknown, seg0`。该约定与训练及原始评测一致，避免 CSV 行号意外进入模型输入。

## 推理效率记录

低显存、单卡高吞吐和三卡高吞吐实验均使用完整 3,025 条私有测试数据与全新
Python 进程。三卡实验使用空闲 RTX 3090 GPU 0/3/6，按完整对话分片并在结束
时恢复原始输入顺序。全进程计时从入口启动到退出，包含解释器导入、输入与参数
校验、模型加载、全部前向、冻结后处理和结果写盘；三卡时间还包含分片、逐行
集合校验与合并，没有把模型加载时间排除。共享主机上没有强制清空操作系统页
缓存，基座权重在实验前已位于本地磁盘。

结构化结果见
[`benchmarks/inference_efficiency_3025.json`](../benchmarks/inference_efficiency_3025.json)，
每 0.5 秒的 `nvidia-smi` 原始轨迹见
[`benchmarks/inference_efficiency_3025/`](../benchmarks/inference_efficiency_3025/)。
这些记录不含私有音频、ASR、标签或逐条预测。实验协议与结果解释见
[`docs/INFERENCE_EFFICIENCY.md`](INFERENCE_EFFICIENCY.md)。

## 训练代码与实际数据来源

GitHub 补充最终训练、验证选择与审计代码，见 [TRAINING.md](TRAINING.md) 和 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)。训练记录中包含教师预测一致性过滤，19,277 条参与 KD，3,085 条仅 CE；本说明按原训练 manifest 与代码核定。
