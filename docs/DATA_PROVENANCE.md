# 最终 DACE / ACCR 的实际数据来源

本说明以训练程序、保存的训练 manifest、实际 CSV 和发布权重 SHA256 为依据。统计脚本只输出聚合数量与文件哈希。

## 两个方法对应同一次训练

最终两个方法都使用 `abc_qwen2audio_directhead_fullr16_kd040_twoepoch_valonly_20260722` 训练轨迹：primary 来自 `checkpoint-700`，secondary 来自 `checkpoint-1400`。两个 LoRA 和两个分类头逐文件与 ModelScope 发布版本一致。DACE 和 ACCR 的差异在验证集冻结的读出和路由参数。

训练输入为 `train_timesplit.csv`，共 22,362 条；真实读取路径由每行 `source_audio_path` 决定。

| 原始批次 | 原始记录数 | 三分类有效数 | 用于梯度训练 | 用于验证 | 用于测试 |
|---|---:|---:|---:|---:|---:|
| `audio_1w5` / `data_1w5.csv` | 15,934 | 15,930 | 15,930 | 0 | 0 |
| `audio_1w2` / `data_1w2.csv` | 12,737 | 12,653 | 6,432 | 3,196 | 3,025 |
| 合计 | 28,671 | 28,583 | 22,362 | 3,196 | 3,025 |

`data_1w5.csv` 排除 4 条“开心”；`data_1w2.csv` 排除 84 条 `happy`。当时预处理记录的音频缺失数、重复文件名数均为 0。

## audio 目录的含义

`data/prepare_timesplit_abc_data.py` 从两批原始目录筛选有效样本，调用 `link_or_copy` 将其组织为 `audio`。原预处理摘要记录创建了 28,583 个硬链接。当前工作目录快照中的文件 inode 已各自独立，因此不能把当前状态称为仍然共享 inode 的硬链接。

本次对所有 28,583 对音频做 SHA256 比较，内容全部相同。`audio` 因而是合并镜像，没有带来第三批新增样本。训练表中直接指向 `audio` 的路径为 0 条，直接指向 `audio_1w5` 的有 15,930 条，直接指向 `audio_1w2` 的有 6,432 条。

## 时间划分

| 划分 | 时间 | 样本数 | 对话数 | A | B | C |
|---|---|---:|---:|---:|---:|---:|
| 训练 | 2024-07 至 2025-05 | 22,362 | 11,607 | 10,994 | 8,576 | 2,792 |
| 验证 | 2025-06 | 3,196 | 1,446 | 1,408 | 1,447 | 341 |
| 测试 | 2025-07 | 3,025 | 1,442 | 1,302 | 1,340 | 383 |

三组之间的 `dialogue_key` 交集均为 0。验证数据用于校准、路由和平滑参数选择；测试数据用于冻结后的评估。验证/测试样本不能计入 22,362 条梯度训练样本。

早期 `train.csv`（9,196 条）、`val.csv`（300 条）、`test.csv`（500 条）属于旧实验划分，不是最终训练命令的输入。

## 教师软标签的真实使用方式

训练使用离线 MVEF 教师文件：

```text
eval_outputs/abc_c25_rootclosed4_replay_step9_c25_final_trainproxy_valonly_20260709/train_oof_predictions.csv
```

文件有 22,200 条训练集 OOF 三类概率，SHA256 为 `3f17d2ad94f1ee75fcb17a761539dd799d04e257c658e51a672fe47f820777b6`。

实际训练代码在 `read_teacher` 对齐后，会将教师 argmax 与人工标签不一致的样本的 `teacher_weight` 设为 0。保存的训练 manifest 与本次独立复算均为：

- 19,277 条：人工标签交叉熵与软标签 KL 蒸馏，`alpha=0.4`、`T=2.0`。
- 2,923 条：教师概率存在但 argmax 错误，仅人工标签交叉熵。
- 162 条：教师概率缺失，仅人工标签交叉熵。

所有 22,362 条都参与训练，但实际有非零蒸馏权重的为 19,277 条。原项目 `PROJECT_INNOVATIONS.md` 以及 ModelScope 文档中关于“保留教师预测错误样本进行蒸馏”的描述与实际代码不符；公开代码仓库采用训练记录口径。

这里的“数据集”指项目微调、蒸馏与评估使用的私有中国移动通话数据。Qwen2-Audio 上游预训练语料不由本项目控制，也不能据本目录完整追溯。

## 复核

在拥有原项目数据的环境中运行：

```bash
python scripts/audit_training_data.py \
  --project-root /path/to/original/ChinaMobile \
  --output work/training_data_audit.json --compare-audio
```

聚合证据见 [training_data_audit.json](../benchmarks/training_data_audit.json)；完整音频镜像核查见 [audio_mirror_audit.json](../benchmarks/audio_mirror_audit.json)。这两个文件不包含通话标识、转写内容或逐条标签。
