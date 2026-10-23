# ChinaMobile SER：DACE 与 ACCR

中文客服通话三分类情绪识别的核心代码，包含数据预处理、学生训练、验证集校准与路由选择、单卡/多卡推理及训练数据来源审计。

模型权重：[ModelScope · chengyangyu03/ChinaMobile-SER-DACE-ACCR](https://modelscope.cn/models/chengyangyu03/ChinaMobile-SER-DACE-ACCR)。

标签为 **A=中性、B=厌恶、C=愤怒**。输入为原始音频与 ASR 转写，并使用同一通话同一说话方的相邻 ASR 上下文。当前流程适用于具有完整通话上下文的离线分析。

| 方法 | 2025-07 私有测试准确率 | Macro-F1 | 每条大模型前向 |
|---|---:|---:|---:|
| DACE：双适配器校准集成 | 80.2314%（2427/3025） | 76.7331% | 2.0000 |
| ACCR：自适应置信度校准路由 | 80.0661%（2422/3025） | 76.4714% | 平均 1.3954 |

两种方法共享一个 4-bit Qwen2-Audio-7B-Instruct 基座，以及同一训练轨迹的 step 700 / step 1400 两组 LoRA 和四池化分类头。DACE 总是运行两组适配器，ACCR 仅在校准置信度不高于冻结阈值时调用第二组。推理不运行 MVEF 教师。

## 安装与推理

GitHub 保存代码、冻结参数和聚合实验指标。约 424 MB 的项目权重从 ModelScope 下载，并逐文件验证固定 SHA256；官方 7B 基座另行下载。

```bash
git clone https://github.com/chengyangyu03/ChinaMobile-SER-DACE-ACCR.git
cd ChinaMobile-SER-DACE-ACCR

# 仅需 Python 标准库；下载两个 LoRA 和两个分类头
python scripts/download_project_models.py

bash scripts/create_env.sh chinamobile-ser-dace-accr
conda activate chinamobile-ser-dace-accr
python scripts/download_base_model.py
python scripts/verify_install.py

# 参数：方法、GPU、音频目录、ASR表、输出目录、batch size、音频上限秒数
bash scripts/run_pipeline.sh dace 0 /path/to/audio /path/to/asr.xlsx work/dace 2 16
bash scripts/run_pipeline.sh accr 0 /path/to/audio /path/to/asr.xlsx work/accr 2 16

# 多 GPU：按完整对话分片，最后恢复原始顺序
bash scripts/run_multigpu_pipeline.sh accr 0,1,2 \
  /path/to/audio /path/to/asr.xlsx work/accr_multi 2 16
```

最小 ASR CSV（示例为人工构造）：

```csv
wav_name,ASR
call001_20260724090000_right_1.wav,您好我想咨询一下业务
call001_20260724090000_right_2.wav,为什么一直没有处理
```

可选 `label` 列仅用于运行结束后的计分。推荐文件名为 `对话ID_left/right_序号.wav`；也可显式提供 `dialogue_key`、`segment`。每次运行创建独立时间戳目录，输出预测 CSV 和包含输入/模型哈希、耗时、显存与路由率的 summary JSON。

完整 3,025 条测试中，batch size 1 的整进程显存峰值约 7.11 GiB，建议 12 GB 或更大显存；24 GB 显卡可使用较大 batch。环境与效率记录见 [推理效率](docs/INFERENCE_EFFICIENCY.md)。

## 实际训练数据

**DACE 和 ACCR 共用同一套训练数据，不是分别训练的两个基座。**

| 原始音频目录 | 训练 | 验证 | 测试 | 有效样本合计 |
|---|---:|---:|---:|---:|
| `audio_1w5` + `data_1w5.csv` | 15,930 | 0 | 0 | 15,930 |
| `audio_1w2` + `data_1w2.csv` | 6,432 | 3,196 | 3,025 | 12,653 |
| 合计 | **22,362** | **3,196** | **3,025** | **28,583** |

`audio` 是两批有效音频的合并镜像，不是第三份独立数据集。最终训练表中的 `source_audio_path` 直接指向 `audio_1w5` / `audio_1w2`。对三个目录共 28,583 对文件的 SHA256 核查表明，合并目录内容与对应原文件完全相同。

训练月份为 2024-07 至 2025-05；验证为 2025-06；测试为 2025-07。原始两批共 28,671 条，排除三分类之外的 88 条开心标签后得到 28,583 条。私有音频、ASR、逐条标注和 OOF 概率不随代码公开。

训练同时使用人工硬标签和 MVEF 训练集 OOF 概率。**实际实现包含教师与人工标签一致性过滤**：19,277 条使用 CE + KD，2,923 条教师预测错误样本以及 162 条无教师样本仅使用 CE。ModelScope 原说明中“22,200 条均参与蒸馏”的概述不够准确；本仓库按训练代码与保存的 manifest 说明。

可核查的聚合统计、表哈希及权重对应关系见 [训练数据审计](benchmarks/training_data_audit.json) 和 [数据来源说明](docs/DATA_PROVENANCE.md)。

## 核心代码

| 路径 | 用途 |
|---|---|
| `data/prepare_timesplit_abc_data.py` | 合并两批数据、规范标签、按月份划分 |
| `training/train_qwen2audio_directhead_student_valonly_abc.py` | 四池化分类头、rank-16 LoRA、过滤后的 OOF 蒸馏 |
| `training/train_qwen2audio_mvef_student_valonly_abc.py` | 音频读取、上下文构建、教师概率匹配等公共函数 |
| `training/eval_qwen2audio_directhead_components_valonly_abc.py` | 导出检查点的验证集概率 |
| `training/select_qwen2audio_omni_nested_jointlogistic_valonly_abc.py` | DACE 双适配器嵌套校准与平滑选择 |
| `training/select_qwen2audio_budget_cascade_nested_valonly_abc.py` | ACCR 校准、路由、平滑的嵌套选择 |
| `src/ser_runtime.py` | 单基座双适配器推理、冻结校准和对话平滑 |
| `scripts/` | 下载、安装、训练入口、单卡/多卡推理、审计 |
| `models/dace/`、`models/accr/` | 与 ModelScope 相同的冻结参数 |

训练和校准的原始 Python 源文件保持原样，包含其本地模块依赖，文件哈希见 [source_provenance.json](benchmarks/source_provenance.json)。历史选择器的 `omni` / `qwen2audio` 参数名在 DACE 中分别表示 step 700 / step 1400，同属 Qwen2-Audio。验证选择 JSON 保留历史 gate 字段；这些字段不是重新申请测试权限的流程。

复现训练命令、私有输入格式与冻结参数导出见 [TRAINING.md](docs/TRAINING.md)。MVEF 是离线软教师输入；本仓库提供最终学生和 DACE/ACCR 的核心训练链，不包含历次教师实验的全部脚本与检查点。

## 文档与验证

- [方法](docs/METHODS.md)、[输入输出](docs/INPUT_OUTPUT.md)、[故障排查](docs/TROUBLESHOOTING.md)
- [训练](docs/TRAINING.md)、[数据来源](docs/DATA_PROVENANCE.md)、[复现边界](docs/REPRODUCIBILITY.md)
- [私有测试聚合指标](benchmarks/private_test_metrics.json)、[完整时间划分](benchmarks/data_split_protocol.json)

```bash
sha256sum -c SHA256SUMS
python scripts/download_project_models.py --check-only
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

公开准确率来自私有数据集，第三方需要自己的有标注数据进行评估。仅有公开代码和权重不能复算私有准确率，也不能在缺少原训练数据与教师 OOF 概率时重建同一组权重。

## 许可

沿用 ModelScope 发布包的 Apache-2.0 许可，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。官方 Qwen2-Audio 基座另须遵守其许可。
