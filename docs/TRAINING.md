# 训练与校准复现

最终学生训练脚本与依赖保持原始内容，见 `training/`。以下入口只去除了原服务器的绝对路径和 GPU 调度脚本，模型训练逻辑没有重写。超参数与聚合训练记录见 [training_manifest.json](../benchmarks/training_manifest.json)。

## 数据预处理

需要自备原项目两批私有数据，目录布局为：

```text
private_data/
  audio_1w2/
  audio_1w5/
  data_1w2.csv
  data_1w5.csv
  train_oof_predictions.csv
```

`data_1w2.csv` 列为 `wav_path,Emotion,text`；`data_1w5.csv` 列为 `wav_name,label,ASR`。使用三分类标签；文件名中包含 `对话ID_14位年月日时分秒_片段标识.wav`。

```bash
python -m pip install -r requirements-training.txt
python data/prepare_timesplit_abc_data.py --root /absolute/path/to/private_data
```

脚本生成 `train_timesplit.csv`、`val_202506.csv`、`test_202507.csv` 与 `audio` 合并目录。训练 CSV 字段为：

```text
wav_name,label,ASR,timestamp,month,dialogue_key,segment,source,source_csv,source_audio_path
```

最终训练器直接读取 `source_audio_path` 指向的原始批次目录。音频转为单声道、16 kHz，并截取开头最多 16 秒；相邻上下文来自同一对话同侧的前一段与后一段 ASR，每段最多 320 个字符。

## 训练学生

教师文件最少包含 `wav_name,prob_A,prob_B,prob_C`。概率必须来自训练集 OOF 预测，不能混入验证或测试样本。缺失教师的样本继续参与人工标签训练；教师预测错误的样本关闭 KD。原项目的教师 OOF 文件及全部教师检查点未公开。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash scripts/train_student.sh \
  "$PWD/models/base/Qwen2-Audio-7B-Instruct" \
  /absolute/path/to/private_data/train_timesplit.csv \
  /absolute/path/to/private_data/val_202506.csv \
  /absolute/path/to/private_data/train_oof_predictions.csv \
  "$PWD/work/student"
```

原训练配置：4 GPU，每卡 batch 2，梯度累积 4，有效 batch 32，1,400 个优化步；AdamW 学习率 `7.5e-5`，betas `(0.9,0.95)`，weight decay `0.01`，梯度裁剪 `1.0`。LoRA rank 16、alpha 32、dropout 0.05；四池化分类头每块投影宽度 64；种子 `20260722`。训练数据和教师路径含 `test` 时会被拒绝，历史训练月份不能晚于 2025-05。

保存步数为 350、700、1050、1400；最终发布 primary 为 700，secondary 为 1400。更改 GPU 数会改变全局 batch 和样本顺序，不应声称仍然逐位复现原训练权重。

## 导出验证概率

```bash
for STEP in 700 1400; do
  CUDA_VISIBLE_DEVICES=0 python training/eval_qwen2audio_directhead_components_valonly_abc.py \
    --model models/base/Qwen2-Audio-7B-Instruct \
    --adapter "work/student/checkpoint-$STEP" \
    --val-csv /absolute/path/to/private_data/val_202506.csv \
    --output-dir "work/validation_step$STEP" \
    --batch-size 2 --max-audio-seconds 16 --include-context
done
```

该评估器导出总三分类概率及分类头的分量 logits；最终 DACE/ACCR 使用总概率。

## DACE 校准与 ACCR 路由

两个选择器在 2025-06 验证集上按完整 `dialogue_key` 执行外 4 折 / 内 3 折选择。DACE 的历史参数名 `--omni` 表示 primary，`--qwen2audio` 表示 secondary；这两个实际输入都是 Qwen2-Audio 检查点。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python training/select_qwen2audio_omni_nested_jointlogistic_valonly_abc.py \
  --omni work/validation_step700/component_predictions.csv \
  --qwen2audio work/validation_step1400/component_predictions.csv \
  --meta /absolute/path/to/private_data/val_202506.csv \
  --output-dir work/dace_selection

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python training/select_qwen2audio_budget_cascade_nested_valonly_abc.py \
  --primary work/validation_step700/component_predictions.csv \
  --secondary work/validation_step1400/component_predictions.csv \
  --meta /absolute/path/to/private_data/val_202506.csv \
  --output-dir work/accr_selection

python scripts/export_frozen_parameters.py --method dace \
  --selection work/dace_selection/selection.json \
  --output work/frozen/dace_frozen_parameters.json
python scripts/export_frozen_parameters.py --method accr \
  --selection work/accr_selection/selection.json \
  --output work/frozen/accr_frozen_parameters.json
```

导出器仅规范 ACCR 的部署协议名称，并检查选择记录未使用测试数据；它不重新训练或修改数值参数。原始发布参数已放在 `models/dace/` 与 `models/accr/`，Shell 推理入口默认使用这些固定参数。自己的再训练结果应使用自己的校准器，通过 `src/dace_inference.py` 或 `src/accr_inference.py` 的 `--selection`、`--expected-selection-sha256`、`--primary-adapter`、`--secondary-adapter` 指定。其余输入参数可用 `--help` 查看。

所有训练与选择输出目录都应使用新目录。脚本拒绝覆盖已有结果，以保留复核记录。

## 复现范围

本仓库可以独立运行已发布模型，也提供最终学生训练、验证校准与路由代码。精确重建训练权重仍需要原始私有音频、ASR、标签、MVEF OOF 概率、同一基座与相同训练环境；GPU 运算也可能存在非确定性。上游 MVEF 历次训练、OOF 生成和规则实验未作为最终部署代码打包，不能据此宣称已公开完整教师训练系统。
