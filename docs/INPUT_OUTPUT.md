# 输入与输出契约

## 输入表

支持 UTF-8/UTF-8-BOM CSV 与 XLSX。默认列名：

| 列 | 必需 | 说明 |
|---|---|---|
| `wav_name` | 是 | 音频文件名；在 `--audio-dir` 下查找 |
| `ASR` | 是 | 当前片段的 ASR 文本，可为空字符串 |
| `label` | 否 | 全部提供或整列省略；用于本地计分 |
| `dialogue_key` | 否 | 同一通话的稳定 ID |
| `segment` | 否 | 建议为 `left_1`、`right_3` 等 |

自定义列名可传给 `prepare_input.py`：

```bash
python scripts/prepare_input.py \
  --audio-dir /data/wav \
  --asr-file /data/asr.xlsx \
  --output-dir work/input \
  --wav-column filename \
  --asr-column transcript \
  --label-column emotion \
  --dialogue-column call_id \
  --segment-column turn
```

## 标签归一化

匹配英文时忽略大小写和首尾空白：

| 内部标签 | 支持输入 |
|---|---|
| A | `A`、`中性`、`neutral` |
| B | `B`、`厌恶`、`disgust`、`disgusted` |
| C | `C`、`愤怒`、`anger`、`angry` |

label 列若部分为空会直接报错，避免把缺失标签误当成 A。

## 对话信息自动恢复

推荐文件名：

```text
<任意对话ID>_<left|right>_<正整数>.<扩展名>
```

例如 `call001_20260724090000_right_3.wav` 会解析为：

- `dialogue_key=call001_20260724090000`
- `segment=right_3`

若文件名不符合该规则，程序把每个文件视为独立对话。此时模型仍可推理，但不会触发有效对话平滑；无法得到显式序号时，prompt 使用与训练协议一致的 `side=unknown, seg0`，不会把 CSV 全局行号写入 prompt。要复现完整方法，请在 ASR 表显式提供对话列，并优先把片段列写成 `left_1`、`right_3` 等带序号格式。

## 音频处理

- 使用 `soundfile` 读取；双声道自动取均值。
- 非 16 kHz 音频用 polyphase 方法重采样到 16 kHz。
- 默认只保留前 16 秒，与训练设置一致。
- 音频损坏、表中重名、文件缺失都会在加载模型前报错。

## 输出

预测 CSV 主要字段：

| 字段 | 说明 |
|---|---|
| `pred` / `pred_zh` | A/B/C 与中文预测 |
| `prob_A/B/C` | 校准与对话平滑后的最终概率 |
| `called_second` | 是否运行 secondary adapter；DACE 恒为 1 |
| `model_forwards` | 此条样本的物理大模型前向次数 |
| `confidence` | 最终三类概率最大值 |
| `correct` | 输入有标签时为 0/1，否则为空 |

summary 中的 `timing_seconds.total` 包含输入验证、模型加载、前向、冻结后处理和
文件写盘。`peak_cuda_memory_bytes` 是当前进程的 PyTorch 峰值**已分配**
tensor 显存，不含 CUDA 上下文、缓存分配器保留空间和其他 CUDA 库；选购显卡
应参考完整效率实验中由 `nvidia-smi` 采样的整进程峰值，而不是只参考这个字段。

多卡入口最终只输出一份按原始 CSV 顺序合并的预测文件。多卡 summary 以
`timing_seconds.total_to_summary_write` 记录入口总耗时，并分别记录 GPU 列表、
每片行数、每卡 PyTorch 峰值与各 shard summary 路径；隐藏工作目录保留每个
worker 的日志，便于排查单卡失败。
