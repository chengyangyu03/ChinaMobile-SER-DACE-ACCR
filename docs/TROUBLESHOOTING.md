# 常见问题

## CUDA out of memory

先把 batch size 从 2 改为 1：

```bash
bash scripts/run_accr.sh 0 work/input/manifest.json work/output 1 16
```

确保指定 GPU 没有其他大进程，不要在同一张卡上并发启动 DACE 和 ACCR。完整
3,025 条实测中，batch size 1 的 `nvidia-smi` 整进程峰值约 7.11 GiB，
建议使用至少 12 GB 显存；batch size 56 的单卡高吞吐峰值约 22.50 GiB，
三卡实验的单卡最高峰值约 22.84 GiB，只适合 24 GB 卡。更多口径说明见
[`INFERENCE_EFFICIENCY.md`](INFERENCE_EFFICIENCY.md)。

## 找不到基座

运行：

```bash
python scripts/download_base_model.py
python scripts/verify_install.py
```

基座必须位于 `models/base/Qwen2-Audio-7B-Instruct`。不要只下载配置文件；需要完整五个 safetensors 分片。

## ModelScope 命令返回成功但 SHA256 失败

网络中断时，ModelScope 客户端可能保留 `.incomplete` 文件但返回 0。对项目仓库直接重新执行原下载命令，客户端会续传缺失部分：

```bash
modelscope download chengyangyu03/ChinaMobile-SER-DACE-ACCR \
  --local_dir ChinaMobile-SER-DACE-ACCR
cd ChinaMobile-SER-DACE-ACCR
sha256sum -c SHA256SUMS
```

不要忽略 SHA256 失败。`download_base_model.py` 已内置最多三次续传与五个权重分片精确大小检查。

## 音频找不到

`wav_name` 必须与音频文件夹内文件名一致。当前输入准备脚本按文件名查找，不递归扫描子目录。若音频分散在多个目录，请先建立平铺目录或符号链接目录。

## ACCR 平均前向不是 1.3954

正常。1.3954 是私有测试集上的平均值。新数据只要 primary 校准置信度分布改变，路由比例就会改变。检查 summary 的 `router`、`called_second`、`call_rate`。

## 小批量结果与完整对话结果不同

冻结后处理包含同一通话同一说话方的 label-free 概率平滑。若把一个对话拆成多次独立任务，跨批次片段不会彼此平滑。建议一次提交完整对话集合；这里的“batch size”只是 GPU 小批大小，不会切断运行内的后处理。

## 标签无法识别

支持 A/B/C、中性/厌恶/愤怒以及 neutral/disgust/disgusted/anger/angry。若 label 列存在，必须每行都有合法值；纯预测时请移除 label 列。

## 输出目录已存在

脚本故意拒绝覆盖。默认时间戳精确到秒；等待一秒重试，或给不同的 `WORK_DIR`。不要手动复用旧输出目录。
