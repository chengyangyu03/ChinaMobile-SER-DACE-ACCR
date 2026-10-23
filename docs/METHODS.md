# DACE 与 ACCR 方法说明

## 1. 共享主干

DACE 和 ACCR 都使用一个 4-bit `Qwen2-Audio-7B-Instruct` 基座。项目训练权重由同一条训练轨迹的两个 checkpoint 组成：

| 发布名称 | 训练轨迹位置 | LoRA | 直接分类头 |
|---|---:|---|---|
| primary adapter | step 700 | rank 16，alpha 32 | 四池化三分类头 |
| secondary adapter | step 1400 | rank 16，alpha 32 | 四池化三分类头 |

LoRA 覆盖语言/音频注意力、FFN 和多模态投影的 `q/k/v/o/out_proj`、`gate/up/down_proj`、`fc1/fc2`、`linear`。四池化头读取：

1. 回答位置 hidden state；
2. 音频 token hidden-state 均值；
3. 音频 token hidden-state 标准差；
4. 非音频文本 token hidden-state 均值。

训练 prompt 同时提供当前原始音频、当前 ASR、同一对话同一说话方的前一段和后一段 ASR。训练目标是硬标签交叉熵与 MVEF 训练集 OOF 软概率蒸馏的组合；22,362 条训练样本中 22,200 条可匹配 OOF 软标签；实际训练经过教师 argmax 与人工标签一致性过滤，只有 19,277 条使用非零 KD，2,923 条教师预测错误样本和 162 条无教师样本仅使用真实标签交叉熵。蒸馏权重为 0.4，温度为 2.0。MVEF 只在训练阶段提供软教师，不参与 DACE/ACCR 推理。

## 2. DACE

**DACE = Dual-Adapter Calibrated Ensemble**。

每条样本依次通过 primary 和 secondary adapter，各执行一次完整 Qwen2-Audio 前向。两个三类概率分别转换为中心化 log-probability：

```text
z = log(p) - mean(log(p))
```

拼接后得到 6 维特征，由验证集冻结的三分类多项逻辑回归读取。它包含 `3×6` 个系数和 3 个截距，共 21 个标量参数。之后执行冻结的 label-free same-side 对话平滑：

- alpha = 0.4
- group_min = 3
- max_margin = 0.2
- temperature = 1.0
- 类别 bias = 0

因此 DACE 的大模型成本严格为每条 2.0 次前向。运行时只加载一个 7B 基座；两个 LoRA 在同一进程内依次激活。

## 3. ACCR

**ACCR = Adaptive Confidence-Calibrated Routing**。

ACCR 首先对所有样本运行 primary adapter，并通过冻结的三分类逻辑回归校准。路由规则固定为：

```text
if max(calibrated_primary_probability) <= 0.7793420477762057:
    run secondary adapter
else:
    keep primary result
```

被路由样本的 primary/secondary 概率使用冻结的 6 维联合逻辑回归重新校准；未路由样本保留 primary 校准概率。全体样本最后使用与 DACE 相同的 same-side 平滑参数。

在 3,025 条私有测试集上，1,196 条调用了 secondary：

```text
call_rate = 1196 / 3025 = 0.3953719008
mean_forwards = 1 + call_rate = 1.3953719008
```

新数据上的平均前向不是常数。领域变化、语音质量和类别边界都会改变 primary 置信度分布；请以运行 summary 的 `called_second`、`call_rate` 和 `average_model_forwards` 为准。

## 4. 无模型对话平滑

两种方法的最终平滑不加载语言模型、音频模型或分类器，也不使用标签。程序按 `dialogue_key + side` 分组，仅在组内至少 3 条且当前样本类别 margin 不超过 0.2 时，将单条概率与该组均值按 0.6/0.4 混合。

完整对话批量输入最符合验证协议。若只输入零散片段，方法仍可运行，但少于 3 条的组不会平滑；若文件名无法恢复对话和说话方，应显式提供 `dialogue_key`、`segment`。
