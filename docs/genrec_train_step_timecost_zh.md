# GenRec 训练单步耗时报告（H20）

测试时间 2026-09-21，跑的是真实的 `GenRecCausalLMModel`，不是替身模型。
分支 `bench/genrec-attn-kernel-h20`，基于 PR #663（`3d82728`）。

## 结论先行

在**真实 JHK 序列长度下，attention 只占训练单步的 7.0%**，不是 12.4%。
12.4% 是 2048 token 的压测点，不是线上口径。PR #663 把前向打包（packed）之后，
单步耗时由线性层主导（60%），attention 只是其中一小块。

| 指标 | 真实 JHK 长度（max 743） | 2048 |
| --- | ---: | ---: |
| attention 占单步比例（FA2） | **7.0%** | **12.4%** |
| FA3 相对 FA2 的端到端收益 | **+2.5%** | **+5.0%** |
| FA2 相对 SDPA 的端到端收益 | **1.92x** | **2.55x** |

## 测试环境

| | |
| --- | --- |
| GPU | 1x NVIDIA H20 96GB（SM 9.0），驱动 580.95.05 |
| 软件栈 | torch 2.13.0+cu129、transformers 5.17.0、FA2 2.8.3.post1、FA3 3.0.0（由 `hopper/` 源码编译） |
| 模型 | `GenRecCausalLMModel` + Qwen2.5-0.5B：hidden 896、24 层、14 Q 头 / 2 KV 头（GQA 7:1）、head_dim 64、intermediate 4864、词表 resize 到 152,448 |
| 配置 | JHK：codebook 256x256x256、`max_length` 1024、`batch_size` 20、`lm_parameter_dtype: FP32` + BF16 autocast |
| 数据 | `jhk_top300_nan_repro_20260827/samples/full/*.parquet`，真实的 `item_list_with_sid` / `single_label` |
| 优化器 | 未 fuse 的 `torch.optim.AdamW`（见"口径说明"） |

组装后长度 = 71 个静态 prompt token + 历史 SID code + 3 个 label code。

## 1. 单步总耗时

### batch 扫描（真实 JHK 长度）

| batch | padding 浪费 | SDPA | FA2 | FA3 | SDPA 峰值 | FA2 峰值 | FA3 峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 53.4% | 213.21 ms | 103.42 | **101.71** | 18,107.5 MiB | 11,918.4 | 11,918.4 |
| 20 | 43.1% | 448.84 ms | 233.39 | **227.71** | 31,515.2 MiB | 20,147.2 | 20,147.2 |
| 40 | 50.0% | 920.43 ms | 402.75 | **393.29** | 58,939.5 MiB | 31,274.2 | 31,274.1 |

FA2 相对 SDPA：2.06x / 1.92x / 2.29x。FA3 相对 FA2：+1.68% / +2.49% / +2.41%。
**FA3 完全不省显存**，与 FA2 的差异在 0.1 MiB 以内。

### 长度扫描（batch 20）

| max_len | mean_len | padding 浪费 | SDPA | FA2 | FA3 | FA3 相对 FA2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 743（真实） | 423 | 43.1% | 449.16 ms | 233.84 | 228.22 | **+2.46%** |
| 1022 | 568 | 44.4% | 634.47 ms | 303.40 | 294.91 | **+2.88%** |
| 2048 | 1102 | 46.2% | 1486.69 ms | 582.43 | 554.94 | **+4.95%** |

峰值显存：SDPA 31.5 / 40.9 / **76.8 GB**（96 GB 卡上已接近 OOM），
FA2 与 FA3 均为 20.1 / 24.7 / **41.1 GB**。

## 2. 分阶段拆解（FA2，batch 20）

| 阶段 | 743 | 占比 | 2048 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| 前向合计 | 73.51 ms | 31.2% | 187.91 ms | 32.1% |
| &nbsp;&nbsp;`build_input`（embedding group + prompt 投影） | 0.11 ms | 0.0% | 0.16 ms | 0.0% |
| &nbsp;&nbsp;backbone 前向 | 72.64 ms | 30.8% | 186.69 ms | 31.9% |
| &nbsp;&nbsp;gather / labels / pad | 0.76 ms | 0.3% | 1.07 ms | 0.2% |
| 反向 | 147.45 ms | **62.5%** | 381.85 ms | **65.3%** |
| 优化器 | 14.97 ms | 6.3% | 15.30 ms | 2.6% |
| **合计** | **235.93 ms** | | **585.06 ms** | |

这张表有两个要点：

- **反向是前向的 2 倍。** 任何"反向弱于前向"的 kernel 都会按这个比例被稀释，
  而 FA3 正是这种情况（见第 4 节）。
- **prompt 相关逻辑不花钱。** `build_input` 负责 embedding group、`PromptProjection`
  以及把投影结果 `index_copy` 到 hole 位置，总共 **0.11 ms，占单步 0.05%**，
  没有优化价值。

## 3. 时间花在哪里（CUDA kernel 分桶）

### 真实 JHK 长度（max_len 743，batch 20）

| 分桶 | SDPA | FA2 | FA3 |
| --- | ---: | ---: | ---: |
| gemm（线性层） | 233.74 ms (52.8%) | **136.58 (60.1%)** | 136.67 (61.6%) |
| cast/copy（精度转换） | 61.15 ms (13.8%) | **32.06 (14.1%)** | 32.27 (14.5%) |
| elementwise | 41.91 ms (9.5%) | **23.75 (10.4%)** | 23.13 (10.4%) |
| **attention** | **82.14 ms (18.5%)** | **15.91 (7.0%)** | **10.78 (4.9%)** |
| 优化器 | 13.43 ms (3.0%) | 13.43 (5.9%) | 13.42 (6.0%) |
| norm/激活 | 7.85 ms (1.8%) | 3.75 (1.6%) | 3.75 (1.7%) |
| 其他 | 2.42 ms | 1.72 | 1.73 |
| softmax / CE | 0.18 ms | 0.17 | 0.18 |
| embedding | 0.05 ms | 0.06 | 0.06 |

### 2048

| 分桶 | SDPA | FA2 | FA3 |
| --- | ---: | ---: | ---: |
| gemm（线性层） | 630.15 ms (42.4%) | 344.38 (59.6%) | 344.56 (62.6%) |
| **attention** | **547.46 ms (36.9%)** | **71.95 (12.4%)** | **45.89 (8.3%)** |
| cast/copy | 159.95 ms (10.8%) | 76.37 (13.2%) | 76.23 (13.9%) |
| elementwise | 107.26 ms (7.2%) | 58.80 (10.2%) | 57.26 (10.4%) |
| norm/激活 | 21.60 ms (1.5%) | 9.60 (1.7%) | 9.60 (1.7%) |
| 优化器 | 13.42 ms (0.9%) | 13.45 (2.3%) | 13.44 (2.4%) |

**FA3 只改变了一行。** 其余每个分桶与 FA2 的差异都在 0.2 ms 以内。

## 4. attention 细分

FA2 与 FA3 的 attention kernel 耗时（batch 20，max_len 743）：

| | FA2 | FA3 | 加速比 |
| --- | ---: | ---: | ---: |
| 前向 | 4.425 ms | 2.649 ms | **1.67x** |
| 反向 | 11.504 ms | 8.133 ms | **1.41x** |
| 合计 | 15.93 ms | 10.78 ms | **1.48x** |

反向占 attention 总耗时的 **72%**，而这恰恰是 FA3 收益最小的地方。
到 2048 时 attention 加速比提升到 1.57x（71.95 -> 45.89 ms）。

端到端数字可以直接推出来。2048 时 attention 占 12.4%，FA3 对它加速 1.57x，
则 `12.4% x (1 - 1/1.57) = 4.5%`，与**实测 4.95%** 吻合。

另外，SDPA 这一路根本没走到 flash 后端：它的 kernel 是
`fmha_cutlassB/F_bf16_aligned_64x64_k64_sm80`，也就是 memory-efficient 后端。
原因是左填充布局必须传 4D mask，而 flash 后端拒绝 4D mask。注意后缀是
`sm80`——在 Hopper 上跑的是 Ampere 世代的 kernel。

## 5. 为什么 attention 不可能成为抓手

GEMM 这一桶**没有余量**。batch 20、8,458 个真实 token 时，线性层大约是
18.2 TFLOP 的前向+反向矩阵乘，实测耗时 136.58 ms：

```
每 token 每层：2 x (896x1152 QKV + 896x896 O + 3 x 896x4864 MLP) = 29.82 MFLOP
x 24 层 x 3（前向+反向）x 8,458 token                            = 18.16 TFLOP
18.16 TFLOP / 0.13658 s                                          = 133 TFLOP/s
```

H20 的 BF16 峰值约 148 TFLOP/s，即 GEMM 已经跑到**约 90% 的峰值**。
这占单步 60% 的部分没法再优化，只能减少——更少的 token、更少的参数，或者上 FP8。

60% 已经打满、attention 只占 7% 的情况下，即使 attention kernel 快到无穷大，
单步最多也只能提升 7%。

## 6. 剩下的优化空间在哪

1. **`cast/copy` 占 14.1%，是 attention 的两倍。** 这是 `lm_parameter_dtype: FP32`
   在 BF16 混合精度下产生的 bf16 <-> fp32 转换。proto 注释写明 FP32 是刻意选择
   （避免 Adam 小更新量在 bf16 下的 ULP 下溢），所以这是真实的精度取舍——
   但每一步都有 14% 花在精度转换上。**把 BF16 参数做一次精度/吞吐对比实验，
   是目前单项收益最大的方向。**
2. **`elementwise` 占 10.4%。** RoPE、残差相加、mask。适合做算子融合或对
   decoder block 上 `torch.compile`。
3. **优化器在 743 时占 6.3%，2048 时只占 2.6%。** 它与长度无关（各长度都是
   13.4 ms CUDA 时间），因为它只随参数量变化，不随 token 数变化。
4. **attention 占 7%。** FA2 已经把这块收走了，FA3 再加 2.5%。

## 7. 建议

- **采用 FA2。** 线上 batch 20 下 1.92x 加速、峰值显存 -36%，且
  `requirements/cu129.txt` 里已经钉好了 wheel。
- **暂不采用 FA3。** 线上长度下只有 +2.5%、显存 0 收益，而且需要源码编译
  Hopper 版本（没有现成 wheel），并且链接的是 torch 次版本 ABI——
  torch 一升级就会在 import 阶段直接挂掉。该分支上 FA3 已接好并有测试，
  以后要复测只是改一行配置。
- **序列变长再回头看 FA3。** 从 743 到 2048，它的收益翻了一倍。
- **下一个目标不是 attention**，是那 14% 的 cast/copy。

## 8. 复现方式

```bash
FD=<存放 flash_attn 2 wheel 的目录>
PYTHONPATH=.:$FD python -m tzrec.tools.genrec_attn_kernel_bench \
    --backbone /mnt/data/Qwen2.5-0.5B \
    --data-glob '/mnt/data/aop_lab/collision_exp/jhk_top300_nan_repro_20260827/samples/full/*.parquet' \
    --kernels SDPA,FLASH_ATTENTION_2,FLASH_ATTENTION_3 \
    --batch-sizes 10,20,40
# 更长序列：
    --sequence-length 3000 --max-length 4096 --hist-scale 2.95
```

## 9. 口径说明

- **超过 897 token 的序列是人造的。** 真实 JHK 数据最长 897。1022 和 2048 两个点
  是把每行自己的真实 SID code 重复拼接得到的（`--hist-scale`），
  因此 code 取值和 batch 的长短不齐都是真实的，只有长度是人造的。
- **优化器是未 fuse 的 `torch.optim.AdamW`**，不是 tzrec 配置的 dense optimizer。
  它与长度无关，所以在短序列下会抬高分母，略微低估其他各桶的占比。
- **单卡、无 DDP。** 不含梯度 all-reduce、梯度累积（线上是 4 个 microbatch）
  和数据加载，只衡量计算步本身。
- **权重是按真实 `config.json` 随机初始化的。** 形状因而耗时是准确的，
  只有数值不同，不影响吞吐。
- 分桶之和是 CUDA 时间，略低于墙钟的阶段合计（FA2 在 743 时为 227.4 ms
  对 235.9 ms），差值是 kernel launch 间隙和 CPU 时间。
