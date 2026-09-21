# GenRec training step — time-cost report (H20)

Measured 2026-09-21 on the real `GenRecCausalLMModel`, not a stand-in backbone.
Branch `bench/genrec-attn-kernel-h20`, on top of PR #663 (`3d82728`).

## Headline

At the **real JHK sequence length, attention is 7.0% of the training step** — not 12.4%.
The 12.4% figure is the 2048-token stress point, not production. After PR #663 packs the
forward, the step is dominated by the linear layers (60%), and attention is a small slice.

| what                           | at real JHK length (max 743) |   at 2048 |
| ------------------------------ | ---------------------------: | --------: |
| attention share of step (FA2)  |                     **7.0%** | **12.4%** |
| FA3 gain over FA2, end to end  |                    **+2.5%** | **+5.0%** |
| FA2 gain over SDPA, end to end |                    **1.92x** | **2.55x** |

## Setup

|           |                                                                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU       | 1x NVIDIA H20 96GB (SM 9.0), driver 580.95.05                                                                                                               |
| Stack     | torch 2.13.0+cu129, transformers 5.17.0, FA2 2.8.3.post1, FA3 3.0.0 (built from `hopper/`)                                                                  |
| Model     | `GenRecCausalLMModel` over Qwen2.5-0.5B: hidden 896, 24 layers, 14 Q heads / 2 KV heads (GQA 7:1), head_dim 64, intermediate 4864, vocab resized to 152,448 |
| Config    | JHK: codebook 256x256x256, `max_length` 1024, `batch_size` 20, `lm_parameter_dtype: FP32` + BF16 autocast                                                   |
| Data      | `jhk_top300_nan_repro_20260827/samples/full/*.parquet`, real `item_list_with_sid` / `single_label`                                                          |
| Optimizer | `torch.optim.AdamW` unfused (see Caveats)                                                                                                                   |

Assembled length = 71 static prompt tokens + history codes + 3 label codes.

## 1. Step totals

### Batch sweep, real JHK length

| batch | pad waste |      SDPA |    FA2 |        FA3 |    SDPA peak | FA2 peak | FA3 peak |
| ----: | --------: | --------: | -----: | ---------: | -----------: | -------: | -------: |
|    10 |     53.4% | 213.21 ms | 103.42 | **101.71** | 18,107.5 MiB | 11,918.4 | 11,918.4 |
|    20 |     43.1% | 448.84 ms | 233.39 | **227.71** | 31,515.2 MiB | 20,147.2 | 20,147.2 |
|    40 |     50.0% | 920.43 ms | 402.75 | **393.29** | 58,939.5 MiB | 31,274.2 | 31,274.1 |

FA2 vs SDPA: 2.06x / 1.92x / 2.29x. FA3 vs FA2: +1.68% / +2.49% / +2.41%.
**FA3 saves no memory at all** — identical to FA2 to within 0.1 MiB.

### Length sweep, batch 20

|    max_len | mean_len | pad waste |       SDPA |    FA2 |    FA3 | FA3 vs FA2 |
| ---------: | -------: | --------: | ---------: | -----: | -----: | ---------: |
| 743 (real) |      423 |     43.1% |  449.16 ms | 233.84 | 228.22 | **+2.46%** |
|       1022 |      568 |     44.4% |  634.47 ms | 303.40 | 294.91 | **+2.88%** |
|       2048 |     1102 |     46.2% | 1486.69 ms | 582.43 | 554.94 | **+4.95%** |

Peak memory: SDPA 31.5 / 40.9 / **76.8 GB** (close to OOM on a 96 GB card),
FA2 and FA3 both 20.1 / 24.7 / **41.1 GB**.

## 2. Stage breakdown (FA2, batch 20)

| stage                                                 |        at 743 |     share |       at 2048 |     share |
| ----------------------------------------------------- | ------------: | --------: | ------------: | --------: |
| forward total                                         |      73.51 ms |     31.2% |     187.91 ms |     32.1% |
| ├ `build_input` (embedding group + prompt projection) |       0.11 ms |      0.0% |       0.16 ms |      0.0% |
| ├ backbone forward                                    |      72.64 ms |     30.8% |     186.69 ms |     31.9% |
| └ gather / labels / pad                               |       0.76 ms |      0.3% |       1.07 ms |      0.2% |
| backward                                              |     147.45 ms | **62.5%** |     381.85 ms | **65.3%** |
| optimizer                                             |      14.97 ms |      6.3% |      15.30 ms |      2.6% |
| **TOTAL**                                             | **235.93 ms** |           | **585.06 ms** |           |

Two things to read off this:

- **Backward is 2x forward.** Any kernel whose backward is weaker than its forward is
  penalised accordingly — which is exactly FA3's situation (section 4).
- **The prompt machinery is free.** `build_input`, which runs the embedding group, the
  `PromptProjection` modules and the `index_copy` into the hole positions, costs
  **0.11 ms, 0.05% of the step**. It is not worth optimising.

## 3. Where the time goes (CUDA kernel buckets)

### At real JHK length (max_len 743, batch 20)

| bucket        |                 SDPA |                FA2 |              FA3 |
| ------------- | -------------------: | -----------------: | ---------------: |
| gemm (linear) |    233.74 ms (52.8%) | **136.58 (60.1%)** |   136.67 (61.6%) |
| cast/copy     |     61.15 ms (13.8%) |  **32.06 (14.1%)** |    32.27 (14.5%) |
| elementwise   |      41.91 ms (9.5%) |  **23.75 (10.4%)** |    23.13 (10.4%) |
| **attention** | **82.14 ms (18.5%)** |   **15.91 (7.0%)** | **10.78 (4.9%)** |
| optimizer     |      13.43 ms (3.0%) |       13.43 (5.9%) |     13.42 (6.0%) |
| norm/act      |       7.85 ms (1.8%) |        3.75 (1.6%) |      3.75 (1.7%) |
| other         |              2.42 ms |               1.72 |             1.73 |
| softmax / CE  |              0.18 ms |               0.17 |             0.18 |
| embedding     |              0.05 ms |               0.06 |             0.06 |

### At 2048

| bucket        |                  SDPA |               FA2 |              FA3 |
| ------------- | --------------------: | ----------------: | ---------------: |
| gemm (linear) |     630.15 ms (42.4%) |    344.38 (59.6%) |   344.56 (62.6%) |
| **attention** | **547.46 ms (36.9%)** | **71.95 (12.4%)** | **45.89 (8.3%)** |
| cast/copy     |     159.95 ms (10.8%) |     76.37 (13.2%) |    76.23 (13.9%) |
| elementwise   |      107.26 ms (7.2%) |     58.80 (10.2%) |    57.26 (10.4%) |
| norm/act      |       21.60 ms (1.5%) |       9.60 (1.7%) |      9.60 (1.7%) |
| optimizer     |       13.42 ms (0.9%) |      13.45 (2.3%) |     13.44 (2.4%) |

**FA3 changes exactly one row.** Every other bucket matches FA2 to within 0.2 ms.

## 4. Attention detail

FA2 vs FA3 attention kernel time, batch 20, max_len 743:

|          |       FA2 |      FA3 |   speedup |
| -------- | --------: | -------: | --------: |
| forward  |  4.425 ms | 2.649 ms | **1.67x** |
| backward | 11.504 ms | 8.133 ms | **1.41x** |
| total    |  15.93 ms | 10.78 ms | **1.48x** |

Backward is **72% of attention time**, and that is where FA3 gains least. At 2048 the
attention speedup improves to 1.57x (71.95 -> 45.89 ms).

The end-to-end number follows arithmetically. At 2048: attention is 12.4% of the step and
FA3 is 1.57x on it, so `12.4% x (1 - 1/1.57) = 4.5%` predicted against **4.95% measured**.

Note the SDPA arm never reaches the flash backend: its kernels are
`fmha_cutlassB/F_bf16_aligned_64x64_k64_sm80`, the memory-efficient backend, because
the left-padded layout requires a 4D mask which the flash backend rejects. They are also
`sm80` — Ampere-generation kernels running on Hopper.

## 5. Why attention cannot be the lever

The GEMM bucket has **no headroom**. At batch 20 / 8,458 real tokens the linear layers are
about 18.2 TFLOP of forward+backward matmul, executed in 136.58 ms:

```
per token per layer: 2 x (896x1152 QKV + 896x896 O + 3 x 896x4864 MLP) = 29.82 MFLOP
x 24 layers x 3 (fwd+bwd) x 8,458 tokens                               = 18.16 TFLOP
18.16 TFLOP / 0.13658 s                                                = 133 TFLOP/s
```

H20 peak BF16 is about 148 TFLOP/s, so the GEMMs run at roughly **90% of peak**. That 60%
of the step cannot be optimised — only reduced, by fewer tokens, fewer parameters, or FP8.

With 60% saturated and attention at 7%, even an infinitely fast attention kernel caps out
at a 7% step improvement.

## 6. Where the remaining headroom is

1. **`cast/copy`, 14.1% — twice attention.** These are the autocast bf16 \<-> fp32
   conversions caused by `lm_parameter_dtype: FP32` under BF16 mixed precision. The proto
   comment records FP32 as deliberate (it avoids bf16-ULP underflow of Adam's small
   updates), so this is a real accuracy trade — but 14% of every step is dtype conversion.
   An accuracy-vs-throughput test of BF16 parameters is the single largest available win.
1. **`elementwise`, 10.4%.** RoPE, residual adds, masking. A fusion pass or
   `torch.compile` on the decoder block targets this.
1. **`optimizer`, 6.3% at 743 but 2.6% at 2048.** It is length-invariant (13.4 ms of CUDA
   at every length) because it scales with parameter count, not tokens.
1. **Attention, 7%.** Already collected by FA2. FA3 adds 2.5%.

## 7. Recommendation

- **Adopt FA2.** 1.92x and -36% peak memory at production batch 20, with a pinned wheel
  already in `requirements/cu129.txt`.
- **Do not adopt FA3 yet.** +2.5% at production length, 0% memory, and it needs a
  from-source Hopper build (no wheel exists) that links against the torch minor ABI, so a
  torch bump breaks it at import. It is wired and tested on this branch, so re-measuring
  later is a one-line config change.
- **Revisit FA3 if sequences grow.** Its advantage doubles from 743 to 2048 tokens.
- **Next target is not attention.** It is the 14% cast/copy bucket.

## 8. Reproduce

```bash
FD=<dir containing the flash_attn 2 wheel>
PYTHONPATH=.:$FD python -m tzrec.tools.genrec_attn_kernel_bench \
    --backbone /mnt/data/Qwen2.5-0.5B \
    --data-glob '/mnt/data/aop_lab/collision_exp/jhk_top300_nan_repro_20260827/samples/full/*.parquet' \
    --kernels SDPA,FLASH_ATTENTION_2,FLASH_ATTENTION_3 \
    --batch-sizes 10,20,40
# longer sequences:
    --sequence-length 3000 --max-length 4096 --hist-scale 2.95
```

## 9. Caveats

- **Sequences above 897 tokens are manufactured.** Real JHK data maxes at 897. The 1022 and
  2048 points repeat each row's own real SID codes (`--hist-scale`), so the codes and the
  batch raggedness stay real and only the length is synthetic.
- **The optimizer is unfused `torch.optim.AdamW`**, not tzrec's configured dense optimizer.
  It is length-invariant, so it inflates the denominator at short lengths and slightly
  understates every other bucket's share there.
- **Single GPU, no DDP.** Gradient all-reduce, gradient accumulation (production uses 4
  microbatches) and the dataloader are all excluded. This measures the compute step only.
- **Weights are randomly initialised** from the real `config.json`. Shapes and therefore
  timings are exact; only the values differ, which does not affect throughput.
- Kernel-bucket sums are CUDA time and fall slightly below the wall-clock stage totals
  (for FA2 at 743: 227.4 ms vs 235.9 ms); the difference is launch gaps and CPU time.
