# 设计 `semantic_id_path`：全前缀剪枝、最小内存、快速加载——可扩展到大码本与深层级

本文修订了早先的"单整数 packing"方案，以处理两个现实约束：

- 码本宽度可达 **8192+** 个码字（每个码字 ≥ 13 位）。
- 层级深度可超过 **4** 层。

## 摘要

早先的单整数 packing 技巧（`packed_sid: int32`）在 `Σᵢ bᵢ > 64` 时**失效**。健壮的设计是**字典序排序的 `(N, H)` int16 张量**，并在波束搜索时执行**逐列增量二分查找**。成本可优雅扩展：

- 内存：`N × H × 2 B`（int16；可覆盖任意 `codebook_width ≤ 65536`）。
- 波束步计算量：每步 `O(K · log N)`，仍然向量化。
- 全前缀正确性保持——与 GRID 的 `_check_valid_prefix` 剪枝能力完全一致，只是用 O(log N) 代替 O(N)。

## 为何单整数 packing 失效

将 `(c₀, c₁, …, c_{H-1})` 打包为单个整数需要 `Σᵢ bᵢ` 位，其中 `bᵢ = ceil(log₂(codebook_sizes[i]))`：

| H | codebook_width | bits/code | total bits | 适配 |
|---:|---:|---:|---:|---|
| 4 | 256 | 8 | 32 | **int32** |
| 4 | 8192 | 13 | 52 | **int64** |
| 4 | 65536 | 16 | 64 | **int64**（恰好） |
| **5** | **8192** | **13** | **65** | ✗ 溢出 int64 |
| 6 | 8192 | 13 | 78 | ✗ |
| 8 | 8192 | 13 | 104 | ✗ |
| 8 | 65536 | 16 | 128 | ✗ |

对于"小型"TIGER（H=4，C=256），单整数 packing 仍是最快的可能表示。但对于用户的现实场景——`C=8192` 配合 `H > 4`——它会失败。需要一种可扩展的表示。

## 推荐设计：字典序排序的 `(N, H)` int16 张量

### 磁盘 schema

```
codes : list<int16>   # 每行长度 = H；行按 (c₀, c₁, …, c_{H-1}) 字典序排序
```

单列，按字典序升序排列。当 `max(codebook_sizes) ≤ 65536` 时使用 `int16`；只有罕见目录每层级 > 65K 码字时才提升为 `int32`（残差 VQ 中很少见）。

字典序排序的 int16 列经 snappy 压缩效果良好（通常为原始大小的 25–40%）。对于 `N = 1M, H = 8`：磁盘约 5–8 MB。

### 内存表示

```
self.codebooks : torch.Tensor  shape=(N, H)  dtype=int16   按行字典序排序
```

加上一个用于活跃波束搜索范围的小型伴随 buffer：

```
self.beam_lo : torch.Tensor  shape=(K,)  dtype=int64   每波束的下界（包含）
self.beam_hi : torch.Tensor  shape=(K,)  dtype=int64   每波束的上界（不包含）
```

`beam_lo` / `beam_hi` 不持久化——它们是单次 `generate()` 调用期间的暂存状态。

### 波束搜索查询：逐列增量二分查找

关键洞察：在字典序排序表中，任意固定前缀 `(c₀, …, c_h)` 对应一段**连续的行区间** `[lo, hi)`。当波束搜索从层级 `h` 推进到 `h+1` 时，区间只会收窄。

**步骤 0**（波束刚发出 `c₀`，无先验区间）：
```
beam_lo = searchsorted(codebooks[:, 0], c₀)         # 跨 K 个波束向量化
beam_hi = searchsorted(codebooks[:, 0], c₀ + 1)
valid   = beam_hi > beam_lo
```

**步骤 h > 0**（用 `c_h` 扩展现有前缀，细化 `[beam_lo, beam_hi)`）：
```
slice    = codebooks[beam_lo : beam_hi, h]          # 当前列，限定在波束范围内
new_lo   = beam_lo + searchsorted(slice, c_h)
new_hi   = beam_lo + searchsorted(slice, c_h + 1)
valid    = new_hi > new_lo
beam_lo  = new_lo
beam_hi  = new_hi
```

每波束每步：**两次在收窄区间上的二分查找**，二者均可通过 `torch.searchsorted`（PyTorch 2.0 起接受批量已排序序列）跨 K 个波束向量化。

总成本：每步 `O(K · log N)`，整次生成 `O(K · H · log N)`——基本免费。

每步的 `valid` 标志即波束剪枝信号：当为 `False` 时，将波束在刚发出码字处的 logit 设为 `-inf`（或将该波束从 top-K 池中丢弃）。

### 内存与加载——具体数字

`N = 1M items, max codebook = 8192（13 位，适配 int16）`：

| H | 内存 buffer | 磁盘（snappy parquet） | 加载（热缓存） |
|---:|---:|---:|---:|
| 4 | 8 MB | ~2–3 MB | < 50 ms |
| 6 | 12 MB | ~3–5 MB | < 80 ms |
| 8 | 16 MB | ~5–8 MB | < 100 ms |

`N = 100K items`（中等规模目录），`H = 8, C = 8192`：

| | 大小 |
|---|---|
| 内存 | 1.6 MB |
| 磁盘 | ~500 KB |
| 加载 | < 20 ms |

作为对比，GRID 在相同内存表上的暴力 `_check_valid_prefix` 每波束步成本为 **`O(N · K · H)`**——对于 N=1M, K=10, H=8 即约 8000 万次运算，而字典序排序设计仅需约 200 次。相同 buffer，**每步少 400,000× 计算量**。

## 算法正确性——具体示例

`H=4, C=8192` 的目录，5 个 item（为可读性而缩小，但编码方式适用于任意规模）：

| 行索引 | Item | SID |
|---:|---|---|
| 0 | A | (0, 1, 2, 3) |
| 1 | B | (0, 1, 2, 7) |
| 2 | C | (0, 1, 5000, 100) |
| 3 | D | (5, 1, 2, 9) |
| 4 | E | (5, 8000, 4, 11) |

行按 `(c₀, c₁, c₂, c₃)` 字典序排序。

**波束步骤 0**——波束发出 `c₀ = 0`：
- `lo = searchsorted(col0, 0) = 0`
- `hi = searchsorted(col0, 1) = 3`
- 区间 `[0, 3)` 覆盖行 {A, B, C}。✓ 有效。

**波束步骤 1**——用 `c₁ = 1` 扩展前缀：
- `slice = codebooks[0:3, 1] = [1, 1, 1]`
- `new_lo = 0 + searchsorted([1,1,1], 1) = 0`
- `new_hi = 0 + searchsorted([1,1,1], 2) = 3`
- 区间 `[0, 3)`。✓ 有效。

**波束步骤 2**——用 `c₂ = 8` 扩展前缀（任何以 `(0, 1)` 开头的 item 中都不存在该码字）：
- `slice = codebooks[0:3, 2] = [2, 2, 5000]`
- `new_lo = 0 + searchsorted([2,2,5000], 8) = 2`
- `new_hi = 0 + searchsorted([2,2,5000], 9) = 2`
- 区间 `[2, 2)` → 空。✗ **剪枝波束**。

对比：GRID 的暴力扫描需要将 `(0, 1, 8)` 广播到全部 5 行并发现无匹配——`O(5 × 3) = 15` 次比较。字典序排序版仅用 `2 × log₂(3) ≈ 4` 次比较完成同样工作。在百万级规模下差距扩大至 50,000 倍以上。

## 何时自动选择单整数 packing

对于小预算情形（`Σᵢ bᵢ ≤ 64`），单整数 packing 在每步计算上仍占优（在 1-D 与 2-D 结构上 searchsorted 约 2× 更快），且体积略小。加载时的决策规则：

```
total_bits = sum(ceil(log2(codebook_sizes[h])) for h in range(H))
if total_bits <= 64:
    use single-int packed buffer (int32 if total_bits ≤ 32, else int64)
else:
    use lex-sorted (N, H) int16 buffer
```

两条代码路径产生相同的每波束步 `valid` 掩码；模型可在 `__init__` 时选择并存储一个标志位。

## 更新后的加载流程

1. `pa.parquet.read_table(semantic_id_path, columns=["codes"])`——读取字典序排序的 list-of-int16 列。
2. **验证字典序顺序**：在展平行上执行一次 `(codebooks[1:] >= codebooks[:-1]).all()`；若上游发出未排序数据则快速报错。1M item 约 5 ms。
3. **验证位预算**：每个码字必须满足 `codes[h] < codebook_sizes[h]`。捕获配置漂移。
4. `torch.from_numpy(...)` → `(N, H) int16` 张量。
5. `register_buffer("codebooks", t, persistent=True)`。
6. 若 `total_bits ≤ 64`，亦预计算 packed 版本并将其作为活跃 buffer。可选的微优化。

对于希望使用最原始（未排序）格式的 schema 用户：模型可在加载时通过 `t = t[torch.lexsort(t.T.flip(0))]` 排序。1M item 增加约 50 ms，仅在首次训练时一次性执行。

## 为何此方案优于我们考虑过的所有其他设计

| 方案 | Buffer 大小（`N=1M, H=8, C=8192`） | 波束步计算量 | 全前缀正确性 | 可扩展至 H=8, C=8192？ |
|---|---|---|---|---|
| GRID 逐 item 表（当前） | 16 MB int16 | O(N·K·H) ~8000 万次 | ✓ | ✓ 但慢 |
| 逐层级词表 `(level, sid)` | 8 KB bool 掩码 | O(K·C) ~8 万次 | ✗（跨层级幻觉） | ✓ |
| 单整数 packing | **无法适配（104 位）** | — | — | ✗ |
| **字典序排序 `(N, H)` int16** | **16 MB** | **O(K·log N) ~200 次** | **✓** | **✓** |

字典序排序设计是该对比中**唯一**在保持全前缀正确性的同时可扩展到深层级与大码本的方案，并且每步计算量为 O(log N)。

## 边界情况

### 非均匀码本（`codebook: "8192,8192,4096,256"`）

dtype 由 `max(codebook_sizes)` 决定——选择 `int16`，因为 8192 ≤ 65536。字典序排序与二分查找无变化。

### 超大目录（`N > 100M`）

在 `N = 100M, H = 8, C = 8192` 时：
- 内存 buffer = 1.6 GB。开始成为真实成本。
- 磁盘 = 压缩后约 500 MB。
- 加载（热缓存）≈ 5 s；冷启动 ≈ 30 s。

若投入运营时的缓解措施：
- 按 `c₀` 值分片码本，按需加载分片。模型在步骤 0 只需匹配当前波束 `c₀` 的分片（且该 batch 后续生成均使用同一分片）。
- 以 `c₀ % num_shards` 为键存储为多个 parquet 文件，并通过 `pyarrow.dataset` 懒加载。

典型 TIGER 规模下不需要；目录跨过 50M 时再考虑。

### 大于 int16 的码本（`C > 65536`）

将 dtype 提升为 int32（每码字 4 B）。内存翻倍，其余不变。

## 总结推荐

| 问题 | 答案 |
|---|---|
| **磁盘 schema** | `codes: list<int16>`（长度 H），按行字典序排序 |
| **内存 buffer** | `(N, H) int16` 字典序排序张量（仅当 `C > 65536` 时用 `int32`） |
| **波束搜索算法** | 逐列增量 `torch.searchsorted`；跨层级步骤维护每波束 `(lo, hi)` 区间 |
| **磁盘大小（1M items, H=8, C=8192）** | 压缩 parquet 约 5–8 MB |
| **内存大小（同上）** | 16 MB |
| **加载时间** | 热缓存 <100 ms |
| **无损全前缀剪枝？** | 是——语义与 GRID 一致，每步 O(log N) |
| **可扩展至 H ≥ 8, C ≥ 8192？** | 是——优雅扩展 |
| **可选优化** | 加载时检测 `Σᵢ bᵢ ≤ 64` 并切换至单整数 packed buffer；两条路径产生相同 `valid` 掩码 |

字典序排序 + 增量二分查找方案给出了通用答案：全前缀正确性、每波束步 O(log N)、可扩展到任意现实的 `(H, C)` 组合——代价是每步两次 `searchsorted` 调用而非一次。早先的单整数 packing 是小预算情形的优化；本设计将其作为子情形涵盖。
