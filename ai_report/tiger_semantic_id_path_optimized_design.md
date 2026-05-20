# Designing `semantic_id_path` for full-prefix pruning, minimal memory, fast load — scaling to large codebooks and deep hierarchies

This revises the earlier "single-int packing" recipe to handle two realistic constraints:

- Codebook width can reach **8192+** codes (≥ 13 bits per code).
- Hierarchy depth can exceed **4** layers.

## TL;DR

The earlier single-int packing trick (`packed_sid: int32`) **breaks** when `Σᵢ bᵢ > 64`. The robust design is a **lex-sorted `(N, H)` int16 tensor**, with **incremental column-wise binary search** during beam search. Costs scale gracefully:

- Memory: `N × H × 2 B` (int16; covers any `codebook_width ≤ 65536`).
- Beam-step compute: `O(K · log N)` per step, still vectorized.
- Full-prefix correctness preserved — identical pruning power to GRID's `_check_valid_prefix`, just O(log N) instead of O(N).

## Why single-int packing breaks

Packing `(c₀, c₁, …, c_{H-1})` into one int requires `Σᵢ bᵢ` bits, where `bᵢ = ceil(log₂(codebook_sizes[i]))`:

| H | codebook_width | bits/code | total bits | fits in |
|---:|---:|---:|---:|---|
| 4 | 256 | 8 | 32 | **int32** |
| 4 | 8192 | 13 | 52 | **int64** |
| 4 | 65536 | 16 | 64 | **int64** (just) |
| **5** | **8192** | **13** | **65** | ✗ overflows int64 |
| 6 | 8192 | 13 | 78 | ✗ |
| 8 | 8192 | 13 | 104 | ✗ |
| 8 | 65536 | 16 | 128 | ✗ |

For "small" TIGER (H=4, C=256) single-int packing remains the fastest possible representation. For the user's realistic scenarios — `C=8192` with `H > 4` — it fails. Need a representation that scales.

## Recommended design: lex-sorted `(N, H)` int16 tensor

### On-disk schema

```
codes : list<int16>   # length = H per row; rows lex-sorted by (c₀, c₁, …, c_{H-1})
```

Single column, sorted ascending in lex order. Use `int16` when `max(codebook_sizes) ≤ 65536`; bump to `int32` only for exotic catalogs with > 65K codes per layer (rare for residual VQ).

Parquet on a lex-sorted int16 column compresses well with snappy (typically 25–40% of raw). For `N = 1M, H = 8`: ~5–8 MB on disk.

### In-memory representation

```
self.codebooks : torch.Tensor  shape=(N, H)  dtype=int16   lex-sorted by row
```

Plus a small companion buffer for the active beam-search ranges:

```
self.beam_lo : torch.Tensor  shape=(K,)  dtype=int64   inclusive lower bound per beam
self.beam_hi : torch.Tensor  shape=(K,)  dtype=int64   exclusive upper bound per beam
```

`beam_lo` / `beam_hi` are not persistent — they're scratch state during a single `generate()` call.

### Beam-search query: incremental column-wise binary search

The key insight: in a lex-sorted table, any fixed prefix `(c₀, …, c_h)` corresponds to a **contiguous row range** `[lo, hi)`. As beam search advances from level `h` to level `h+1`, the range only narrows.

**Step 0** (beam has just emitted `c₀`, no prior range):
```
beam_lo = searchsorted(codebooks[:, 0], c₀)         # vectorized over K beams
beam_hi = searchsorted(codebooks[:, 0], c₀ + 1)
valid   = beam_hi > beam_lo
```

**Step h > 0** (extend existing prefix with `c_h`, refining `[beam_lo, beam_hi)`):
```
slice    = codebooks[beam_lo : beam_hi, h]          # current column, restricted to the beam's range
new_lo   = beam_lo + searchsorted(slice, c_h)
new_hi   = beam_lo + searchsorted(slice, c_h + 1)
valid    = new_hi > new_lo
beam_lo  = new_lo
beam_hi  = new_hi
```

Per beam per step: **two binary searches over a shrinking range**, both vectorizable across K beams via `torch.searchsorted` (which accepts batched sorted sequences as of PyTorch 2.0).

Total cost: `O(K · log N)` per step, `O(K · H · log N)` for a full generation pass — essentially free.

The `valid` flag at each step is the beam-pruning signal: when `False`, the beam's logit at the just-emitted code is set to `-inf` (or the beam is dropped from the top-K pool).

### Memory & load — concrete numbers

`N = 1M items, max codebook = 8192 (13 bits, fits int16)`:

| H | In-memory buffer | Disk (snappy parquet) | Load (hot cache) |
|---:|---:|---:|---:|
| 4 | 8 MB | ~2–3 MB | < 50 ms |
| 6 | 12 MB | ~3–5 MB | < 80 ms |
| 8 | 16 MB | ~5–8 MB | < 100 ms |

`N = 100K items` (mid-size catalog), `H = 8, C = 8192`:

| | size |
|---|---|
| In-memory | 1.6 MB |
| Disk | ~500 KB |
| Load | < 20 ms |

For comparison, GRID's brute-force `_check_valid_prefix` against the same in-memory table costs **`O(N · K · H)`** per beam step — for N=1M, K=10, H=8 that's ~80M ops vs the lex-sorted design's ~200 ops. Same buffer, **400,000× less compute per step.**

## Algorithm correctness — concrete example

Catalog with `H=4, C=8192`, 5 items (made small for legibility, but the encoding works at any scale):

| Row idx | Item | SID |
|---:|---|---|
| 0 | A | (0, 1, 2, 3) |
| 1 | B | (0, 1, 2, 7) |
| 2 | C | (0, 1, 5000, 100) |
| 3 | D | (5, 1, 2, 9) |
| 4 | E | (5, 8000, 4, 11) |

Rows are lex-sorted by `(c₀, c₁, c₂, c₃)`.

**Beam step 0** — beam emits `c₀ = 0`:
- `lo = searchsorted(col0, 0) = 0`
- `hi = searchsorted(col0, 1) = 3`
- Range `[0, 3)` covers rows {A, B, C}. ✓ valid.

**Beam step 1** — extends prefix with `c₁ = 1`:
- `slice = codebooks[0:3, 1] = [1, 1, 1]`
- `new_lo = 0 + searchsorted([1,1,1], 1) = 0`
- `new_hi = 0 + searchsorted([1,1,1], 2) = 3`
- Range `[0, 3)`. ✓ valid.

**Beam step 2** — extends prefix with `c₂ = 8` (a code not in any item starting with `(0, 1)`):
- `slice = codebooks[0:3, 2] = [2, 2, 5000]`
- `new_lo = 0 + searchsorted([2,2,5000], 8) = 2`
- `new_hi = 0 + searchsorted([2,2,5000], 9) = 2`
- Range `[2, 2)` → empty. ✗ **prune beam**.

Compare: GRID's brute-force scan would have to broadcast `(0, 1, 8)` against all 5 rows and find no match — `O(5 × 3) = 15` comparisons. The lex-sorted version did the same with `2 × log₂(3) ≈ 4` comparisons. The gap widens to 50,000×+ at million-item scale.

## When to autoselect single-int packing

For the small-budget case (`Σᵢ bᵢ ≤ 64`), single-int packing still wins on per-step compute (~2× faster searchsorted on a 1-D vs 2-D structure) and is marginally smaller. Decision rule at load:

```
total_bits = sum(ceil(log2(codebook_sizes[h])) for h in range(H))
if total_bits <= 64:
    use single-int packed buffer (int32 if total_bits ≤ 32, else int64)
else:
    use lex-sorted (N, H) int16 buffer
```

Both code paths produce the same `valid` mask per beam step; the model can pick at `__init__` and store a flag.

## Updated load procedure

1. `pa.parquet.read_table(semantic_id_path, columns=["codes"])` — read the lex-sorted list-of-int16 column.
2. **Validate lex-sorted order** with one `(codebooks[1:] >= codebooks[:-1]).all()` over flat rows; error fast if upstream emitted unsorted data. ~5 ms for 1M items.
3. **Validate bit budget**: every code must satisfy `codes[h] < codebook_sizes[h]`. Catches config drift.
4. `torch.from_numpy(...)` → `(N, H) int16` tensor.
5. `register_buffer("codebooks", t, persistent=True)`.
6. If `total_bits ≤ 64`, also precompute the packed version and pick it as the active buffer. Optional micro-optimization.

For schema users who want the rawest format (un-sorted): the model can sort at load with `t = t[torch.lexsort(t.T.flip(0))]`. Adds ~50 ms for 1M items, one-shot at first training.

## Why this beats every other design we considered

| Approach | Buffer size (`N=1M, H=8, C=8192`) | Beam-step compute | Full-prefix correctness | Scales to H=8, C=8192? |
|---|---|---|---|---|
| GRID per-item table (current) | 16 MB int16 | O(N·K·H) ~80M ops | ✓ | ✓ but slow |
| Per-level vocab `(level, sid)` | 8 KB bool mask | O(K·C) ~80K ops | ✗ (cross-level hallucinations) | ✓ |
| Single-int packed | **does not fit (104 bits)** | — | — | ✗ |
| **Lex-sorted `(N, H)` int16** | **16 MB** | **O(K·log N) ~200 ops** | **✓** | **✓** |

The lex-sorted design is the **only** option in the comparison that keeps full-prefix correctness while scaling to deep hierarchies and large codebooks, and it does so with O(log N) per-step compute.

## Edge cases

### Non-uniform codebooks (`codebook: "8192,8192,4096,256"`)

The dtype is sized by `max(codebook_sizes)` — picks `int16` since 8192 ≤ 65536. Lex sorting and binary search work unchanged.

### Very large catalogs (`N > 100M`)

At `N = 100M, H = 8, C = 8192`:
- In-memory buffer = 1.6 GB. Starts to become a real cost.
- Disk = ~500 MB compressed.
- Load (hot cache) ≈ 5 s; cold ≈ 30 s.

Mitigations if this becomes operational:
- Shard the codebook by `c₀` value and load shards on demand. The model only ever needs the shard matching the current beam's `c₀` at step 0 (and the same shard for the rest of generation in that batch).
- Store as multiple parquet files keyed by `c₀ % num_shards` and lazy-load via `pyarrow.dataset`.

Not needed for typical TIGER scales; deferred until catalogs cross 50M.

### Codebooks larger than int16 (`C > 65536`)

Bump dtype to int32 (4 B per code). Memory doubles, everything else unchanged.

## Summary recommendation

| Question | Answer |
|---|---|
| **On-disk schema** | `codes: list<int16>` (length H), lex-sorted by row |
| **In-memory buffer** | `(N, H) int16` lex-sorted tensor (`int32` only if `C > 65536`) |
| **Beam-search algorithm** | Incremental column-wise `torch.searchsorted`; maintain per-beam `(lo, hi)` ranges across hierarchy steps |
| **Disk size (1M items, H=8, C=8192)** | ~5–8 MB compressed parquet |
| **In-memory size (same)** | 16 MB |
| **Load time** | <100 ms hot cache |
| **Lossless full-prefix pruning?** | Yes — same semantics as GRID, O(log N) per step |
| **Scales to H ≥ 8, C ≥ 8192?** | Yes — gracefully |
| **Optional optimization** | Detect `Σᵢ bᵢ ≤ 64` at load and switch to single-int packed buffer; both paths produce the same `valid` mask |

The lex-sorted-with-incremental-binary-search approach gives the universal answer: full-prefix correctness, O(log N) per beam step, scales to any realistic `(H, C)` combination — at the cost of doing two `searchsorted` calls per step instead of one. The earlier single-int packing was an optimization for the small-budget case; this design subsumes it.
