# Beam-search pruning: GRID full-prefix vs per-level vocab

Context: choosing between two designs for the `constrained_beam_search` option in the tzrec port of TIGER.

## TL;DR

Per-level vocab pruning is **strictly weaker** than GRID's full-prefix check, but it's the **only option compatible with the new `semantic_id_path` schema** (`codebook_level, sid`) — that schema can't reconstruct per-item SID tuples.

## What each rule actually checks

| | GRID's current design (`_check_valid_prefix`) | Per-level vocab pruning |
|---|---|---|
| Rule | Prefix `(c₀, …, c_h)` must equal the first `h+1` codes of **at least one real item** | Just-emitted code `c_h` must appear **somewhere** at hierarchy `h` |
| Data needed | Full per-item table `(num_items, num_hierarchies)` | Per-level mask `(num_hierarchies, codebook_width)` bool |
| Cross-hierarchy consistency | **Yes** | **No** — each level checked alone |

## Concrete example

Catalog of 4 items:

| Item | SID |
|---|---|
| A | (0, 1, 2, 3) |
| B | (0, 1, 2, 7) |
| C | (5, 1, 2, 9) |
| D | (5, 8, 4, 11) |

Per-level vocab derived from this catalog:

| codebook_level | sids used |
|---|---|
| 0 | {0, 5} |
| 1 | {1, 8} |
| 2 | {2, 4} |
| 3 | {3, 7, 9, 11} |

Beam emits prefix `(0, 8)` at step 2:

- **GRID**: no item starts with `(0, 8)` → **prune**.
- **Per-level vocab**: `8 ∈ {1, 8}` at level 1 → **keep going**. Could complete to `(0, 8, 4, 11)` — every code is in its level's vocab, but no real item has this SID. **Hallucination missed.**

## Cost comparison

`N` = catalog size, `K` = beam width, `H` = hierarchies, `C` = codebook width.

| | GRID full-prefix | Per-level vocab |
|---|---|---|
| Buffer memory | `N × H × 8 B` (e.g. 1M items × 4 × 8 = 32 MB) | `H × C` (e.g. 4 × 256 = 1 KB) |
| Compute per beam step | `O(N × K × H)` — broadcast catalog against beam prefixes | `O(K × C)` — single `masked_fill_` |
| 1M items, K=10, H=4 | ~40M comparisons per step | ~2.5K ops per step |
| Scales with catalog size? | **Yes** (linear in N) | **No** (independent of N) |

GRID's code itself flags the full-prefix scan as a TODO at `tiger_generation_model.py:215` because of this cost.

## Why per-level fits the new parquet schema

Your new `semantic_id_path` has only two columns: `codebook_level`, `sid`. **There is no per-item grouping key.** Implications:

- Per-level vocab: **supported** — just project rows to a `(num_hierarchies, codebook_width)` bool mask.
- Full-prefix: **not supported** — would need a third column (e.g. `item_id` or `tuple_id`) to re-group rows into full SIDs.

If you want GRID-strength constraint back, the parquet needs a grouping column. With only `(codebook_level, sid)`, you're committed to per-level.

## Where per-level still helps vs no constraint at all

Per-level is weaker than GRID's check but **not useless** vs unconstrained beam search:

- Catches "this code never appears at this level" hallucinations — most useful for **sparse hierarchies** (e.g. the dedup digit, where only a small subset of `[0, codebook_width)` is actually used).
- Cost is one `masked_fill_` per decode step — effectively free.
- Doesn't help against cross-level hallucinations (the `(0, 8, …)` example above).

## Three-way summary

| Approach | Buffer | When to pick |
|---|---|---|
| GRID full-prefix | Per-item `(N, H)` | Catalog ≤ few 100K items AND you need provably-real SIDs at decode time |
| **Per-level vocab pruning** | `(H, C)` bool mask | Catalog > 1M items but you still want some pruning; **matches the new `(codebook_level, sid)` schema** |
| No constraint + offline join | None | You trust the offline retrieval step to filter hallucinated SIDs; simplest model |

Given your three design changes (history pre-encoded, schema = `(codebook_level, sid)`, beam search returns SIDs for offline retrieval), the realistic choice is between **per-level vocab pruning** and **no constraint**. The full-prefix option is off the table because the schema doesn't carry the data needed for it.
