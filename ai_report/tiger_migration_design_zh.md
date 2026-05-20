# TIGER → TorchEasyRec 迁移设计

状态：设计草案。**本文档不包含代码** — 这是一个结构化方案，解决在编写迁移 PR 之前需要回答的问题。

## 0. 范围

我们将 `GRID/TIGER_implementation.md` 的 **§2.3 – §4.4** 移植到 TorchEasyRec 中。具体而言：

| 本设计覆盖的范围 | 不在范围内 |
|---|---|
| §2.3 — 模型输入契约（`input_ids`、`attention_mask_encoder`、`user_id`、目标 SID） | §1 — Hydra 配置（由 proto 替代，见下方 §2） |
| §2.4 — 三重职责的 attention mask | §2.1 — TFRecord 读取器（由 tzrec 的 `ParquetReader`/`OdpsReader` 替代） |
| §3.x — 模块组合（encoder、decoder、SID embedding 表、BOS/SEP、decoder MLP、T5MultiLayerFF、可选用户 embedding） | §2.2 — `collate_with_sid_causal_duplicate`（§3.4 中提出的离线预展开） |
| §4.1 — encoder 前向传播（偏移技巧 + SEP 注入） | §5 — Lightning 训练循环（由 tzrec 的 `TrainPipelineSparseDist` 替代） |
| §4.2 — decoder 前向传播 + 逐层级 CE loss 求和 | §6 — 参数量计算（仅供参考） |
| §4.3 — 约束波束搜索 | §7 — 配置旋钮（整合到 proto 中，见下方 §2） |
| §4.4 — `SIDRetrievalEvaluator`（`NDCG`/`Recall@K`） | |

目标形态：一个新的 `Tiger` 子类继承自 `tzrec.models.model.BaseModel`（`tzrec/models/tiger.py`），由新的 `tzrec/protos/models/tiger.proto` 消息驱动，完全参与 `ai_report/TZREC_PIPELINE.md` 中记录的训练/评估/预测生命周期。

---

## 0.1 版本范围 — v1 包含什么、v2 延后什么

本设计覆盖 **v1**。以下特性有意**延后到 v2**，以保持首版实现的精简：

| 延后特性 | 它会带来什么 | 为何从 v1 延后 |
|---|---|---|
| **约束波束搜索** | 波束搜索剪枝那些前缀不与任何真实 item 匹配的波束，在解码时消除幻觉的 SID。GRID 的 `_check_valid_prefix` 是参考。 | 需要逐 item 或逐前缀的查找结构（参见 `tiger_semantic_id_path_optimized_design.md` 中通过字典序排序 `(N, H)` 张量给出的可扩展设计）。生命周期（训练时从磁盘引导、评估/预测时从 checkpoint 恢复）需要模式检测的管道工作；在剪枝行为真正交付之前我们不希望引入这些工作。 |
| **`semantic_id_path` proto 字段** | 从上游 `SidRq*` 预测输出引导前缀有效性表。 | 仅供约束波束搜索使用。v1 中删除；v2 中与所选剪枝数据结构一同重新引入。 |
| **`codebooks` 持久化 buffer** | 前缀有效性表的内存表示（在 v1 草稿中曾兼作 item_id → SID 查找表，但该角色现已被预先 SID 编码的数据废弃；见下方 §3）。 | 同上 — 仅为支持剪枝而存在。 |
| **模式检测机制**（向 `BaseModel.__init__` 传递 `Mode`） | 让模型知道是从磁盘引导（训练）还是从 checkpoint 恢复（评估/预测）。 | 仅 codebooks-buffer 生命周期需要。没有该 buffer，v1 在 `__init__` 中就没有依赖模式的行为。 |
| **模型内的 Item → SID 查找** | GRID 通过 `map_sparse_id_to_semantic_id` 进行的运行时目录查找。 | v1 通过离线预编码数据规避此问题：训练/评估/预测的 parquet 行直接携带 SID token（见 §3.1）。模型从不接触原始 `item_id`。 |

**v1 的波束搜索行为**：纯无约束 top-K 波束解码。输出 `(B, K, num_hierarchies)` 的 SID 元组 + `(B, K)` 分数。一个独立的**离线检索步骤（不在本设计范围内）**负责把每个发出的 SID 与目录 join 以恢复 item ID — 它会自然地丢弃任何不能解码为 item 的幻觉 SID。评估时的 `recall@K` / `ndcg@K` 指标基于与持出 ground-truth SID 的精确 SID 匹配工作，因此即使没有约束波束搜索它们也保持良定义（幻觉波束自然计为未命中）。

**前向兼容承诺**：以下记录的每个 v1 设计决策都被选择为：添加 v2 特性是纯加法的 — proto、模型类、磁盘行 schema、引擎集成都没有破坏性变更。具体地：

- `Tiger` proto 消息将在 v2 中获得 `semantic_id_path` 和 `constrained_beam_search` 字段，不重新编号或删除现有字段。
- 模型类将获得 `codebooks` buffer 和 `_check_valid_prefix` 等价辅助方法，不改变 `predict()` / `loss()` / `update_metric()` 签名。
- 数据 parquet schema（SID 编码的历史 + 标签）在 v1 与 v2 之间保持完全相同。

延后设计工作的参考文档：
- `ai_report/tiger_beam_search_pruning_comparison.md` — 比较剪枝策略（GRID 全前缀 vs 逐层级词表 vs 无约束）。
- `ai_report/tiger_semantic_id_path_optimized_design.md` — 在任意 `(H, C)` 规模下做全前缀剪枝的可扩展字典序排序 `(N, H)` 张量设计。

---

## 1. 架构映射一览

| GRID 概念 | TorchEasyRec 对应物 | 备注 |
|---|---|---|
| `Hydra @package _global_` YAML | `tzrec.protos.models.tiger.Tiger` proto | 见 §2。 |
| `LightningModule` + `TransformerBaseModule` | `tzrec.models.model.BaseModel` 子类 | 见 §5。 |
| Hydra `_target_` 实例化 | 通过 `BaseModel.create_class(name)` 的类注册表 | 使用注册表元类声明时自动注册。 |
| `SemanticIDDatasetConfig` + `TFRecordIterator` + `map_sparse_id_to_semantic_id` | `ParquetDataset`/`OdpsDataset` + `feature_configs`（携带**已 SID 编码**整数 token 的 sequence_feature） — **v1 中无运行时 item→SID 查找** | 见 §3。 |
| `collate_with_sid_causal_duplicate`（在线连续子序列增强） | **已放弃。** 每用户序列 → 一个训练对（完整历史 → 持出最后一个 item） | 见 §3.4。 |
| `NextKTokenMasking(next_k=num_hierarchies)` 标签变换 | 一个 `list<int64>` 标签列 = 直接的持出 item SID 元组（无需映射） | 见 §3.1。 |
| GRID `Batch`（`SequentialModelInputData` + `SequentialModuleLabelData`） | `tzrec.datasets.utils.Batch`，含稀疏特征 KJT + 标签张量 | 见 §4。 |
| `T5EncoderModel` + `T5Stack`（decoder） | 相同的 HF 类，在 `__init__` 中实例化 | 见 §5.2。 |
| `item_sid_embedding_table_encoder`（1024×128，填充形式 `max × num_hierarchies`） | `self.sid_embedding`，一个普通 `nn.Embedding(sum(codebook_sizes), embed_dim)` — 紧凑形式，无填充行。**不是** TorchRec `EmbeddingCollection`。 | 见 §5.3。 |
| `decoder_mlp: ModuleList[Linear]` 每层级一个 | 相同的 `ModuleList` 稠密 `Linear` 头 | 见 §5.4。 |
| 约束波束搜索使用的 `codebooks` 张量 | **延后到 v2** — 见 §0.1 | — |
| `eval_step` 运行 `generate` + 检索指标 | `update_metric(predictions, batch)` 在 `torch.no_grad()` 内调用 `generate` | 见 §7。 |
| GRID `ckpt_path` + Lightning checkpoint | tzrec `model_dir/model.ckpt-N/`（由引擎处理） | 无需模型特定处理。 |

---

## 2. Proto 定义

### 2.1 新文件：`tzrec/protos/models/tiger.proto`

定义单个消息 `Tiger`，携带所有 TIGER 特定的配置旋钮。**字段命名遵循 `sid_model.proto` 惯例**（proto2，`optional ... [default = X]`，snake_case，逗号分隔字符串表示逐层列表，布尔字段不加 `should_` / `do_` 前缀），使得同时包含 `SidRqvae` 块和下游 `Tiger` 块的配置读起来端到端一致。

从 `SidRqvae`/`SidRqkmeans` 直接借鉴的两个设计选择：

- **`codebook` 为单个逗号分隔字符串**，而非两个数值字段。`codebook: "256,256,256,256"` 表示"4 个层级，每个 256 码字"。列表长度**即**层级数；值给出逐层级码字数。这样做：(a) 消除了 `num_hierarchies × codebook_width` 一致性检查的陷阱，(b) 免费支持非均匀码本（如 `"256,256,256,128"`——去重位只需 128 个桶），(c) 与上游 SID 生成器配置声明码本的方式完全匹配——用户直接复制粘贴同一字符串。
- **`hidden_dims` 为单个逗号分隔字符串**描述前馈块，而非两个字段。`hidden_dims: "1024"` → 标准 T5 单隐层 FF；`hidden_dims: "1024,1024"` → 双隐层 `T5MultiLayerFF` 替换（旧的 `d_ff + mlp_layers` 对）。长度 = FF 隐藏层数；值 = 逐层 FF 宽度。与 `SidRqvae.hidden_dims` 一致。

#### 字段列表（proto2）

所有字段均为 `optional`，使消息向前/向后兼容。显示字段号以明确 proto 文件顺序。

**骨干网络形状**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 1 | `embed_dim` | `uint32` | `128` | 模型的隐藏宽度。替代 GRID 的 `d_model`。与 `SidRqvae.embed_dim` 对齐。 |
| 2 | `num_heads` | `uint32` | `6` | 每个 T5 block 的注意力头数。 |
| 3 | `d_kv` | `uint32` | `64` | 每头 key/value 维度（T5 标准名；保留）。 |
| 4 | `hidden_dims` | `string` | `"1024"` | 逗号分隔的 FF 隐藏宽度。长度控制 FF 块的隐藏层数；若 > 1，将每个 `T5LayerFF` 替换为 `T5MultiLayerFF`。 |
| 5 | `num_encoder_layers` | `uint32` | `4` | encoder 中 `T5Block` 的数量。 |
| 6 | `num_decoder_layers` | `uint32` | `4` | decoder 中 `T5Block` 的数量。 |
| 7 | `dropout_rate` | `float` | `0.15` | attention + FF 内的 Dropout。T5 标准名；保留。 |

**SID 结构**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 10 | `codebook` | `string` | `"256,256,256,256"` | 逐层级码字数。列表长度 → 层级数；值 → 逐层宽度。驱动 SID embedding 表大小、逐层级偏移步长以及逐层级 `decoder_mlp` 头宽度。最大值也作为占位值传给 `T5Config.vocab_size`（见 §5.2 了解为何该字段仍需设置）。必须与生成训练 parquet 中 `history_sids` 和 `label_sids` 列的上游离线 pipeline 所产出的 SID 编码相匹配（见 §3.1）。 |

**码本来源** — *（延后到 v2，见 §0.1；v1 没有 `semantic_id_path` 字段）*

**可选用户特征**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 12 | `num_user_bins` | `uint32` | `0` | `0` 完全禁用用户 ID 分支。`> 0` 启用取余哈希的 `nn.Embedding(num_user_bins, embed_dim)` 前置到 encoder 输入。 |

**前向传播开关**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 20 | `use_sep_token` | `bool` | `true` | 在 encoder 序列中每个 item 的 `n` 个 SID token 之间插入一个可学习 SEP。替代 GRID 的 `should_add_sep_token`。 |

**波束搜索 / 生成**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 30 | `beam_width` | `uint32` | `10` | 无约束自回归解码时的波束大小。替代 GRID 的 `top_k_for_generation`。 |

*（v1 省略约束波束搜索标志 — 延后到 v2，见 §0.1。字段号 31 为 v2 添加保留以保持 proto 向后兼容。）*

**输出命名** — *无 proto 字段*

v1 中已移除。predict 模式的输出列名由**模型硬编码**：`Tiger.predict()`（推理模式下）返回

```
{
  "generated_sids": (B, K, num_hierarchies) int64,   # 0-indexed（与 label_sids 一致），按边际得分降序排序
  "step_scores"   : (B, K, num_hierarchies) float32, # 所选码字在逐层级的 softmax 概率
}
```

（完整规范见 §7.2 步骤 6。）用户通过向 `tzrec.predict` / `tzrec.predict_checkpoint` 传递 `--reserved_columns user_id` 来包含来自输入 parquet 的行标识列（如 `user_id`）— 引擎的 `_write_predictions`（`tzrec/main.py:998`）会将这些保留列与模型预测合并写入输出 parquet。两侧均不需要 proto 字段。这与 `SidRqkmeans` 的惯例一致（其返回 `{"codes": ...}`，用户传 `--reserved_columns "id"` 以透传 item ID）。

**特征关联**

| # | 名称 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| 50 | `feature_groups` | `repeated FeatureGroupConfig` | — | 两个命名组：`"history"`（过去原始 `item_id` 序列）和 `"user"`（user_id 特征；仅 `num_user_bins > 0` 时需要）。tzrec 惯例；模型通过 `self.get_features_in_feature_groups(...)` 消费它们。 |

#### 字段命名变更 vs 早期"GRID 直译"方案

对于曾看过使用 GRID 风格字段名早期草案的读者，以下是重命名日志：

| 旧名（GRID 直译） | 新名（RQVAE 对齐） | 原因 |
|---|---|---|
| `d_model` | `embed_dim` | 匹配 `SidRqvae.embed_dim`；更具描述性——它是 embedding/hidden 宽度，T5 只是叫它 `d_model`。 |
| `d_ff` + `mlp_layers` | `hidden_dims`（string） | 镜像 `SidRqvae.hidden_dims`。一个字段代替两个；长度编码 `mlp_layers`，值编码逐层 `d_ff`。非均匀 FF 宽度免费获得。 |
| `num_hierarchies` + `codebook_width` | `codebook`（string） | 镜像 `SidRqvae.codebook` / `SidRqkmeans.codebook`。消除 `num_hierarchies × codebook_width` 一致性检查陷阱；非均匀码本免费获得。 |
| `should_add_sep_token` | `use_sep_token` | 按 `SidRqvae` 风格去掉动词前缀（`shared_codebook`、`normalize_residuals`、`rotation_trick`）。 |
| `top_k_for_generation` | `beam_width` | 标准 NMT 术语；更短；描述其本质。 |
| `should_check_prefix` | *（延后到 v2 作为 `constrained_beam_search`）* | — |

### 2.2 接入 `tzrec/protos/model.proto`

顶层 `ModelConfig` 是一个 `oneof model { ... }`，为每个支持的模型包含一个条目。我们添加：

```
  Tiger tiger = <next_free_field_number>;
```

并在 `model.proto` 顶部添加 `import "tzrec/protos/models/tiger.proto";`。`_pb2.py` 和 `_pb2.pyi` 通过 `bash scripts/gen_proto.sh` 重新生成（按 `CLAUDE.md` 规范）。

### 2.3 用户编写的 Pipeline 配置布局

最小化的用户侧配置如下：

```
model_dir: "experiments/tiger_amazon_beauty"
train_input_path: ".../taobao_or_amazon_prefix_pairs/*.parquet"
eval_input_path:  ".../taobao_or_amazon_eval/*.parquet"
train_config { sparse_optimizer { ... }  dense_optimizer { adam_optimizer { lr: 1e-3 } } num_epochs: 10 }
eval_config  { num_steps: ... }
data_config  { batch_size: 32  dataset_type: ParquetDataset  fg_mode: FG_NONE
               label_fields: "label_sids"  num_workers: 8 }
feature_configs { sequence_feature { feature_name: "history_sids" sequence_length: 120 ... } }    # SID-token 流，长度 = items × num_hierarchies
feature_configs { id_feature       { feature_name: "user_id"      num_buckets: ... embedding_dim: ... } }
model_config { tiger {
    embed_dim: 128
    num_heads: 6
    d_kv: 64
    hidden_dims: "1024,1024"          # 双隐藏层 T5MultiLayerFF；仅 "1024" → 标准 T5
    num_encoder_layers: 4
    num_decoder_layers: 4
    dropout_rate: 0.15
    codebook: "256,256,256,256"       # 4 层级 × 每层 256 码字
    num_user_bins: 0                  # 禁用 user-id 分支
    beam_width: 10
    feature_groups { ... }
    # v1 没有 semantic_id_path / constrained_beam_search 字段 — 延后到 v2（§0.1）
} }
```

默认使用 `fg_mode: FG_NONE`，因为 TIGER 不需要即时 FG（无分桶、无组合交叉、无哈希）— 所有特征已经是整数 item ID，由模型自身在运行时转换为 SID token。若 `num_user_bins > 0` 也可接受 `FG_DAG`（开销低，且允许用户后续添加用户侧原始特征）。

---

## 3. 数据管道映射（替代 GRID §2.1、§2.2）

### 3.1 磁盘上的行契约

**每用户一行。** 每行携带用户的完整交互历史（**已编码为扁平的 SID-token 流**），加上持出的下一个 item 的 **SID 元组**。item_id → SID 映射由上游离线 pipeline 完成（不在本设计范围内）；模型从不接触原始 item ID。

| 列 | dtype | 含义 |
|---|---|---|
| `history_sids` | `list<int64>` | 用户历史，作为**扁平的 SID-token 序列**：`num_items_in_history × num_hierarchies` 个整数，码字按 `[c₀^{item₀}, c₁^{item₀}, …, c_{H-1}^{item₀}, c₀^{item₁}, …]` 连续布局。长度始终为 `num_hierarchies` 的倍数。**码字为 0-indexed** — `c_h ∈ [0, codebook_sizes[h])`（见 §3.6）。 |
| `user_id` | `int64`（可选） | 仅 `num_user_bins > 0` 时使用。 |
| `label_sids` | `list<int64>`，长度 = `num_hierarchies` | 持出最后一个 item 的 SID 元组 — `(c₀, c₁, …, c_{H-1})`，按同一约定为 **0-indexed**。 |
| `__source_id` / `__row_idx` | （由 `ParquetReader` 注入） | 行级 dataloader checkpoint 元数据。 |

`reserved_columns`（predict 模式）可包含 `user_id`，以便 writer 输出 `(user_id, generated_sids, scores)` 行；下游离线检索将这些 SID 回 join 到 item_id。

### 3.6 约定：码字以 0-indexed 存储，逐层级偏移在模型内部发生

**`history_sids` 和 `label_sids` 中的 SID 码字以 0-indexed 存储** — 每个逐层级码字位于 `[0, codebook_sizes[h])`。同一约定端到端适用：`label_sids`（输入）、`generated_sids`（§7.2 中的 predict 输出）、`target_sids`（§8.2 中的评估比较）都是 0-indexed。在任何接口上都不进行 `+1` / `-1` 转换。

到紧凑统一 embedding 表索引范围的逐层级偏移仅在**模型内部**发生于 `_add_repeating_offset_to_rows`：

```
lookup_idx[h] = c[h] + cumsum(codebook_sizes)[h]
              = c[h] + sum(codebook_sizes[:h])
```

以 `codebook: "128,256,512"` 为例的具体示例：

| 输入中的 SID 元组 | 模型内部的查找索引 |
|---|---|
| `[0, 0, 0]` | `[0, 128, 384]`（每个层级的首个码字） |
| `[1, 2, 3]` | `[1, 130, 387]` |
| `[127, 255, 511]` | `[127, 383, 895]`（每个层级的最后一个合法行） |
| `[0, 0, 0, 0]`（填充位） | mask 乘法后为 `[0, 0, 0, 0]`；见 §6.2 职责 1 |

**tzrec 中的填充识别依赖 KJT lengths，而非哨兵值。** 从 KJT lengths 通过 `lengths_to_mask` 推导出的 attention mask 才是标记填充位的东西；填充位中实际存放的值无关紧要 — 无论如何都会在 mask 乘法中被置零（职责 1）并被 attention 阻止（职责 3）。因此 `to_padded_dense` 恰好以 `0` 填充（而 `0` 也是层级 0 的合法码字 0）这一点并无危害：冲突存在但 mask 阻止了这些位置的任何贡献。

为何采用 0-indexed：

1. **约定一致。** PyTorch（`embedding_dim`、`num_embeddings`）、`F.cross_entropy`（0-indexed 类别目标）、`torch.argmax`（0-indexed 结果）、HuggingFace tokenizer（0-indexed `vocab_size`）——栈中每一个相邻部件都使用 0-indexed 整数 ID。采用 1-indexed 会迫使在每个边界上都进行 `+1` / `-1` 转换（输入偏移、CE 目标、输出还原）——额外三个算术操作，额外三个可能产生 off-by-one bug 的地方。
2. **热路径上更少的算术操作。** 偏移技巧变为 `mask * (c + offsets)` — 比 `mask * ((c-1) + offsets)` 少一个操作。
3. **与 `SequenceFeature` 语义匹配。** 当 `history_sids` 声明为 `sequence_feature` 且 `num_buckets = max(codebook_sizes)` 时，自然的桶范围是 `[0, num_buckets)` — 声明 `num_buckets: 257` 以为"填充=0"留一个哨兵位是别扭的。
4. **相对 1-indexed 无信息损失。** 填充识别不需要哨兵值，因为 KJT lengths 已经编码了该信息；两种约定下 attention mask 都能正确传播。选择 0-indexed 在获得同样正确性的同时减少了索引约定的漂移。

### 3.2 `feature_configs`

- **`history_sids`** 声明为 int64 SID token 的 `sequence_feature`：
  - 序列最大长度 = pipeline 的 `sequence_length` 旋钮，以 **token** 为单位（不是 item，例如 30 items × 4 hierarchies = `sequence_length: 120`），
  - 稀疏 / id-feature 类型，
  - 放在 `data_group: "history"` 中。每 item 的边界是隐式的：每 `num_hierarchies` 个连续 token 属于同一个 item — 模型在 SEP 注入期间内部进行逐 item 的 reshape（§6.2）。
- **`user_id`** 作为 `id_feature` 放在 `data_group: "user"` 中，`num_buckets = num_user_bins`，`embedding_dim` 被我们忽略 — 我们在模型内部使用自己的取余哈希用户 embedding，而非 tzrec `EmbeddingGroup` 产出的 embedding。（替代方案：依赖 tzrec 的 `EmbeddingGroup` 并移除模型的 `user_embedding`；见 §10 — 待决事项。）
- **无 `label_fields` 的 feature_config 条目** — `data_config.label_fields = "label_sids"` 足以让 parser 抓取 int64 列表并产出正确张量。

### 3.3 落入 `Batch` 的内容

经 `DataParser.parse` + `to_batch` 后，模型接收的 `Batch` 包含：

- `batch.sparse_features["history"]`：一个 `KeyedJaggedTensor`，含键 `"history_sids"`。每样本的 lengths 描述该历史有多少个 SID token（始终为 `num_hierarchies` 的倍数）。values 是 **0-indexed** 的逐层级码字，位于 `[0, codebook_sizes[h])`（见 §3.6 约定）。来自 `to_padded_dense` 的填充位保持为 `0` — 无危害，因为无论是否与层级 0 的真实码字 0 冲突，attention 都会对其进行掩蔽（职责 3）。模型在查找时应用 `c + cumsum_h` 偏移变换（§6.2 职责 1），将其映射到统一的 `sum(codebook_sizes)` 行的 SID embedding 表中。
- `batch.sparse_features["user"]`：一个含键 `"user_id"` 的 `KeyedJaggedTensor`，每样本 length 为 1。若 `num_user_bins == 0` 则跳过。
- `batch.labels["label_sids"]`：`(B, num_hierarchies)` 的 int64 张量（`DataParser` 将 `list<int64>` 标签列物化为 `label_sids.values` + `label_sids.lengths`；因为每行长度固定，模型 reshape/`view` 为 `(B, num_hierarchies)`）。
- `batch.checkpoint_info`：行级恢复元数据（由引擎处理；模型忽略）。

### 3.4 不做连续子序列增强

**决策：完全放弃 `collate_with_sid_causal_duplicate`。** 每个用户成为一条训练行（完整历史 → 持出最后一个 item，即 §3.1 的行契约）。不做在线展开、不做离线预展开工具、不做自定义 `BaseDataset` 子类。

理由：

- **框架适配。** tzrec 的管线假设"一个输入行 = 一个 batch 样本"；增强策略要么与 `ParquetReader` 的行级 checkpointing 冲突（每输入行产出可变数量记录的自定义 dataset 会打破 `(__source_id, __row_idx)` 记账），要么需要在每次数据刷新时重新生成 parquet。
- **运维简单。** 工作流中无预计算步骤。在新 SID 生成上重新训练只需一条 `torchrun -m tzrec.train_eval ...` 命令。
- **足够的信号。** TIGER 的放大技巧在原始序列短且稀少时最有价值。对于 Amazon Beauty / Sports / Toys 数据集及类似生产规模的用户序列，留一法基线已经很强；GRID 论文自身报告的主要数字并未显示可归因于增强的逐步方差。

接受的权衡：相同原始数据下，训练看到的 `(prefix, next-item)` 样本比 GRID 少。若与 GRID 参考出现 benchmark 差距，可重新引入增强：(a) 在第一个里程碑落地后作为自定义 dataset 子类重新实现，或 (b) 在 parquet 准备时作为独立的一次性离线工具增强——两者均暂缓。

### 3.5 为何使用 `fg_mode: FG_NONE`（或空 FG 的 `FG_DAG`）

TIGER 不分桶、不哈希、不做组合交叉。历史已由离线 pipeline 进行 SID 编码；标签已是 SID 元组。encoder 只是通过自己的表查找 SID-token embedding（模型侧的偏移技巧；§6.2）。FG 对该模型是空操作。

---

## 4. 张量表示：GRID 名称 → tzrec 名称

此表是本文档其余部分的罗塞塔石碑。

| GRID（`SemanticIDEncoderDecoder`） | tzrec `Batch` 来源 | 形状 | 类型 |
|---|---|---|---|
| `input_ids`（逐层级 SID token，经 GRID `map_sparse_id_to_semantic_id` 后） | 直接为 `batch.sparse_features["history"]["history_sids"]` — 已由离线 pipeline 进行 SID 编码。KJT values 是扁平 SID token；KJT lengths 给出每样本 token 数（始终为 `num_hierarchies` 的倍数）。模型通过 `to_padded_dense` 致密化为 `(B, L_tokens)`。 | `(B, L_tokens)` | int64 |
| `attention_mask_encoder`（1=真实，0=填充） | 通过标准 `lengths_to_mask` 技巧从 KJT lengths 派生 → **直接得到** `(B, L_tokens)`，**无需**逐 item-到-token 的展开步骤。 | `(B, L_tokens)` | int64 / bool |
| `user_id` | `batch.sparse_features["user"]["user_id"]`（KJT，每样本 length 1）→ squeeze 为 `(B,)` | `(B,)` | int64 |
| `future_ids`（目标 item 的 SID）— 仅训练 | `batch.labels["label_sids"]` reshape 为 `(B, num_hierarchies)` — **无需 codebooks 查找**，标签已经是 SID 元组。 | `(B, num_hierarchies)` | int64 |
| `attention_mask_decoder` | `None`（目标 SID 始终满长；与 GRID 一致） | — | — |
| `codebooks`（用于 `_check_valid_prefix` 的逐 item SID 表） | *（延后到 v2 — 见 §0.1；v1 没有 `codebooks` buffer）* | — | — |

**关键洞察**：在 tzrec 中填充由 **KJT lengths** 表示，而非 GRID 的哨兵值 `-1`。因此 `attention_mask_encoder` 从 `lengths` 构建而非 `(input_ids != -1)`。功能上完全等价；仅构建步骤不同。

**第二个关键洞察（v1 特有）**：因为历史已预先 SID 编码且标签直接携带 SID 元组，encoder 管线比 GRID 短一步。相对 GRID 消失的步骤：`history_item_id → SID` 查找（曾是 `encoder_forward_pass` 的步骤 2）和 `label_item_id → SID` 查找（曾是 `model_step` 的步骤 5）。仅当我们希望约束波束搜索查询逐 item 表时，它们才在 v2 中重新出现。

---

## 5. 模型架构（替代 GRID §3.x）

### 5.1 类骨架（无代码，仅结构）

新类 `Tiger(BaseModel)` 位于 `tzrec/models/tiger.py`。构造函数签名遵循 `BaseModel` 契约：

`__init__(model_config, features, labels, sample_weights, sampler_type, **kwargs)`

`__init__` 内部的操作顺序：

1. 读取 `self._model_config = model_config.tiger`（proto 子消息）。
2. 从 proto 暂存超参数：
   - 解析 `codebook`（逗号分隔字符串）为整数列表 → 推导 `num_hierarchies = len(codebook_sizes)` 和逐层级宽度 `codebook_sizes[h]`。SID embedding 表大小取 `sum(codebook_sizes)`（紧凑形式，无填充行；见 §5.3）。预计算 `self.code_offsets = torch.tensor([0, *cumsum(codebook_sizes[:-1])])` 供逐层级索引偏移使用。`T5Config.vocab_size` 占位值为 `1`（空操作；见 §5.2）。
   - 解析 `hidden_dims` 为整数列表 → 推导 `mlp_layers = len(hidden_dims_list)`。若 `mlp_layers == 1`，保留标准 `T5LayerFF`；若 > 1，使用列表中的宽度执行 `T5MultiLayerFF` 替换。
   - 读取 `embed_dim`、`num_heads`、`d_kv`、`num_encoder_layers`、`num_decoder_layers`、`dropout_rate`、`num_user_bins`、`beam_width`、`use_sep_token`。
3. 构建下方 §5.2–§5.7 的模块。

*（v1 没有 codebooks buffer — 按 §0.1 延后到 v2。模型在 `__init__` 时没有要从中引导的磁盘 parquet。）*

注意：TIGER 的 embedding 表保持为普通 `nn.Embedding` 实例（而非 TorchRec 的 `EmbeddingCollection`）。在 TIGER 的规模下（不到 1MB 的 embedding 表），TorchRec 的分片 / 稀疏优化器机制提供不了真正的收益反而增加摩擦 — 偏移变换（§3.6）需要被迫移到查表之前的 KJT 操作中。普通 `nn.Embedding` 让查表保持为单个张量操作，并将偏移技巧保留在模型内部。

**无需重写 `sparse_parameters()`。** `BaseModel.sparse_parameters()`（`tzrec/models/model.py:151`）仅收集 `EmbeddingBagCollectionInterface` / `EmbeddingCollectionInterface` 的实例；普通 `nn.Embedding` 会被自动忽略，因此对 TIGER 而言该函数无需任何模型侧代码即返回两个空列表。TorchRec planner 仍会运行但产出一个无分片模块的平凡计划；随后 `create_train_pipeline` 会回退到 `TrainPipelineBase`（见 `TZREC_PIPELINE.md` §2.1），这正是对 TIGER 而言的正确选择。

### 5.2 T5 encoder & decoder（对应 GRID §3.1）

从 proto 派生的超参数构建 `T5Config`。映射：`d_model = embed_dim`、`num_heads`、`d_kv`、`d_ff = max(hidden_dims_list)`、`num_layers = num_encoder_layers`（对应 `num_decoder_layers`）、`dropout_rate`。**`T5Config.vocab_size = 1`** — 这是一个真正的空操作（T5 使用 `vocab_size` 的唯一位置是 `T5Stack.__init__` 中初始化 `embed_tokens` 的大小，而我们随后立即删除该表；一旦 `embed_tokens` 被移除，T5 forward 始终使用 `inputs_embeds=...`）。设为 `1` 可避免在构造时临时分配一个即将丢弃的 embedding 张量。

- `self.encoder_t5 = T5EncoderModel(config=enc_cfg)`
- `self.decoder_t5 = T5Stack(dec_cfg, embed_tokens=<临时>)` — `T5Stack` 构造函数需要一个 `embed_tokens`；传入一个临时 `nn.Embedding(1, embed_dim)` 后立即丢弃。

然后**删除 T5 内部的 embedding 表**：

- `delete_module(self.encoder_t5, "shared")`
- `delete_module(self.encoder_t5.encoder, "embed_tokens")`
- `delete_module(self.decoder_t5, "embed_tokens")`

并**从头重新初始化**其余 encoder/decoder 权重（我们不加载预训练 T5 权重 — TIGER 从未这样做）。对剩余子模块递归执行 `reset_parameters` 与 GRID 行为一致。

### 5.3 SID embedding 表

单个紧凑 embedding：

```
self.sid_embedding = nn.Embedding(
    num_embeddings = sum(codebook_sizes),      # 不是 max × num_hierarchies——无填充行
    embedding_dim  = embed_dim,
)
```

总行数是逐层级码字数的**总和**，而非 GRID 使用的填充后形式 `max × num_hierarchies`。层级 `h` 的码字占据一个连续块：

```
层级 h 拥有行 [cumsum_h, cumsum_h + codebook_sizes[h])
其中 cumsum_h = sum(codebook_sizes[:h])
```

因此对于 `codebook: "128,256,512"`，表有 `128 + 256 + 512 = 896` 行（而填充形式需 `3 × 512 = 1536` 行——小 42%）。对于均匀码本如 `"256,256,256,256"`，两种形式都产生 1024 行；紧凑形式永远不会更大。

偏移张量 `self.code_offsets = torch.tensor([0, *cumsum(codebook_sizes[:-1])])`（shape `(num_hierarchies,)` int64）在 `__init__` 中预计算一次，由 `_add_repeating_offset_to_rows` 使用（§3.6 给出公式）。

**encoder 和 decoder 共用。** 同一个 `self.sid_embedding` 在 `encoder_forward_pass`（以 embedding 历史 SID 流）和 `decoder_forward_pass`（以 embedding teacher-forced 目标 SID + BOS token）中被查阅。encoder–decoder 共享是 TIGER 中唯一的"绑定"关系，且为硬编码 — 无 proto 标志控制。为何丢弃 GRID 的 `weight_tying` 标志，见 `tiger_weight_tying_explanation.md`。

### 5.4 逐层级 decoder 头（对应 GRID §3.5）

`self.decoder_mlp = nn.ModuleList([nn.Linear(embed_dim, codebook_sizes[h], bias=False) for h in range(num_hierarchies)])`。无偏置（与 GRID 一致）。每个头输出**其自身逐层级宽度** — 对均匀情况所有头为 `Linear(embed_dim, 256)`；对 `codebook: "256,256,256,128"` 最后一个头为 `Linear(embed_dim, 128)`。每个头在恰好一个解码步使用。

### 5.5 BOS / SEP 参数、可选用户 embedding（对应 GRID §3.2、§3.6）

- `self.bos_token = nn.Parameter(torch.randn(1, embed_dim))` — decoder 起始 token。
- `self.sep_token = nn.Parameter(torch.randn(1, embed_dim))` 若 `use_sep_token`，否则 `None`。
- `self.user_embedding = nn.Embedding(num_user_bins, embed_dim)` 若 `num_user_bins > 0`，否则 `None`。

### 5.6 codebooks buffer — *延后到 v2*

v1 中模型**没有 `codebooks` 持久化 buffer**，且 `__init__` 时不读取任何磁盘上的 SID 表。GRID 中需要逐 item 查找的两个用例都被消除：

- 历史 → SID 展开：由上游 pipeline 离线完成；v1 模型直接从 batch 读取 `history_sids`。
- 标签 → 目标 SID：同样 — `batch.labels["label_sids"]` 已携带 SID 元组。

第三个 GRID 用例（通过 `_check_valid_prefix` 进行约束波束搜索前缀检查）连同支持它的 buffer 一并延后到 v2。见 §0.1 以及在以下文档中的设计探索：

- `ai_report/tiger_beam_search_pruning_comparison.md` — GRID 全前缀 vs 逐层级词表剪枝 vs 无约束。
- `ai_report/tiger_semantic_id_path_optimized_design.md` — 在任意 `(H, C)` 规模下支持全前缀剪枝、每波束步 `O(K · log N)` 计算的可扩展字典序排序 `(N, H)` 张量设计。

**前向兼容影响**：v1 没有此 buffer 意味着 `BaseModel.__init__` 中无需模式检测管道工作（每种模式都走相同构造器）；无 `load_state_dict` 重写；启动时无 rank-0 parquet IO。v2 的 PR 将连同 `semantic_id_path` proto 字段以及所选剪枝实现一起添加这些内容。

### 5.7 `T5LayerFF → T5MultiLayerFF` 替换（对应 GRID §3.4）

仅当 `len(hidden_dims_list) > 1` 时触发。辅助方法 `_swap_t5_ff_with_multilayer(hidden_dims_list)` 在构造后遍历 `self.named_modules()`，定位每个 `T5LayerFF` 实例，并替换为 MLP 宽度恰为 `hidden_dims_list` 的 `T5MultiLayerFF`。`T5MultiLayerFF` 本身复用自一个小的内部模块文件（`tzrec/modules/tiger_ff.py`，概念位置），镜像 GRID 的 `T5MultiLayerFF`（layernorm + dropout + `MLP[embed_dim → hidden_dims[0] → ... → hidden_dims[-1] → embed_dim]` + 残差）。当 `len(hidden_dims_list) == 1` 时，不执行替换，底层 T5 使用标准 `T5LayerFF`，`d_ff = hidden_dims_list[0]`。

---

## 6. 前向传播（替代 GRID §4.1、§4.2）

### 6.1 入口点：`predict(self, batch)` — 训练与 teacher-forced 评估

`BaseModel.predict` 是 `TrainWrapper.forward` 调用的方法（见 `TZREC_PIPELINE.md` §1.5）。对于 TIGER，`predict` 执行：

1. **从 batch 提取输入。**
   - 从 `batch.sparse_features["history"]`：读取 `values`（扁平 SID-token 流 — **已是逐层级码字**，无需 item-id 查找）和 `lengths`（`(B,)`，每样本 token 数，始终为 `num_hierarchies` 的倍数）。
   - 通过 `KeyedJaggedTensor.to_padded_dense` 致密化 → `(B, L_tokens)` 的 int64 张量。填充位置持有值 `0`（默认填充）；这些位置上的逐层级语义正确性由 attention mask 保护（§6.2 职责 1）。
   - 从 `batch.sparse_features["user"]`（若已配置）：读取 user_id（每样本单个整数，squeeze 为 `(B,)`）。
   - 从 `batch.labels["label_sids"]`（仅训练/评估 — 纯推理时为 `None`；见 §7）→ reshape 为 `(B, num_hierarchies)`。这就是 `future_ids`；无需进一步转换。

2. **构建 encoder attention mask。**
   - 从 KJT `lengths` → `attention_mask_encoder: (B, L_tokens)`，通过标准 `lengths_to_mask`。无需逐 item-到-token 展开步骤（lengths 已经以 token 为单位）。
   - 这就是 GRID 的 `attention_mask_encoder` 张量，承担**三重职责**（从 §2.4 原样保留）：

#### 6.2 `attention_mask_encoder` 的三重职责（保留 §2.4）

| 职责 | 时机 | 实现备注 |
|---|---|---|
| 1 — 保护 SID embedding 查表 | 在 `_add_repeating_offset_to_rows` 内部 | 计算 `lookup_idx = mask * (c + self.code_offsets[h])`（在位置维度广播）。对于真实位置（mask=1）：产出 `c + cumsum_h ∈ [cumsum_h, cumsum_h + codebook_sizes[h])` — 即紧凑 `sum(codebook_sizes)` embedding 表中的正确行。对于填充位置（mask=0）：产出 `0` — 一个确定性的合法行（层级 0 的首个码字），这是安全的，因为 attention 会阻止这些位置参与 softmax（职责 3）。mask 乘法将填充-vs-真实的歧义崩塌为同一个安全的行，不论输入值为何。 |
| 2 — 随 SEP 注入同步扩展 | 在 `_inject_sep_token_between_sids` 内部 | 逻辑与 GRID 完全一致。扩展后的 mask 从 `encoder_forward_pass` 返回。 |
| 3 — 喂入 T5 self-attention | 在最终 `self.encoder_t5(...)` 调用时 | T5 将 0 转为 attention logits 上的 `-inf` 加性偏置。与 GRID 相同。 |

3. **运行 `encoder_forward_pass`**（逻辑逐字来自 GRID `:584`）：应用偏移技巧 → SID embedding 查表 → 可选 SEP 注入 → 可选 user-id 前置 → T5 encoder。返回 `(encoder_output, attention_mask_for_encoder)`。

4. **运行 `decoder_forward_pass`**，传入 `future_ids`（来自步骤 1 — 已经是来自 `batch.labels["label_sids"]` 的 SID 元组）、`encoder_output` 和扩展后的 encoder mask。返回 decoder 隐藏状态 `(B, num_hierarchies + 1, embed_dim)`（`+1` 是 BOS 前置）— 丢弃最后一个位置以与目标对齐。

5. **返回 `predictions` 字典**，包含下游代码需要的字段：
   - `decoder_hidden`：`(B, num_hierarchies, embed_dim)` 隐藏状态（去除 BOS 后）。`loss()` 方法读取此项并应用逐层级头。
   - `encoder_output` 和 `attention_mask_for_encoder`：可选，仅当引擎还会调用 `generate()`（如评估时）包含，避免重复计算。见 §7.2。
   - （推理时，额外包含 `generated_ids` 和 `marginal_probs` — 见 §7。）

### 6.3 Loss（`loss(self, predictions, batch)`）

实现 GRID §4.2 的训练目标：

- 令 `target_sid = batch.labels["label_sids"].view(B, num_hierarchies).long()`。按 §3.6 约定这是 **0-indexed** 的；逐层级头产出的 0-indexed logits 与之范围一致。无需转换。
- 对 `h in range(num_hierarchies)`：
  - `logits_h = self.decoder_mlp[h](predictions["decoder_hidden"][:, h, :])` → `(B, codebook_sizes[h])`，索引范围 `[0, codebook_sizes[h])`。
  - `loss_h = F.cross_entropy(logits_h, target_sid[:, h])`。
- 返回 `{"tiger_ce": sum(loss_h for h in range(num_hierarchies))}`。

返回的字典键需要**精确匹配** proto 级 `LossConfig`（或我们自己的约定）所知晓的一个条目，用于指标初始化。约定建议：选择单个字符串如 `"tiger_ce"`，以便 `init_loss/init_metric` 可注册 `MeanMetric("tiger_ce")`。

`init_loss(self)` — 对 TIGER 而言我们实际上没有需要注册的 loss 模块（损失是纯函数式的），因此这可以是空操作，或注册一个占位的 `nn.Identity` 以满足任何框架内省需求。`TrainWrapper` 在训练开始前无论如何都会调用 `init_loss`。

---

## 7. 推理 / 波束搜索（替代 GRID §4.3）

### 7.1 波束搜索运行的位置

波束搜索在两种情况下运行：

1. **评估期间**（在 `update_metric` 内）— 我们需要 predictions 来评分检索指标。`update_metric` 由引擎在 `_evaluate`（`main.py:_evaluate`）内的 `torch.no_grad()` 下调用。实现：调用 `self.generate(batch)` 方法，1:1 镜像 GRID 的 `generate` 但从 `Batch` 读取输入（与 `predict()` §6.1 步骤 1–3 相同的提取逻辑）。
2. **预测期间**（当引擎处于 `predict` 或 `predict_checkpoint` 时）— `predict_checkpoint` 无需额外工作即可运行，因为它通过 `PredictWrapper.forward → BaseModel.predict` 即时执行；我们只需让 `predict()` 在 `batch.labels` 为空时（即推理）行为不同：调用 `self.generate(batch)` 而非 teacher-forced decoder，并将结果放入 predictions 字典。

### 7.2 `generate(self, batch)` — GRID `:738` 的移植

伪代码级映射（无代码）：

1. 从 batch 提取序列 + user_id，与 §6.1.1 完全相同。
2. 运行一次 `encoder_forward_pass` → `(encoder_output, attention_mask_for_encoder)`。
3. 初始化 `generated_sids=None`、`marginal_log_prob=None`、`step_scores=None`（`(B, K, num_hierarchies)` 累加器）、`past_key_values = EncoderDecoderCache(self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache())`。
4. 对 `h in range(num_hierarchies)`：
   - 若 `h > 0`：reshape 并 repeat-interleave encoder 输出/mask 以匹配当前波束宽度（`beam_width`）。
   - 调用 `decoder_forward_pass`，`use_cache=True` 且使用累积的 `past_key_values`。
   - 读取最后一个位置的隐藏状态 → 应用 `self.decoder_mlp[h]` → logits `(B*K, codebook_sizes[h])`，自然为 0-indexed。
   - 调用 `_beam_search_one_step`（逻辑改编自 GRID `:253`，**v1 中移除 `_check_valid_prefix` 掩码分支**）：softmax → 逐步概率；排序/top-K；步骤 0 的波束扩展；后续步骤通过 `past_key_values.reorder_cache(replace_indices)` 的 KV-cache 重排；更新 `generated_sids`（始终为 0-indexed — 与磁盘约定一致）和 `marginal_log_prob`。**同时将每个波束所选码字的逐步 softmax 概率**记入 `step_scores[:, :, h]`。
5. **无需还原步骤** — `generated_sids` 始终为 0-indexed，与磁盘输入约定（§3.6）一致。模型的 argmax 输出与输入数据端到端处于同一索引空间。
6. 返回：

   | 输出键 | Shape | Dtype | 含义 |
   |---|---|---|---|
   | `generated_sids` | `(B, K, num_hierarchies)` | int64 | Top-K 波束以 0-indexed SID 元组表示（与 `label_sids` 和 `history_sids` 同一约定），**按 `marginal_log_prob` 降序排序**。 |
   | `step_scores` | `(B, K, num_hierarchies)` | float32 | 逐层级上所选码字的 softmax 概率，波束顺序与 `generated_sids` 一致。供下游 / 诊断检查波束从哪里开始偏离真实值。 |

发出的 `generated_sids` 是预测模式下**模型的最终输出**（推理模式下从 `Tiger.predict()` 返回；通过 `predict_step` 传递给 writer）。下游离线检索（不在范围内）将每个 SID 元组与目录 join 以映射回 item ID；不对应任何 item 的 SID 被 join 自然跳过，从而过滤掉无约束搜索可能产生的幻觉波束。

### 7.3 约束前缀检查 — *延后到 v2*

GRID 的 `_check_valid_prefix`（及其使用逐 item 或逐前缀表的 tzrec 等价实现）延后到 v2 — 见 §0.1。v1 波束搜索发出 decoder 预测的任何 SID 元组；离线检索步骤负责过滤。v2 设计空间见 `ai_report/tiger_beam_search_pruning_comparison.md`。

### 7.4 `tzrec.predict`（导出路径）的 JIT 脚本化注意事项

快速 `tzrec.predict` 路径（非 `predict_checkpoint`）加载由 `tzrec.export` 产出的**脚本化**模型。含有以下特征的波束搜索代码：

- 跨层级的 Python `for` 循环，
- 使用 `reorder_cache(...)` 的动态 KV-cache 重排，
- HuggingFace 的 `EncoderDecoderCache` 对象，

**不能直接 TorchScript 化**。三个逃生方案：

1. **推荐（第一个里程碑）**：仅导出 encoder。输出一个脚本化产物，从 batch dict 产出 `(encoder_output, attention_mask_for_encoder)`；在 predict 侧用 Python 做波束搜索，使用相同的 `generate` 代码路径。类似 `MatchModel` 在 tzrec 中按塔导出的方式（见 `TZREC_PIPELINE.md` §6）。具体实现：在 `export()` 编排中特殊处理 TIGER（类似 `MatchModel` 分支），导出 `TigerEncoderWrapper` 而非完整模型。
2. 使用 `predict_checkpoint` 而非 `predict` — 原始 checkpoint 路径使用 `PredictWrapper`（无 JIT），完整波束搜索原样运行。比脚本化路径慢但最简单。
3. （长期）手动将波束搜索改为 script-friendly（用两个并行 `DynamicCache` 替代 `EncoderDecoderCache`，内联循环体等）。延后处理。

proto 字段 `is_scripted_inference`（布尔，默认 `false`）可在后续控制此行为；第一个里程碑不需要。

---

## 8. 评估时评分（替代 GRID §4.4）

### 8.1 指标注册：`init_metric(self)`

读取 proto 的评估块（或硬编码，因为 TIGER 的指标集是固定的）并为每个 `(metric_name, top_k)` 对注册一个 torchmetrics 模块，镜像 GRID 的 `SIDRetrievalEvaluator`：

- 对 `k in [5, 10]`（可配置）：注册 `RetrievalNormalizedDCG(top_k=k)` 和 `RetrievalRecall(top_k=k)` 到 `self._metric_modules` 中，名称如 `"recall@5"`、`"ndcg@5"`、`"recall@10"`、`"ndcg@10"`。这些是标准 torchmetrics 检索模块（已是依赖项）。
- 同时注册 `self._metric_modules["tiger_ce"] = MeanMetric()`，使引擎的平均 loss 日志正常工作。

### 8.2 更新指标：`update_metric(self, predictions, batch, losses=None)`

引擎的 `_evaluate` 在 `torch.no_grad()` 下对每个 batch 调用此方法。实现：

1. 若提供了 `losses`（引擎会传入），`self._metric_modules["tiger_ce"].update(losses["tiger_ce"], batch_size)`。
2. 调用 `predictions = self.generate(batch)` → 返回 §7.2 步骤 6 描述的同一字典，其中 `generated_sids` 为 **0-indexed**（与磁盘约定一致）。
3. 读取 `target_sids = batch.labels["label_sids"].view(B, num_hierarchies)` — 同为 0-indexed（§3.6 约定）。两边在同一索引空间；无需转换。
4. 计算匹配图：`match = (generated_sids == target_sids.unsqueeze(1)).all(dim=2)` → `(B, K)` bool — K 个波束中是否有任何一个等于真实 SID？
5. 通过归约逐层级步评分计算逐波束边际得分：`marginal_probs = step_scores.prod(dim=-1)` → `(B, K)`（链式法则下标准的逐步概率乘积）。从 `generate()` 出来的波束已按此值降序排序。
6. 构建 torchmetrics 检索模块所需的 `(preds, target, indexes)` 三元组（与 GRID `eval_metrics.py:281-313` 相同的 reshape）：
   - `preds = marginal_probs.reshape(-1)`，
   - `target = match.reshape(-1)`，
   - `indexes = repeat([0..B-1], K).reshape(-1)`。
7. 将此三元组前向传入每个已注册的 NDCG/Recall 指标：`metric.update(preds, target, indexes=indexes)`。

### 8.3 计算指标：`compute_metric(self)`

继承自 `BaseModel.compute_metric`（`tzrec/models/model.py:123`）— 遍历 `self._metric_modules`，对每个调用 `.compute()`，重置，返回字典。引擎将字典以每行一个 JSON 对象的格式写入 `train_eval_result.txt` / `eval_result.txt`。

### 8.4 训练时监控指标

对于周期性训练步日志（`_log_train`），tzrec 暴露 `update_train_metric` / `compute_train_metric`。我们可在此注册单个轻量指标（如步级 CE 均值），使 tensorboard 获得实时标量而无需每步运行波束搜索。完整检索指标仅在评估-checkpoint 节奏下运行。

---

## 9. 训练 / 评估 / 预测生命周期集成

将 §5 – §8 放入 `ai_report/TZREC_PIPELINE.md` 的上下文中：

| 引擎步骤 | TIGER 的接入点 |
|---|---|
| `_create_features` | 读取 §3.2 中声明的 `feature_configs`；产出含一个 `SequenceFeature`（history）和零/一个 `IdFeature`（user）的列表。 |
| `create_dataloader` | 加载预展开的 parquet（§3.4 方案 1）；无模型特定代码。 |
| `_create_model` | 通过 `BaseModel.create_class("Tiger")` 从 proto oneof 选取我们的 `Tiger` 类。 |
| `TrainWrapper(model, mixed_precision=...)` | 照常包裹。**不预期 autocast 异常**（使用 `bf16-mixed`）；若约束波束搜索 softmax 中出现数值问题（`TIGER_implementation.md` §7.4 的 GRID 警告），建议首次运行使用 `precision: "fp32"`。 |
| TorchRec planner | 看到零个 `ShardedModule`（按 §5.1）→ 产出平凡计划。`DistributedModelParallel` 仍包裹模型 — 没问题。 |
| 优化器组装 | 所有参数走**稠密优化器**（sparse-parameters 重写返回空）。标准 Adam-with-weight-decay 与 GRID 一致。 |
| `create_train_pipeline` | 回退到 `TrainPipelineBase`（无稀疏模块）。吞吐量受稠密计算约束；pipeline 重叠不太关键。 |
| `_train_and_evaluate` 步循环 | `pipeline.progress(iterator) → TrainWrapper.forward → Tiger.predict + Tiger.loss → backward`。与任何其他 tzrec 模型相同的模式。 |
| `_evaluate` 步循环 | `pipeline.progress(iterator) → TrainWrapper.forward`（仍计算 loss），然后 `model.update_metric(predictions, batch, losses)` 内部运行 `generate`。 |
| `predict_checkpoint` | 加载 checkpoint，调用 `PredictWrapper.forward → Tiger.predict`，检测到空 labels 时路由到 `generate`。写出 `(user_id, generated_sids)` 行。 |
| `predict`（已导出） | 按 §7.4：第一个里程碑输出仅 encoder 产物；writer 侧用 Python 做波束搜索。 |
| `export` | 同样的注意事项：在 `export()` 中特殊处理 TIGER，导出 `TigerEncoderWrapper`（仅 encoder + 偏移技巧 + SEP 注入）；完整模型的常规路径留作未来工作。 |
| Checkpointing | v1 无特殊处理（除可训练参数外无额外持久化 buffer）。`TZREC_PIPELINE.md` §7.1 的恢复语义原样适用。v2 将向状态字典添加 codebooks buffer；见 §0.1。 |
| `on_train_end` | 对 TIGER 为空操作（不像 `SidRqkmeans` 用它做 FAISS 拟合）。 |

---

## 10. 实现前待决事项

以下是设计有意留给后续评审的事项：

1. **用户 embedding 来源**（§3.2、§5.5）：保留 TIGER 自己的取余哈希 `nn.Embedding`（与 GRID 一致）vs 通过 tzrec 的 `EmbeddingGroup` 路由（更地道，但丢失取余哈希语义）。建议：保持 TIGER 的以确保对等；当用户侧特征超出 `user_id` 时再重新评估。
2. **分片 vs 稠密 SID 表**（§5.1、§5.3）：`self.sid_embedding` 为 `sum(codebook_sizes) × embed_dim` ≈ 典型配置下 100K 参数。普通 `nn.Embedding` 正确；无需分片。记录原因以防未来贡献者通过切换到 TorchRec `EmbeddingCollection` 来"修复"此处。
3. **JIT 脚本化波束搜索**（§7.4）：第一个里程碑仅导出 encoder vs 脚本化完整波束搜索。建议：仅 encoder。
4. **Loss 键名**（§6.3）：`"tiger_ce"` vs `"cross_entropy"` vs `"summed_cross_entropy_per_hierarchy"`。该字符串仅影响 tensorboard 标量名和 `_metric_modules` 中的指标键。建议：`"tiger_ce"`。

（上游已解决：连续子序列增强 — 见 §3.4。`weight_tying` 标志 — 作为空操作已移除，见 `tiger_weight_tying_explanation.md`。`prediction_key_name` / `prediction_value_name` 字段 — 已改为硬编码输出列名 + `--reserved_columns`，见 §2.1 "输出命名"。延后到 v2：码本存储、约束波束搜索默认值、模式检测机制 — 见 §0.1。）

---

## 11. 文件 / proto 交付清单

迁移 PR 中创建或修改的内容：

| 路径 | 状态 | 用途 |
|---|---|---|
| `tzrec/protos/models/tiger.proto` | **新建** | 消息 `Tiger`，所有模型旋钮。 |
| `tzrec/protos/model.proto` | 编辑 | 在 `model` oneof 中添加 `Tiger tiger = N;` 并 import。 |
| `tzrec/protos/{model,tiger}_pb2.py` & `*_pb2.pyi` | 重新生成 | 运行 `bash scripts/gen_proto.sh`。提交生成的文件。 |
| `tzrec/models/tiger.py` | **新建** | `Tiger(BaseModel)` — 本设计的 §5、§6、§7、§8。 |
| `tzrec/models/tiger_test.py` | **新建** | 单元测试；使用类似 `tzrec/tests/configs/tiger_mock.config` 的 mock 配置。 |
| `tzrec/modules/tiger_ff.py` | **新建** | `T5MultiLayerFF`，从 `GRID/src/models/modules/semantic_id/tiger_generation_model.py:1080` 移植。 |
| `tzrec/models/__init__.py` | 编辑 | Import `Tiger` 使注册表可见。 |
| `tzrec/tests/configs/tiger_mock.config` | **新建** | 单元测试用小配置；镜像 `dssm_mock.config` 的布局。 |
| `tzrec/main.py` | 编辑（仅采用 §7.4 方案 1 时） | 在 `export()` 中添加 TIGER 分支，输出仅 encoder 产物。 |
| `docs/source/models/tiger.md` | **新建** | 用户向文档；可为本设计的精简版。 |
| `examples/tiger_amazon.config` | **新建** | 用户可复制的端到端示例配置。 |

---

## 12. 验收标准（"完成"的定义）

- `bash scripts/gen_proto.sh` 运行无错；新的 `Tiger` 消息可从 `ModelConfig.tiger` 到达。
- `python -m unittest tzrec.models.tiger_test` 在 mock 配置和合成 SID 编码数据上通过 — v1 没有 `semantic_id_path` 也没有 codebooks buffer，因此测试只需合成 `history_sids` + `label_sids` 的 parquet 行。
- `torchrun -m tzrec.train_eval --pipeline_config_path examples/tiger_amazon.config --train_input_path .../user_sequences/*.parquet` 使用真实 Amazon Beauty 第 3 阶段 SID 输出至少运行一个完整 epoch（每用户一行，无预展开）。
- `python -m tzrec.predict_checkpoint --pipeline_config_path examples/tiger_amazon.config --predict_input_path .../predict_in.parquet --predict_output_path .../predict_out.parquet` 写出含 `(user_id, semantic_ids)` 列的 parquet。
- 评估时 `recall@5 / recall@10` 指标在相同 Amazon Beauty 数据和等量训练计算后，与对应 GRID checkpoint 在小容差范围内匹配。
- 仅 Encoder 的 `tzrec.export → tzrec.predict` 路径产出的 encoder 隐藏状态与 `tzrec.predict_checkpoint` 在数值精度内一致（覆盖 §7.4 方案 1）。

以上成为 PR 描述的检查清单。
