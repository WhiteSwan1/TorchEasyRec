# TIGER 中 `weight_tying` 实际做了什么？

简短答案：在 v1（以及 GRID）中，**`weight_tying` 是一个历史遗留标志 — 对模型没有可观察的影响**。它是一个我们接受但忽略的 proto 字段。TIGER 中实际存在的"绑定"无论该标志值如何都是硬编码的。

## weight tying 通常意味着什么

在标准 encoder–decoder Transformer（如 T5）中，`weight_tying=True` 表示**输入 embedding 表被复用为输出投影**：

```
shared.weight  ←————┐
                    ├── encoder.embed_tokens.weight
                    ├── decoder.embed_tokens.weight
                    └── lm_head.weight  (transposed)
```

一个 `(vocab_size, d_model)` 张量，三种角色。其好处众所周知（Press & Wolf, 2017）：参数量更小，且具备"输入与输出 token 空间一致"的强归纳偏置。T5-base 通过这种方式节省约 25 M 参数（其 `lm_head` 绑定到 32128×768 的 `shared` embedding — 见 `TIGER_vs_T5base_params.md`）。

## TIGER 中 `weight_tying` 实际做了什么

**没有任何可观察的效果**。GRID 的 `SemanticIDEncoderDecoder` 无论该标志如何都硬编码了两种关系：

1. **encoder 与 decoder 共享 SID embedding 表**。见 `tiger_generation_model.py:863-884` — `get_embedding_table("encoder")` 和 `get_embedding_table("decoder")` 都返回 `self.item_sid_embedding_table_encoder`。encoder 在 `encoder_forward_pass` 期间从中读取；decoder 在 `decoder_forward_pass` 期间从中读取。始终如此。
2. **输出头**不**与 embedding 表绑定**。`decoder_mlp[h]` 是逐层级各自独立的 `Linear(embed_dim, codebook_sizes[h], bias=False)` — 独立的可训练参数，从不绑定到输入 embedding 的行。

因此 TIGER 中存在的唯一"绑定"是第 1 点，无论 `weight_tying=True` 还是 `weight_tying=False` 它都会发生。该标志被父类 `TransformerBaseModule`（`transformer_base_module.py:66-70` 的 `get_embedding_table`）读取，但那段代码路径是为通用的 embedding-vs-decoder 检索打分服务的 — 而 `SemanticIDEncoderDecoder` 重写了 `model_step`，因此父类中使用该标志的逻辑**永远不会被触达**。

GRID 自身的配置设置 `weight_tying: true`，模型训练正常；将其翻转为 `false` 会产出完全相同的训练模型（第 1 点的 embedding 表共享仍然存在）。它实际上是死配置。

## 为何在 v1 的 proto 中保留它

两个原因，均记录在 `tiger_migration_design.md` 的 §2.1 第 21 行和 §5.3：

1. **proto 级兼容性**。若用户将 GRID YAML 整体复制到其 tzrec pipeline 配置中（按 §2.1 重命名 codebook 字段后），让 `weight_tying: true` 通过 proto parser 而不报错可使迁移摩擦保持最小。删除该字段会迫使用户额外 grep 并删除一行。
2. **前向兼容**。若未来版本决定该标志应有特定含义（见下一节），我们已经把字段管道铺好了。

## 若我们激活它，它可以是什么含义（延后的决定）

设计文档 §10 决策 5 把这留作开放问题。最自然的语义是：

> 当 `weight_tying=True` 时，将每个 `decoder_mlp[h].weight` 绑定到 `item_sid_embedding_table_encoder` 的一个切片 — 具体而言，是覆盖 `[h × max(codebook_sizes), h × max(codebook_sizes) + codebook_sizes[h])` 行的切片。

这将在每个层级内部镜像 T5 的 `shared ↔ lm_head` 配方：层级 `h` 处码字 `c` 的输入 embedding 被复用为 `Linear` 权重的对应行，用于产出层级 `h` 处码字 `c` 的 logit。从概念上很有吸引力 — 它强制"在层级 h 预测码字 c 的得分依赖于 c 在层级 h 的输入 embedding"。

参数节省量为 `num_hierarchies × codebook_width × embed_dim`。在 GRID 默认下（4 × 256 × 128 = 131 K 参数），约占模型 ~13 M 参数的 1% — 称不上有意义的压缩。该归纳偏置可能有助于 recall@K，但这是一个经验问题。

**§10 决策 5 的建议**：v1 中保留 GRID 风格的模糊行为（标志被接受但忽略），仅当未来消融实验显示真正的 T5 风格绑定能改善指标时再重新评估。在 proto 注释中醒目记录，使用户不会以为该标志在做什么事情。

## TL;DR

| 方面 | 取值 |
|---|---|
| 在 v1 中该标志是否改变模型行为？ | **否** |
| 在 GRID 中该标志是否改变模型行为？ | **否**（尽管 GRID YAML 默认为 `true`） |
| 无论标志如何，*实际*存在的绑定是什么？ | encoder 和 decoder 共享 `item_sid_embedding_table_encoder` — 在 `get_embedding_table` 中硬编码 |
| 为何仍在 proto 中？ | 与 GRID 配置向后兼容 + 若我们决定赋予其语义则向前兼容 |
| 能否移除？ | 可以，安全 — 但保留它的代价只是一个被忽略的字段，移除则使复制粘贴 GRID 配置略微更难 |
| 若 v2 在 decoder 头内部激活 T5 风格绑定能节省的参数 | GRID 默认下约 131 K（≈ 总量的 1%）；不显著 |

## 建议

保留该字段，v1 中静默接受，在 proto 注释中记录其空操作状态。若未来消融实验显示将 `decoder_mlp[h]` 绑定到 SID embedding 切片可改善指标，则在 v2 中以同一字段激活 — 无需 proto 迁移。
