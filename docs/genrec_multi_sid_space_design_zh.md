# GenRec 多 SID 空间设计

> 状态：实现契约，代码落地分支为 `feat/genrec-multi-sid-space`。
> 基线：Alibaba `upstream/master` 的 `7392f51`。
> 目标实验：行为序列使用原始 SID，监督 label 使用碰撞处理后的 SID；两套 SID 使用独立 token 前缀和独立词表区间。

## 1. 结论

本方案将当前单一 `SidSpace` 扩展为多空间，并冻结以下契约：

1. `PromptConfig.sid_space` 从 singular 改为 repeated，保留字段名和 tag `5`。
1. 每个 `SidSpace` 必须声明唯一 `name`；`name` 与实际 INLINE 特征字段或 label 字段精确匹配。
1. 每个空间拥有独立的 `base_vocab` 和 `token_format`，在全局 LM vocabulary 中占据互不重叠的连续区间。
1. 离线样本生成继续产出按层展平后的 0-based SID。Assembler 只增加该字段所属空间的 `base_vocab`，不重复增加层内 offset。
1. response 对应的 SID 空间是唯一目标空间。训练 label、生成约束 band、beam 层数及生成结果解码都只使用该空间。
1. 第一阶段所有 INLINE 都必须命中一个 `SidSpace`。非 SID INLINE 的表达与偏移方式留待后续扩展；当前不做隐式猜测。
1. 原始 SID 与碰撞后 SID 是两个 token namespace。即使它们描述同一 item，也不要求 token ID、前缀或末层 code 相同。
1. 不保留 legacy single-space 分支。即使配置只有一个空间，也必须走 repeated、named、按字段精确绑定的统一逻辑。

改造面属于中等：需要修改 proto、prompt compiler/types、assembler、GenRec 模型、训练/预测组装、导出和测试；数据解析器与 causal-LM loss 主体无需重写。

## 2. 背景与目标

当前训练数据中的行为序列与 `single_label` 共用一套 SID 编码。例如历史 item 和目标 item 都以同一套 `<|sid_{i}|>` token 进入模型。

新实验要求：

- 行为序列从原始 RQ-VAE 映射取 SID；
- label 从碰撞处理后的映射取 SID；
- 两者允许拥有不同的 item-to-SID 映射；
- 两者使用不同 token 前缀和不同全局 vocabulary 区间；
- 模型根据字段所属的 SID 空间选择正确偏移，而不是让所有 INLINE 共用一个偏移。

这里的 `SidSpace` 只定义一套 SID 的坐标系和 token namespace，不负责把 item 映射成 SID。原始映射与碰撞映射的匹配由本地离线数据生产链路完成。

### 2.1 非目标

第一阶段不处理以下能力：

- 非 SID 的 INLINE token 特征；
- 一个字段内混合多套 SID 空间；
- 在框架内把原始 SID 动态转换成碰撞后 SID；
- 多个独立生成 label 或在一次 decode 中切换 SID 空间；
- 根据数值范围、字段内容或 token 前缀猜测字段属于哪个空间。

## 3. 当前实现及限制

| 环节        | 当前行为                                                     | 单空间假设造成的问题            |
| ----------- | ------------------------------------------------------------ | ------------------------------- |
| Proto       | `required SidSpace sid_space = 5`                            | 只能声明一套 codebook 和前缀    |
| INLINE 判定 | 单个 sequence 特征且无 embedding 即为 INLINE                 | INLINE 没有 SID 类型或空间身份  |
| Label 判定  | response 占位符命中 `data_config.label_fields` 后强制 INLINE | label 是否为 SID 仅由其位置决定 |
| Prompt 编译 | response 宽度取唯一 `sid_space.num_levels`                   | 无法使用独立 label 空间         |
| Assembler   | 所有 INLINE 都执行 `value + sid_space.base_vocab_size`       | 行为和 label 必须共用前缀与偏移 |
| 模型        | embedding resize、level offsets、decode bands 都取唯一空间   | 生成不能只约束到碰撞后空间      |
| 导出        | 只导出单空间扩展后的 tokenizer/vocab                         | serving 无法恢复多空间绑定关系  |

当前 single label 并不是通过 SID 元数据识别的。它先由 response 占位符匹配到 `label_fields`，随后被无条件当作 SID INLINE。类似地，普通无 embedding 的 sequence feature 也会被当作 SID。因此多空间实现必须把“INLINE 的空间归属”变成显式编译结果。

### 3.1 与 PromptGeneration 抽象的关系

`PromptGeneration.pdf` 将直接 token lookup 与 embedding 投影视为不同的 prompt 填充方式，并在生成推荐中保留显式的多 token SID。该抽象支持 SID 通过 INLINE 直接进入 LM token 空间，但没有规定“所有 INLINE 都是 SID”，也没有规定多个 INLINE 必须共享一个 offset。

因此本设计把两个概念拆开：

- INLINE 是值进入 prompt 的传输方式；
- `SidSpace` 是该 INLINE 值所属的 SID 坐标系与 vocabulary namespace。

只有同时满足 INLINE 且通过 name 绑定到 `SidSpace` 的字段，才能执行 SID offset。第一阶段尚未定义第二种 INLINE namespace，所以未绑定项直接失败。

## 4. 术语与 0-based 数据契约

### 4.1 `codebook` 表示容量，不表示最大 code

对空间 `s`，设：

```text
codebook_s = [K_s,0, K_s,1, ..., K_s,L-1]
```

其中 `K_s,l` 是第 `l` 层的容量，合法 local code 为：

```text
0 <= c_s,l < K_s,l
```

因此，如果口头所说的“每层 SID 空间是 `[3, 3, 3]`”表示每层最大 code 为 `3`，那么 proto 中的容量必须写成 `[4, 4, 4]`，合法 code 是 `0..3`。

### 4.2 数据表中的 SID 已包含层 offset

每个空间独立计算层 offset：

```text
level_offset_s,l = sum(codebook_s[0:l])
flat_code_s,l    = level_offset_s,l + local_code_s,l
```

数据表中的行为 SID 和 label SID 都保存 `flat_code`，不是未经处理的 local code。Assembler 再做一次空间级映射：

```text
token_id_s,l = base_vocab_s + flat_code_s,l
```

框架不得再次增加 `level_offset`，也不得对异常值执行 clamp、取模或自动修正。

### 4.3 展平顺序

一个 item 的各层 SID 按层顺序排列：

```text
[item0.level0, item0.level1, ..., item0.levelL-1,
 item1.level0, item1.level1, ..., item1.levelL-1, ...]
```

行为序列每行的元素数必须是所属空间 `num_levels` 的整数倍。第一阶段 response 只允许一个目标 item，因此 label 每行必须恰好包含 `num_levels` 个元素。

## 5. Proto 设计

建议保留 `sid_space` 的字段名和 tag `5`，直接改变 cardinality：

```proto
message PromptConfig {
    required string tokenizer_path = 1;
    required string prompt = 2;
    optional string response = 3;
    repeated PromptSlot slots = 4;

    repeated SidSpace sid_space = 5;

    optional uint32 max_length = 6 [default = 0];
    optional string sentinel_token = 7 [default = "<|pg_hole|>"];
}

message SidSpace {
    repeated uint32 codebook = 1;
    optional string token_format = 2 [default = "<|sid_{i}|>"];
    optional uint32 vocab_pad_to_multiple_of = 3 [default = 128];
    optional string manifest_path = 4;

    required string name = 5;
}
```

`repeated` 与 `required` 不能同时使用，因此“至少一套空间”由 `compile_prompt` 显式校验。

`SidSpace.name` 在 proto 和业务语义中都必填。`compile_prompt` 仍需显式校验 name 非空且全局唯一，以提供明确错误信息。不存在匿名空间、默认空间或按空间数量触发的特殊分支。

由于 name 直接匹配底层字段，一个空间当前只绑定一个字段。典型的 SID history + SID label 配置应声明两套空间；即使两者 codebook 相同，也分别拥有自己的 name、prefix 和 `base_vocab`。

### 5.1 为什么不在 `PromptSlot` 增加空间名

第一阶段直接用底层字段名绑定：

- body INLINE：使用 `PromptSlot.feature_names[0]`；
- response INLINE：使用其唯一 label 字段名；
- `SidSpace.name` 必须与上述字段名完全相同。

例如占位符叫 `history`，但显式 slot 的底层字段是 `item_list_with_sid`，则空间名必须是 `item_list_with_sid`，而不是 `history`。

该方案不需要同时修改 FeatureConfig、label schema 和 PromptSlot，且 body 与 response 在 prompt compiler 中已经统一成 slot。若未来需要多个字段共享同一 token namespace，再单独引入显式 `sid_space_name` 引用；本阶段不提前扩展。

## 6. 配置示例

```proto
data_config {
    label_fields: "single_label"
}

feature_configs {
    sequence_raw_feature {
        feature_name: "item_list_with_sid"
        expression: "user:item_list_with_sid"
    }
}

prompt_config {
    tokenizer_path: "/path/to/tokenizer.json"
    prompt: "History: {{history}} Predict:"
    response: "{{single_label}}"

    slots {
        name: "history"
        feature_names: "item_list_with_sid"
    }

    sid_space {
        name: "item_list_with_sid"
        codebook: 8192
        codebook: 8192
        codebook: 8192
        token_format: "<|raw_sid_{i}|>"
        manifest_path: "/path/to/raw_sid_manifest.json"
    }

    sid_space {
        name: "single_label"
        codebook: 8192
        codebook: 8192
        codebook: 8192
        token_format: "<|resolved_sid_{i}|>"
        manifest_path: "/path/to/resolved_sid_manifest.json"
    }
}
```

这里的两个 manifest 只用于校验各自的 codebook/映射版本，不参与逐 item 转换。训练表仍需提前生成：

- `item_list_with_sid`：原始 SID 映射；
- `single_label`：碰撞处理后 SID 映射。

## 7. 全局 vocabulary 布局

### 7.1 `base_vocab` 的精确定义

`base_vocab_s` 是空间 `s` 在全局 LM token ID 中的绝对起始位置，不是所有空间共享的“原 tokenizer 大小”。

设基础 tokenizer 大小为 `B`，第 `s` 个空间的 token 数为：

```text
S_s = sum(codebook_s)
```

按 `sid_space` 的声明顺序分配：

```text
base_vocab_0 = B
base_vocab_s = B + sum(S_j, j < s)
```

空间 `s` 的完整 token 区间为：

```text
[base_vocab_s, base_vocab_s + S_s)
```

第 `l` 层的生成 band 为：

```text
band_lo_s,l = base_vocab_s + level_offset_s,l
band_hi_s,l = band_lo_s,l + K_s,l - 1
```

所有 SID token 注册完成后，再注册一次可选 sentinel，最后对总 vocabulary 做一次 padding。不同空间之间不插入 padding 空洞。

`vocab_pad_to_multiple_of` 当前属于 `SidSpace`。多空间第一阶段要求所有空间配置相同的值，并将该共同值应用于最终全局 vocabulary；值不一致时直接报错。

### 7.2 token 字符串

`{i}` 仍表示当前空间内的 `flat_code`，范围为 `[0, S_s)`。不同空间通过不同 literal prefix 区分，例如：

```text
<|raw_sid_0|>       ... <|raw_sid_24575|>
<|resolved_sid_0|>  ... <|resolved_sid_24575|>
```

编译器必须验证：

- `token_format` 包含 `{i}`；
- 同一空间内渲染结果唯一；
- 不与基础 tokenizer 已有 token 冲突；
- 不与其他 SID 空间或 sentinel 冲突。

只要最终 token 字符串全局唯一，不强制格式包含 `{name}`。

### 7.3 0-based 示例

假设每层 local code 范围为 `0..3`，则两个空间的 `codebook` 都是 `[4, 4, 4]`，每个空间大小为 `12`，层 offset 为 `[0, 4, 8]`。

同一 item 的原始 local SID 为 `[1, 1, 1]`，碰撞处理后为 `[1, 1, 0]`：

```text
原始 flat SID:   [1, 5, 9]
碰撞后 flat SID: [1, 5, 8]
```

若基础 tokenizer 大小为 `B`，原始空间先声明，碰撞空间后声明：

```text
raw.base_vocab      = B
resolved.base_vocab = B + 12

行为 token ids = [B + 1,  B + 5,  B + 9]
label token ids = [B + 13, B + 17, B + 20]
```

因此行为与 label 即使前两层 local code 相同，也位于不同的 embedding/lm-head 行。这正是本实验需要的隔离语义。

## 8. 编译与运行时设计

### 8.1 编译流程

```text
实际 feature/label 字段名
          │
          ▼
按 SidSpace.name 精确查找 ──未命中/多义──► 编译失败
          │
          ▼
生成 SlotSeg.sid_space_index
          │
          ├──► Assembler 选择对应 base_vocab
          │
          └──► response 记录 target_sid_space_index
                         │
                         └──► width / bands / beam / decode
```

`compile_prompt` 应完成：

1. 校验空间数量、名称、codebook、manifest、padding 参数和 token 唯一性；
1. 按声明顺序扩展 tokenizer，并为每个空间计算绝对 `base_vocab`；新增 token 的实际 ID 必须与预期连续区间一致，否则编译失败；
1. 对每个 INLINE slot 使用实际字段名查找唯一空间；
1. 将 name 查找结果固化为整数 `sid_space_index`，避免 assembler 热路径做字符串字典查询；
1. 将 response 对应空间记录为 `target_sid_space_index`；
1. 用目标空间的 `num_levels` 计算 response width；
1. 所有 SID token 与可选 sentinel 添加完后，计算一次全局 `target_vocab_size`。

### 8.2 编译产物

建议把单数结构调整为：

```text
ResolvedSidSpace
  name
  codebook
  num_levels
  base_vocab                 # 当前实现可沿用 base_vocab_size 字段名
  level_offsets
  band_lo / band_hi

SlotSeg
  ...
  sid_space_index            # INLINE 必填；PROJECTED 为空

CompiledPrompt
  sid_spaces                 # 按 proto 声明顺序
  target_sid_space_index
  target_vocab_size
  sentinel_token_id
  eos_token_id / pad_token_id
  prompt_plan
  projection_plan
```

全局属性不再挂在任意一个 `ResolvedSidSpace` 上，避免多个空间各自携带互相矛盾的 `target_vocab_size` 或特殊 token ID。

删除 singular `compiled_prompt.sid_space` 入口。所有消费者统一读取 `sid_spaces`，并通过 slot 的 `sid_space_index` 或 `target_sid_space_index` 选择空间；即使只有一套空间，也不提供默认取首项的旁路。

### 8.3 Assembler

当前单一 `self.id_shift` 改为按空间索引保存的 shift 表：

```text
sid_base_vocabs = [space.base_vocab for space in sid_spaces]
```

每个 INLINE segment 的组装规则为：

```text
output_values = input_flat_codes + sid_base_vocabs[segment.sid_space_index]
```

训练 response 也走同一规则，因此 causal-LM labels 会自然变成碰撞空间 token，无需重写 loss 构造。PROJECTED slot 不绑定 SID 空间，也不执行 SID 偏移。

### 8.4 模型与生成

模型需要：

- 将所有空间的 `base_vocab` 按声明顺序保存为 `_sid_base_vocabs`；
- embedding/lm-head 只 resize 一次，目标行数为全局 `target_vocab_size`；
- 只为目标空间构造生成 schedule；
- `beam_widths` 数量必须等于目标空间的 `num_levels`；
- 每一步 logits 只允许目标空间相应层的 `[band_lo, band_hi]`；
- 生成 token 转回 local code 时，减去目标空间的 `base_vocab` 和 `level_offsets`。

模型不得通过“最后声明的空间”“数值落在哪个区间”或固定 label 名推断目标空间。

## 9. 数据、训练、评估与 serving 契约

### 9.1 数据生产

本次实验全部使用本地文件，不依赖 SQL 或 ODPS。离线样本生成程序分别读取两套映射：

```text
历史 item ──join 原始 map────► item_list_with_sid
目标 item ──join 碰撞后 map──► single_label
```

两个字段都遵守各自空间的 0-based flat SID 契约。框架不从历史 SID 推导 label，也不要求两个映射的 prefix 相等。

### 9.2 训练

最终序列中：

- prompt body 的历史 SID 使用原始空间 token；
- response 位置使用碰撞空间 token；
- response mask 与 causal shift 逻辑保持现状；
- CE 在全局 vocabulary 上计算，但有效目标行属于碰撞空间。

### 9.3 评估与候选映射

生成结果属于目标的碰撞后 SID 空间。因此：

- direct SID HR 使用碰撞后 `single_label`；
- SID-to-item 展开、候选 catalog 和 Item-ID HR 必须使用碰撞后映射；
- 不得拿生成结果去查询原始 SID map。

### 9.4 导出与 serving

导出物必须包含同一份扩展 tokenizer，并固化：

- SID 空间声明顺序；
- 每个空间的 `name`、`codebook`、`base_vocab`、`token_format` 和 bands；
- `target_sid_space_index`；
- 全局 `target_vocab_size`、sentinel/EOS/PAD ID。

scripted front-end 使用 `SlotSeg.sid_space_index` 进行偏移；在线解码使用目标空间 bands。训练、预测和 serving 必须复用同一编译逻辑，不能各自重算空间顺序。

## 10. 校验与失败策略

### 10.1 配置期必须失败

以下情况由 `compile_prompt` 直接拒绝：

- 没有任何 `sid_space`；
- 任一空间缺少 name、name 为空或 name 重复；
- codebook 为空或任一容量不大于 0；
- manifest 与声明的 codebook 不一致；
- 多空间的 padding multiple 不一致；
- token format 缺少 `{i}`；
- 任意 SID token 与基础 tokenizer、其他空间或 sentinel 重名；
- INLINE feature/label 没有匹配的空间；
- 一个 INLINE slot 包含多个底层字段；
- response 不是唯一 SID label，或目标空间不唯一；
- response label 上声明 projection。

### 10.2 数据期必须失败

数据必须满足：

- body SID 行长度是空间层数的整数倍；
- label 行长度恰好等于目标空间层数；
- 第 `l` 层 flat code 位于该层的 `[level_offset_l, level_offset_l + K_l)`；
- 不包含负数、OOV、缺层或多层。

出现错误时应报告字段名、空间名及预期范围；不得通过取模、截断、补齐或换用另一空间继续训练。具体逐值范围检查可以放在数据边界或可控的校验阶段，正常热路径不得依赖数值推断空间身份。

## 11. 版本边界与 checkpoint 规则

### 11.1 Proto/config 断代

将 tag `5` 从 singular message 改为 repeated message 后：

- Python API 从 `cfg.sid_space.codebook` 变为 `cfg.sid_space[0].codebook`，所有消费者必须同步升级；
- `HasField("sid_space")` 对 repeated 字段不再合法，改为长度检查；
- 旧二进制不能安全读取包含多个 tag-5 message 的新配置，因为它可能把多个 message 合并成一套错误的 codebook。
- 旧配置缺少 required `SidSpace.name`，不再支持直接加载；
- 旧的 singular `CompiledPrompt`、assembler 和模型入口全部删除，不增加适配层。

GR 尚未形成正式兼容契约，因此本次按断代升级处理。多空间配置、runtime、训练包和导出物必须成套发布；已有单空间配置需要显式改写，已有单空间 GR checkpoint 不承诺可续训。

### 11.2 多空间 ABI

以下内容属于 checkpoint/serving ABI：

- 空间声明顺序；
- `name`；
- `codebook`；
- `token_format`；
- `base_vocab`；
- 目标空间；
- sentinel/EOS/PAD ID 和最终 vocabulary 大小。

实现还会记录两类内容摘要：完全扩展后的 tokenizer 规范化 JSON 的
`tokenizer_sha256`，以及每个配置了 manifest 的空间对应的
`manifest_sha256`。这样，即使 vocabulary 大小和 codebook 没变，基础 tokenizer
行映射变化或同容量 SID 映射被替换也会被识别。

checkpoint 将规范化后的完整 `sid_abi` 写入 `hf_export_meta.json`，HF 导出的
`config.json` 同样携带该字段。续训、评估、预测在恢复权重前，以及 HF 导出在
转换权重前，都会把当前编译结果与 checkpoint 中的 `sid_abi` 做严格相等比较；
缺少该字段或任一字段不一致都直接失败。

同一多空间实验续训或加载导出模型时必须一致。空间重命名、重排、插入、删除或修改 codebook/prefix 都视为 vocabulary 变化，必须重新训练或使用单独的显式迁移工具；第一阶段不提供 row remap，也不静默加载不匹配的 checkpoint。

未配置 manifest 的空间，其 `manifest_sha256` 为 `null`；此时只能校验声明和
tokenizer ABI，框架无法证明外部数据使用的 SID-to-item 映射没有被同容量替换。

## 12. 代码改造范围

| 模块                                     | 主要改动                                                            |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `tzrec/protos/prompt.proto`              | `sid_space` 改 repeated，`SidSpace` 增加 `name`                     |
| `tzrec/prompt/types.py`                  | 多 `ResolvedSidSpace`、slot 空间索引、全局 vocab 属性、目标空间索引 |
| `tzrec/prompt/compile.py`                | name 绑定、多空间 tokenizer 扩展、全局 padding、目标空间解析与校验  |
| `tzrec/prompt/assembler.py`              | 从全局 `id_shift` 改为 per-segment shift                            |
| `tzrec/datasets/dataset.py`              | 向 assembler 传递多空间编译产物                                     |
| `tzrec/models/model.py`                  | serving front-end 使用多空间 assembler                              |
| `tzrec/models/genrec_model.py`           | 全局 resize、保存所有 base、目标空间反解码                          |
| `tzrec/models/genrec_causal_lm_model.py` | beam/band 绑定目标空间                                              |
| `tzrec/utils/hf_export_util.py`          | 导出多空间与目标空间 metadata                                       |
| 对应 tests/config fixtures               | 更新 repeated API，并增加双空间覆盖                                 |

`.proto` 落地后必须执行：

```bash
bash scripts/gen_proto.sh
```

生成的 `_pb2.py` 与 `_pb2.pyi` 不手工修改。

## 13. 测试矩阵

### 13.1 编译与 vocabulary

- 仅声明一个 named 空间时，仍走 repeated 路径并正确完成 name 绑定、base 分配和 decode；
- 两个空间使用相同 flat code 时，全局 token ID 与 token 字符串不重叠；
- 两空间不同 `token_format`、不同/相同 codebook 均能正确分配 base；
- duplicate/empty/unknown name、token 冲突、manifest mismatch、padding mismatch 均失败；
- sentinel 只添加一次，最终 vocabulary 只 padding 一次。

### 13.2 Assembler

- 同一 batch 中多个 INLINE slot 分别使用正确 base；
- 原始 history 与碰撞 label 具有不同 prefix 时，逐元素断言 `input_ids`；
- response mask 只覆盖碰撞 label token；
- 非 SID INLINE 明确失败；
- body 长度非层数整数倍、label 长度错误和越界 code 明确失败。

### 13.3 模型与生成

- embedding/lm-head 行数等于全局 target vocabulary；
- 模型保存的 `_sid_base_vocabs` 与编译结果一致；
- beam 层数、逐层 band 和 token-to-local-code 全部取目标空间；
- 生成 token 不会落入原始 history 空间；
- train、eval、predict、export、scripted serving round trip 一致。

### 13.4 端到端实验断言

构造同一 item：

```text
raw flat SID      = [1, 5, 9]
resolved flat SID = [1, 5, 8]
```

端到端断言：

1. history 使用 raw 空间对应 token IDs；
1. label 使用 resolved 空间对应 token IDs；
1. loss mask 只监督 resolved token；
1. decode 输出 `[1, 1, 0]` 的 local collision SID；
1. Item-ID 评估查询 resolved SID-to-item map。

## 14. 推荐落地顺序

1. 修改 proto 与编译数据结构，生成 bindings；
1. 实现多空间解析、tokenizer 扩展和 name-to-index 绑定；
1. 修改 assembler，使每个 INLINE 使用自己的 base；
1. 将模型 resize、beam、bands 和反解码切到全局 vocab + 目标空间；
1. 更新 dataset、serving 与 HF export；
1. 补齐单个 named 空间测试和双空间端到端测试；
1. 最后接入本地生成的原始 history / 碰撞 label 数据做小流量实验。

完成标准是：所有 SID 配置统一使用 repeated、named 空间，同一训练样本可以让行为 SID 与 label SID 使用不同前缀、不同全局 token 区间和不同 item-to-SID 映射，并且训练、生成、评估和 serving 全链路都明确以 label 对应空间作为目标空间。
