# SID 码本防碰撞设计

## 目标

为语义 ID（SID）生成流程增加一个离线防碰撞步骤。该步骤在 SID 批量预测之后执行，保证每个最终 SID/码本桶中的物品数量不超过配置上限，例如 5。工具需要支持 CSV、Parquet 和 MaxCompute 输入输出，并保留原始 SID 与防碰撞后的 SID，便于审计和回溯。

本设计参考了 `/mnt/fangtinglin/al_sid/SID_generation/ID_collision.sql`。

## SQL 脚本评审

现有 SQL 的核心策略是合理的：

1. 构建 `item_codebook_info(item_id, origin_codebook, codebook, index)`。
2. 对每个原始 `codebook_index` 只保留前 `capacity` 个物品。
3. 将其余物品视为溢出物品。
4. 将溢出物品的近邻候选 SID（`sorted_index`）展开为带优先级的候选行。
5. 迭代地将溢出物品分配到尚未满载的候选 SID，并保证每个最终 SID 最多保留 `capacity` 个物品。

生产实现不建议直接照搬该 SQL。当前脚本存在一些问题：循环需要手动控制、反复覆盖中间表、使用 `rand()` 导致结果不可复现、候选字符串编码存在歧义、部分别名不够严谨、没有显式收敛检查、缺少诊断输出，并且不支持本地 CSV/Parquet。

## 建议工作流

1. 按现有流程训练并导出或评估 SID 模型。
2. 使用 `python -m tzrec.predict` 做离线预测，保留 item id，并输出 SID codes。
3. 如果需要重分配，必须显式提供候选 SID codes，可以来自模型预测输出，也可以来自独立候选表。候选生成/输出不作为默认行为开启。
4. 运行新的防碰撞工具，建议入口为 `python -m tzrec.tools.sid.collision_prevention`。
5. 写出最终映射表：

```text
item_id, origin_codebook, codebook, index
```

`origin_codebook` 是模型输出的原始 SID。`codebook` 是容量约束后的最终 SID。`index` 是该物品在最终 SID 桶内的 1-based 编号。

## 输入输出 Schema

防碰撞工具应将所有输入归一化为两张逻辑表。

`raw_sid`：

```text
item_id: string 或 int64
origin_codebook: string，例如 "111,222,333"
```

工具也应支持 SID 预测输出中的 `codes` 字段，类型可以是 `ARRAY<BIGINT>` / `list<int64>`，也可以是 `code_0, code_1, code_2` 这样的拆分列。内部统一使用可配置分隔符转换为 `origin_codebook`。对于 CSV，codes 必须保存为字符串；CSV 输出不支持嵌套数组/list。

只有当原始输出已经满足容量约束时，`candidate_sid` 才可以省略。它没有隐式默认来源；如果存在溢出物品但没有提供候选行，工具应直接报错，并给出清晰错误。

```text
item_id
origin_codebook
candidate_codebook
priority: int64       # 越小优先级越高
score: double         # 可选，距离或相似度
```

为了兼容现有 SQL，工具可以支持紧凑的字符串 `sorted_index` 字段，但内部规范格式应使用上面的长表格式。这样可以避免 SID 本身使用逗号拼接时，与候选列表分隔符产生歧义。

主输出保持与 SQL 兼容：

```text
item_id
origin_codebook
codebook
index
```

可选诊断输出应包含总物品数、原始碰撞桶数量、最终碰撞桶数量、重分配数量、未分配数量、迭代次数和最终最大桶大小。

## 分配算法

CSV/Parquet 与 MaxCompute 应使用同一套确定性分配语义。

1. 将每行原始 SID 归一化为规范字符串。
2. 初始分配：按 `origin_codebook` 分组，使用确定性 tie-breaker 排序，例如 `hash(seed, item_id)`，保留前 `capacity` 行并设置 `index = row_number`。
3. 其余物品标记为未分配。
4. 只对未分配物品展开或读取候选行。
5. 每轮迭代：
   - 去掉已经分配物品的候选。
   - 去掉目标 `candidate_codebook` 已满的候选。
   - 按 `(priority, score, hash(seed, item_id))` 对候选排序。
   - 对每个目标 codebook，最多选择其剩余容量数量的候选。
   - 如果同一物品被多个目标 codebook 选中，只保留该物品的最优候选。
   - 将接受的候选追加到 `item_codebook_info`，并分配下一个可用 `index`。
6. 当全部物品已分配、某一轮没有新分配，或达到 `max_iters` 后停止。

默认未分配策略应为 `error`，因为静默保留超容量 SID 会破坏防碰撞语义。调试场景可以提供 `keep_original`，分析场景可以提供 `drop`。

分配器必须保证以下不变量：

- 每个 `item_id` 在最终表中最多出现一次；
- 每个最终 `codebook` 的 `count(*) <= capacity`；
- `origin_codebook` 永不被覆盖；
- 相同输入和相同 seed 必须产生相同输出。

## CSV 和 Parquet 后端

本地模式应复用 PyArrow 读写能力，与现有 `CsvReader`、`ParquetReader`、`CsvWriter`、`ParquetWriter` 的行为保持一致。初版可以是单进程内存实现，因为防碰撞分配需要全局 group by 和重复 join。工具应根据配置的内存上限或估算行数提前失败，并给出清晰错误。

更大规模的本地数据后续可以考虑外部排序、SQLite 或 DuckDB，但初版不应强制引入新的执行引擎依赖。

示例：

```bash
python -m tzrec.tools.sid.collision_prevention \
  --input_path /path/to/sid_predict/*.parquet \
  --output_path /path/to/final_sid \
  --reader_type ParquetReader \
  --writer_type ParquetWriter \
  --item_id_field item_id \
  --code_field codes \
  --candidate_input_path /path/to/sid_candidates/*.parquet \
  --max_items_per_codebook 5 \
  --seed 2026
```

## MaxCompute 后端

MaxCompute 模式应通过 PyODPS 执行生成 SQL，而不是将全量数据下载到 Python。脚本应创建带可配置临时前缀和生命周期的中间表，并编排与本地分配器一致的循环逻辑。

建议中间表：

- `${prefix}_raw_sid`
- `${prefix}_candidate_sid`
- `${prefix}_assigned`
- `${prefix}_selected`
- `${prefix}_diagnostics`

生成 SQL 时应使用确定性 hash 排序替代 `rand()`。工具需要提供 `--dry_run_sql`，用于只打印 SQL 而不执行；每轮迭代之间应执行 `SELECT COUNT(*)` 检查，用于判断是否收敛。

示例：

```bash
python -m tzrec.tools.sid.collision_prevention \
  --backend odps \
  --input_path odps://project/tables/raw_sid_table/ds=20260630 \
  --candidate_input_path odps://project/tables/sid_candidates/ds=20260630 \
  --output_path odps://project/tables/item_codebook_info/ds=20260630 \
  --max_items_per_codebook 5 \
  --temp_prefix tmp_sid_collision_20260630 \
  --odps_lifecycle 7
```

## 候选生成

防碰撞质量取决于候选覆盖率。v1 脚本只支持两类候选来源：

1. 已有候选表，例如当前 SQL 中的 `sorted_index_lv3`。
2. 显式开启的模型 KNN 候选输出，推荐方案。

对 SID 而言，v1 候选策略采用最后一级 KNN。保持贪心前缀 `codes[:-1]` 不变，计算最后一级量化前 residual 到最后一级 codebook 的 top-k 最近 code id，然后只替换最后一级 code，生成完整候选 SID 元组。该方案与现有 SQL 中“对最后一级码本 ID 防碰撞”的假设一致，也避免了全层组合搜索的复杂度。

v1 只支持显式候选表和显式开启的最后一级 KNN 候选输出。beam search 不进入第一版；v1 不新增 beam-search API、配置字段或实现路径。

## `sid_model` 需要增加的能力

防碰撞工具应保持为后处理工具，不应进入训练 loss。`sid_model` 和 SID quantizer 需要补充的能力是：在推理时输出足够好的候选 SID，供后处理分配器选择。

建议新增：

- 增加一个由 `SidRqvae` 和 `SidRqkmeans` 共用的候选输出开关；该开关默认必须关闭：
  - `enabled`：推理时是否输出候选；默认行为仍然只输出 `codes`；
  - `topk`：每个 item 输出多少个候选 SID；
  - `strategy`：v1 只支持 `last_layer_knn`；
  - `target_layer`：默认 `-1`，表示最后一层；
  - `include_origin`：是否将原始 SID 作为 priority 0 候选。
- 增加 quantizer API：
  - `get_codes(input)` 已存在，继续作为原始 SID 输出；
  - `get_code_candidates(input, topk, strategy, target_layer)`，返回形状为 `[B, K, L]` 的 `candidate_codes` 和形状为 `[B, K]` 的 `candidate_scores`；
  - 内部暴露每层量化前 residual，避免最后一级 KNN 重复计算完整残差路径。
- 只有显式开启候选输出时，预测输出才增加：
  - `codes`: `[B, L]`，保持现有行为；
  - `candidate_codes`: `[B, K, L]`；
  - `candidate_scores`: `[B, K]`。
- 对 CSV 消费场景，SID codes 只输出或转换为字符串。`code_0`, `code_1`, ... 这样的拆分列可以作为输入兼容，但 CSV 输出应将 `origin_codebook` 和 `codebook` 保存为规范字符串。
- 不要将 `max_items_per_codebook` 放进 `sid_model` 或任何 SID proto。容量上限是业务后处理策略，必须只作为 `python -m tzrec.tools.sid.collision_prevention` 的参数传入。

现有 `unique_sid_ratio` 指标仍然有价值，但它是 batch 级近似指标。精确的全局碰撞统计应由防碰撞工具输出，而不是放进 `sid_model`。

## 实施计划

1. 新增 `tzrec/tools/sid/collision_prevention.py`，包含后端无关的分配器接口。
2. 实现本地 CSV/Parquet 模式，使用 Arrow 批量加载并复用现有 writer。
3. 实现 MaxCompute 模式，包括 SQL 生成器和 PyODPS 执行器。
4. 增加候选归一化工具，支持 `codes`、`code_0...`、长表候选和紧凑字符串候选列表。
5. 在配置开关控制下，为 SID quantizer/model 增加候选输出。
6. 增加单元测试：确定性分配、满桶过滤、重复 item 防护、未分配策略、CSV/Parquet 往返。
7. 按现有 ODPS 测试模式，增加依赖环境变量的 MaxCompute gated tests。

## 待定问题

- 候选输出应通过 SID prediction CLI 参数开启，还是通过模型 export/predict config 开启；但默认必须关闭。
- 审计字段应默认写入主输出，还是只写入 diagnostics 表。
