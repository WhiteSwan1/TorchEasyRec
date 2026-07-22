# SID 碰撞工具 32 核利用方案

## 1. 结论

当前工具无法通过设置 `OMP_NUM_THREADS=32` 或继续调整单个 Parquet Reader 的 `use_threads` 来占满 32 核。实测 `use_threads=True` 后平均仍约为 1.00 个逻辑核，因为 42 GiB 的 `candidate_codes` 实质上是一个宽 Parquet 叶子列，Arrow 的“多列并行”没有足够的列级并行度。

推荐分两阶段实施：

1. **先复用 Arrow 的有序 Dataset Scanner。** 将冻结后的 Parquet row group 按原始顺序构造成 fragments，由 Scanner 的 C++ 线程池并行解码，再把有序 raw batch 送回现有 rechunk 和 candidate 处理逻辑。该路径比自行维护线程池、异常归并和 pandas 并发访问更简单。
1. **再按实测决定计算并行。** 先给 lookup、plan、resolve 和 write 增加独立计时；只有 `prepare_collision_plan` 在 candidate 加速后成为显著瓶颈，才按 band 范围评估并行 hash 和局部排序。

最新的 96 row-group 热缓存微基准中，Arrow CPU pool 上限为 16 的配置最快：原始扫描达到 10.58 倍、平均使用 13.11 核。上限为 32、但 `fragment_readahead` 仍为 16 的补跑耗时从 1.614 秒回退到 1.832 秒。它说明 16 附近值得优先测试，但不能证明“32 核必然更慢”。完整链路应重复测试 8、16、24、32，并以 wall time 而非 CPU 占用率选型。

本报告只提供设计和实验计划，没有修改生产代码。

## 2. 已知运行事实

| 项目                   |                                                     实测值 |
| ---------------------- | ---------------------------------------------------------: |
| CPU                    |                  32 个物理核，单 socket、单 NUMA、无超线程 |
| 内存                   |                                           128 GiB，无 swap |
| 存储                   |                            NFS v3，挂载自 `192.168.0.31:/` |
| 输入行数               |                                                 42,864,043 |
| Parquet row group      |                                 1,312 个，每组约 32,768 行 |
| `candidate_codes` 宽度 |               每行 600 个 `int64`，即 200 个三层 candidate |
| Parquet 编码页大小     | codec 后 41,895,630,133 字节，codec 前 42,147,374,634 字节 |
| `id` 大小              |                                             压缩约 0.38 GB |
| overflow 行数          |                                1,403,836，约占输入的 3.27% |
| 最终版本总耗时         |                                                 434.214 秒 |
| candidate 扫描估算耗时 |                                         316.7 秒，约占 73% |
| 其余阶段合计           |                                                约 117.5 秒 |
| 进程 RSS               |                                                  约 11 GiB |

42.1 GB 是移除压缩 codec 后、仍经过 RLE/bit-pack 等 Parquet 编码的 page bytes，不是 Arrow 中 `600 × int64` 的物化大小。两者接近只能说明额外 codec 的收益较小，不能据此判断信息熵。热缓存时主要限制是解码、物化和内存带宽；冷缓存时还会受到 NFS 服务端和网络吞吐限制。

## 3. 当前为什么只有一个核

`CollisionResolutionRunner.run()` 当前按顺序执行：

1. `_load_codes()`：无参调用 `reader.to_batches()`，等价于 `worker_id=0, num_workers=1`。
1. `prepare_collision_plan()`：单线程计算 band、稳定 hash、全局 `np.lexsort` 和 bucket rank。
1. `_load_candidate_last_codes()`：再次无参调用 `reader.to_batches()`，在单一 Python 循环中完成 ID 匹配、candidate 选择和写入。
1. `CollisionResolver.resolve()`：按 overflow 顺序执行贪心 first-fit。
1. 非 `rate_only` 模式下顺序写出 map 和 grouping。

仓库初始化时默认设置 `OMP_NUM_THREADS=1`，但这不是 candidate 扫描只有一个核的根因。该环境变量不控制 PyArrow Reader，也不会自动并行 `np.lexsort`、pandas `Index.get_indexer` 或 Python first-fit 循环。将其改为 32 既不能解决 Reader 分片，也可能在后续外层并行时造成嵌套过订阅。

## 4. 可以复用的框架能力

| 现有能力                                        | 可复用方式                                    | 限制                                                        |
| ----------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------- |
| `BaseReader.to_batches(worker_id, num_workers)` | 作为统一的 Reader 分片契约                    | SID 当前没有传入 worker 参数                                |
| `calc_slice_intervals()`                        | 将全局行序列切为连续、无重叠区间              | 边界可能落在 row group 内，邻接任务会重复解码边界 row group |
| `ParquetReader.to_batches()`                    | 单个大文件也能按全局连续行范围分片            | 尚无同一 Reader 被多线程并发调用的正式测试                  |
| `pyarrow.dataset.Scanner`                       | 内部并行扫描 row-group fragments，并有序返回  | CPU pool 是进程全局配置；需显式构造和测试 fragment 顺序     |
| `OdpsReader.to_batches()`                       | 对固定 Storage API session 按 row offset 分片 | 多 table/session 的有序归并和客户端并发安全需单独验证       |
| `CsvReader.to_batches()`                        | 按文件分配 worker                             | 单 CSV 文件无法扩展，轮转分配还会改变多文件全局顺序         |
| dynamic-embedding 初始化工具                    | 可借鉴 worker 生命周期和有界 Queue 背压       | 使用多进程且无序汇聚，不能直接用于 SID                      |
| hitrate 工具                                    | 展示 rank 到 Reader worker 的映射             | SID 有全局容量状态和单一输出，不能直接套用 torchrun         |

`ParquetReader` 初始化时已有一个 `ThreadPoolExecutor(os.cpu_count())`，但它只并行读取文件 metadata，不会并行正文扫描。PyArrow 17 的 C++ `ScanBatches()` 接口明确约定 batch 按 Dataset fragment 顺序到达，Python `Scanner.to_batches()` 直接使用该接口。因此首版应优先复用 Scanner，而不是新增一套通用多 Reader 归并框架；仍需单测传入 `FileSystemDataset` 的 fragment 列表顺序就是冻结后的文件、row-group 顺序。

## 5. P0：有序 Arrow Dataset Scanner

### 5.1 推荐数据流

```mermaid
flowchart LR
    A[冻结文件清单、metadata 和全局行起点] --> B[生成有序 row-group micro-slice]
    B --> C[构造有序 FileSystemDataset fragments]
    C --> D[Scanner C++ 线程池并行解码]
    D --> E[ScanBatches 按 fragment 顺序返回]
    E --> F[BaseReader 恢复现有 batch_size 边界]
    F --> G[现有 match、take 和末层 code 提取]
    G --> H[重复 overflow 广播和 missing 校验]
    H --> I[现有 first-fit resolver]
```

具体设计：

1. 第一次读取前冻结 Parquet 文件清单、文件顺序、每个文件的 row-group metadata 和全局行起点。两遍读取必须使用同一个快照，不能各自重新执行 glob。
1. 按 `(file_index, row_group_index)` 将每个文件拆成 fragments，并依冻结顺序显式构造 `FileSystemDataset`。不要重新 glob 或依赖文件系统枚举顺序。
1. 使用 `Scanner(use_threads=True)` 只投影 `id` 和 `candidate_codes`。Scanner 内部完成并行读取和解码，主 Python 线程只消费有序的 raw RecordBatch。
1. 将 Scanner 输出继续送入现有 `BaseReader._arrow_reader_iter()`，跨 fragment 合并并恢复完全相同的逻辑 `batch_size`；随后复用现有 `Index.get_indexer`、`pc.take`、`_candidate_last_matrix`、进度和业务校验逻辑。
1. candidate 变换和矩阵写入首版保持单线程。Scanner 保证 batch 到达顺序后，跨 batch 的重复 source ID 仍由后出现记录覆盖。
1. 所有任务完成后，继续执行现有 `broadcast_duplicate_targets()`、missing 检查和 resolver，不改变下游接口。

该方案直接复用 Arrow 的线程调度、保序和 readahead，新增代码集中在“冻结并构造 fragments”以及 Reader 接口衔接，不需要自行实现 future reorder buffer。Scanner 的 fragment 顺序、每一行内容和现有 batch 重切结果仍必须由差分测试确认。

### 5.2 并行度配置

- `pa.set_cpu_count(W)` 调整的是进程全局 Arrow CPU pool，不是某个 Scanner 的 `max_workers`。配置名应为 `parquet_scan_threads`，不能叫 `candidate_reader_workers`。
- SID CLI 是单进程、单任务，可以在扫描前记录原值、设置 W，并在 `finally` 中恢复；测试必须证明异常路径也会恢复。若未来同进程同时运行其他 Arrow 工作，不能在任务中途修改全局值。
- `fragment_readahead` 负责限制同时预读的 row-group fragments，首轮内部取 `min(W, 16)`；`batch_readahead` 先沿用 16。两者影响内存，不必在第一版暴露为客户参数。
- 完整链路按 W=8、16、24、32 重复测量，并补测 W=32/`fragment_readahead=32`，避免把“线程池上限”和“实际同时预读 fragment 数”混为一谈。

### 5.3 后备方案

若 ordered Scanner 在完整链路中无法扩展、不能覆盖目标文件系统，或内存无法受控，再采用显式 row-group `ThreadPoolExecutor`：任务携带全局序号，只返回 raw batch，由协调线程按序 rechunk 和处理。进程池不是首选，因为它需要复制或 IPC 传输宽 Arrow buffer、ID lookup 和约 2.25 GB 的 candidate 矩阵。

只有分段计时证明 `get_indexer`、`take` 或末层提取成为下一瓶颈时，才考虑第二级计算并行。此时必须先验证共享 `pd.Index` hash engine 的线程安全，或使用线程本地 Index；不能用一个全局锁把 lookup 再次串行化。

## 6. 必须保持的结果语义

并行化不能只保证 collision stats 相同，必须保持每一行最终 SID 相同。

### 6.1 重复源 ID

目标语义是：同一 item ID 在源数据中多次出现时，物理输入顺序中最后一次出现的 candidate 生效，然后广播给所有重复 overflow 行。最后一次记录即使是 retained/non-overflow 行，也必须为相同 overflow ID 提供 candidate。

Scanner 的有序 batch 保留跨 batch 的覆盖顺序；同一 batch 内多个 source row 命中同一 target 时，不能依赖 NumPy 重复高级索引赋值的隐式顺序。实现应携带全局 source row，只保留每个 target 最大 source row 后再写代表位置，最后执行现有 duplicate broadcast。

### 6.2 文件与任务顺序

全局顺序必须明确为：冻结后的文件顺序、文件内 row-group 顺序、row-group 内行顺序。多文件不能用 `files[worker_id::num_workers]` 后再按 worker 汇总，因为这会把 `0,1,2,3` 的顺序变成 `0,2,1,3`。

### 6.3 错误与取消

- candidate 字段、宽度和跨 batch top-k 等现有业务校验仍在 ordered Scanner 输出完成 rechunk 后执行，并保持“只校验命中 candidate 行”的当前范围，不能无意扩展为全表校验。
- `ScanBatches()` 保证成功 batch 的返回顺序，但没有承诺多个并发读取错误按最早 fragment 抛出。读取异常应补充 file/row-group 上下文，保证不创建部分输出，并在 `finally` 中释放 iterator 和恢复 Arrow 全局配置；不要承诺逐字相同的错误文本或强制取消所有已启动任务。
- 进度只在有序 batch 进入现有处理点时由主线程更新；Scanner 内部任务不输出进度，因而调度顺序不会改变日志计数。

## 7. 内存和背压设计

本数据每个输入行包含 600 个 `int64` candidate，原始 payload 为 4,800 字节；命中后保留 200 个末层 code，即 1,600 字节。最终 overflow candidate 矩阵约为：

```text
1,403,836 × 200 × 8 ≈ 2.25 GB
```

一个 row group 约 32,768 行，物化 600 个 `int64` 的 candidate payload 约为 157 MiB。仅按同时预读 fragment 数估算：4、8、16 个 row group 分别约为 0.63、1.26、2.52 GiB；这还不含编码页、offset、临时 Arrow/NumPy 数组、Scanner batch readahead 和最终约 2.25 GB 的 candidate 矩阵。

Scanner 自带有界 readahead，不会一次物化全部 1,312 个 fragments，但没有按字节设置上限的公开参数。首轮以 `fragment_readahead<=16` 为约束，目标是 decoded buffer 估算不超过 4 GiB，并实际记录 RSS 与 Arrow allocated bytes；若其他数据集的 row group 更大，应自动降低 readahead。不要假定峰值一定是 raw payload 的固定倍数。

`batch_size` 只控制 Scanner 输出和现有 rechunk 的 I/O batch，并不等于并行度。客户侧只需看到 `parquet_scan_threads`；readahead 先作为受内存约束的内部策略。若未来必须提供严格字节背压，再启用 5.3 的显式任务池后备方案。

## 8. 实测扩展性与性能上限

### 8.1 96 row-group 热缓存微基准

下表把同一 Parquet 文件的前 96 个 row group 构造成 96 个 Dataset fragments，只读取 `id` 和 `candidate_codes`，共 3,145,728 行。W 是 `pa.set_cpu_count(W)` 设置的 Arrow 全局 CPU pool 上限，不是 Python worker 数；`fragment_readahead=min(W,16)`、`batch_readahead=16`。实验只校验行数，不包含 lookup、candidate 提取、有序业务处理、resolve 或写出，也没有校验内容 digest。

| Arrow CPU pool 上限 | fragment readahead |     耗时 |      吞吐 | 相对 W=1 | 平均使用核数 |
| ------------------: | -----------------: | -------: | --------: | -------: | -----------: |
|                   1 |                  1 | 17.081 s | 184.2 K/s |    1.00x |         1.01 |
|                   4 |                  4 |  5.116 s | 614.9 K/s |    3.34x |         3.78 |
|                   8 |                  8 |  2.548 s | 1.235 M/s |    6.70x |         7.39 |
|                  16 |                 16 |  1.614 s | 1.949 M/s |   10.58x |        13.11 |
|                32\* |                 16 |  1.832 s | 1.717 M/s |    9.32x |        13.08 |

`32*` 是在新 Python 进程中单独补跑的单次观测，其 fragment readahead 仍为 16；其耗时比 W=16 配置高约 13.5%。因此这组热缓存数据只说明 W=16 值得优先验证，并暗示当前路径可能在约 13 个有效核心附近遇到内存带宽或调度限制，不能据此断言 32 核配置固有退化。

### 8.2 端到端乐观上限

最终版本总耗时为 434.214 秒，其中 candidate 约 316.721 秒，其余阶段约 117.493 秒。若只并行 candidate，则 Amdahl 上限为：

```text
T(S) = 117.493 + 316.721 / S
```

| 套用的 raw Scanner 加速比 | 端到端乐观耗时 | 端到端乐观加速 |
| ------------------------: | -------------: | -------------: |
|                     1.00x |        434.2 s |          1.00x |
|              3.34x（W=4） |        212.4 s |          2.04x |
|              6.70x（W=8） |        164.7 s |          2.64x |
|            10.58x（W=16） |        147.4 s |          2.95x |
|           9.32x（W=32\*） |        151.5 s |          2.87x |
|     32x（不现实的理想值） |        127.4 s |          3.41x |
|                    无限快 |        117.5 s |          3.70x |

这些是把热缓存原始扫描加速直接套到完整 candidate 阶段的**乐观上限**，不是性能预测。完整阶段还包含 ID 转换、哈希匹配、`take`、candidate 矩阵转换和有序归并，实际加速会更低；冷缓存还会受 NFS 限制。

验收目标应是端到端耗时，而不是 CPU 使用率：若 W=16 已达到重复实验峰值吞吐的 95%，则没有必要为了显示 32 核满载而使用 W=32。

## 9. P1 候选：并行 `prepare_collision_plan`

现有日志中的 candidate 前后空窗还包含 concatenate、plan、`_ItemIdLookup` 构建、Reader 初始化和首个 batch，无法证明 plan 独占约 85～95 秒。P0 首先应分别计时；只有 plan 在优化后超过端到端耗时的 10%，才进入这一阶段。

可验证的设计方向是利用“不同 band 之间绝不互相迁移”的性质：

1. 先用 O(N) histogram 统计各 band 行数，按累计行数选择连续、尽量均衡且不拆分 band 的范围；极端单 band 数据必须回退串行。
1. 按行分块并行计算 `stable_order_hash(item_ids)` 和 `_band_ids(codes)`，再按上述范围物化局部行索引；需单独测量这一分区过程，避免新增 O(N) copy 抵消排序收益。
1. 每个任务只对本范围执行 `np.lexsort((order_hash, last_code, band_id))`，生成局部 bucket rank、bucket count 和代表行。
1. 主线程用 bucket 数量前缀和添加全局 bucket ID 偏移，并按 band 范围顺序拼接和散射结果。

开始实现前必须用微基准确认并发 `np.lexsort` 是否释放 GIL、分区是否均衡，以及局部排序加分区的总耗时确实优于一次全局排序；否则保持串行。不要为此直接新增 Polars、DuckDB、Dask 或 Ray 依赖。

## 10. 后续低优先级阶段

### 10.1 First-fit resolver

first-fit 只占当前总耗时约 1%～2%。它可以按 band 并行，因为不同 band 的目标 bucket 不相交；每个 band 内仍必须保持 overflow 顺序。除非 candidate 和 plan 优化后它达到 10% 以上，否则不值得优先增加复杂度。

### 10.2 首遍 SID 读取

`_load_codes()` 当前看起来只占较小比例。可以复用同一个 ordered Scanner，让主线程按序 concatenate ID/code chunk 并统一校验 Arrow ID 类型，但应先独立计时，并放在 candidate 之后。

### 10.3 输出写入

当前性能实验使用 `--rate_only`，没有测量约 9 GiB 输出的真实写入瓶颈。不能根据 rate-only 日志推断 Writer 是否需要 32 核。后续应先独立记录三个输出的构造、压缩和 NFS 写入耗时；若确有必要，可让三个独立输出 writer 并发，但不能并发写同一个 Parquet 文件。

## 11. 数据格式分阶段支持

- **Parquet：** P0 首发范围。它已有 metadata、row-group 和连续全局行区间，能够在单大文件上扩展。
- **CSV：** P0 保持串行。当前 Reader 只按文件轮转分片，单文件无法扩展，多文件还需要恢复原始文件顺序。
- **ODPS：** P0 保持串行。Storage API 支持 row range 并发，但必须复用同一 read session，并按 `(table, session, start)` 有序提交；还需验证客户端线程安全、quota 和服务端限流。不能为每个 worker 新建独立 Reader/session，否则快照和行序可能变化。
- **random 策略：** 不读取 candidate，因此 `parquet_scan_threads` 不应影响这一路径。先根据阶段计时决定是否需要优化 plan；random candidate 生成和 first-fit 当前不是优先项。

## 12. 实验矩阵与停止条件

### 12.1 P0 实验顺序

1. 在现有串行实现中增加阶段计时，但不改变计算结果；分别记录 read、plan、lookup 构造、Parquet decode、hash match、`take`/转换、commit、resolve 和 write。
1. 使用同一文件快照和热缓存，比较当前 Reader 基线与 ordered Scanner 的 `W = 1, 4, 8, 16, 24, 32`；每个配置至少重复三次并交错顺序。
1. W=32 分别测试 `fragment_readahead=16/32`，并和 W=16/readahead=16 对照；Scanner 均使用 `use_threads=True`。
1. 记录 candidate wall time、端到端 wall time、user/system CPU、平均及峰值核数、RSS、Arrow allocated bytes、NFS/进程读字节、fragment/batch readahead 和输出 digest。
1. 对选中的 W 补跑冷缓存或独立文件快照，确认收益不是 page cache 偶然结果。
1. 最后只对最佳配置做一次真实输出实验，记录并清理新增的大文件。

### 12.2 停止条件

- W 翻倍后 candidate 吞吐提升小于 10%，或端到端提升小于 5%。
- RSS 超过物理内存的 75%，出现显著 page-cache 抖动或 Arrow allocation 峰值失控。
- NFS 吞吐已饱和，提高 W 只增加 system CPU 或延迟。
- 任意最终数组、输出 digest、异常类型或重复 ID 行为与 W=1 reference 不一致。

最终默认值应选择“达到峰值吞吐 95% 所需的最小 W”，而不是固定为 32。

## 13. 必需测试

所有新增函数和能力都需要相应 unittest，并在收尾阶段检查测试是否放在职责对应的文件中。

### 13.1 Candidate 并行

- W=1、4、16 在整数和字符串 ID 下的 candidate 矩阵、`resolved_last_codes`、`slot_indices`、unresolved rows 和 stats 逐项一致。
- 显式验证 `FileSystemDataset` fragment 顺序、逐 batch content digest 和全局行覆盖与串行 Reader 一致。
- 重复源 ID 位于同一 batch、跨 batch、不同 row group 和不同文件时，仍由全局 source row 最大者生效；最后记录是 retained/non-overflow 行时也一样。
- 重复 overflow ID 广播与串行版本一致。
- row group 小于/大于 `batch_size`、`batch_size=1` 和任务边界穿过逻辑 batch 时，重组后的 batch 内容与串行版本逐项一致。
- candidate 缺失、null、ragged、宽度变化、top-k 跨 fragment 变化时，业务校验的类型、消息和触发范围与当前实现一致。
- 无 overflow 时不创建 Scanner；W 大于 row-group 数时不产生空任务错误。
- Scanner 读取异常包含 fragment 上下文、不产生部分输出，并释放 iterator；不要求内部尚未开始的 C++ 任务具备可观测的取消行为。
- 多文件清单冻结、两遍扫描快照一致、任务覆盖无空洞无重叠、总行数严格一致。
- 进度计数单调且最终等于实际扫描行数。
- `pa.set_cpu_count()` 在正常、异常和提前返回路径均恢复原值，测试之间不泄漏全局状态。
- random/CSV/ODPS 路径不进入 Scanner 并行分支。

### 13.2 Plan 并行

- 随机生成不同 band 分布、极端单 band、空输入和大量相同 SID，对串行/并行 plan 的所有数组逐项比较。
- band 分区边界恰好穿过相同 band 时必须自动调整，不能拆分该 band。
- string/int ID 的 stable hash、bucket rank、overflow 顺序和代表行保持一致。
- W=1 回退必须走与当前串行实现等价的路径。

### 13.3 真实数据验收

`--rate_only` 不写出 map，因此不能只比较六个 stats。应在 result 被释放前计算 `resolved_last_codes`、`slot_indices`、`unresolved_rows` 和 bucket metadata 的稳定 digest，并和 W=1 reference 比较；最终候选版本还应至少完成一次真实输出的逐文件/逐列校验。

## 14. 预计涉及的文件

以下只是未来实现范围，本报告没有修改它们：

- `tzrec/tools/sid/resolve_sid_collisions.py`：Scanner 并行度配置、阶段计时和 candidate 顺序处理。
- `tzrec/datasets/parquet_dataset.py`：可选的 ordered Scanner 路径和冻结后的 Parquet fragment 构造能力。
- `tzrec/utils/sid/collision.py`：后续按 band 并行 plan 和可选 resolver。
- 对应的 `*_test.py`：Reader fragment/Scanner 测试、SID 顺序语义测试和可选 plan 差分测试。

## 15. 推荐落地顺序

1. **PR-A：** 阶段计时、结果 digest 和冻结的 Parquet fragment 清单，不改变默认执行路径。
1. **PR-B：** 可选 ordered Scanner 路径和单一 `parquet_scan_threads` 配置；完成 W=1/4/8/16/24/32 的重复实验后再决定默认值。
1. **PR-C：** 只有独立计时证明必要且微基准确认有效时，才按 band 并行 `prepare_collision_plan`。
1. **PR-D：** 只有测量证明必要时，再考虑显式任务池、ODPS、CSV、first-fit 和 Writer。

现有单次方向性数据把 W=16/readahead=16 指向首要候选，而不是证明它就是默认值。若完整链路在 16 附近达到峰值，就应接受部分核心空闲；强行提高 CPU 占用反而可能增加 wall time。只有阶段计时确认其他串行计算值得并行时，才继续扩展 plan 等阶段。
