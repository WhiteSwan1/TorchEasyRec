# SID 碰撞逐轮仲裁与多进程重构验证报告

## 1. 本轮复测结论

本报告以提交 `2c970ff` 为基线，并记录其上的未提交分片传输改造。小规模性能
数字仍来自基线提交；第 6 节单独对比改造前的 object 分发失败和改造后的
Tensor 分块传输复测，避免把两套实现混为一谈。

500 万行数据上的主要结论如下：

- first-fit 从 1 进程增加到 8 进程，只从 19.81 秒降至 18.45 秒，即
  1.07 倍加速，却额外占用约 9.46 GiB 聚合峰值内存。默认应使用单进程。
- iterative 的 1、8、32 进程分别耗时 167.59、37.25、26.89 秒。8 进程相对
  单进程加速 4.50 倍，收益明确，适合作为默认并行起点。
- iterative 从 8 增加到 32 进程只再加速 1.39 倍，却额外占用约
  28.77 GiB。只有时延优先且内存充足时才建议使用 32 进程。
- 所有并发度下，同一策略的聚合统计一致；但本轮使用 `rate_only`，没有输出
  map，因此该性能实验本身不证明逐 item 结果逐位一致。
- 上述并发建议只适用于 500 万行、每条 16 个候选的基准。基线实现在真实
  255,172,938 行、每个 overflow item 200 个候选的 8 rank 实验中，于任务
  分发阶段达到 125.60 GiB RSS 后失败。本轮改造以 94.34 GiB 全程峰值完成
  相同任务，wall time 为 101 分 33 秒，并完整写出三份结果。
- 与 first-fit 的逐 item 流式比较表明，两者迁移率分别为 99.4750% 和
  99.5630%，但有 8,448,288 个 item 的最终 SID 不同。策略切换会影响实际
  结果，不能只看接近的聚合迁移率。

## 2. 当前实现

### 2.1 策略语义

- `strategy=candidate|random` 决定候选如何产生。
- `placement_policy=first_fit|iterative` 决定如何使用候选。
- first-fit 按 item 依次检查候选列，将其放入第一个未满且不等于原始位置的
  桶。
- iterative 按完整 SID 前缀 band 独立执行逐轮仲裁，同一个 band 不跨
  rank 拆分。

iterative 每轮先删除已分配 item 的边和指向满桶的边，再按候选优先级、稳定
item hash 和原始行号选择 proposal。一个 item 同轮被多个桶接受时，只提交
优先级最高的目标；没有新 winner 时结束。无法迁移的 item 保留原始 SID。

### 2.2 多进程数据流

rank 0 独占 TorchEasyRec Reader、Writer 和 commit：

1. 读取全量 `item_id`、原始 `codes` 和所需候选，生成
   `CollisionPlan`。
1. 在读取候选前按完整 SID prefix band 划分 overflow 行。候选宽度对所有行
   相同，因此分片成本使用 `candidate_count=1`；测试确认 K=1 与 K=200 的
   分片边界一致。
1. `_ShardedCandidates` 为每个 rank 独立分配候选矩阵。candidate 输入扫描
   后直接写入目标 rank 所属矩阵；random 策略同样按 shard 分块生成，不再先
   构造连续的全量候选矩阵再复制。
1. 重复 string/int item ID 延续既有 last-wins 语义；代表位置的候选会复制
   到其他重复 overflow 行，包括跨 batch 和跨 rank 的情况。
1. rank 0 按 shard 计算 iterative 所需的 `order_hashes`。
   `build_collision_work` 复用已经对齐且连续的 candidate/hash 数组，只复制
   该 shard 所需的 plan 数组。
1. `send_object_list` 只发送 band、配置、shape 和 dtype。7 个数值数组通过
   `dist.send/recv` 以不超过 128 MiB 的 Tensor 片段传输，接收端直接预分配
   最终数组。
1. `take(rank)` 转移候选 shard 所有权；远端 shard 发送完成后立即删除对应
   Work。所有 rank 接收完成后通过 barrier 同步，再开始仲裁。
1. 各 rank 只解析自己的 shard。
1. 使用 `dist.gather_object` 将全部 `CollisionShardResult` 收集到 rank 0，
   合并后由 rank 0 写输出。
1. 使用 `dist.broadcast_object_list` 将最终统计返回给所有 rank。

因此 CSV、Parquet、ODPS 都不会被多个 rank 重复读取或写入，也不需要
`distributed_work_dir`。数值任务不再整体 pickle；结果 gather 仍会让
rank 0 同时持有所有 shard 结果。当前实现没有独立的 READY/NACK、取消或
流式结果协议，通信失败由 Gloo 超时和 `torchrun` 进程管理处理。

### 2.3 内存与 I/O

本次改造限制的是传输片段和部分候选临时矩阵，不是整个程序的总内存：

- 单次 Tensor 传输片段不超过 128 MiB，数值数组不再整体 pickle；
- 重复候选复制和 random 候选生成使用按候选宽度计算的分块行数；
- 发送每个远端 shard 后，rank 0 可立即释放该 shard，不再同时保留原矩阵、
  分片副本和序列化副本。

仍随数据规模增长的内存包括：

- 候选扫描结束、发送开始前，rank 0 仍持有所有独立 shard，其总量仍为
  `O(overflow_count × K)`；
- rank 0 仍持有全量 item ID、codes 和 plan，每个 worker 仍持有自己的完整
  candidate shard；
- candidate I/O batch、NumPy 高级索引和 iterative 逐 band 临时数组仍会
  产生额外内存；
- rank 0 的结果 `gather_object` 仍随结果规模增长。

因此准确表述是“分块 Tensor 传输与逐 shard 释放”，不是“整体内存有界”。
`batch_size` 只控制 Reader 的 I/O batch，不限制最终候选、plan 或 resolver
工作集。每个 worker 还会导入 TorchEasyRec 框架模块；进程数增加时，固定
的进程初始化内存也不可忽略。

### 2.4 主要代码落点

| 文件                                        | 职责                                   |
| ------------------------------------------- | -------------------------------------- |
| `tzrec/utils/sid/collision.py`              | 通用 plan、shard、first-fit 和候选生成 |
| `tzrec/utils/sid/iterative_collision.py`    | SQL 风格逐轮仲裁                       |
| `tzrec/utils/sid/collision_sharding.py`     | 按完整 band 和工作量进行连续分片       |
| `tzrec/utils/sid/distributed_collision.py`  | Gloo 初始化、Tensor 任务传输和结果收集 |
| `tzrec/tools/sid/resolve_sid_collisions.py` | rank 0 I/O、任务分发、结果合并和 CLI   |

## 3. RQ-VAE candidate scores

RQ-VAE 开启候选输出时会同时生成：

- `candidate_codes`：量化器中为 `(B, K, L)`，模型输出展平为
  `(B, K * L)`；
- `candidate_scores`：形状为 `(B, K)`，与第 `K` 个候选对齐。

候选已按 score 升序排列。当前碰撞工具不直接读取 `candidate_scores`：
first-fit 使用候选列顺序，iterative 使用列号作为 priority，再以稳定 hash
解决同优先级竞争。这与原始 `ID_collision.sql` 仅使用 priority 的核心语义
一致。

## 4. 正确性验证

本次分片传输改造新增或扩展了以下覆盖：

- candidate/random 与 first-fit/iterative 四种组合的分布式结果、统计和输出
  与单进程一致；
- string 重复 item ID 在同 batch、跨 batch 和跨 rank 时保持既有候选匹配
  语义；
- int/string random 候选分片与原单矩阵结果逐项一致；
- 空 shard、独立内存和所有权只能转移一次；
- K=1 与 K=200 的分片边界一致；
- shard-local candidate/hash 构建 Work 时不额外复制；
- 元数据协议、128 MiB 分块上限、空数组、uint64、dtype/连续性校验和
  barrier；
- 分布式路径不会调用原来的全量候选矩阵构造函数。

本轮相关 unittest 96 项通过，全部 SID 相关回归 209 项通过；另完成真实
4 rank 全输出 smoke test和 2 rank Gloo uint64 往返测试。`pre-commit`、
`git diff --check` 和 Pyre 均通过。

下表属于基线提交的 500 万行性能复测。分布式编排的单测通过 mock 通信验证，
性能复测则真实执行了 8 rank 和 32 rank `torchrun`。五组运行返回码均为 0，
同一策略在不同 rank 数下得到完全相同的统计：

| 策略      | 原始超容量桶 |    迁移数 |  未解决数 | 最终超容量桶 | 最大桶 |
| --------- | -----------: | --------: | --------: | -----------: | -----: |
| first-fit |       80,000 | 3,000,000 | 1,600,000 |       40,000 |     60 |
| iterative |       80,000 | 3,000,000 | 1,600,000 |       80,000 |     44 |

这里的“桶”是完整 SID 相同的一组 item，而不是某一层的单个 code。本次容量
`max_items_per_codebook=5`，即解析目标是让每个完整 SID 不超过 5 个 item；
未解决 item 仍会保留，所以最终桶可能继续超容量。各字段含义如下：

- **策略**：使用候选 SID 的放置方式。first-fit 按 item 顺序选择第一个可用
  候选；iterative 通过多轮 proposal 和仲裁决定目标。
- **原始超容量桶（`raw_collision_buckets`）**：处理前 item 数严格大于容量
  5 的完整 SID 数量。80,000 表示原始数据中有 80,000 个 SID 各自包含至少
  6 个 item，并不表示只有 80,000 个 item 发生碰撞。
- **overflow item**：每个原始桶按稳定顺序保留前 5 个 item，其余 item
  才需要重新放置。它没有单独列在表中，数量恒等于
  `迁移数 + 未解决数`。
- **迁移数（`relocated_count`）**：成功放入其他可用末层 code 的 overflow
  item 数量。SID 前缀保持不变，只有最后一层 code 被替换；这些 item 不再
  占用原始超容量位置。
- **未解决数（`unresolved_count`）**：遍历完可用候选后仍找不到容量的
  overflow item 数量。工具不会删除它们，而是保留其原始 SID，因此原始桶
  可能继续超过容量。
- **最终超容量桶（`final_collision_buckets`）**：全部迁移结束后，item 数
  仍严格大于 5 的完整 SID 数量。它衡量还有多少个碰撞桶，不等于未解决 item
  数量。
- **最大桶（`max_final_bucket_size`）**：最终所有完整 SID 中包含 item
  最多的那个桶的实际 item 数，用来观察最严重的热点。容量虽然是 5，但未
  解决 item 保留在原桶，所以该值仍可能大于 5。

本组数据共有 4,600,000 个 overflow item：
`3,000,000 + 1,600,000 = 4,600,000`。两种策略都成功迁移约 65.22%，未解决
约 34.78%，并且所有 5,000,000 个输入 item 都保留在结果中。

“最终超容量桶更多”与“最大桶更小”并不矛盾：两种策略遗留 item 总数相同，
first-fit 将遗留项集中在较少原始桶，iterative 则把它们分散到更多桶。评估
效果时应联合观察 `unresolved_count`、`final_collision_buckets` 和
`max_final_bucket_size`。

## 5. 500 万行性能复测

### 5.1 环境与方法

- 代码：`feat/sid_collision_torchrun`，提交 `2c970ff`
- CPU：Intel Xeon Platinum 8369B，32 物理核、64 逻辑核
- 内存：369 GiB
- 输入：5,000,000 行 Parquet，378,293,269 bytes（约 361 MiB）
- Schema：int64 `item_id`、2 层 int64 `codes`、每条 16 个候选（展平后
  32 个 int64）
- 输入 SHA-256：
  `48ec90c59d7e274076af4caa5e3411edd2e618bd8149817482fed11590691aea`
- 配置：`codebook=20000,64`、容量 5、candidate 策略、每条 16 个候选
- I/O：`batch_size=100000`、`progress_interval=1000000`
- 环境：`OMP_NUM_THREADS=1`
- 模式：`rate_only`，包含输入读取、plan、候选扫描、分发和解析，但不包含
  map/group 输出写入

使用同一输入和同一提交顺序执行五组测试。Wall time 包含 Python/torchrun
启动；CPU time 与 RSS 均聚合根进程及其子进程。平均核数按
`CPU time / Wall time` 计算。数据页可能已进入系统缓存，且每组只执行一次，
因此小于约 5% 的差异不应被解释为稳定收益。

原始计量结果保存在 gitignored 的
`experiments/rerun_2c970ff_*_5m.json`，便于复核命令和未舍入数据。

### 5.2 本轮结果

| 策略      | 进程 | Wall time | CPU time | 平均核数 |  峰值 RSS |
| --------- | ---: | --------: | -------: | -------: | --------: |
| first-fit |    1 |   19.81 s |  19.69 s |     0.99 |  2.50 GiB |
| first-fit |    8 |   18.45 s |  58.94 s |     3.19 | 11.96 GiB |
| iterative |    1 |  167.59 s | 167.47 s |     1.00 |  2.50 GiB |
| iterative |    8 |   37.25 s | 208.06 s |     5.59 | 11.87 GiB |
| iterative |   32 |   26.89 s | 390.33 s |    14.52 | 40.64 GiB |

对比关系：

- first-fit 8 rank 相对单进程快 1.07 倍，wall time 下降 6.86%，但 CPU
  time 增加 39.25 秒、内存增加 9.46 GiB。
- iterative 8 rank 相对单进程快 4.50 倍，wall time 下降 77.77%，内存
  增加 9.37 GiB。
- iterative 32 rank 相对单进程快 6.23 倍；相对 8 rank 仅快 1.39 倍，
  wall time 再下降 10.36 秒，CPU time 增加 182.27 秒、内存增加
  28.77 GiB。
- 32 rank 平均只使用 14.52 核，说明串行 I/O、rank 0 分发/汇总、负载尾部
  和进程启动仍限制了 32 核利用率。

### 5.3 为什么 first-fit 多进程收益很小

这里不是“多进程没有并行”，而是可并行的 first-fit 核心太短，节省时间被
串行阶段和分布式固定开销抵消。

当前 first-fit 的端到端流程中，只有
`CollisionResolver._resolve_first_fit_shard` 按 band 分片并行。以下工作仍由
rank 0 串行完成：

1. 首次读取 `item_id` 和 `codes`，再构建全量 `CollisionPlan`；
1. 第二次扫描输入、匹配 overflow item 并生成完整候选矩阵；
1. 分片复制并依次发送任务；
1. gather、合并、校验，以及非 `rate_only` 模式下的全部输出写入。

额外的无源码修改阶段计时显示：单进程总耗时 19.94 秒，其中首次读取
0.83 秒、plan 2.07 秒、候选扫描 6.09 秒，first-fit 解析 4,600,000 个
overflow item 需要 6.29 秒。resolver 只占端到端时间约 31.5%。即使假设
8 rank 能将它完美缩短为八分之一，并且通信完全免费，Amdahl 上限也只有：

```text
T8_ideal = (19.94 - 6.29) + 6.29 / 8 = 14.44 s
speedup_ideal = 19.94 / 14.44 = 1.38x
```

8 rank 阶段计时为 18.15 秒：rank 0 的首次读取 0.79 秒、plan 2.00 秒和
候选扫描 5.89 秒，共 8.68 秒完全串行。每个 shard 的 resolver 只需
0.755–0.775 秒，但 rank 0 的 7 次 `send_object_list` 顺序发送共需约
1.08 秒，worker 因而错峰启动，使 resolver 并行窗口被拉长到 1.86 秒，
有效平均并发只有约 3.28，而不是 8。随后 gather 和 merge 还需约
0.63 秒。

多进程把 load-to-finish 阶段从 15.28 秒降至 11.46 秒，节省 3.82 秒；
但启动、重复框架导入、Gloo 初始化和退出等区间从 4.66 秒增至 6.69 秒，
又增加 2.03 秒，最终只快约 1.10 倍。该诊断用于解释阶段占比，小数会受
单次波动影响；其结论与正式测试的 1.07 倍一致。

这些额外开销具体来自：

- rank 0 为所有 shard 复制约 4,600,000 × 16 个 int64 候选，候选本身约
  561.5 MiB，三个 overflow 对齐数组还要复制约 105 MiB；
- `send_object_list` 需要 pickle 和复制对象，并由 rank 0 依次向 7 个
  worker 发送；
- `gather_object` 将所有结果同时收回 rank 0，随后还要校验和合并；
- 每个进程都要导入框架。大部分 I/O 阶段只有 rank 0 工作，其余 worker
  等待任务或最终 broadcast。

因此平均使用 3.19 核并不是共享 GIL 或 `OMP_NUM_THREADS=1` 阻止了
多进程：每个 worker 有独立 Python 解释器，first-fit 的 Python 循环可以并发
执行。它本身是 Python 嵌套循环、`tolist()` 和 `dict` get/set 组成的标量
计算，并非可由 OMP 自动展开的 NumPy/BLAS 内核。当前每个 rank 只处理约
575,000 个 overflow item，单个 shard 不足 0.8 秒，计算粒度不足以摊薄固定
成本。iterative 的单进程核心耗时远长于 first-fit，因而可以摊薄相同的启动和
通信成本，8 rank 才能取得 4.50 倍端到端加速。

非 `rate_only` 场景还会增加 rank 0 串行写出时间，因此 first-fit 的并行
占比通常会进一步下降。是否值得并行应看 resolver 占端到端耗时的比例，而
不能只看总行数；当前配置应继续使用单进程 first-fit。

阶段计时原始记录保存在 gitignored 的
`experiments/diagnostic_2c970ff_firstfit_*`。

### 5.4 与旧报告数字的关系

| 策略      | 进程 |   旧报告 | 本轮复测 |    变化 |
| --------- | ---: | -------: | -------: | ------: |
| first-fit |    1 |  20.66 s |  19.81 s |  -4.10% |
| first-fit |    8 |  17.23 s |  18.45 s |  +7.10% |
| iterative |    1 | 232.35 s | 167.59 s | -27.87% |
| iterative |    8 |  36.23 s |  37.25 s |  +2.82% |
| iterative |   32 |  27.65 s |  26.89 s |  -2.74% |

旧报告来自不同中间实现，传输方式和部分代码均已变化，不能作为严格 A/B。
其中 iterative 单进程基线变化较大，说明旧基线已经失效；其余几组的单次波动
也再次表明应以本轮同一提交内的横向对比为准，不能把差值全部归因于某一个
函数。

## 6. 真实 2.55 亿行远端实验

### 6.1 实验设置

在一台 32 核、128 GiB 内存、无 swap 的机器上创建了独立 shallow clone，
并将本轮 5 个已修改源码/测试文件复制到该实验目录：

- 基线分支：`feat/sid_collision_torchrun`
- 基线提交：`2c970ff`
- 实验代码：上述提交加本轮未提交分片传输改造
- 输入：6 个 Parquet，255,172,938 行，合计 512.44 GiB
- Schema：string `id`、3 层 `codes`、每条 200 个候选 SID
- 配置：`codebook=8192,8192,8192`、容量 5、candidate 策略
- first-fit：使用现有单进程完整产出作为对照
- iterative：只运行 8 rank `torchrun`，没有运行单进程
- 磁盘：启动前约剩余 90 GiB；输入通过只读软链接复用，没有复制

iterative 输出被配置到独立的 `sid_collision_iterative_bounded`，不会覆盖
已有 first-fit 结果。运行前 94 项相关 unittest 全部通过。

### 6.2 first-fit 单进程基线

现有 first-fit 完整产出来自旧提交 `c99127a`，参数与输入一致，没有使用
`rate_only`：

| 总 item     | 原始超容量桶 |     迁移数 | 未解决数 | 最终超容量桶 | 最大桶 |
| ----------- | -----------: | ---------: | -------: | -----------: | -----: |
| 255,172,938 |    3,719,947 | 29,264,462 |  128,456 |        1,946 |  3,054 |

overflow item 共 29,392,918，迁移成功率约 99.563%，未解决率约
0.437%。三个输出均完整：

| 输出               |        行数 |      大小 |
| ------------------ | ----------: | --------: |
| item → SID map     | 255,172,938 | 4.186 GiB |
| 原始 SID → itemids | 167,939,082 | 2.529 GiB |
| 最终 SID → itemids | 177,583,296 | 2.572 GiB |

`history.log` 从首个读取进度点到最终统计为 5,983.41 秒，即 99 分 43 秒；
日志未记录进程启动时刻、CPU 和 RSS，因此这只是接近完整 wall time 的历史
参考。它来自旧提交，不能与最新 iterative 耗时构成严格的策略性能 A/B。

### 6.3 改造前 object 分发失败基线

基线提交 `2c970ff` 的旧分发实现，在完成全量候选扫描后、进入 shard 序列化
和发送时失败，尚未进入逐轮仲裁，也没有创建任何输出：

| 指标             | 结果                       |
| ---------------- | -------------------------- |
| 返回码           | 1                          |
| 失败前 wall time | 3,387.09 秒（56 分 27 秒） |
| 聚合 CPU time    | 3,300.18 秒                |
| 聚合峰值 RSS     | 125.60 GiB                 |
| cgroup 内存上限  | 128 GiB                    |
| cgroup failcnt   | 351                        |
| 终止信号         | rank 0 收到 SIGTERM        |

主要阶段约为：

1. 首次读取 2.55 亿行：约 1 分钟；
1. rank 0 构建和排序 `CollisionPlan`：约 12 分钟；
1. rank 0 串行扫描 candidate：约 41 分钟，约 10 万行/秒；
1. 分片复制和顺序发送：内存达到上限后终止。

真实输入每条 `candidate_codes` 有 600 个 int64，即 200 个三层 SID。
29,392,918 个 overflow item 的末层候选矩阵单份就需要：

```text
29,392,918 × 200 × 8 bytes = 43.80 GiB
```

旧版 rank 0 在分发完成前一直保留这份全量矩阵；所有 worker 的 shard 合计又
形成约一份完整候选矩阵，同时还有 plan、string item ID、分片对齐数组、
pickle/Gloo 临时副本和每个 worker 的框架内存。因此基础数据与两份候选矩阵
已经接近或超过 128 GiB。实测失败前 rank 0 约 94.6 GiB，已有 3 个 worker
各约 6.9 GiB。

这不是 iterative 仲裁循环导致的 OOM，而是仲裁前 object collective 分发
造成的重复驻留。减少 rank 数不会消除“全量候选 + 全部 shard”这两份总量，
还会增大单次 pickle payload，所以不能把 2 rank 或 4 rank 视为可靠修复。

### 6.4 分块 Tensor 传输复测

相同的 255,172,938 行、29,392,918 个 overflow item、每条 200 个末层候选
重新使用 8 rank 运行。7 个远端 shard 全部传输完成后，8 个 rank 均进入
iterative 仲裁，已越过旧实现的 OOM 位置：

| 指标       | 改造前                     | 改造后传输完成时    |
| ---------- | -------------------------- | ------------------- |
| rank 0 RSS | 约 94.6 GiB                | 约 37.9 GiB         |
| worker     | 已接收 3 个，约 6.9 GiB/个 | 7 个，约 7.3 GiB/个 |
| 聚合 RSS   | 峰值 125.60 GiB 后失败     | 约 88.1 GiB         |
| 仲裁状态   | 未进入                     | 8 个 rank 均已进入  |

进入仲裁后，聚合 RSS 稳定在约 89.2–89.8 GiB。相对旧峰值，传输完成时观察
到的驻留内存降低约 37.5 GiB。`cgroup failcnt` 是机器启动以来的累计分配失败
计数，本次从历史初值 351 增至 2618，说明候选扫描和页缓存仍有明显内存压力；
不能将本次结果表述为“从未触碰内存限制”。

全量运行成功结束：

| 指标           | 结果                        |
| -------------- | --------------------------- |
| 返回码         | 0                           |
| wall time      | 6,092.67 秒（101 分 33 秒） |
| 聚合 CPU time  | 19,191.27 秒                |
| 平均使用核数   | 3.15                        |
| 全程峰值 RSS   | 94.34 GiB                   |
| cgroup failcnt | 2618（任务结束时累计值）    |

全程峰值高于传输完成时的 88.1 GiB，因为计量覆盖首次 plan、候选扫描、仲裁、
排序和输出的全部阶段。主要阶段约为：

1. 首次读取：约 1 分钟；
1. 构建并排序 plan：约 13 分钟；
1. candidate 扫描：约 42 分钟；
1. 传输和 8 rank iterative 仲裁：约 34 分钟；
1. 三份输出准备与写入：约 12 分钟，其中 resolved grouping 的 rank 0
   全量排序约 8 分 50 秒。

最终统计为：

| 总 item     | 原始超容量桶 |     迁移数 | 未解决数 | 最终超容量桶 | 最大桶 |
| ----------- | -----------: | ---------: | -------: | -----------: | -----: |
| 255,172,938 |    3,719,947 | 29,238,615 |  154,303 |        3,077 |  3,086 |

三个输出的行数和大小为：

| 输出               |        行数 |      大小 |
| ------------------ | ----------: | --------: |
| item → SID map     | 255,172,938 | 4.186 GiB |
| 原始 SID → itemids | 167,939,082 | 2.529 GiB |
| 最终 SID → itemids | 178,016,249 | 2.574 GiB |

这证明分块 Tensor 传输和逐 shard 释放解决了旧实现的分发阶段 OOM，但不证明
总内存与数据量无关。候选扫描结束时 rank 0 仍需持有合计 43.80 GiB 的候选
shard，worker 在仲裁期间也需保留自己的完整 shard。

### 6.5 与 first-fit 的结果比较

两种运行使用同一输入、codebook、candidate 字段，并都设置
`max_items_per_codebook=5`。iterative 输出完成后，使用低内存脚本按对齐的
Parquet batch 精确比较两份 item map。比较成功覆盖全部 255,172,938 行，并
得到以下质量统计：

| 指标             |   first-fit |   iterative | iterative - first-fit |
| ---------------- | ----------: | ----------: | --------------------: |
| 原始超容量桶     |   3,719,947 |   3,719,947 |                     0 |
| overflow item    |  29,392,918 |  29,392,918 |                     0 |
| 迁移数           |  29,264,462 |  29,238,615 |               -25,847 |
| 未解决数         |     128,456 |     154,303 |               +25,847 |
| 迁移率           |    99.5630% |    99.4750% |           -0.0880 pct |
| SID 冲突率       |    30.4067% |    30.2370% |           -0.1697 pct |
| 最终超容量桶     |       1,946 |       3,077 |                +1,131 |
| cap=5 超容量桶率 |   0.001096% |   0.001728% |         +0.000633 pct |
| 最大桶           |       3,054 |       3,086 |                   +32 |
| 最终占用 SID 桶  | 177,583,296 | 178,016,249 |              +432,953 |

逐 item 比较还显示：

- item 顺序、总行数和输入覆盖完全一致；
- 原始 SID 差异为 0，两种策略都没有改动 SID 前缀；
- 8,448,288 个 item（约 3.31%）的最终 SID 不同；
- 11,987,936 个 item 的 slot index 不同，其中 5,343,327 个 item 的最终
  SID 相同但组内 index 不同。

first-fit 在这份数据上的迁移数多 25,847，迁移率高约 0.088 个百分点，最终
超容量桶和最大桶也略少；iterative 则使用了多 432,953 个最终 SID 桶。两者
聚合迁移率接近，但 844.83 万个最终 SID 差异说明策略切换会实质改变逐 item
结果，不能将其视为等价实现。

这里的 SID 冲突率沿用仓库质量指标定义：
`1 - 最终占用 SID 桶数 / item 总数`，不使用 cap=5；它衡量因 SID 没有做到
一物一码而产生的重复冗余比例，但不等于“落在非单例桶中的 item 比例”。
`cap=5 超容量桶率`则是
`最终超容量桶 / 最终占用 SID 桶`，只衡量容量约束仍未满足的桶比例。因此
iterative 虽有更低的总体 SID 冲突率，但 cap=5 后仍超容量的桶比例略高。
原始输入的 SID 冲突率为 34.1862%；first-fit 和 iterative 分别降低了
3.7795 和 3.9492 个百分点。

比较方法包括：

- 验证 item 顺序和总行数；
- 逐 item 统计原始/最终 SID、末层 code 和 slot index 差异；
- 从稠密且从 1 开始的 slot index 精确还原桶大小分布，进而计算迁移数、
  未解决数、超容量桶和最大桶；
- 扫描本轮 original grouping 计算原始桶统计；
- 校验 first-fit/iterative grouped output 行数与 map 推导出的占用桶数一致。

该方法会精确比较 map 和聚合指标，但不会逐项展开两份 grouped output 中的
`codebook/itemids` 内容。因此最终结论应表述为“map 精确比较 + grouped
row count 校验”，不能称为 grouped output 内容逐项一致。

first-fit 历史日志约 99 分 43 秒，本轮 iterative 计量为 101 分 33 秒，表面
仅相差约 1 分 50 秒；但两者来自不同提交，且 first-fit 日志不含准确进程启动
时刻，不能据此作严格的 wall time A/B。

### 6.6 iterative 循环次数

iterative 不存在一个覆盖全量数据的固定全局轮数。每个完整 SID prefix band
独立执行 `_resolve_one_band()`，`round_index` 从 1 重新开始：

1. 每个有效轮重新检查该 band 所有未分配 item 的 200 个候选；
1. 发生 winner 时更新桶占用并进入下一轮；
1. 最后再执行一次 `winner_count == 0` 的终止轮。

所以单个 band 的实际循环次数等于“成功分配轮数 + 1 个终止轮”；200 个候选
不表示执行 200 轮，也不是严格的轮数上限。本次生产路径使用
`collect_diagnostics=False`，日志和最终 map/group 均未保存 round 信息，
因此已完成运行的最大轮数、P50/P95/P99 和 band-round 总数无法事后精确反推，
报告中不能填写一个猜测值。

若后续需要该指标，应在每个 band 完成时在线累计
`len(round_stats) - 1` 的直方图，只汇总 band 数、成功轮数总和、最大值和
P50/P95/P99；不应保留 2.55 亿行的全量诊断。该改造额外内存很小，但获取本次
真实数据的精确值仍需重新执行 candidate 扫描和 iterative 仲裁。

## 7. 建议与后续验证

生产建议：

- first-fit 使用普通单进程命令；
- iterative 可从 8 rank 开始；大规模运行前仍须按
  `overflow_count × candidate_count × 8` 估算一份候选总量，并为 plan、
  item ID、每个 worker 的 shard 和框架进程预留内存；
- 不要通过增加 `batch_size` 期望降低 resolver 内存，它只影响 I/O batch；
- `OMP_NUM_THREADS=1` 可避免多进程下底层线程过量竞争。当前热点主要是 Python
  / NumPy 仲裁和进程通信，放大该值不等于提升 CPU 利用率。

并发缩放数字来自 500 万行 `rate_only` 单次运行；2.55 亿行完整输出也只运行
一次。上线前仍应在目标机器上至少执行三次，报告中位数与 P95，并补充 CSV
和 ODPS 的完整输出耗时。数值任务传输已不再使用 object collective；若需
继续降低峰值或 wall time，应优先评估候选扫描期间的流式构建/落盘、
iterative 逐 band 临时数组、rank 固定内存、resolved grouping 的 rank 0
全量排序，以及避免 rank 0 同时 gather 全部结果。

推荐命令：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  -m tzrec.tools.sid.resolve_sid_collisions \
  --input_path input.parquet \
  --output_path output/resolved_map \
  --resolved_sid_groups_output_path output/resolved_groups \
  --codebook 8192,8192,8192 \
  --max_items_per_codebook 5 \
  --strategy candidate \
  --placement_policy iterative
```
