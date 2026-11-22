# KernelTrace：notify、dispatch 与 combine 采集

在 Ascend950 Global Clos 上，给训练进程设置 `DEEPEP_TRACE_DIR` 即可开启。目录应由当前作业独占并允许各 rank 写入。普通调用方式与上游相同。采集 eager V2 调用实际启动的下列 kernel，沿用相同的 Python 后台线程和日志格式。

| op_name | kernel_name | 范围 |
| --- | --- | --- |
| notify_dispatch | notify_dispatch_kernel | fresh dispatch 的计数、元数据及通知阶段 |
| cached_notify_dispatch | cached_notify_dispatch_kernel | cached dispatch 原有的通知同步 |
| dispatch | v1_dispatch_udma_kernel | fresh / cached dispatch 通信及 kernel 内部输出处理 |
| cached_notify_combine | cached_notify_combine_kernel | 仅在执行路径实际启动独立 notify 时记录 |
| combine | v1_combine_udma_kernel | 融合通信和归约 |
| combine | v1_combine_udma_comm_kernel / v1_combine_udma_reduce_kernel | 分离的通信和归约分别记录 |
| combine | v1_combine_udma_early_kernel / v1_combine_udma_remaining_kernel / v1_combine_udma_pipeline_kernel | 所选策略实际执行的阶段分别记录 |

普通 fused/communication combine 的 Begin 同步在通信 kernel 内，其耗时包含在 combine 记录中，不额外生成 notify 记录。采集不改变策略，也不额外启动通知 kernel。layout、expand、aggregate、缓存计划重建、显式 Buffer barrier 和 NPU Graph 捕获/重放不采集。图捕获时不预留槽位，避免重放旧序号。

## 运行与校验

按[构建指南](build.md)在含匹配 CANN、Torch、torch_npu 和 SHMEM SDK 的环境安装本分支的 wheel。Host 与 Device 两部分均需重新构建。在单机四卡上可使用现有 benchmark 验证。

```bash
set -e
RUN="$(mktemp -d "$PWD/kernel-trace-XXXXXX")"
export DEEPEP_TRACE_DIR="$RUN/trace"
python -u benchmarks/clos/bench_v2_bandwidth.py \
    --ep-size 4 --nnodes 1 --aiv 32 --dtypes bf16 --expand 0 \
    --num-tokens 2048 --hidden 6144 --num-topk 8 --experts-per-rank 32 \
    --warmup 2 --iterations 5 --no-flush-l2-cache --output "$RUN/bench"
python scripts/check_kernel_trace.py "$RUN/trace" --ranks 4 --require-ops notify_dispatch dispatch combine --require-metadata
```

每个文件名包含主机、PID、设备和文件创建时间。校验器检查各 rank 是否有指定类型的成功完成记录、launch 与 end 是否配对、时钟换算是否一致，以及是否丢样。一次 API 调用可能启动多个 kernel，基准的预热和准备也会产生记录，因此记录数不等于 iterations。增加 eager 调用数后可用 `--require-reuse` 要求每个文件超过 128 条完成记录；此检查证明跨越槽池容量，逐核 done_seq 的代际校验还需原生协议测试。性能验收需使用相同输入、资源和启动方式做开关交替对照，并同时记录丢样数。

## 时间含义

| 记录 | 来源及含义 |
| --- | --- |
| calibration | 首个启用采集的 UDMA Buffer 在当前 ACL context 创建采集器，短 kernel 读取设备 tick。Host 使用私有流包围这次采样，同步一次，保存主机单调时钟窗口的宽度和墙钟偏移。共享该采集器的 Buffer 共用同一偏移。 |
| launch | Host 在调用相应 kernel 启动封装前读取 Unix 墙钟，单位 ns。封装返回后把时间与当前线程的 ACL 状态交给日志线程。 |
| end | 同一个 gid 的完成记录。每个参与 AIV 在本核实现的入口和尾部读取 tick。start_tick 取各核最小值，end_tick 取各核最大值，覆盖本 rank 参与核的执行区间。 |
| launch_error / incomplete | 提交路径报告异常，或在采集器关闭时仍缺少有效 tick。这些记录保留 gid 以便定位。 |
| summary | 正常关闭时汇总已完成数、未完成数和队列竞争导致的丢样数。 |

Ascend950 按 1 GHz 换算，令校准核采样为 $C_0$，Host 单调时钟包围采样的中点为 $M_0$，另一次相邻采样所得墙钟和单调时钟为 $W_a,M_a$，所有 Host 时间使用 ns。

$$
O=(W_a-M_a)+M_0-C_0,\qquad W(C)\approx C+O
$$

`start_ns` 和 `end_ns` 为上述参考墙钟。`elapsed_ns = end_tick - start_tick` 独立于墙钟偏移。跨 AIV 取极值的前提是各核 GetSystemCycle 共享计时基准；设备验收需与独立 profiler 对照。校准窗口宽度与设备速率漂移都会影响跨时钟比较，launch 到 start 的差值用于参考，不能作为精确排队时延。

## 上报和资源

Python KernelTrace 在每个进程内共享一个后台线程，管理各活跃 ACL context 的原生采集器。C++ 提供 calibration、drain_trace_json、synchronize 接口。Python 每轮休眠约 2 ms 后调用原生侧回收；原生侧释放 GIL，完成记录合并、JSONL 编码和原生对象清理，返回一块文本。Python 负责批量写入和刷新。

每个采集器分配 128 槽的 Host 映射内存，每槽预留 64 个 AIV，每个 AIV 占用 256 字节，总计约 2 MiB。核内起始 tick 保存在局部变量。结束时 dispatch/notify 复用已释放的 UB，combine/cached notify 在工作结束后使用 32 字节 UB，将 start_tick、end_tick 写入偏移 0，等待 MTE3_S 后，在偏移 128 发布 done_seq。第二次写出也用 MTE3_S 保护 UB 生命周期。combine/cached notify 在读取结束 tick 前用本核 V_S、MTE3_S 将流水完成推进到标量侧；不增加全卡或全核同步。每个 AIV 独占自己的区域。

Host 为成功预留的槽分配从 1 开始的进程内序号，kernel 按此序号发布。只有全部参与核的 done_seq 等于本轮序号才消费数据。回收后保留旧 tick 和序号，下一轮靠新序号识别数据。占用的槽会导致当前采集跳过并记录 dropped。提交异常的槽隔离到销毁。

最后一个共享采集器的 Python Buffer 释放时同步设备，排空记录并关闭日志。正常退出也会停止线程并排空记录。强制退出可能丢失末尾数据。Python 调度与 GIL 会影响日志延迟和回收速度，开销及丢样率需实测。

文件名为 kernel-主机-PID-d设备-创建时间.jsonl。记录使用 event、op_name、kernel_name，gid 为进程内序号，同一文件中的 launch 与 end 按 gid 关联。notify、dispatch、combine 的每次启动分别预留 ticket，独立发布 done_seq；不同阶段可以交错完成，不能按文件相邻行配对。分离 combine 的阶段耗时不可直接当作完整 Python API 耗时。early 阶段只表示该 kernel 的提交工作结束，异步 UDMA 传输仍可能在进行，其完成等待由后续阶段负责。设备验证应包含超过 128 次调用的槽位复用、不同 AIV 数量、精度和开关性能对照。


## Host 元数据与调用关联

每条启用 V2 元数据的 launch/end 记录增加 metadata_version=1。元数据在 Python API 的 Host 路径构造，C++ 在 Reserve 时为对应槽保留不可变快照；设备仍只写起止 tick 和 done_seq，Host 映射区布局和 kernel 参数均未因本次元数据扩展而改变。后台线程调用原生侧合并记录并编码，不从 NPU tensor 回读参数。

| 字段 | 含义与来源 |
| --- | --- |
| ep_size | 当前 EP 组大小 |
| num_tokens | 原始路由 tensor 的行数；不是有效通信行数或 padding 后实际有效条数 |
| hidden_size / element_bytes / dtype | API 输入特征维度（元素数）、元素字节数、数据类型；FP8 dispatch 使用 x 而非 scales |
| num_experts / topk | 模型路由专家总数、路由 tensor 列数；cached dispatch/combine 从 handle 取，combine 不带 weight 时仍保留真实 top-k |
| api_input_rows | 当前 API 输入 tensor 的行数；展开后的 combine 可以不同于 num_tokens |
| kernel_input_rows / kernel_output_rows | dispatch/combine 启动点实际传入的输入/输出 tensor 行数；notify 不附加这两个字段。固定容量输出记录的是分配行数，不冒充有效接收行数 |
| cached / do_expand | 是否 cached dispatch、是否使用展开模式，0/1 |
| token_capacity | Buffer/handle 的每 rank token 容量，与实际 token 数分开 |
| comm_id | 每次 Buffer 初始化，由 EP rank 0 生成并发给组内所有 rank 的 UUID；同组此次 Buffer 实例一致，重新创建 Buffer 会生成新 ID |
| buffer_id | 本进程内的 Python Buffer 标识；即使共享原生采集器也独立 |
| process_id / hostname / pid | 进程生命周期 UUID、主机名、操作系统 PID；PID 可复用，因此跨文件身份使用 process_id |
| rank / global_rank | EP 组内 rank；global_rank 仅在已初始化的 PyTorch distributed 环境中获取，其他环境省略，不用组内 rank 冒充 |
| device | ACL 逻辑设备编号，不能当作物理 PCI bus ID |
| stream | 实际 kernel 启动点的 ACL stream 句柄，以十进制字符串保存，避免无符号地址精度或溢出问题；只在本进程/设备生命周期内解释 |
| aiv | 本次实际启动核数；combine 通信与归约允许不同核数 |
| api_name / api_call_id | dispatch 或 combine，以及本进程内独立的调用编号；一次 API 内 notify/通信/归约共用该编号 |
| handle_id | 路由 handle 身份；fresh dispatch 预留并随返回 handle 保存，cached dispatch/combine 复用；与执行轮次不同 |
| gid | 原有的每 kernel 事件序号；一次 API 可以对应多个 gid |

身份作用域：事件用 (process_id, gid)，API 用 (process_id, api_call_id)，handle 用 (process_id, handle_id) 关联。各 rank 的 api_call_id/handle_id 不保证一致，不能直接据此跨 rank 配对。comm_id 提供通信组关联，但没有宣称增加了全组执行轮次编号。若 handle 没有历史采集标识，首次被采集使用时分配 handle_id，不虚构其创建记录。

同一 handle 被 cached dispatch 重复使用时，handle_id 保持不变，每次生成新 api_call_id。例：

- fresh dispatch：api_call_id=A，handle_id=H；notify 与 dispatch 各有独立 gid。
- cached dispatch：api_call_id=B，handle_id=H；cached notify 与 dispatch 关联到 B。
- combine：api_call_id=C，handle_id=H；通信及延后执行的 reduce 都关联到 C。

C++ 元数据上下文采用线程局部存储和作用域恢复，异常退出也恢复上一层。Buffer 固定字段只转换一次，按 shape、dtype、API、模式等缓存最多 64 份不可变模板。每次 API 创建只含两个 ID 和模板引用的小快照，使用原生句柄恢复上下文；字段在后台展开，Reserve 只保存共享引用和本次启动的 stream/行数等标量，不逐 kernel 重复复制整张元数据表，也不保留 Python 可变字典的引用。combine continuation 另保留调用时快照，执行时恢复该快照，避免其他调用覆盖。

初始化增加一次 Host 元信息 all-gather，用于检查各 rank 的采集开关一致性并分发 comm_id；UDMA 路径即使关闭采集也参加此一致性检查。业务调用不新增集合通信或设备同步。tensor.size、dtype 和 element_size 只读取 Host 元信息。元数据构造、复制与 JSONL 增量仍有 Host 开销，不能仅据此承诺小于 1%。

只用初始化时的配置决定是否启用采集。多个 Buffer 可以共享日志线程和原生槽池，但各自的 comm_id/buffer_id 随调用快照保存。Graph 捕获/重放仍不产生 kernel trace，调用编号因此可能有空缺。

本次不补充 GPU NVLink 域 rank、GPU kernel 内部 iter、训练 step/microbatch 编号或物理 PCI bus ID；也不增加 Python API 周期汇总、文件轮转或版本配置日志。它们不能由现有 kernel gid 推导出来。当前元数据接入 V2 Python API；直接调用底层原生接口不保证有完整 Python API/handle 元数据。

新数据建议使用 --require-metadata 校验，检查字段完整性、launch/end 元数据相等、同一 API 的元数据一致性及 handle 路由信息稳定性。旧日志不带此参数仍可校验。远端验证应覆盖 fresh/cached/无 weight 与带 weight 的 combine、expand=0/1、超过 128 槽的复用、不同通信/归约 AIV 数，以及多个 Buffer 的身份隔离。必须重新构建 Host 扩展；性能开销需要相同输入下 trace 开关对照。

## 新元数据与开销验收用例

`tests/runtime/test_kernel_trace.py` 同时提供 CPU 回归和可选择运行的 NPU 验收入口，不需要新建 benchmark 脚本。

CPU 回归检查元数据关联、错误字段、遗漏 API、槽位代际协议，以及开销判定的通过、失败和不确定分支：

```bash
python -m pytest tests/runtime/test_kernel_trace.py -q
```

在已安装本分支 wheel、Torch、torch_npu、pytest 的 Linux 单机 UDMA 环境，空闲的四张 NPU 上执行：

```bash
python -u tests/runtime/test_kernel_trace.py --npu \
    --ep-size 4 --first-device 0 \
    --num-tokens 2048 --hidden 6144 --num-topk 8 --experts-per-rank 32 \
    --aiv 32 --reduce-aiv 32 \
    --pairs 6 --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/kernel_trace_acceptance_$(date +%Y%m%d_%H%M%S)"
```

输出目录必须尚不存在。也可设置 `DEEPEP_RUN_TRACE_NPU=1` 后通过 pytest 运行 `test_npu_metadata_and_overhead`；该入口使用上述默认配置。普通 pytest 默认跳过实机测试。

此用例固定 BF16、随机路由、expand=0、无 bias，不清理 L2。依次覆盖 fresh dispatch、cached dispatch、无 weight 和带 weight 的 combine。weight 是 combine 的 topk_weights 输入，未增加 bias。精度用独立 CPU reference 检查，发生在计时之前。

采集开启时，检查所有 rank 的真实 JSONL：

- 字段完整、launch/end 配对、时钟换算一致，且 summary 中 dropped/incomplete 均为零。
- shape、dtype、原始 token 数、专家数、容量、实际输入/输出行数、PID、device 和 global_rank 与测试输入/进程相符。
- comm_id 覆盖完整 EP 组；每次 fresh 调用产生不同 handle_id，cached dispatch 和两种 combine 引用准备阶段的同一个 handle。
- API 调用数与准备、预热和正式执行数一致，每个日志超过 128 条完成记录。代际旧值拒收由 CPU 原生协议测试另行验证。

每对 A/B 使用独立的 torchrun 进程组，交替执行 off→on、on→off，默认共 6 对、12 次进程组/Buffer 初始化。子进程清除继承的 DEEPEP_TRACE_DIR 和 PYTHONPATH，在输出目录运行以加载已安装 wheel；其他运行环境保持一致。初始化、精度检查、预热和退出排空不计入稳态开销。

计时不启用 profiler、不捕获 Graph，每个窗口连续执行多次 eager API。窗口前设备同步和 Host barrier 在计时之外，窗口末设备同步计入耗时；逐次只插入正常的事件流依赖，不额外逐次做 Host 同步。计时包含 Host 提交、设备执行、业务路径原有同步和同时运行的 Python 日志线程影响。每个窗口取所有 rank 的最大耗时，各窗口取中位数，得到一次进程试验的每 API 耗时。

每种操作独立计算配对开销：

$$
\Delta_i = 100\left(\frac{T_{\mathrm{on},i}}{T_{\mathrm{off},i}}-1\right)\%.
$$

对进程级配对样本的中位数做固定随机种子的 10,000 次 bootstrap，输出每种操作的 95% 经验区间。区间上界严格小于 1% 为 PASS；下界大于等于 1% 为 FAIL；其余为 INCONCLUSIVE。四种操作均 PASS 且精度/日志校验通过时退出码为 0，否则非零。窗口不作为独立试验增加样本数，不剔除慢样本。

这是指定 shape 和运行环境的稳态验收，不是所有训练负载的开销保证，也不保证 bootstrap 区间消除系统性频率/负载漂移。波动较大时可增加 `--pairs 12 --windows 5 --iterations 500` 并保持机器空闲。初始化校准与最后日志排空另存耗时，不被宣称满足稳态的 1% 阈值。expand=1、多 Buffer、不同拓扑和 Graph 不在这个实机入口的验收范围内。

产物如下：

- `acceptance.json`：总结果、四种操作的开销、区间及 PASS/FAIL/INCONCLUSIVE；异常时保存失败原因。
- `config.json`、`runs.json`：配置、各对 A/B 的汇总原始值。
- `pairXX_on/trace/`：真实 Python 落盘日志；off 目录不应有 trace。
- `pairXX_on/` 和 `pairXX_off/` 下的 `rankN.json`：每 rank 每窗口耗时、初始化/退出排空耗时、wheel 路径和精度结果；`run.log` 保存 torchrun 输出。

单次 torchrun 默认超时 1800 秒，可用 `--timeout` 调整。超时或中断时终止该次创建的进程组，不影响其他任务。未实际执行 NPU 验收前，CPU 测试通过不能视为开销已满足 1%。

## 分层开销诊断

当完整采集开销明显超过阈值时，同一个测试入口支持 `--diagnostic`。这是归因工具，不是性能验收，也不修改 kernel 的打点、done_seq 或 Host 映射内存协议。

需要重新构建安装本分支的 Host 原生扩展；只复制 Python 文件不够。采集启动会检查原生扩展的版本标识（当前为 3），拒绝混用不支持这些开关的旧扩展。

| 模式 | 实际工作 | 输出 |
| --- | --- | --- |
| off | 完全关闭采集 | rank 耗时/精度 JSON |
| production | 普通采集路径，只设置 trace 目录，清除两个诊断环境变量；无诊断调用计数或批次计数 | kernel JSONL，无 diagnostic JSON |
| metadata | 仅执行 API 元数据构造、原生上下文交换与恢复；不分配采集环、不启动校准 kernel 或采集线程 | rank JSON 中的元数据调用统计 |
| native | 元数据、设备打点、Python 线程轮询原生 Drain；不把每条记录转换成 Python 字典 | 退出时的 diagnostic JSON |
| json_discard | 与 json 一样在释放 GIL 时完整编码；随后在 C++ 释放文本，只返回计数和空字符串 | 退出时的 diagnostic JSON |
| json | native 的工作，加释放 GIL 的原生 JSONL 编码；返回整批文本后丢弃 | 退出时的 diagnostic JSON |
| full | 原生批量编码和 Python JSONL 写入/刷新 | kernel JSONL 和退出时的 diagnostic JSON |

native 仍构造原有 C++ TraceRecord，使用原来的 Drain 和 done_seq 回收逻辑；每批只向 Python 返回少量计数。它不是“只有设备打点”的纯设备测试，包含原生记录处理、轮询及批次统计成本。json/full 在此基础上返回整批 JSONL 文本，不返回逐事件 Python 字典。三种模式都检查完成、launch/end 数、丢样和退出 summary，不能用关闭回收或填满槽池来减少被采集的事件。

第一轮不打开细分计时，比较端到端 API 耗时：

```bash
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --rounds 3 --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_diagnosis_$(date +%Y%m%d_%H%M%S)"
```

默认每轮运行全部五种模式，轮转并反转运行顺序，3 轮共 15 次独立进程组/Buffer 初始化。每种模式与同轮 off 比较；保留全部窗口和逐轮比值。需要更多证据时增加 rounds。`--modes` 可以选取子集，但必须包含 off 和至少一种开启模式。

第二轮另建目录，沿用相同参数并增加 `--diagnostic-timing`，通常 `--rounds 1` 即可获得初步分项数据。此轮在 Host 增加计时调用，只用于定位，不能混入第一轮或正式 1% 验收。两种计时设置下，full 都由原生侧编码整批文本，再交给 Python 写入。

统计说明：

- `rankN.json.metadata_statistics`：fresh_dispatch/cached_dispatch/combine 的调用数、构造与进入原生上下文的 `enter_ns`、恢复上下文的 `restore_ns`；包括准备、精度和预热调用。timing 关闭时计时值为零，不代表没有成本。
- `trace/diagnostic-*.json.counters`：polls、records、launches、ends、summaries、completed、incomplete、dropped、launch_errors。native/json 不生成逐事件 kernel JSONL，这是有意省略；full 仍做完整日志校验。
- `drain_ns`：原生 Drain 墙钟耗时，执行时释放 GIL。
- `convert_ns`：整批原生文本转 Python 字符串的墙钟耗时；native 返回空字符串。版本 1 中此字段测量逐条字典转换，含义不同。
- `drain_call_wall_ns`：Python 发起整个 drain 调用到返回的墙钟耗时，包含转换、批次计数、调度以及可能的 GIL 等待；不能把它减去 drain_ns 后直接称为 GIL 等待时间。
- `serialize_ns`、`write_flush_ns`：释放 GIL 期间原生 JSONL 编码、Python 写入及 flush 的累计墙钟耗时；只在对应模式且 timing 开启时有值。flush 不是 fsync，不表示持久化到物理磁盘。

`diagnosis.json` 汇总五种模式相对 off 的开销，`diagnostic_valid=true` 只表示诊断证据通过完整性检查，`performance_acceptance` 始终为 false。完整性失败返回非零并保留原因；性能变慢本身不使诊断任务失败。native/json/full 的计数校验针对本用例当前 eager 非展开的 notify+dispatch、communication+reduce 路径（每次 API 两个 kernel）；改变执行策略后若 kernel 数不同，应先核对实际路径，不放宽丢样检查。

诊断环境变量只供本功能排查，并在初始化时确定：`DEEPEP_TRACE_DIAGNOSTIC_MODE=metadata|native|json_discard|json|full`，`DEEPEP_TRACE_DIAGNOSTIC_TIMING=0|1`，开启模式还需设置 DEEPEP_TRACE_DIR。EP 内配置必须一致；不支持运行中修改。测试控制器自动管理并清除继承值，正常 `--npu` 验收也会清除诊断变量。没有诊断变量时保持原有完整采集路径。production 只是测试控制器的模式名称，不是诊断环境变量的合法值。

相邻模式的差值包含工作重叠、线程调度变化及小额诊断统计成本，不能当作各模块的精确独占时间。优先用未加细分计时的五组结果定位增量，再结合计时轮次解释。json 与 native 的差值现在主要对应原生编码和批量文本传递，不能沿用旧版的逐事件 Python 转换/编码归因。最终优化必须回到原始完整采集开关 A/B 验收。


## 普通采集对照和编码返回链路定位

诊断 full 即使关闭细分计时，仍包含每 API 调用计数、每批原生计数返回及 Python 累加；它与普通采集并非同一路径。先比较普通采集和诊断 full，再拆分编码与文本返回，两个步骤均沿用同一个 worker、shape、精度检查及计时边界。

```bash
# 第一步：普通 off/on 与诊断 off/full，六轮共 24 次进程组初始化。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off production full --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_production_control_$(date +%Y%m%d_%H%M%S)"

# 第二步：编码与 Python 文本返回，六轮共 36 次进程组初始化。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off native json_discard json --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_encoding_control_$(date +%Y%m%d_%H%M%S)"
```

`--paired-baselines` 为每个开启模式运行独立且相邻的 off；轮转开启模式顺序，并交替 off→mode、mode→off。`plan.json` 记录执行顺序，`runs.json` 的 pair 标识对应本模式的 off。所有试次顺序执行，不共享或并发使用 Buffer。关闭该选项时仍使用原有同轮共享 off 的方式。

`diagnosis.json.layer_increments` 提供 production→full、native→json_discard、json_discard→json 等可用对照的逐轮原始差值、各自 off 校正后的差值、中位数、范围及正增量轮数；单位 ns。这些差值不是独占耗时，不能因中位数为正就忽略符号不一致的轮次。production 与 full 都校验完整逐事件日志；production 还要求无诊断统计和诊断报告。带 production 的比较拒绝 `--diagnostic-timing`，避免把加计时的诊断路径与普通路径混为一谈。

json_discard 保留相同的设备打点、done_seq 检查、回收、2 ms 轮询和 C++ 编码函数；只改变文本离开原生层的方式。它仍返回小型计数字典和空 Python 字符串，不是无 GIL 或无 Python 线程的模式。与 json 的差值包括文本转换、传递、释放位置及调度变化，不能直接称为 GIL 等待。没有关闭回收或故意填满槽位来省略采集。

所有原生诊断批次新增 `encoded_bytes` 和 `returned_text_bytes`（UTF-8 文本字节累计）：native 两者为 0，json_discard 前者大于 0、后者为 0，json/full 两者相等且大于 0。控制器检查这些条件以及原有 completed/launch/end/dropped/incomplete，避免把未编码或错误返回文本当作有效对照。新计数仅存在于诊断路径，不改变普通采集输出。

如需详细计时，可另跑第二步的模式组合，加 `--diagnostic-timing --rounds 1` 并使用新目录；此结果仅辅助定位，不纳入主性能比较。最终低于 1% 的验收仍需运行不带 `--diagnostic` 的普通 A/B 用例。

## 元数据缓存与原生批量编码的验证

缓存只保留标量和原生快照，不保留用户 tensor 或 handle 对象。shape、dtype、容量或模式变化会使用新模板；超过 64 种时清空模板缓存。正在执行或延迟执行的 kernel 持有 shared_ptr 父快照引用，不受缓存清空影响。上下文仍是线程局部的，嵌套调用及异常退出使用原生句柄恢复。

每次调用仍读取 Host tensor 属性、更新 ID、预留槽位和记录 launch 时间，并非零开销。kernel 的 tick、done_seq 和满槽跳过机制保持不变。Python 仍拥有后台线程、文件和退出排空流程。JSON 保留所有字段和整数精度，转义控制字符、引号与反斜杠，UTF-8 可直接输出；字段顺序、空格和非 ASCII 转义形式可能变化，按 JSON 解析后的值兼容。

重新构建安装 Host 扩展后，先运行本文 CPU 回归，再运行原始 --npu 完整采集开关 A/B，不加诊断参数。检查精度、元数据、零丢样和实际开销。诊断的 json/full 已采用新编码实现，分项不能直接套用旧版本含义。本次优化尚未完成 NPU 实测，不能承诺开销低于 1%。
