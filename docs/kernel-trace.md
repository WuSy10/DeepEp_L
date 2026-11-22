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
python scripts/check_kernel_trace.py "$RUN/trace" --ranks 4 --require-ops notify_dispatch dispatch combine
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

Python KernelTrace 在每个进程内共享一个后台线程，管理各活跃 ACL context 的原生采集器。C++ 提供 calibration、drain_trace、synchronize 接口。Python 每轮休眠约 2 ms 后批量读取新记录，负责 JSONL 序列化、写入和刷新。

每个采集器分配 128 槽的 Host 映射内存，每槽预留 64 个 AIV，每个 AIV 占用 256 字节，总计约 2 MiB。核内起始 tick 保存在局部变量。结束时 dispatch/notify 复用已释放的 UB，combine/cached notify 在工作结束后使用 32 字节 UB，将 start_tick、end_tick 写入偏移 0，等待 MTE3_S 后，在偏移 128 发布 done_seq。第二次写出也用 MTE3_S 保护 UB 生命周期。combine/cached notify 在读取结束 tick 前用本核 V_S、MTE3_S 将流水完成推进到标量侧；不增加全卡或全核同步。每个 AIV 独占自己的区域。

Host 为成功预留的槽分配从 1 开始的进程内序号，kernel 按此序号发布。只有全部参与核的 done_seq 等于本轮序号才消费数据。回收后保留旧 tick 和序号，下一轮靠新序号识别数据。占用的槽会导致当前采集跳过并记录 dropped。提交异常的槽隔离到销毁。

最后一个共享采集器的 Python Buffer 释放时同步设备，排空记录并关闭日志。正常退出也会停止线程并排空记录。强制退出可能丢失末尾数据。Python 调度与 GIL 会影响日志延迟和回收速度，开销及丢样率需实测。

文件名为 kernel-主机-PID-d设备-创建时间.jsonl。记录使用 event、op_name、kernel_name，gid 为进程内序号，同一文件中的 launch 与 end 按 gid 关联。notify、dispatch、combine 的每次启动分别预留 ticket，独立发布 done_seq；不同阶段可以交错完成，不能按文件相邻行配对。分离 combine 的阶段耗时不可直接当作完整 Python API 耗时。early 阶段只表示该 kernel 的提交工作结束，异步 UDMA 传输仍可能在进行，其完成等待由后续阶段负责。设备验证应包含超过 128 次调用的槽位复用、不同 AIV 数量、精度和开关性能对照。
