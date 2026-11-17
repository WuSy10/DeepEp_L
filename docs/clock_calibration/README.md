# Ascend 950 时钟校准与单 AIV 连续采集：方案 A 回传

独立校准命令建立当前 NPU `GetSystemCycle()` 计数与本机 CPU
`CLOCK_MONOTONIC` 纳秒之间的映射。单 AIV 校准 kernel 把原始设备计数
**直接写到注册映射的 Host 内存**；时间戳不经过 HBM→Host 拷贝。这就是
本文所说的方案 A 回传。校准验收后，可运行本文末尾的单 AIV 连续采集实验，
获取每次算子的 NPU 起止计数及 CPU launch 时间。它尚未接入业务算子。

## 与 NCCL 的对应关系

NCCL 初始化校准用 CPU 读钟、GPU 读钟、流同步构造 Host 包围区间。
NCCL Profiler 则预分配映射 Host 缓冲，GPU 写时间戳和工作序号，Host
读取设备发布的值。本实验结合了前者的 `H0 → D → H1` 包围采样和后者的
映射 Host 回传。NCCL 自身的初始化校准包含 D2H；这里的设备时间戳
没有 D2H。参考：[NCCL 校准](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/init.cc#L608)、
[映射 Host 分配](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/include/alloc.h#L131)、
[设备打点](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/device/common.h#L246)、
[Host 读取](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/plugin/profiler.cc#L895)。

## 构建与快速验收

要求 Linux、Ascend 950、匹配 `torch_npu` 的 CANN 9.2、Python ≥3.11，
以及本仓库固定的 SHMEM SDK。先完成[构建指南](../build.md)的环境准备，
然后在仓库根目录执行：

```bash
unset DEEPEP_BUILD_EP_GMM_FUSED DEEPEP_BUILD_MEGAMOE
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

`SHMEM_ROOT` 未设置时默认使用 `third-party/shmem/install/shmem`。
先用最短参数检查映射和 kernel：

```bash
python -m deep_ep.clock_calibration --device 7 --samples 4 --warmup 1 \
  --interval 1 --output clock-smoke.json
```

再执行完整的三批校准：

```bash
python -m deep_ep.clock_calibration --device 7 --samples 32 --warmup 5 \
  --interval 10 --output clock-v2.json
```

`--device` 是当前进程可见的设备编号。前两批相隔 `--interval` 秒，
用于拟合走速；第三批再间隔同样时间，只作独立验收。每批丢弃开头的
`--warmup` 个预热样本，保留 `--samples` 个样本。验收未通过时仍写原始
结果到 JSON，但命令返回非零。

## 一次样本的完整路径

1. Host 查询当前设备 AIV 的映射内存访问能力。`aclrtMallocHost` 分配
   锁页内存，在其中选一段 4 KB 对齐的区域，调用
   `aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED, &device_ptr)`。
   CPU 使用原 Host 地址，kernel 使用返回的设备地址。
2. CPU 在专用流提交校准 kernel 前读取 `H0`。kernel 以 `<<<1,32>>>`
   运行于单 AIV，读取 `GetSystemCycle()` 为 `D`。
3. kernel 在 UB 中构造 32 字节记录：`sequence`、`D`、
   `commit=sequence+1`、固定 magic。它用 MTE3 `DataCopy` 把完整记录
   一次写到映射 Host 区，并等待 MTE3 完成。这与当前上游 Clos notify
   向映射 Host 内存发布计数所用的 MTE3 路径一致。
4. CPU 同步**这条校准流**，随后读取 `H1`，检查记录的序号、commit、
   magic、非零计数，并保存 `(H0,D,H1)`。

同步只为确认这一校准样本的设备任务完成，建立 `H0 ≤ H(D) ≤ H1`。
`commit` 在本实验中是同步之后的完整性校验值；CPU 不在 kernel 执行中
并发轮询。记录直接从 NPU 写入映射 Host 内存，因此同步不改变方案 A
的回传方式。相关运行时约束见[昇腾 Host 内存注册文档](https://github.com/hicann/runtime/blob/master/docs/zh/api_ref/11-02_host_memory_management.md)。

## 时间映射和 JSON 内容

每批保留全部原始三元组，并选 `H1-H0` 最短的样本作为锚点。前两批的
Host 中点为 `Ha`、`Hb`，设备计数为 `Da`、`Db`：

```text
rate = (Hb - Ha) / (Db - Da)          # Host ns / NPU tick
host_ns(d) = Ha + round((d - Da) * rate)
```

转换用整数分子、分母，避免大计数转浮点损失精度。第三批只验收，不参与
拟合。JSON 包含设备、SoC、三批原始样本、锚点、走速、留出验收结果和
`transport=mapped_host_mte3`。时间轴是本机单调时钟，不是 Unix 日期；
跨机器不能直接对比。`mapping_uncertainty_ns` 只传播锚点的 Host 包围
区间误差，不覆盖长期漂移、不同 AIV 的计数器差异或设备复位。Context、
进程或设备复位后须重新校准。

## 失败定位

能力查询失败会在提交 kernel 前报错。若 `aclrtSynchronizeStream` 返回
`507035`，它表示 Vector Core 执行异常，异常会包含样本序号和 CANN
最近错误信息；仍需结合相同时间的 CANN plog 查具体故障指令。查询支持
映射访问，只能证明该硬件单元声明具备能力，不能代替实机写入验证。

实现入口：[设备 kernel](../../csrc/kernels/clock_calibration.cpp)、
[Host 绑定](../../csrc/ops/clock_calibration.cpp)、
[Python 拟合](../../deep_ep/clock_calibration.py)。

## 单 AIV 连续算子实验

校准命令运行正常后，先用小批量检查构建和日志：

```bash
python -m deep_ep.clock_probe --device 0 --samples 4 --warmup 1 \
  --interval 1 --probe-warmup 2 --iterations 8 --work-iters 32 \
  --output clock-probe-smoke.jsonl
```

再运行完整批次：

```bash
python -m deep_ep.clock_probe --device 0 --samples 32 --warmup 5 \
  --interval 10 --probe-warmup 20 --iterations 1000 --work-iters 32 \
  --output clock-probe.jsonl
```

`--device` 必须是当前进程可见的设备编号。命令先按本页方法执行两批校准
和第三批留出验收；留出样本不通过就停止，不启动探针。随后 `--probe-warmup`
次和 `--iterations` 次单 AIV kernel 连续提交到一条新建的专用流。
`--work-iters` 控制每个 kernel 对 32 个 `int32` 元素执行向量加法的次数，
范围 1..4096。第 0 个元素的预期结果为 `sequence + 1 + work_iters`；
该工作负载用于生成可在 CPU 上独立复算的结果，
不代表业务算子的向量性能。
`iterations + probe-warmup` 最大为 16384。

Host 为本批每次 launch 预留一个**独立的 64 字节记录位置**，这些位置位于
同一块 4 KB 对齐并注册映射的锁页 Host 内存中。kernel 参数带上位置和
序号；单 AIV 在工作前读取 `GetSystemCycle()`，初始化片上向量，并反复
执行加法；等待向量流水完成、读取结果后，再读取结束计数。随后通过 MTE3 写入
`sequence / start_tick / end_tick / result / sequence+1 / magic / work_iters /
~sequence`。记录直接落在映射 Host 内存，没有中间 HBM 时间戳缓冲或
D2H 拷贝。每个位置只使用一次，本实验**不复用槽位**。

C++ 在每次底层 launch 调用前后用 `CLOCK_MONOTONIC` 记录
`cpu_launch_begin_ns`、`cpu_launch_end_ns`。它们表示 Host **提交调用**
的时间，不是 NPU 执行起止时间，也不是整个 Python API 耗时。所有探针
提交完毕后，Host **只同步一次探针流**，再核对包括预热在内的每条记录：
序号、起止顺序、CPU 复算结果、完成值、magic 和迭代次数。任一条失败
便报错，不输出成功日志。`sequence+1` 在本实验中仍是同步后的完整性
检查字段，不是 CPU 并发轮询时可依赖的发布标志。

输出为 JSONL：首行 `type=calibration` 保存三批原始校准样本、映射参数、
留出验收结果、负载参数及 `calibration_id`；之后每行 `type=operator`
对应一个保留的算子，不包含预热算子。每条算子行保存：

- `sequence`、`calibration_id`、`device`、`work_iters`、`result`；
- 原始 `cpu_launch_begin_ns`、`cpu_launch_end_ns` 和二者之差；
- 原始 `npu_start_tick`、`npu_end_tick` 和二者之差；
- 用本次校准映射得到的 `npu_start_host_ns`、`npu_end_host_ns`、
  `npu_duration_ns`，以及两个时刻的锚点包围区间传播误差。
  这些也是 `CLOCK_MONOTONIC` 时间轴，不是日期时间；误差值不覆盖
  后续漂移或设备复位。

成功时终端打印 `clock probe passed: device=... operators=...`；
`wc -l clock-probe.jsonl` 应为 `iterations + 1`，因为首行是校准记录。
`npu_duration_ticks` 包含探针的向量加法及完成等待，不包含随后通过
MTE3 写 64 字节日志的时间。校准本身逐样本同步；探针批次只在全部提交后
同步一次。目前没有后台采集、批量异步落盘、槽位轮转或溢出丢弃逻辑；JSONL 在批次验证完成
后由当前进程写出。设备或进程重置后须重新运行校准。

实现入口：[探针 kernel](../../csrc/kernels/clock_probe.cpp)、
[Host 批量提交与校验](../../csrc/ops/clock_probe.cpp)、
[日志输出](../../deep_ep/clock_probe.py)。

## 后台收集与批量落盘实验

`clock_probe_async` 是下一阶段的独立命令。它继续使用上面的校准与
`mapped_host_mte3` 方案 A 回传，但在后续批次继续提交时，由后台线程
检查前面批次的完成情况。原来的 `clock_probe` 命令及其 `iterations+1`
行输出不变，可用来对照。本实验的进度 Event 路径要求 Ascend 950PR。

### 运行

构建新版本后，先运行小批次：

```bash
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
python -m deep_ep.clock_probe_async --device 7 --samples 4 --warmup 1 \
  --interval 1 --probe-warmup 2 --iterations 32 --work-iters 32 \
  --batch-size 4 --slot-batches 2 --queue-batches 2 \
  --output clock-probe-async-smoke.jsonl
```

再运行较长的连续采集：

```bash
python -m deep_ep.clock_probe_async --device 7 --samples 32 --warmup 5 \
  --interval 10 --probe-warmup 20 --iterations 10000 --work-iters 32 \
  --batch-size 16 --slot-batches 8 --queue-batches 8 \
  --flush-ms 100 --submit-interval-us 1000 \
  --output clock-probe-async.jsonl
```

命令结束时打印 `attempted`、`logged`、`dropped`。正常成功文件包含
1 行 `calibration`、`logged` 行 `operator` 和最后 1 行 `summary`，因此
`wc -l` 应为 `logged+2`。始终有 `attempted=logged+dropped`。
当 `dropped=0` 时，1 万次运行应输出 10002 行。小批次也可能因设备
冷启动或 Host 调度而出现丢弃；丢弃是可见的容量行为，不会伪造记录。
可以用下面的检查同时核对文件结构和计数：

```bash
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("clock-probe-async.jsonl").open()]
header, summary = rows[0], rows[-1]
operators = rows[1:-1]
assert header["type"] == "calibration" and summary["type"] == "summary"
assert all(row["type"] == "operator" for row in operators)
assert len(operators) == summary["logged"] == summary["accepted"]
assert summary["attempted"] == summary["accepted"] + summary["dropped"]
assert all(row["calibration_id"] == header["calibration_id"] for row in operators)
assert all(row["generation"] == row["sequence"] + 1 for row in operators)
print(summary)
PY
```

### 内存和完成协议

固定容量为 `batch-size × slot-batches` 个 64 字节映射 Host 位置，最大
16384 个；默认 `16 × 8 = 128` 个位置。每批占用一组位置。kernel 的
8 个字段仍是 `sequence/start_tick/end_tick/result/sequence+1/magic/
work_iters/~sequence`。本实验的 `generation=sequence+1`，每次提交
严格递增；即使一整批因满而跳过，其序号也会前进，日志中的缺口可与
`dropped` 计数相互核对。

每批 kernel 后面，在**同一条探针流**记录一个进度 Event。提交线程
先完成 `aclrtRecordEvent`，再把批次放入待查询队列；后台收集线程只
查询已经入队的 Event。Event 返回 `COMPLETE` 之后才读取映射记录，
逐条检查预期序号、`sequence+1`、magic、tick 顺序、工作次数和 CPU
复算结果。`NOT_READY` 不释放位置；查询失败或字段不匹配会停止运行，
也不会授权位置复用。这里 Event 是并发读取的门槛，`sequence+1` 是
本轮 generation 完整性字段；**不在设备运行时仅凭这个字段轮询读取**。
进度 Event 使用可重复记录的 `ACL_EVENT_CAPTURE_STREAM_PROGRESS`，
只在前一轮已完成且记录被搬走后再为同一位置组记录下一轮。
这遵循昇腾的[跨线程 Event 查询示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0024.html)；
文档也提醒不同线程中的记录与查询可能发生顺序问题，因此本实现先
记录 Event，再在互斥保护下将批次加入收集队列，查询线程只处理已入队
批次。查询线程还会绑定提交线程保存的 ACL Context。
[CANN 9.2 beta.2 API 状态说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_0086.html)。

原生 `poll()` 先把整批有效记录复制成独立的 CPU 数据，继续保留原
映射位置组。Python 收集线程把这份 CPU 数据成功放进有界队列后，
才调用 `release()` 将对应位置组标记为空闲。
队列满时暂停收集，不提前释放映射位置；这会把背压传到固定容量的
映射槽位。提交路径若找不到空闲位置组，跳过**这一批的新采集**并
增加 `dropped`，不等待设备、也不覆盖正在被设备使用的位置。

### 换算、写盘与退出

第二个后台线程消费 CPU 自有队列，在这时调用校准映射换算原始 tick，
保留原始值和映射不确定度，再按 `--flush-records` 条或
`--flush-ms` 毫秒批量写 JSONL。默认分别为 256 条与 100 ms；
没有逐条 `fsync`，业务提交线程不写磁盘。首行校准 header 在启动
提交之前写出。最后一行 `summary` 包含 `attempted/accepted/dropped/logged`；
写入线程检查 `accepted=logged` 才写 summary。

`--submit-interval-us` 默认 1000 微秒，用于这个独立实验的提交节奏。
设为 0 可压测固定容量，例如 `--batch-size 16 --slot-batches 2
--queue-batches 1 --submit-interval-us 0`，预期可能出现 `dropped>0`。
一条完整的拥塞命令如下：

```bash
python -m deep_ep.clock_probe_async --device 7 --samples 4 --warmup 1 \
  --interval 1 --probe-warmup 2 --iterations 10000 --work-iters 32 \
  --batch-size 16 --slot-batches 2 --queue-batches 1 \
  --submit-interval-us 0 --output clock-probe-async-pressure.jsonl
```

正常退出时等待收集线程排空待完成批次，再等待写入线程写完尾部；
最终关闭探针时有一次保护性流同步。`--drain-timeout` 默认 60 秒，
只用于判断排空失败；超时不会授权复用。若设备状态不明，代码保留
映射页与事件资源直至进程结束，以避免仍在运行的设备任务写入已释放内存。
强制终止进程无法保证尾部完整；缺少最终 `summary` 的文件应视为
未完整收尾。

这仍是独立的单 AIV 实验，尚未接入业务算子；也没有自动周期重校准。
时钟重置、进程重启或设备重置后必须重新运行校准。映射不确定度只
传播锚点包围区间误差，不保证长时间漂移上界。实现入口：
[原生槽位和 Event 管理](../../csrc/ops/clock_probe_async.cpp)、
[Python 收集与写入](../../deep_ep/clock_probe_async.py)。
