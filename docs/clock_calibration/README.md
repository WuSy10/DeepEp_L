# Ascend 950 独立时钟校准：方案 A 回传

本实验仅建立当前 NPU `GetSystemCycle()` 计数与本机 CPU `CLOCK_MONOTONIC`
纳秒之间的映射，不启动业务算子，也不采集批量算子日志。单 AIV kernel
把原始设备计数**直接写到注册映射的 Host 内存**；时间戳不经过 HBM→Host
拷贝。这就是本文所说的方案 A 回传。

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
python -m py_compile deep_ep/clock_calibration.py
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

`SHMEM_ROOT` 未设置时默认使用 `third-party/shmem/install/shmem`。
若环境已安装 `pytest`，还可运行
`python -m pytest -q tests/test_clock_calibration.py` 检查纯 CPU 换算逻辑。
语法和 CPU 单元测试不验证映射内存；实机运行才验证 AIV 写入。

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
