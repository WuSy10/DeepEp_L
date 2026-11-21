#include "kernel_operator.h"
#include "launch.hpp"

namespace {
constexpr uint64_t kRecordMagic = 0x434c4f434b4d4150ULL;
}  // namespace

// 单个 AIV 读取设备计数 D，并通过 MTE3 把 32 字节记录直接写到已注册映射的 Host 内存。
// sequence、sequence+1 和 magic 只供同步后的 Host 做记录校验，不能代替流同步；
// 真正用于拟合的是 D 与 Host 侧的 H0、H1。
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void deepep_capture_clock(
    GM_ADDR mapped_record, uint64_t sequence)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 32);
    auto local = buffer.Get<uint64_t>();
    local.SetValue(0, sequence);
    local.SetValue(1, static_cast<uint64_t>(AscendC::GetSystemCycle()));
    local.SetValue(2, sequence + 1);
    local.SetValue(3, kRecordMagic);
    // 标量写入片上缓冲完成后，才允许 MTE3 把四个 uint64 字段搬到映射地址。
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint64_t> record;
    record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(mapped_record), 4);
    AscendC::DataCopy(record, local, 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

extern "C" DEEPEP_EXPORT void deepep_capture_clock_do(void* stream, void* mapped_record, uint64_t sequence)
{
    // 在传入的校准流上启动一次单 AIV 采样；流同步由 Host 调用方负责。
    deepep_capture_clock<<<1, 32, stream>>>(static_cast<uint8_t*>(mapped_record), sequence);
}
