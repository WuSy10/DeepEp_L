#include "kernel_operator.h"
#include "launch.hpp"

namespace {
constexpr uint64_t kRecordMagic = 0x434c4f434b4d4150ULL;
}  // namespace

// One AIV writes a 32-byte sample directly to registered Host memory via MTE3.
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
    deepep_capture_clock<<<1, 32, stream>>>(static_cast<uint8_t*>(mapped_record), sequence);
}
