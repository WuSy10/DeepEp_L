#include "kernel_operator.h"
#include "launch.hpp"

namespace {
constexpr uint64_t kProbeMagic = 0x434c4f434b50524fULL;
}

// One AIV computes a runtime-dependent vector result and returns its timing record
// directly to mapped Host memory. Each launch owns a separate 64-byte record.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void deepep_clock_probe(
    GM_ADDR mapped_record, uint64_t sequence, uint32_t work_iters)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ones_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> record_buffer;
    pipe.InitBuffer(values_buffer, 128);
    pipe.InitBuffer(ones_buffer, 128);
    pipe.InitBuffer(record_buffer, 64);
    auto values = values_buffer.Get<int32_t>();
    auto ones = ones_buffer.Get<int32_t>();
    auto local = record_buffer.Get<uint64_t>();

    const uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
    AscendC::Duplicate(values, static_cast<int32_t>(sequence + 1), 32);
    AscendC::Duplicate(ones, int32_t(1), 32);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < work_iters; ++i) {
        AscendC::Add(values, values, ones, 32);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const int32_t result = values.GetValue(0);
    const uint64_t end = static_cast<uint64_t>(AscendC::GetSystemCycle());

    local.SetValue(0, sequence);
    local.SetValue(1, start);
    local.SetValue(2, end);
    local.SetValue(3, static_cast<uint64_t>(result));
    local.SetValue(4, sequence + 1);
    local.SetValue(5, kProbeMagic);
    local.SetValue(6, static_cast<uint64_t>(work_iters));
    local.SetValue(7, ~sequence);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint64_t> record;
    record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(mapped_record), 8);
    AscendC::DataCopy(record, local, 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

extern "C" DEEPEP_EXPORT void deepep_clock_probe_do(
    void* stream, void* mapped_record, uint64_t sequence, uint32_t work_iters)
{
    deepep_clock_probe<<<1, 320, stream>>>(static_cast<uint8_t*>(mapped_record), sequence, work_iters);
}
