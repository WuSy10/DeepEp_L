#pragma once
#include "kernel_operator.h"
#include "common/kernel_trace_layout.h"

namespace ascend_deepep::trace {
using namespace AscendC;
// Trace publication reuses the first 32 UB bytes after the final reader.
// MTE3_S drains the tick payload before publishing the generation marker.
__aicore__ inline void PublishTraceTick(uint64_t address, LocalTensor<uint64_t> scratch, uint64_t tick, uint64_t end = 0,
                                      uint64_t third = 0, uint64_t fourth = 0)
{
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address), 4);
    output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    scratch.SetValue(0, tick);
    scratch.SetValue(1, end);
    scratch.SetValue(2, third);
    scratch.SetValue(3, fourth);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(output, scratch, 4);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// Called after the implementation has consumed its event credits.
// Reduction may finish with MTE3_V waits: bridge V and stores to the scalar
// timer before reusing UB. These are local pipeline fences, not collectives.
__aicore__ inline void Finish(uint64_t slot, uint64_t sequence, uint32_t cores, uint64_t start)
{
    if (!slot || cores != uint32_t(get_block_num() * get_subblockdim())) return;
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    const auto end = GetSystemCycle();
    const uint32_t core = get_block_idx() * get_subblockdim() + get_subblockid();
    TPipe pipe;
    TBuf<TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 32);
    const auto line = slot + uint64_t(core) * kCoreBytes;
    PublishTraceTick(line, buffer.Get<uint64_t>(), start, end);
    PublishTraceTick(line + kLineBytes, buffer.Get<uint64_t>(), sequence);
}
}
