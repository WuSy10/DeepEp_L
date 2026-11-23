#pragma once
#include "kernel_operator.h"
#include "common/kernel_trace_layout.h"

namespace ascend_deepep::trace {
using namespace AscendC;
// 等最后一个读取者用完 UB 再复用这 32 字节。每次写入都等 MTE3 完成后才返回。
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

// 在 kernel 主体用完事件后调用。归约路径可能以 MTE3_V 等待结束，
// 这里用本核流水线事件等待 V 和 MTE3，再读取结束 tick 并复用 UB。
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
    // 先写起止 tick，再在独立的缓存行写完成序号，供 Host 判断这轮数据是否就绪。
    PublishTraceTick(line, buffer.Get<uint64_t>(), start, end);
    PublishTraceTick(line + kLineBytes, buffer.Get<uint64_t>(), sequence);
}
}
