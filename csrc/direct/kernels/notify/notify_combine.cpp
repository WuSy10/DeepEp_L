#include "kernels/common/kernel_trace.h"
#include "shmem.h"

#include "common/launch.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __schedmode__(1) __global__ __aicore__ void cached_notify_combine_kernel(
    uint64_t ffts_addr, uint32_t instance_id, uint64_t slot, uint64_t sequence, uint32_t cores)
{
    const uint64_t start = slot ? AscendC::GetSystemCycle() : 0;
    aclshmemx_instance_ctx_set(instance_id);
    util_set_ffts_config(ffts_addr);
    aclshmemx_sync_vec_all();
    ascend_deepep::trace::Finish(slot, sequence, cores, start);
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_combine_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint32_t instance_id)
{
    cached_notify_combine_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(32), stream>>>(
        ffts_addr, instance_id, 0, 0, block_dim);
}

extern "C" ASCEND_DEEPEP_EXPORT void cached_notify_combine_trace_kernel_do(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint32_t instance_id, uint64_t slot, uint64_t sequence)
{
    cached_notify_combine_kernel<<<block_dim, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(32), stream>>>(
        ffts_addr, instance_id, slot, sequence, block_dim);
}
