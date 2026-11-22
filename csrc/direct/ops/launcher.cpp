#include "launcher.h"

#include <acl/acl.h>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "runtime/kernel_trace.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "common/launch.h"

namespace {

aclrtStream GetCurrentAclStream()
{
    return c10_npu::getCurrentNPUStream().stream();
}

uint8_t* BytePtr(void* ptr)
{
    return static_cast<uint8_t*>(ptr);
}

uint8_t* OptionalTensorPtr(const c10::optional<torch::Tensor>& tensor)
{
    return tensor.has_value() ? BytePtr(tensor->data_ptr()) : nullptr;
}

// Use a launch-local tiling copy: notify and communication must never share a ticket.
template <class Tiling, class Launch>
void TraceLaunch(ascend_deepep::trace::KernelTrace* sink, Tiling tiling, const char* op_name,
                 const char* kernel_name, Launch launch, int64_t input_rows = -1, int64_t output_rows = -1)
{
    const auto ticket = sink ? sink->Reserve(tiling.rank, tiling.instance_id, tiling.block_dim, op_name, kernel_name,
                                              reinterpret_cast<uint64_t>(GetCurrentAclStream()), tiling.world_size,
                                              input_rows, output_rows)
                             : ascend_deepep::trace::KernelTrace::Ticket{};
    tiling.kernel_trace_slot = ticket.address;
    tiling.kernel_trace_seq = ticket.sequence;
    const int64_t launch_ns = ticket.address
        ? std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()
        : 0;
    try {
        launch(tiling);
    } catch (...) {
        if (ticket.address) sink->Submit(ticket, launch_ns, -1);
        throw;
    }
    if (ticket.address) {
        const auto status = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        sink->Submit(ticket, launch_ns, status);
        TORCH_CHECK(status == ACL_SUCCESS, kernel_name, " submission reported ACL status ", status);
    }
}

}  // namespace

void LaunchDispatchLayoutKernel(torch::Tensor topk_idx, torch::Tensor num_tokens_per_rank,
                                torch::Tensor num_tokens_per_expert, torch::Tensor is_token_in_rank,
                                const ascend_deepep::DispatchLayoutTilingData& tiling, torch::Tensor scratch_buffer,
                                uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    dispatch_layout_kernel_do(tiling.block_dim, stream, BytePtr(topk_idx.data_ptr()),
                              BytePtr(scratch_buffer.data_ptr()), BytePtr(num_tokens_per_rank.data_ptr()),
                              BytePtr(num_tokens_per_expert.data_ptr()), BytePtr(is_token_in_rank.data_ptr()),
                              reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchLayoutTilingData*>(&tiling)),
                              ffts_addr);
}

void LaunchNotifyDispatchKernel(torch::Tensor num_tokens_per_rank, torch::Tensor num_tokens_per_expert,
                                torch::Tensor is_token_in_rank, torch::Tensor rank_prefix_matrix,
                                torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix,
                                torch::Tensor num_recv_tokens, torch::Tensor num_recv_tokens_per_expert,
                                const c10::optional<torch::Tensor>& dispatch_plan, torch::Tensor send_head,
                                const ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer,
                                uint64_t ffts_addr, ascend_deepep::trace::KernelTrace* trace_sink)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    TraceLaunch(trace_sink, tiling, "notify_dispatch", "notify_dispatch_kernel", [&](auto& td) {
        notify_dispatch_plan_kernel_do(
            td.block_dim, stream, BytePtr(num_tokens_per_rank.data_ptr()), BytePtr(num_tokens_per_expert.data_ptr()),
            BytePtr(is_token_in_rank.data_ptr()), BytePtr(symmetric_buffer), BytePtr(rank_prefix_matrix.data_ptr()),
            BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
            BytePtr(num_recv_tokens.data_ptr()), BytePtr(num_recv_tokens_per_expert.data_ptr()),
            OptionalTensorPtr(dispatch_plan), BytePtr(send_head.data_ptr()),
            reinterpret_cast<uint8_t*>(&td), ffts_addr);
    });
}

void LaunchCachedNotifyDispatchKernel(const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr,
                                      ascend_deepep::trace::KernelTrace* trace_sink)
{
    aclrtStream stream = GetCurrentAclStream();
    TraceLaunch(trace_sink, tiling, "cached_notify_dispatch", "cached_notify_dispatch_kernel", [&](auto& td) {
        if (td.kernel_trace_slot) {
            cached_notify_dispatch_trace_kernel_do(td.block_dim, stream, ffts_addr, td.instance_id,
                                                    td.kernel_trace_slot, td.kernel_trace_seq);
        } else {
            cached_notify_dispatch_kernel_do(td.block_dim, stream, ffts_addr, td.instance_id);
        }
    });
}

void LaunchCachedDispatchPlanKernel(torch::Tensor routes, torch::Tensor plan, torch::Tensor send_head,
                                    const ascend_deepep::DispatchTilingData& tiling, uint64_t ffts_addr)
{
    cached_dispatch_plan_kernel_do(
        tiling.block_dim, GetCurrentAclStream(), BytePtr(routes.data_ptr()), BytePtr(plan.data_ptr()),
        send_head.numel() ? BytePtr(send_head.data_ptr()) : nullptr,
        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
}

void LaunchV1DispatchKernel(
    torch::Tensor x, const c10::optional<torch::Tensor>& x_scales, const c10::optional<torch::Tensor>& topk_idx,
    const c10::optional<torch::Tensor>& topk_weights, const c10::optional<torch::Tensor>& num_tokens_per_rank,
    torch::Tensor is_token_in_rank, const c10::optional<torch::Tensor>& num_tokens_per_expert, torch::Tensor recv_x,
    const c10::optional<torch::Tensor>& recv_x_scales, const c10::optional<torch::Tensor>& recv_topk_idx,
    const c10::optional<torch::Tensor>& recv_topk_weights, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor recv_channel_prefix_matrix, torch::Tensor recv_src_idx,
    torch::Tensor send_head, const c10::optional<torch::Tensor>& dispatch_plan,
    ascend_deepep::DispatchTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr, bool use_udma,
    ascend_deepep::trace::KernelTrace* trace_sink)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    if (use_udma) {
        TraceLaunch(trace_sink, tiling, "dispatch", "v1_dispatch_udma_kernel", [&](auto& td) {
            v1_dispatch_udma_plan_kernel_do(
                td.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales), OptionalTensorPtr(topk_idx),
                OptionalTensorPtr(topk_weights), OptionalTensorPtr(num_tokens_per_rank),
                BytePtr(is_token_in_rank.data_ptr()), OptionalTensorPtr(num_tokens_per_expert), BytePtr(symmetric_buffer),
                BytePtr(recv_x.data_ptr()), OptionalTensorPtr(recv_x_scales), OptionalTensorPtr(recv_topk_idx),
                OptionalTensorPtr(recv_topk_weights), BytePtr(rank_prefix_matrix.data_ptr()),
                BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
                BytePtr(recv_src_idx.data_ptr()), BytePtr(send_head.data_ptr()), OptionalTensorPtr(dispatch_plan),
                reinterpret_cast<uint8_t*>(&td), ffts_addr);
        }, x.size(0), recv_x.size(0));
    } else {
        v1_dispatch_kernel_do(
            tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales), OptionalTensorPtr(topk_idx),
            OptionalTensorPtr(topk_weights), OptionalTensorPtr(num_tokens_per_rank),
            BytePtr(is_token_in_rank.data_ptr()), OptionalTensorPtr(num_tokens_per_expert), BytePtr(symmetric_buffer),
            BytePtr(recv_x.data_ptr()), OptionalTensorPtr(recv_x_scales), OptionalTensorPtr(recv_topk_idx),
            OptionalTensorPtr(recv_topk_weights), BytePtr(rank_prefix_matrix.data_ptr()),
            BytePtr(channel_prefix_matrix.data_ptr()), BytePtr(recv_channel_prefix_matrix.data_ptr()),
            BytePtr(recv_src_idx.data_ptr()), BytePtr(send_head.data_ptr()),
            reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::DispatchTilingData*>(&tiling)), ffts_addr);
    }
}

void LaunchV1CombineKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& bias_0, const c10::optional<torch::Tensor>& bias_1,
                           torch::Tensor src_idx, torch::Tensor rank_prefix_matrix, torch::Tensor channel_prefix_matrix,
                           torch::Tensor send_head, torch::Tensor recv_x,
                           const c10::optional<torch::Tensor>& recv_topk_weights,
                           const ascend_deepep::CombineTilingData& tiling, void* symmetric_buffer, uint64_t ffts_addr,
                           bool use_udma, CombinePhase phase, ascend_deepep::trace::KernelTrace* trace_sink)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    // Profiling may separate arrival skew into an all-rank/all-AIV notify.
    // The benchmark enables this process-local switch only during capture;
    // every rank must use the same setting. Ordinary V1/V2 UDMA relies on
    // Begin, which retains its control-header reset and synchronization.
    const char* profile_notify = std::getenv("DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY");
    const bool separate_notify = profile_notify && std::strcmp(profile_notify, "1") == 0;
    const bool has_udma_begin =
        use_udma && (phase == CombinePhase::Fused || phase == CombinePhase::Communication) && !separate_notify;
    if (!has_udma_begin && phase != CombinePhase::Reduction && phase != CombinePhase::Remaining) {
        TraceLaunch(use_udma ? trace_sink : nullptr, tiling, "cached_notify_combine",
                    "cached_notify_combine_kernel", [&](auto& td) {
            if (td.kernel_trace_slot) {
                cached_notify_combine_trace_kernel_do(td.block_dim, stream, ffts_addr, td.instance_id,
                                                      td.kernel_trace_slot, td.kernel_trace_seq);
            } else {
                cached_notify_combine_kernel_do(td.block_dim, stream, ffts_addr, td.instance_id);
            }
        });
    }
    const char* kernel_name = "v1_combine_udma_kernel";
    auto* combine_kernel_do = use_udma ? &v1_combine_udma_kernel_do : &v1_combine_kernel_do;
    if (phase == CombinePhase::Communication) {
        kernel_name = "v1_combine_udma_comm_kernel";
        combine_kernel_do = use_udma ? &v1_combine_udma_comm_kernel_do : &v1_combine_comm_kernel_do;
    } else if (phase == CombinePhase::Reduction) {
        kernel_name = "v1_combine_udma_reduce_kernel";
        combine_kernel_do = use_udma ? &v1_combine_udma_reduce_kernel_do : &v1_combine_reduce_kernel_do;
    } else if (phase == CombinePhase::Pipeline) {
        kernel_name = "v1_combine_udma_pipeline_kernel";
        TORCH_CHECK(use_udma, "aggregate pipeline requires UDMA");
        combine_kernel_do = &v1_combine_udma_pipeline_kernel_do;
    } else if (phase == CombinePhase::EarlySingles) {
        kernel_name = "v1_combine_udma_early_kernel";
        TORCH_CHECK(use_udma, "early single-source combine requires UDMA");
        combine_kernel_do = &v1_combine_udma_early_kernel_do;
    } else if (phase == CombinePhase::Remaining) {
        kernel_name = "v1_combine_udma_remaining_kernel";
        TORCH_CHECK(use_udma, "early single-source combine requires UDMA");
        combine_kernel_do = &v1_combine_udma_remaining_kernel_do;
    }
    // Communication retains every original completion wait. Reduction reads
    // the same SHMEM slots on the same stream after the optional event wait.
    TraceLaunch(use_udma ? trace_sink : nullptr, tiling, "combine", kernel_name, [&](auto& td) {
        combine_kernel_do(td.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(topk_weights),
                          OptionalTensorPtr(bias_0), OptionalTensorPtr(bias_1), BytePtr(src_idx.data_ptr()),
                          BytePtr(rank_prefix_matrix.data_ptr()), BytePtr(channel_prefix_matrix.data_ptr()),
                          BytePtr(send_head.data_ptr()), BytePtr(symmetric_buffer), BytePtr(recv_x.data_ptr()),
                          OptionalTensorPtr(recv_topk_weights),
                          reinterpret_cast<uint8_t*>(&td), ffts_addr);
    }, x.size(0), recv_x.size(0));
}

void LaunchExpandKernel(torch::Tensor x, const c10::optional<torch::Tensor>& x_scales,
                        const c10::optional<torch::Tensor>& topk_idx, const c10::optional<torch::Tensor>& topk_weights,
                        torch::Tensor expanded_x, const c10::optional<torch::Tensor>& expanded_x_scales,
                        const c10::optional<torch::Tensor>& expanded_topk_weights, torch::Tensor expert_offsets,
                        torch::Tensor actual_expert_counts, torch::Tensor recv_to_expanded_idx,
                        torch::Tensor scratch_buffer, const ascend_deepep::ExpandTilingData& tiling, uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    expand_kernel_do(tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(x_scales),
                     OptionalTensorPtr(topk_idx), OptionalTensorPtr(topk_weights), BytePtr(expanded_x.data_ptr()),
                     OptionalTensorPtr(expanded_x_scales), OptionalTensorPtr(expanded_topk_weights),
                     BytePtr(expert_offsets.data_ptr()), BytePtr(actual_expert_counts.data_ptr()),
                     BytePtr(recv_to_expanded_idx.data_ptr()), BytePtr(scratch_buffer.data_ptr()),
                     reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::ExpandTilingData*>(&tiling)), ffts_addr);
}

void LaunchAggregateKernel(torch::Tensor x, const c10::optional<torch::Tensor>& topk_weights,
                           const c10::optional<torch::Tensor>& expanded_values, torch::Tensor recv_to_expanded_idx,
                           torch::Tensor aggregated_x, const c10::optional<torch::Tensor>& aggregated_values,
                           const ascend_deepep::AggregateTilingData& tiling, uint64_t ffts_addr)
{
    aclrtStream stream = GetCurrentAclStream();
    aggregate_kernel_do(tiling.block_dim, stream, BytePtr(x.data_ptr()), OptionalTensorPtr(topk_weights),
                        OptionalTensorPtr(expanded_values), BytePtr(recv_to_expanded_idx.data_ptr()),
                        BytePtr(aggregated_x.data_ptr()), OptionalTensorPtr(aggregated_values),
                        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::AggregateTilingData*>(&tiling)),
                        ffts_addr);
}

void LaunchElasticBarrierKernel(int64_t block_dim, uint64_t ffts_addr, uint32_t instance_id)
{
    TORCH_CHECK(block_dim > 0, "elastic barrier block_dim must be positive");
    cached_notify_combine_kernel_do(static_cast<uint32_t>(block_dim), GetCurrentAclStream(), ffts_addr, instance_id);
}
