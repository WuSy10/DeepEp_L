#pragma once
#include <cstdint>

namespace ascend_deepep::trace {
// 每个 AIV 独占已注册 Host 内存的两条缓存行：偏移 0 保存起止 tick，
// 偏移 128 保存完成序号 done_seq。MTE3 每次写入 32 字节。
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kCoreBytes = 2 * kLineBytes;
constexpr uint32_t kTickHz = 1000000000;  // Ascend950
// notify 诊断使用独立槽位：先写 20 个 uint64_t 采样值，再写末尾的完成标记。
constexpr uint32_t kNotifyProbeWords = 20;
constexpr uint32_t kNotifyProbeDoneOffset = 224;
static_assert(kNotifyProbeWords * sizeof(uint64_t) <= kNotifyProbeDoneOffset);
static_assert(kNotifyProbeDoneOffset + 32 <= kCoreBytes);
}  // namespace ascend_deepep::trace
