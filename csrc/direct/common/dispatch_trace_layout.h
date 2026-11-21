#pragma once
#include <cstdint>

namespace ascend_deepep::trace {
// 每个 AIV 在已注册的 Host 内存中独占两条缓存行，分别存放起止 tick。
// MTE3 只发布每条缓存行首部对齐的 uint64_t，避免不同核写入相互干扰。
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kCoreBytes = 2 * kLineBytes;
constexpr uint32_t kTickHz = 1000000000;  // Ascend950
}  // namespace ascend_deepep::trace
