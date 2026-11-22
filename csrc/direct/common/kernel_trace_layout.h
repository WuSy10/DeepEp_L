#pragma once
#include <cstdint>

namespace ascend_deepep::trace {
// Each AIV owns two independent cache lines in a registered Host allocation.
// Offset 0 holds start/end ticks; offset 128 holds done_seq. Each MTE3 write is 32 bytes.
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kCoreBytes = 2 * kLineBytes;
constexpr uint32_t kTickHz = 1000000000;  // Ascend950
}  // namespace ascend_deepep::trace
