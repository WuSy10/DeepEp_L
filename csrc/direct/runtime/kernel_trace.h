#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <acl/acl.h>

namespace ascend_deepep::trace {
struct TraceRecord {
    std::string event, op_name, kernel_name;
    std::map<std::string, int64_t> values;
};
class KernelTrace {
public:
    static std::shared_ptr<KernelTrace> Open();
    ~KernelTrace();
    struct Ticket {
        uint64_t address = 0;
        unsigned index = 0;
        uint64_t sequence = 0;
    };
    Ticket Reserve(int rank, uint32_t instance, uint32_t cores,
                   const char* op_name, const char* kernel_name);
    void Submit(Ticket ticket, int64_t launch_ns, int status);
    std::vector<TraceRecord> Drain(bool closing = false);
    TraceRecord Calibration() const;
    void Synchronize();
private:
    enum class State { Free, Reserved, Submitted, Quarantined };
    struct Slot {
        std::atomic<State> state{State::Free};
        uint64_t sequence{};
        int64_t launch_ns{};
        int rank{}, status{};
        uint32_t instance{}, cores{};
        std::string op_name, kernel_name;
        bool logged_launch{};
    };
    static constexpr unsigned kSlots = 128;
    KernelTrace() = default;
    void Init(int device);
    Slot slots_[kSlots];
    std::mutex drain_mutex_;
    std::atomic<unsigned> cursor_{0};
    std::atomic<uint64_t> dropped_{0};
    void* host_{};
    void* mapped_{};
    aclrtContext context_{};
    bool device_accessed_{};
    int device_{};
    int64_t offset_ns_{}, window_ns_{};
    uint64_t calibration_tick_{}, completed_{}, incomplete_{};
};
}
