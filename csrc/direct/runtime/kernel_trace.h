#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <acl/acl.h>
#include "common/kernel_trace_layout.h"

namespace ascend_deepep::trace {
// 每个通信组的本 rank 按 kernel 名维护独立计数器，起始值均为 0。
struct KernelOpCounters {
    static constexpr std::size_t kCount = 10;
    KernelOpCounters()
    {
        for (auto& count : next) count.store(0, std::memory_order_relaxed);
    }
    std::array<std::atomic<int64_t>, kCount> next;
};

// 元数据分为 Buffer 固定字段、形状模板和调用 ID，各层通过父节点共享。
struct TraceMetadata {
    std::shared_ptr<const TraceMetadata> parent;
    std::map<std::string, int64_t> values;
    std::map<std::string, std::string> labels;
    bool has_call_ids = false;
    int64_t api_call_id = 0, handle_id = 0;
    int64_t dispatch_count = -1;  // 无关联的 dispatch 时保留空值。
    std::shared_ptr<KernelOpCounters> op_counters;
};
using TraceMetadataSnapshot = std::shared_ptr<const TraceMetadata>;
TraceMetadataSnapshot MakeCallMetadata(TraceMetadataSnapshot parent, int64_t api, int64_t handle,
                                      int64_t dispatch_count = -1);
TraceMetadataSnapshot CurrentTraceMetadata();
TraceMetadataSnapshot ExchangeTraceMetadata(TraceMetadataSnapshot metadata);
// 临时替换当前线程的元数据，离开作用域时恢复前一份。
class ScopedTraceMetadata {
public:
    explicit ScopedTraceMetadata(TraceMetadataSnapshot metadata);
    ~ScopedTraceMetadata();
    ScopedTraceMetadata(const ScopedTraceMetadata&) = delete;
    ScopedTraceMetadata& operator=(const ScopedTraceMetadata&) = delete;
private:
    TraceMetadataSnapshot previous_;
};
// 一条 JSONL 事件的原生表示，数值和字符串分别保存以维持字段类型。
struct TraceRecord {
    std::string event, op_name, kernel_name;
    std::map<std::string, int64_t> values;
    std::map<std::string, std::string> labels;
};
std::string EncodeTraceRecords(const std::vector<TraceRecord>& records);
// 专用于测试进程，必须在创建 Buffer 前调用。
void ConfigureTraceTestMode(bool metadata_only);
// Opt-in Host diagnostics. Fixed-size samples are reserved before the workload and read afterwards.
// 这些是 Host 诊断的时间点，记录在预分配样本中，测量结束后才导出。
enum class DispatchProbePoint : unsigned {
    NativeEntry, NotifyBegin, NotifyEnd, WaitBegin, FirstPending, FirstCountReady, AllCountsReady,
    WaitEnd, OutputsBegin, OutputsEnd, DispatchBegin, DispatchEnd, NativeExit, Count
};
constexpr unsigned kDispatchProbePoints = static_cast<unsigned>(DispatchProbePoint::Count);
// 同一条样本保存 Host 时间、GIL 状态，以及可选的 notify 设备计时。
struct HostDispatchSample {
    uint64_t ordinal;
    int rank, cached, wait_path; // 1=mapped polling, 2=Tensor.cpu, 3=fixed capacity, 4=cached.
    int64_t api_call_id;
    bool success;
    std::array<int64_t, kDispatchProbePoints> ticks, gil;
    uint64_t notify_address = 0, notify_sequence = 0, notify_done = 0;
    std::array<uint64_t, kNotifyProbeWords> notify_ticks{};
};
struct HostDispatchReport {
    std::vector<HostDispatchSample> samples;
    uint64_t dropped;
};
void ConfigureHostDispatchProbe(std::size_t capacity, int (*gil_state)(), bool device_timing = false);
HostDispatchReport TakeHostDispatchProbe();
// 按一次 dispatch 的生命周期采样，析构时恢复调用线程先前的探针状态。
class HostDispatchProbe {
public:
    HostDispatchProbe(int rank, bool cached);
    ~HostDispatchProbe();
    HostDispatchProbe(const HostDispatchProbe&) = delete;
    HostDispatchProbe& operator=(const HostDispatchProbe&) = delete;
    void Mark(DispatchProbePoint point) { if (enabled_) MarkEnabled(point); }
    void SetWaitPath(int path) { if (enabled_) sample_.wait_path = path; }
    void Finish() { if (enabled_) sample_.success = true; }
    uint64_t NotifyAddress();
    uint64_t NotifySequence() const { return enabled_ ? sample_.notify_sequence : 0; }
    bool Has(DispatchProbePoint point) const { return enabled_ && sample_.ticks[static_cast<unsigned>(point)] >= 0; }
private:
    void MarkEnabled(DispatchProbePoint point);
    bool enabled_ = false;
    int exceptions_;
    HostDispatchProbe* previous_ = nullptr;
    HostDispatchSample sample_;
};
HostDispatchProbe* CurrentHostDispatchProbe();

// 一个 ACL context 共用一个采集器。Host 管理槽位，设备通过映射地址发布计时。
class KernelTrace {
public:
    static std::shared_ptr<KernelTrace> Open();
    ~KernelTrace();
    // address 交给设备、index 供 Host 提交、sequence 用来区分槽位的不同轮次。
    struct Ticket {
        uint64_t address = 0;
        unsigned index = 0;
        uint64_t sequence = 0;
    };
    Ticket Reserve(int rank, uint32_t instance, uint32_t cores,
                   const char* op_name, const char* kernel_name, uint64_t stream = 0, uint32_t ep_size = 0,
                   int64_t input_rows = -1, int64_t output_rows = -1);
    void Submit(Ticket ticket, int64_t launch_ns, int status);
    std::vector<TraceRecord> Drain(bool closing = false, bool recorder = false);
    int Device() const { return device_; }
    int PollIntervalMs() const { return poll_interval_ms_; }
    TraceRecord Calibration() const;
    void Synchronize();
private:
    // 已提交的槽位可被后台线程读取；失败或关闭时未完成的槽位会被隔离。
    enum class State { Free, Reserved, Submitted, Quarantined };
    struct Slot {
        std::atomic<State> state{State::Free};
        uint64_t sequence{};
        int64_t launch_ns{};
        int64_t submit_steady_ns{};  // 提交成功时的单调时间，用于判断超时。
        int rank{}, status{};
        int op = -1;
        int64_t op_count = -1;
        uint32_t instance{}, cores{};
        std::string op_name, kernel_name;
        TraceMetadataSnapshot metadata;
        uint64_t stream{};
        uint32_t ep_size{};
        int64_t input_rows{-1}, output_rows{-1};
        bool logged_launch{};
        bool logged_timeout{};
    };
    TraceRecord DescribeSlot(const Slot& slot) const;
    KernelTrace() = default;
    void Init(int device);
    bool recorder_mode_ = true;
    // 正式采集初始化时读入，开发基准仍使用默认时间。
    int poll_interval_ms_ = 2;
    int timeout_ms_ = 10;
    unsigned slot_count_ = 128;
    std::unique_ptr<Slot[]> slots_;
    std::mutex drain_mutex_;
    std::atomic<unsigned> cursor_{0};
    std::atomic<uint64_t> dropped_{0};
    // 同一块内存的 Host 地址与设备可访问地址，分别用于读回和下发给 kernel。
    void* host_{};
    void* mapped_{};
    aclrtContext context_{};
    bool device_accessed_{};
    int device_{};
    int64_t offset_ns_{}, window_ns_{};
    uint64_t calibration_tick_{}, completed_{}, incomplete_{};
};
}
