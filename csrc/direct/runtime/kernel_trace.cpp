#include "runtime/kernel_trace.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

#include "common/kernel_trace_layout.h"
#include "common/launch.h"
#include "common/platform.h"

namespace ascend_deepep::trace {
namespace {
constexpr size_t kSlotBytes = platform::kMaxNumAiv * kCoreBytes;
constexpr int64_t kNsPerMs = 1'000'000;
// 数组下标就是日志协议中的 op，避免通信方式或启动顺序改变编号。
constexpr std::array<std::string_view, KernelOpCounters::kCount> kRecorderKernelNames{
    "notify_dispatch_kernel", "cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel",
    "cached_notify_combine_kernel", "v1_combine_udma_kernel", "v1_combine_udma_comm_kernel",
    "v1_combine_udma_reduce_kernel", "v1_combine_udma_early_kernel", "v1_combine_udma_remaining_kernel",
    "v1_combine_udma_pipeline_kernel",
};
enum class TraceMode { Recorder, Test, TestMetadataOnly };
std::atomic<TraceMode> trace_mode{TraceMode::Recorder};
// 进程内的 kernel 序号从 1 开始，各 ACL context 共用这个计数器。
std::atomic<uint64_t> next_sequence{1};
thread_local TraceMetadataSnapshot active_metadata;
// 可选 notify 探针的映射区独立于普通 trace 环，释放前需确认设备不再写入。
struct NotifyProbeMemory {
    void* host = nullptr;
    void* mapped = nullptr;
    aclrtContext context = nullptr;
    ~NotifyProbeMemory()
    {
        if (!host) return;
        aclrtContext previous = nullptr;
        if (aclrtGetCurrentContext(&previous) != ACL_SUCCESS ||
            aclrtSetCurrentContext(context) != ACL_SUCCESS || aclrtSynchronizeDevice() != ACL_SUCCESS) {
            std::fprintf(stderr, "DeepEP notify probe cleanup failed; retaining device-accessible memory\n");
            if (previous) (void)aclrtSetCurrentContext(previous);
            return;
        }
        if (!mapped || aclrtHostUnregister(host) == ACL_SUCCESS) (void)aclrtFreeHost(host);
        if (previous) (void)aclrtSetCurrentContext(previous);
    }
};
struct HostProbeState {
    std::size_t capacity = 0;
    uint64_t ordinal = 0, dropped = 0;
    int (*gil_state)() = nullptr;
    HostDispatchProbe* active = nullptr;
    std::vector<HostDispatchSample> samples;
    std::unique_ptr<NotifyProbeMemory> device;
};
// 配置、采样和导出都使用业务线程自己的探针状态。
thread_local HostProbeState host_probe;

template <typename Clock>
int64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

void Check(aclError status, const char* op)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}

// 只接受完整的十进制正整数，避免错误配置让后台忙等或让超时失效。
int ReadRecorderMilliseconds(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    const char* end = value + std::strlen(value);
    int milliseconds = 0;
    const auto parsed = std::from_chars(value, end, milliseconds);
    if (parsed.ec != std::errc{} || parsed.ptr != end || milliseconds <= 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer in milliseconds");
    return milliseconds;
}
}  // namespace

namespace {
// 保留 UTF-8 字节，并转义 JSON 字符串里的引号、反斜杠和控制字符。
void JsonString(std::string& out, const std::string& value)
{
    constexpr char hex[] = "0123456789abcdef";
    out += '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
        else if (ch < 0x20) { out += "\\u00"; out += hex[ch >> 4]; out += hex[ch & 15]; }
        else out += ch; // Preserve UTF-8; the Python file is opened with encoding="utf-8".
    }
    out += '"';
}
}
std::string EncodeTraceRecords(const std::vector<TraceRecord>& records)
{
    std::string out;
    out.reserve(records.size() * 1024);
    for (const auto& record : records) {
        out += '{';
        bool first = true;
        auto key = [&](const std::string& name) {
            if (!first) out += ',';
            first = false; JsonString(out, name); out += ':';
        };
        // 同名字段优先写数值，避免在同一条 JSON 里输出重复键。
        auto label = [&](const std::string& name, const std::string& value) {
            if (record.values.count(name)) return;
            key(name); JsonString(out, value);
        };
        label("event", record.event);
        if (!record.op_name.empty()) label("op_name", record.op_name);
        if (!record.kernel_name.empty()) label("kernel_name", record.kernel_name);
        for (const auto& item : record.labels) {
            if (item.first == "event" || (item.first == "op_name" && !record.op_name.empty()) ||
                (item.first == "kernel_name" && !record.kernel_name.empty())) continue;
            label(item.first, item.second);
        }
        for (const auto& item : record.values) { key(item.first); out += std::to_string(item.second); }
        out += "}\n";
    }
    return out;
}

TraceMetadataSnapshot CurrentTraceMetadata()
{
    return active_metadata;
}
// 每次调用只创建带 ID 和 dispatch 序号的小节点，形状和 Buffer 字段沿父链接读取。
TraceMetadataSnapshot MakeCallMetadata(TraceMetadataSnapshot parent, int64_t api, int64_t handle, int64_t dispatch_count)
{
    auto value = std::make_shared<TraceMetadata>();
    value->parent = std::move(parent);
    value->has_call_ids = true;
    value->api_call_id = api;
    value->handle_id = handle;
    value->dispatch_count = dispatch_count;
    return value;
}
TraceMetadataSnapshot ExchangeTraceMetadata(TraceMetadataSnapshot metadata)
{
    std::swap(metadata, active_metadata);
    return metadata;
}
ScopedTraceMetadata::ScopedTraceMetadata(TraceMetadataSnapshot metadata)
    : previous_(ExchangeTraceMetadata(std::move(metadata)))
{
}
ScopedTraceMetadata::~ScopedTraceMetadata()
{
    ExchangeTraceMetadata(std::move(previous_));
}

void ConfigureTraceTestMode(bool metadata_only)
{
    // 测试工具通过原生绑定显式启用独立模式，旧诊断环境变量不改变正式记录器。
    trace_mode.store(metadata_only ? TraceMode::TestMetadataOnly : TraceMode::Test, std::memory_order_relaxed);
}

std::shared_ptr<KernelTrace> KernelTrace::Open()
{
    const char* enable = std::getenv("DEEPEP_TRACE_RECORDER_ENABLE");
    if (!enable || std::strcmp(enable, "1") != 0) return {};
    const auto mode = trace_mode.load(std::memory_order_relaxed);
    // metadata 测试只准备上下文，不分配采集环，也不校准设备时钟。
    if (mode == TraceMode::TestMetadataOnly) return {};
    try {
        static std::mutex mutex;
        // 以 ACL context 为键，弱引用允许最后一个使用者退出后释放采集器。
        static std::map<aclrtContext, std::weak_ptr<KernelTrace>> sinks;
        std::lock_guard<std::mutex> lock(mutex);
        aclrtContext context{};
        int32_t device{};
        Check(aclrtGetCurrentContext(&context), "get context");
        Check(aclrtGetDevice(&device), "get device");
        if (auto alive = sinks[context].lock()) return alive;
        auto sink = std::shared_ptr<KernelTrace>(new KernelTrace);
        sink->context_ = context;
        sink->recorder_mode_ = mode == TraceMode::Recorder;
        sink->Init(device);
        sinks[context] = sink;
        return sink;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DeepEP kernel trace disabled: %s\n", e.what());
        return {};
    }
}

void KernelTrace::Init(int device)
{
    device_ = device;
    if (recorder_mode_) {
        // 初始化时一次性读取，后续扫描和计时不重新访问环境变量。
        poll_interval_ms_ = ReadRecorderMilliseconds("DEEPEP_TRACE_POLL_INTERVAL_MS", poll_interval_ms_);
        timeout_ms_ = ReadRecorderMilliseconds("DEEPEP_TRACE_TIMEOUT_MS", timeout_ms_);
    }
    // 正式采集使用固定槽数；开发测试才允许改变内存环容量。
    const char* value = recorder_mode_ ? nullptr : std::getenv("DEEPEP_TRACE_SLOTS");
    if (value) {
        char* end{};
        const auto count = std::strtoul(value, &end, 10);
        if (!*value || *value < '0' || *value > '9' || *end || count < 128 || count > 16384)
            throw std::runtime_error("DEEPEP_TRACE_SLOTS must be an integer in [128, 16384]");
        slot_count_ = static_cast<unsigned>(count);
    }
    // slots_ 放 Host 状态和元数据，映射区只放设备写入的 tick 与完成标记。
    slots_ = std::make_unique<Slot[]>(slot_count_);
    Check(aclrtMallocHost(&host_, slot_count_ * kSlotBytes), "allocate trace ring");
    std::memset(host_, 0, slot_count_ * kSlotBytes);
    Check(aclrtHostRegister(host_, slot_count_ * kSlotBytes, ACL_HOST_REGISTER_MAPPED, &mapped_), "map trace ring");

    // 用私有流启动一次校准采样，以同步前后的单调时钟界定采样窗口。
    aclrtStream stream{};
    Check(aclrtCreateStream(&stream), "create calibration stream");
    const auto before = NowNs<std::chrono::steady_clock>();
    device_accessed_ = true;
    kernel_trace_clock_do(stream, static_cast<uint8_t*>(mapped_));
    const auto synced = aclrtSynchronizeStream(stream);
    const auto after = NowNs<std::chrono::steady_clock>();
    const auto destroyed = aclrtDestroyStream(stream);
    Check(synced, "calibrate device clock");
    Check(destroyed, "destroy calibration stream");
    auto* first = static_cast<uint64_t*>(host_);
    const auto tick = __atomic_exchange_n(first, uint64_t{0}, __ATOMIC_ACQ_REL);
    if (!tick) throw std::runtime_error("calibration kernel did not publish a tick");
    const auto anchor0 = NowNs<std::chrono::steady_clock>();
    const auto wall = NowNs<std::chrono::system_clock>();
    const auto anchor1 = NowNs<std::chrono::steady_clock>();
    // 用校准窗口的中点估算设备 tick 对应的 Host 时间，再换算为墙钟时间。
    offset_ns_ =
        wall - (anchor0 + (anchor1 - anchor0) / 2) + before + (after - before) / 2 - static_cast<int64_t>(tick);
    calibration_tick_ = tick;
    window_ns_ = after - before;

}

void ConfigureHostDispatchProbe(std::size_t capacity, int (*gil_state)(), bool device_timing)
{
    if (host_probe.active) throw std::runtime_error("cannot configure an active Host dispatch probe");
    if (host_probe.device) throw std::runtime_error("take previous notify device probe before configuring again");
    if (capacity > 1000000) throw std::runtime_error("Host dispatch probe capacity exceeds 1000000");
    // 仅在请求设备计时且有样本容量时，为每次诊断调用预留独立记录区。
    std::unique_ptr<NotifyProbeMemory> device;
    if (device_timing && capacity) {
        device = std::make_unique<NotifyProbeMemory>();
        Check(aclrtGetCurrentContext(&device->context), "notify probe context");
        Check(aclrtMallocHost(&device->host, capacity * kCoreBytes), "allocate notify probe");
        std::memset(device->host, 0, capacity * kCoreBytes);
        Check(aclrtHostRegister(device->host, capacity * kCoreBytes, ACL_HOST_REGISTER_MAPPED, &device->mapped),
              "map notify probe");
    }
    host_probe.samples.clear();
    host_probe.samples.reserve(capacity);
    host_probe.capacity = capacity;
    host_probe.ordinal = host_probe.dropped = 0;
    host_probe.gil_state = gil_state;
    host_probe.device = std::move(device);
}

HostDispatchReport TakeHostDispatchProbe()
{
    if (host_probe.active) throw std::runtime_error("cannot take an active Host dispatch probe");
    if (host_probe.device) {
        aclrtContext current{};
        Check(aclrtGetCurrentContext(&current), "take notify probe context");
        if (current != host_probe.device->context) throw std::runtime_error("notify probe context changed");
        // 只在显式导出诊断时同步设备，确保读出采样和释放映射前设备已写完。
        Check(aclrtSynchronizeDevice(), "complete notify probe before export");
        for (auto& sample : host_probe.samples) {
            if (!sample.notify_sequence) continue;
            const auto* line = reinterpret_cast<const uint64_t*>(
                static_cast<const uint8_t*>(host_probe.device->host) + (sample.ordinal - 1) * kCoreBytes);
            sample.notify_done = __atomic_load_n(line + kNotifyProbeDoneOffset / sizeof(uint64_t), __ATOMIC_ACQUIRE);
            // 即使已同步也要核对本次序号，避免把没有发布的数据当作有效样本。
            if (sample.notify_done == sample.notify_sequence)
                std::copy_n(line, sample.notify_ticks.size(), sample.notify_ticks.begin());
        }
    }
    HostDispatchReport report{std::move(host_probe.samples), host_probe.dropped};
    host_probe.capacity = 0;
    host_probe.gil_state = nullptr;
    host_probe.device.reset();
    return report;
}

uint64_t HostDispatchProbe::NotifyAddress()
{
    if (!enabled_ || !host_probe.device || sample_.ordinal > host_probe.capacity) return 0;
    // 诊断在显式导出前不会复用槽位，按 1 起始的调用序号计算偏移。
    sample_.notify_sequence = sample_.ordinal;
    sample_.notify_address = reinterpret_cast<uint64_t>(host_probe.device->mapped) +
                             (sample_.ordinal - 1) * kCoreBytes;
    return sample_.notify_address;
}

HostDispatchProbe* CurrentHostDispatchProbe() { return host_probe.active; }

HostDispatchProbe::HostDispatchProbe(int rank, bool cached)
{
    if (!host_probe.capacity) return;
    enabled_ = true;
    exceptions_ = std::uncaught_exceptions();
    sample_.ordinal = ++host_probe.ordinal;
    sample_.rank = rank;
    sample_.cached = cached;
    sample_.wait_path = 0;
    sample_.api_call_id = -1;
    sample_.success = false;
    sample_.ticks.fill(-1);
    sample_.gil.fill(-1);
    previous_ = host_probe.active;
    host_probe.active = this;
    Mark(DispatchProbePoint::NativeEntry);
    for (auto metadata = CurrentTraceMetadata(); metadata; metadata = metadata->parent) {
        if (metadata->has_call_ids) { sample_.api_call_id = metadata->api_call_id; break; }
        auto found = metadata->values.find("api_call_id");
        if (found != metadata->values.end()) { sample_.api_call_id = found->second; break; }
    }
}

void HostDispatchProbe::MarkEnabled(DispatchProbePoint point)
{
    const auto index = static_cast<unsigned>(point);
    sample_.ticks[index] = NowNs<std::chrono::steady_clock>();
    sample_.gil[index] = host_probe.gil_state ? host_probe.gil_state() : -1;
}

HostDispatchProbe::~HostDispatchProbe()
{
    if (!enabled_) return;
    // 作用域异常退出时把样本记为失败，即使此前已经走过成功标记。
    sample_.success = sample_.success && std::uncaught_exceptions() == exceptions_;
    Mark(DispatchProbePoint::NativeExit);
    host_probe.active = previous_;
    if (host_probe.samples.size() < host_probe.capacity) host_probe.samples.push_back(sample_);
    else ++host_probe.dropped;
}

KernelTrace::Ticket KernelTrace::Reserve(int rank, uint32_t instance, uint32_t cores, const char* op_name,
                                       const char* kernel_name, uint64_t stream, uint32_t ep_size,
                                       int64_t input_rows, int64_t output_rows)
{
    const auto& metadata = active_metadata;
    int op = -1;
    int64_t op_count = -1;
    if (recorder_mode_) {
        // 只有成功关联到写盘会话的调用才占用采集槽位。
        const auto* scope = metadata.get();
        while (scope && !scope->op_counters) scope = scope->parent.get();
        if (!scope) return {};
        const auto comm_id = scope->labels.find("comm_id");
        if (comm_id == scope->labels.end() || comm_id->second.empty()) return {};
        const auto kernel = std::find(kRecorderKernelNames.begin(), kRecorderKernelNames.end(), std::string_view(kernel_name));
        if (kernel == kRecorderKernelNames.end()) return {};
        op = static_cast<int>(kernel - kRecorderKernelNames.begin());
        // 计数早于槽位申请。即使采集环已满，业务 kernel 仍会启动并消耗一次编号。
        op_count = scope->op_counters->next[op].fetch_add(1, std::memory_order_relaxed);
    }
    // 环形缓冲区的目标槽位被占用时跳过本次采集，让 kernel 继续启动。
    const auto index = cursor_.fetch_add(1, std::memory_order_relaxed) % slot_count_;
    auto& slot = slots_[index];
    auto expected = State::Free;
    if (cores == 0 || cores > platform::kMaxNumAiv ||
        !slot.state.compare_exchange_strong(expected, State::Reserved, std::memory_order_acquire)) {
        ++dropped_;
        return {};
    }
    slot.sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
    // 0 保留给未初始化的槽位，序号必须能用 JSON 日志中的 int64 字段表示。
    if (slot.sequence == 0 || slot.sequence > uint64_t(std::numeric_limits<int64_t>::max())) {
        slot.state.store(State::Free, std::memory_order_release);
        ++dropped_;
        return {};
    }
    slot.op_name = op_name;
    slot.kernel_name = kernel_name;
    slot.rank = rank;
    slot.instance = instance;
    slot.cores = cores;
    slot.op = op;
    slot.op_count = op_count;
    // 持有当前调用的不可变快照，让后台采集仍能关联到发起 kernel 的 API 和 handle。
    slot.metadata = metadata;
    slot.stream = stream;
    slot.ep_size = ep_size;
    slot.input_rows = input_rows;
    slot.output_rows = output_rows;
    slot.logged_launch = false;
    slot.logged_timeout = false;
    return {reinterpret_cast<uint64_t>(mapped_) + index * kSlotBytes, index, slot.sequence};
}

void KernelTrace::Submit(Ticket ticket, int64_t launch_ns, int status)
{
    auto& slot = slots_[ticket.index];
    slot.launch_ns = launch_ns;
    slot.status = status;
    // 用单调时钟记录 Host 提交成功的时刻，不把排队前的时间或墙钟跳变算作超时。
    if (recorder_mode_ && status == ACL_SUCCESS) slot.submit_steady_ns = NowNs<std::chrono::steady_clock>();
    // 发布 Host 侧提交状态，后台线程看到 Submitted 后就可读取本槽的元数据。
    slot.state.store(State::Submitted, std::memory_order_release);
}

TraceRecord KernelTrace::Calibration() const
{
    return {"calibration", "", "", {{"device", device_}, {"pid", getpid()}, {"tick_hz", kTickHz},
        {"tick", static_cast<int64_t>(calibration_tick_)}, {"offset_ns", offset_ns_}, {"window_ns", window_ns_},
        {"slots", slot_count_}, {"slot_bytes", kSlotBytes}}};
}

void KernelTrace::Synchronize()
{
    // Python 可能从其他线程调用，先切到采集器所属 context，同步后恢复原 context。
    aclrtContext previous{};
    Check(aclrtGetCurrentContext(&previous), "get context");
    Check(aclrtSetCurrentContext(context_), "select trace context");
    const auto status = aclrtSynchronizeDevice();
    const auto restored = aclrtSetCurrentContext(previous);
    Check(status, "drain trace device");
    Check(restored, "restore trace context");
}

TraceRecord KernelTrace::DescribeSlot(const Slot& slot) const
{
    TraceRecord record{"", slot.op_name, slot.kernel_name,
        {{"gid", static_cast<int64_t>(slot.sequence)}, {"rank", slot.rank},
         {"local_instance", slot.instance}, {"launch_ns", slot.launch_ns}, {"aiv", slot.cores},
         {"pid", getpid()}, {"device", device_}}};
    if (slot.op >= 0) record.values["op"] = slot.op;
    if (slot.op_count >= 0) record.values["op_count"] = slot.op_count;
    if (slot.ep_size) record.values["ep_size"] = slot.ep_size;
    if (slot.input_rows >= 0) record.values["kernel_input_rows"] = slot.input_rows;
    if (slot.output_rows >= 0) record.values["kernel_output_rows"] = slot.output_rows;
    // 从调用层往 Buffer 层展开，已写入的 kernel 字段与更近的字段优先。
    for (auto metadata = slot.metadata; metadata; metadata = metadata->parent) {
        if (metadata->has_call_ids) {
            record.values.emplace("api_call_id", metadata->api_call_id);
            record.values.emplace("handle_id", metadata->handle_id);
            if (metadata->dispatch_count >= 0) record.values.emplace("dispatch_count", metadata->dispatch_count);
        }
        record.values.insert(metadata->values.begin(), metadata->values.end());
        record.labels.insert(metadata->labels.begin(), metadata->labels.end());
    }
    record.labels["stream"] = std::to_string(slot.stream);
    return record;
}

std::vector<TraceRecord> KernelTrace::Drain(bool closing, bool recorder)
{
    std::lock_guard<std::mutex> lock(drain_mutex_);
    std::vector<TraceRecord> records;
    const auto poll_ns = recorder ? NowNs<std::chrono::steady_clock>() : 0;
    const auto timeout_ns = static_cast<int64_t>(timeout_ms_) * kNsPerMs;
    for (unsigned index = 0; index < slot_count_; ++index) {
        auto& slot = slots_[index];
        if (slot.state.load(std::memory_order_acquire) != State::Submitted) continue;
        TraceRecord record;
        // 测试路径保留 launch/summary 格式。正式路径只为实际输出的事件展开元数据。
        if (!recorder) record = DescribeSlot(slot);
        // 设备可能还没完成，launch 先记一次，下一次轮询继续等完成标记。
        if (!recorder && !slot.logged_launch) {
            record.event = slot.status ? "launch_error" : "launch";
            record.values["status"] = slot.status;
            records.push_back(record);
            slot.logged_launch = true;
        }
        if (slot.status != ACL_SUCCESS) {
            // 提交失败和已提交但未完成是两种状态，失败不能写成 type=1。
            slot.state.store(State::Quarantined, std::memory_order_release);
            ++incomplete_;
            continue;
        }
        auto* base = static_cast<uint8_t*>(host_) + index * kSlotBytes;
        uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
        bool ready = true;
        for (uint32_t core = 0; core < slot.cores; ++core) {
            auto* line = base + core * kCoreBytes;
            // 每个 AIV 的完成序号匹配当前轮次后，才读取它的起止 tick。
            const auto sequence = __atomic_load_n(
                reinterpret_cast<uint64_t*>(line + kLineBytes), __ATOMIC_ACQUIRE);
            if (sequence != slot.sequence) {
                ready = false;
                break;
            }
            const auto start = __atomic_load_n(reinterpret_cast<uint64_t*>(line), __ATOMIC_RELAXED);
            const auto end = __atomic_load_n(reinterpret_cast<uint64_t*>(line + sizeof(uint64_t)), __ATOMIC_RELAXED);
            first = std::min(first, start);
            last = std::max(last, end);
        }
        if (!ready) {
            if (recorder && recorder_mode_ && !slot.logged_timeout &&
                poll_ns - slot.submit_steady_ns >= timeout_ns) {
                // 超时只报告一次，继续保留此槽；后续收齐标记时用相同 gid 输出 end。
                record = DescribeSlot(slot);
                record.event = "timeout";
                records.push_back(std::move(record));
                slot.logged_timeout = true;
            }
            if (closing) {
                // 关闭仍未收齐则隔离槽位；只有测试日志单独输出 incomplete。
                if (!recorder) {
                    record.event = "incomplete";
                    records.push_back(std::move(record));
                }
                slot.state.store(State::Quarantined, std::memory_order_release);
                ++incomplete_;
            }
            continue;
        }
        // 以各 AIV 的最早开始与最晚结束表示本 rank 的执行区间。
        if (recorder) record = DescribeSlot(slot);
        record.event = "end";
        record.values.insert({{"start_tick", first}, {"end_tick", last},
            {"start_ns", offset_ns_ + static_cast<int64_t>(first)},
            {"end_ns", offset_ns_ + static_cast<int64_t>(last)}, {"elapsed_ns", last - first}});
        records.push_back(std::move(record));
        ++completed_;
        // 全部 AIV 完成本轮写入后归还槽位。下次靠新序号区分旧数据，无需清零。
        slot.state.store(State::Free, std::memory_order_release);
    }
    if (closing) {
        const auto dropped = dropped_.load();
        if (!recorder) {
            // 测试日志在关闭批次末尾输出汇总，供原有校验器使用。
            records.push_back({"summary", "", "", {{"completed", completed_},
                {"incomplete", incomplete_}, {"dropped", dropped}}});
        } else if (incomplete_ || dropped) {
            // 正式文件只包含约定的 type=1/2 行。丢样和关闭时的未完成数另报到 stderr。
            std::fprintf(stderr, "DeepEP trace recorder device %d: incomplete=%llu dropped=%llu\n", device_,
                static_cast<unsigned long long>(incomplete_), static_cast<unsigned long long>(dropped));
        }
    }
    return records;
}

KernelTrace::~KernelTrace()
{
    aclrtContext previous{};
    const bool have_previous = aclrtGetCurrentContext(&previous) == ACL_SUCCESS;
    const bool current = have_previous && previous == context_;
    const bool selected = current || aclrtSetCurrentContext(context_) == ACL_SUCCESS;
    const bool safe = !device_accessed_ || (selected && aclrtSynchronizeDevice() == ACL_SUCCESS);
    // 无法确认设备写完时保留映射，避免设备写入已释放的 Host 内存。
    if (safe && selected && (!mapped_ || aclrtHostUnregister(host_) == ACL_SUCCESS)) {
        if (host_) aclrtFreeHost(host_);
    } else if (host_) {
        std::fprintf(stderr, "DeepEP kernel trace retained an undrained host mapping\n");
    }
    if (!current && selected && have_previous) aclrtSetCurrentContext(previous);
}
}  // namespace ascend_deepep::trace
