#include "runtime/kernel_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

#include "common/kernel_trace_layout.h"
#include "common/launch.h"
#include "common/platform.h"

namespace ascend_deepep::trace {
namespace {
constexpr size_t kSlotBytes = platform::kMaxNumAiv * kCoreBytes;
std::atomic<uint64_t> next_sequence{1};
thread_local TraceMetadataSnapshot active_metadata;
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
}  // namespace

namespace {
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

std::shared_ptr<KernelTrace> KernelTrace::Open()
{
    const char* directory = std::getenv("DEEPEP_TRACE_DIR");
    if (!directory || !*directory) return {};
    // Diagnostic metadata-only mode must not allocate a ring or launch calibration/device probes.
    const char* diagnostic = std::getenv("DEEPEP_TRACE_DIAGNOSTIC_MODE");
    if (diagnostic && std::strcmp(diagnostic, "metadata") == 0) return {};
    try {
        static std::mutex mutex;
        static std::map<aclrtContext, std::weak_ptr<KernelTrace>> sinks;
        std::lock_guard<std::mutex> lock(mutex);
        aclrtContext context{};
        int32_t device{};
        Check(aclrtGetCurrentContext(&context), "get context");
        Check(aclrtGetDevice(&device), "get device");
        if (auto alive = sinks[context].lock()) return alive;
        auto sink = std::shared_ptr<KernelTrace>(new KernelTrace);
        sink->context_ = context;
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
    if (const char* value = std::getenv("DEEPEP_TRACE_SLOTS")) {
        char* end{};
        const auto count = std::strtoul(value, &end, 10);
        if (!*value || *value < '0' || *value > '9' || *end || count < 128 || count > 16384)
            throw std::runtime_error("DEEPEP_TRACE_SLOTS must be an integer in [128, 16384]");
        slot_count_ = static_cast<unsigned>(count);
    }
    slots_ = std::make_unique<Slot[]>(slot_count_);
    Check(aclrtMallocHost(&host_, slot_count_ * kSlotBytes), "allocate trace ring");
    std::memset(host_, 0, slot_count_ * kSlotBytes);
    Check(aclrtHostRegister(host_, slot_count_ * kSlotBytes, ACL_HOST_REGISTER_MAPPED, &mapped_), "map trace ring");

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
        // Only at explicit export, outside timed workload. Do not free a device writer's memory early.
        Check(aclrtSynchronizeDevice(), "complete notify probe before export");
        for (auto& sample : host_probe.samples) {
            if (!sample.notify_sequence) continue;
            const auto* line = reinterpret_cast<const uint64_t*>(
                static_cast<const uint8_t*>(host_probe.device->host) + (sample.ordinal - 1) * kCoreBytes);
            sample.notify_done = __atomic_load_n(line + kNotifyProbeDoneOffset / sizeof(uint64_t), __ATOMIC_ACQUIRE);
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
    const auto index = cursor_.fetch_add(1, std::memory_order_relaxed) % slot_count_;
    auto& slot = slots_[index];
    auto expected = State::Free;
    if (cores == 0 || cores > platform::kMaxNumAiv ||
        !slot.state.compare_exchange_strong(expected, State::Reserved, std::memory_order_acquire)) {
        ++dropped_;
        return {};
    }
    slot.sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
    // Zero is the uninitialized marker; JSON exposes the sequence as int64.
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
    slot.metadata = CurrentTraceMetadata();
    slot.stream = stream;
    slot.ep_size = ep_size;
    slot.input_rows = input_rows;
    slot.output_rows = output_rows;
    slot.logged_launch = false;
    return {reinterpret_cast<uint64_t>(mapped_) + index * kSlotBytes, index, slot.sequence};
}

void KernelTrace::Submit(Ticket ticket, int64_t launch_ns, int status)
{
    auto& slot = slots_[ticket.index];
    slot.launch_ns = launch_ns;
    slot.status = status;
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
    aclrtContext previous{};
    Check(aclrtGetCurrentContext(&previous), "get context");
    Check(aclrtSetCurrentContext(context_), "select trace context");
    const auto status = aclrtSynchronizeDevice();
    const auto restored = aclrtSetCurrentContext(previous);
    Check(status, "drain trace device");
    Check(restored, "restore trace context");
}

std::vector<TraceRecord> KernelTrace::Drain(bool closing)
{
    std::lock_guard<std::mutex> lock(drain_mutex_);
    std::vector<TraceRecord> records;
    for (unsigned index = 0; index < slot_count_; ++index) {
        auto& slot = slots_[index];
        if (slot.state.load(std::memory_order_acquire) != State::Submitted) continue;
        TraceRecord record{"", slot.op_name, slot.kernel_name,
            {{"gid", static_cast<int64_t>(slot.sequence)}, {"rank", slot.rank},
             {"local_instance", slot.instance}, {"launch_ns", slot.launch_ns}, {"aiv", slot.cores},
             {"pid", getpid()}, {"device", device_}}};
        if (slot.ep_size) record.values["ep_size"] = slot.ep_size;
        if (slot.input_rows >= 0) record.values["kernel_input_rows"] = slot.input_rows;
        if (slot.output_rows >= 0) record.values["kernel_output_rows"] = slot.output_rows;
        for (auto metadata = slot.metadata; metadata; metadata = metadata->parent) {
            record.values.insert(metadata->values.begin(), metadata->values.end());
            record.labels.insert(metadata->labels.begin(), metadata->labels.end());
        }
        record.labels["stream"] = std::to_string(slot.stream);
        if (!slot.logged_launch) {
            record.event = slot.status ? "launch_error" : "launch";
            record.values["status"] = slot.status;
            records.push_back(record);
            slot.logged_launch = true;
            if (slot.status) {
                slot.state.store(State::Quarantined, std::memory_order_release);
                ++incomplete_;
                continue;
            }
        }
        auto* base = static_cast<uint8_t*>(host_) + index * kSlotBytes;
        uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
        bool ready = true;
        for (uint32_t core = 0; core < slot.cores; ++core) {
            auto* line = base + core * kCoreBytes;
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
            if (closing) {
                record.event = "incomplete";
                records.push_back(record);
                slot.state.store(State::Quarantined, std::memory_order_release);
                ++incomplete_;
            }
            continue;
        }
        record.event = "end";
        record.values.insert({{"start_tick", first}, {"end_tick", last},
            {"start_ns", offset_ns_ + static_cast<int64_t>(first)},
            {"end_ns", offset_ns_ + static_cast<int64_t>(last)}, {"elapsed_ns", last - first}});
        records.push_back(std::move(record));
        ++completed_;
        // All producers published this generation. Keep old bytes; sequence identifies reuse.
        slot.state.store(State::Free, std::memory_order_release);
    }
    if (closing) records.push_back({"summary", "", "", {{"completed", completed_},
        {"incomplete", incomplete_}, {"dropped", dropped_.load()}}});
    return records;
}

KernelTrace::~KernelTrace()
{
    aclrtContext previous{};
    const bool have_previous = aclrtGetCurrentContext(&previous) == ACL_SUCCESS;
    const bool current = have_previous && previous == context_;
    const bool selected = current || aclrtSetCurrentContext(context_) == ACL_SUCCESS;
    const bool safe = !device_accessed_ || (selected && aclrtSynchronizeDevice() == ACL_SUCCESS);
    // If pending writes cannot be ruled out, leave the mapping to process teardown.
    if (safe && selected && (!mapped_ || aclrtHostUnregister(host_) == ACL_SUCCESS)) {
        if (host_) aclrtFreeHost(host_);
    } else if (host_) {
        std::fprintf(stderr, "DeepEP kernel trace retained an undrained host mapping\n");
    }
    if (!current && selected && have_previous) aclrtSetCurrentContext(previous);
}
}  // namespace ascend_deepep::trace
