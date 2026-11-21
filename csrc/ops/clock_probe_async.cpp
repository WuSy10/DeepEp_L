#include <acl/acl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <time.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" void deepep_clock_probe_do(void* stream, void* mapped_record, uint64_t sequence, uint32_t work_iters);

namespace deepep {
namespace {
constexpr size_t kPage = 4096;
constexpr size_t kRecordBytes = 64;
constexpr uint64_t kProbeMagic = 0x434c4f434b50524fULL;
constexpr uint64_t kMaxSequence = 1000000000ULL;

// 统一把 ACL 返回码和最近的运行时错误转成可定位的异常。
void check(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(operation) + ": " + std::to_string(status);
    const char* detail = aclGetRecentErrMsg();
    if (detail && *detail) message += " (" + std::string(detail) + ")";
    throw std::runtime_error(message);
}

// Host 提交探针时使用的 CLOCK_MONOTONIC，与校准映射的目标时间轴相同。
uint64_t monotonic_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// 异步探针的固定容量日志区：count 个 64 字节槽位，共享一块页对齐映射区。
struct MappedRecords {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    explicit MappedRecords(size_t count)
    {
        // 注册长度按 4 KB 向上取整；多申请不足一页的余量以找对齐起点。
        const size_t bytes = (count * kRecordBytes + kPage - 1) & ~(kPage - 1);
        check(aclrtMallocHost(&allocation, bytes + kPage - 1), "allocate async probe records");
        const auto address = reinterpret_cast<uintptr_t>(allocation);
        host = reinterpret_cast<uint64_t*>((address + kPage - 1) & ~(uintptr_t(kPage - 1)));
        std::memset(host, 0, bytes);
        const aclError status = aclrtHostRegister(host, bytes, ACL_HOST_REGISTER_MAPPED, &device);
        if (status != ACL_SUCCESS) {
            try { check(status, "register async probe records"); }
            catch (...) { aclrtFreeHost(allocation); allocation = nullptr; throw; }
        }
        if (!device || (reinterpret_cast<uintptr_t>(device) & 31U) != 0) {
            aclrtHostUnregister(host);
            aclrtFreeHost(allocation);
            allocation = nullptr;
            device = nullptr;
            throw std::runtime_error("async mapped probe pointer is null or not 32-byte aligned");
        }
    }
    // 只有确认设备不再使用这些槽位后才应正常销毁本对象。
    ~MappedRecords()
    {
        if (device) aclrtHostUnregister(host);
        if (allocation) aclrtFreeHost(allocation);
    }
};

// 单卡异步探针：提交线程、Event 查询线程通过互斥锁共同管理槽位所有权。
class AsyncClockProbe {
public:
    using Row = std::array<uint64_t, 6>;

    AsyncClockProbe(unsigned slot_count, unsigned work_iters)
        : work_iters_(work_iters)
    {
        // 异步进度 Event 实验限定 Ascend 950PR；每个槽位配一个可复用 Event。
        if (slot_count < 1 || slot_count > 16384 || work_iters < 1 || work_iters > 4096)
            throw std::invalid_argument("slot_count 1..16384 and work_iters 1..4096 required");
        int32_t device_id = -1;
        check(aclrtGetDevice(&device_id), "get async probe device");
        const char* soc = aclrtGetSocName();
        if (!soc || std::string(soc).rfind("Ascend950PR", 0) != 0)
            throw std::runtime_error("async progress-event probe requires Ascend950PR");
        device_id_ = device_id;
        soc_ = soc;
        aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
        check(aclrtHostMemMapCapabilities(static_cast<uint32_t>(device_id), ACL_RT_HAC_TYPE_AIV, &capability),
              "query async probe mapped Host capability");
        if (capability != ACL_RT_HOST_MEM_MAP_SUPPORTED)
            throw std::runtime_error("AIV cannot access registered Host memory on this device");
        check(aclrtGetCurrentContext(&context_), "get async probe context");
        if (!context_) throw std::runtime_error("async probe has no current ACL context");
        try {
            records_ = std::make_unique<MappedRecords>(slot_count);
            check(aclrtCreateStream(&stream_), "create async probe stream");
            slots_.resize(slot_count);
            for (auto& slot : slots_) {
                check(aclrtCreateEventExWithFlag(&slot.event, ACL_EVENT_CAPTURE_STREAM_PROGRESS),
                      "create async probe progress event");
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    AsyncClockProbe(const AsyncClockProbe&) = delete;
    AsyncClockProbe& operator=(const AsyncClockProbe&) = delete;
    // 析构时尝试完成流；若同步失败，cleanup 会保留设备可能仍访问的内存。
    ~AsyncClockProbe() { cleanup(); }

    // 两个只读属性供 Python 确认探针与校准发生在同一设备和 SoC。
    int32_t device() const { return device_id_; }
    const std::string& soc() const { return soc_; }

    // 提交一个算子。返回 true 表示保留日志槽位，false 只表示该次日志被丢弃。
    bool submit()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (next_sequence_ >= kMaxSequence)
            throw std::invalid_argument("async probe sequence is out of range");
        const uint64_t sequence = next_sequence_++;
        ++attempted_;
        size_t index = slots_.size();
        for (size_t offset = 0; offset < slots_.size(); ++offset) {
            // 从上次分配位置继续轮转，只选已归还、不会被设备覆盖的槽位。
            const size_t candidate = (next_slot_ + offset) % slots_.size();
            if (!slots_[candidate].in_flight) { index = candidate; break; }
        }
        if (index == slots_.size()) {
            // 槽位满仍提交同样的向量工作；空地址使 kernel 跳过读钟和 MTE3 写日志。
            try {
                deepep_clock_probe_do(stream_, nullptr, sequence, work_iters_);
            } catch (...) {
                poisoned_ = true;
                throw;
            }
            ++dropped_;
            return false;
        }
        next_slot_ = (index + 1) % slots_.size();
        auto& slot = slots_[index];
        slot.in_flight = true;
        slot.sequence = sequence;
        auto* host = reinterpret_cast<uint8_t*>(records_->host) + index * kRecordBytes;
        std::memset(host, 0, kRecordBytes);
        try {
            auto* mapped_record = static_cast<uint8_t*>(records_->device) + index * kRecordBytes;
            const uint64_t before = monotonic_ns();
            deepep_clock_probe_do(stream_, mapped_record, sequence, work_iters_);
            const uint64_t after = monotonic_ns();
            slot.launch_times = {before, after};
            // Event 排在同一条流的 kernel 后；先记录 Event，再公布待查询槽位。
            check(aclrtRecordEvent(slot.event, stream_), "record async probe slot event");
            pending_.push_back(index);
            ++accepted_;
            return true;
        } catch (...) {
            // launch 可能已经引用该槽位，异常时不可贸然释放给下一算子复用。
            poisoned_ = true;
            throw;
        }
    }

    // 查询队首 Event；完成后验证映射记录并复制为 CPU 自有的六元组。
    std::vector<Row> poll()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (pending_.empty()) return {};
        // 查询发生在另一 Host 线程；锁保证只能看到 submit 已记录并入队的 Event。
        const aclError context_status = aclrtSetCurrentContext(context_);
        if (context_status != ACL_SUCCESS) {
            poisoned_ = true;
            check(context_status, "bind async collector context");
        }
        const size_t index = pending_.front();
        auto& slot = slots_[index];
        if (slot.handed_out)
            throw std::runtime_error("async probe slot must be released before polling again");
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        const aclError query_status = aclrtQueryEventStatus(slot.event, &status);
        if (query_status != ACL_SUCCESS) {
            poisoned_ = true;
            check(query_status, "query async probe slot event");
        }
        if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) return {};
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            poisoned_ = true;
            throw std::runtime_error("unexpected async probe event status");
        }
        std::vector<Row> rows;
        const uint64_t sequence = slot.sequence;
        const uint64_t* record = records_->host + index * 8;
        const auto& launch = slot.launch_times;
        if (record[0] != sequence || record[1] == 0 || record[2] < record[1] ||
            record[3] != sequence + 1 + work_iters_ || record[4] != sequence + 1 ||
            record[5] != kProbeMagic || record[6] != work_iters_ || record[7] != ~sequence ||
            launch[1] < launch[0]) {
            poisoned_ = true;
            throw std::runtime_error("invalid async clock probe generation or record at sequence " +
                                     std::to_string(sequence));
        }
        rows.push_back({sequence, launch[0], launch[1], record[1], record[2], record[3]});
        // 虽已复制到 CPU，仍等 Python 将副本成功入队后才允许归还原槽位。
        slot.handed_out = true;
        return rows;
    }

    // Python 已持有 CPU 副本后，按序号归还队首映射槽位。
    void release(uint64_t sequence)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (pending_.empty()) throw std::runtime_error("no async probe slot to release");
        auto& slot = slots_[pending_.front()];
        if (!slot.handed_out || slot.sequence != sequence) {
            poisoned_ = true;
            throw std::runtime_error("async probe release generation mismatch");
        }
        pending_.pop_front();
        slot.handed_out = false;
        slot.in_flight = false;
    }

    // 返回仍等待完成或归还的槽位数，供收集线程判断是否排空。
    size_t pending_records() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return pending_.size();
    }

    // 返回 attempted/accepted/dropped；dropped 统计缺日志的算子，不是未执行算子。
    std::array<uint64_t, 3> counts() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return {attempted_, accepted_, dropped_};
    }

    // 正常收尾：必须先排空记录，再同步整条流以覆盖无日志算子，最后释放资源。
    void close()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) return;
        if (!pending_.empty()) throw std::runtime_error("async probe close requires drained records");
        // 最终流同步保护映射页，也等待没有 Event 的无日志 kernel 完成。
        if (stream_) check(aclrtSynchronizeStream(stream_), "close async probe stream");
        cleanup_after_sync();
        closed_ = true;
    }

    // 异常且设备状态不明时放弃回收映射资源，避免设备写入已释放内存。
    void abandon() noexcept
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) return;
        // 超时或查询失败不能证明设备已停止访问，故资源留到进程结束。
        records_.release();
        stream_ = nullptr;
        for (auto& slot : slots_) slot.event = nullptr;
        closed_ = true;
    }

private:
    struct Slot {
        aclrtEvent event = nullptr;
        bool in_flight = false;
        bool handed_out = false;
        uint64_t sequence = 0;
        std::array<uint64_t, 2> launch_times{};
    };

    // 拒绝在关闭或出错后的探针上继续提交、查询或释放槽位。
    void ensure_open() const
    {
        if (closed_ || poisoned_) throw std::runtime_error("async clock probe is closed or failed");
    }

    // 只在确认流已同步后销毁 Event、流和映射 Host 记录。
    void cleanup_after_sync() noexcept
    {
        for (auto& slot : slots_) {
            if (slot.event) { aclrtDestroyEvent(slot.event); slot.event = nullptr; }
        }
        if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        records_.reset();
    }

    // 析构兜底：尽量同步后释放；同步失败时保留仍可能被设备引用的映射页。
    void cleanup() noexcept
    {
        if (closed_) return;
        if (stream_ && aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
            // 设备可能还在写映射页，宁可保留内存也不能悬空释放。
            records_.release();
            return;
        }
        cleanup_after_sync();
        closed_ = true;
    }

    const unsigned work_iters_;
    int32_t device_id_ = -1;
    std::string soc_;
    aclrtContext context_ = nullptr;
    std::unique_ptr<MappedRecords> records_;
    aclrtStream stream_ = nullptr;
    std::vector<Slot> slots_;
    std::deque<size_t> pending_;
    mutable std::mutex mutex_;
    size_t next_slot_ = 0;
    uint64_t next_sequence_ = 0;
    uint64_t attempted_ = 0;
    uint64_t accepted_ = 0;
    uint64_t dropped_ = 0;
    bool poisoned_ = false;
    bool closed_ = false;
};
}  // namespace

// 把 C++ 生命周期接口暴露给 Python 的提交、收集和写入流水线。
void bind_clock_probe_async(pybind11::module_& module)
{
    namespace py = pybind11;
    py::class_<AsyncClockProbe>(module, "AsyncClockProbe")
        .def(py::init<unsigned, unsigned>(), py::arg("slot_count"), py::arg("work_iters"))
        .def_property_readonly("device", &AsyncClockProbe::device)
        .def_property_readonly("soc", &AsyncClockProbe::soc)
        .def("submit", [](AsyncClockProbe& probe) {
            py::gil_scoped_release release;
            return probe.submit();
        })
        .def("poll", [](AsyncClockProbe& probe) {
            std::vector<AsyncClockProbe::Row> rows;
            { py::gil_scoped_release release; rows = probe.poll(); }
            return rows;
        })
        .def("release", &AsyncClockProbe::release, pybind11::arg("sequence"))
        .def("pending_records", &AsyncClockProbe::pending_records)
        .def("counts", &AsyncClockProbe::counts)
        .def("close", [](AsyncClockProbe& probe) {
            py::gil_scoped_release release;
            probe.close();
        })
        .def("abandon", [](AsyncClockProbe& probe) {
            py::gil_scoped_release release;
            probe.abandon();
        });
}
}  // namespace deepep
