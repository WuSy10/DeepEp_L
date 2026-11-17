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

void check(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(operation) + ": " + std::to_string(status);
    const char* detail = aclGetRecentErrMsg();
    if (detail && *detail) message += " (" + std::string(detail) + ")";
    throw std::runtime_error(message);
}

uint64_t monotonic_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

struct MappedRecords {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    explicit MappedRecords(size_t count)
    {
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
    ~MappedRecords()
    {
        if (device) aclrtHostUnregister(host);
        if (allocation) aclrtFreeHost(allocation);
    }
};

class AsyncClockProbe {
public:
    using Row = std::array<uint64_t, 6>;

    AsyncClockProbe(unsigned batch_size, unsigned slot_batches, unsigned work_iters)
        : batch_size_(batch_size), work_iters_(work_iters)
    {
        if (batch_size < 1 || batch_size > 1024 || slot_batches < 1 || slot_batches > 64 ||
            uint64_t(batch_size) * slot_batches > 16384 || work_iters < 1 || work_iters > 4096)
            throw std::invalid_argument("batch_size 1..1024, slot_batches 1..64, total slots <=16384, work_iters 1..4096 required");
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
            records_ = std::make_unique<MappedRecords>(size_t(batch_size) * slot_batches);
            check(aclrtCreateStream(&stream_), "create async probe stream");
            blocks_.resize(slot_batches);
            for (auto& block : blocks_) {
                block.launch_times.resize(batch_size);
                check(aclrtCreateEventExWithFlag(&block.event, ACL_EVENT_CAPTURE_STREAM_PROGRESS),
                      "create async probe progress event");
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    AsyncClockProbe(const AsyncClockProbe&) = delete;
    AsyncClockProbe& operator=(const AsyncClockProbe&) = delete;
    ~AsyncClockProbe() { cleanup(); }

    int32_t device() const { return device_id_; }
    const std::string& soc() const { return soc_; }

    bool submit(unsigned count)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (count < 1 || count > batch_size_ || next_sequence_ + count > kMaxSequence)
            throw std::invalid_argument("async probe count or sequence is out of range");
        const uint64_t first = next_sequence_;
        next_sequence_ += count;
        attempted_ += count;
        size_t index = blocks_.size();
        for (size_t i = 0; i < blocks_.size(); ++i) {
            if (!blocks_[i].in_flight) { index = i; break; }
        }
        if (index == blocks_.size()) {
            dropped_ += count;
            return false;
        }
        auto& block = blocks_[index];
        block.in_flight = true;
        block.count = count;
        block.first_sequence = first;
        auto* host = reinterpret_cast<uint8_t*>(records_->host) + index * size_t(batch_size_) * kRecordBytes;
        std::memset(host, 0, size_t(count) * kRecordBytes);
        try {
            for (unsigned i = 0; i < count; ++i) {
                auto* slot = static_cast<uint8_t*>(records_->device) +
                             (index * size_t(batch_size_) + i) * kRecordBytes;
                const uint64_t before = monotonic_ns();
                deepep_clock_probe_do(stream_, slot, first + i, work_iters_);
                const uint64_t after = monotonic_ns();
                block.launch_times[i] = {before, after};
            }
            check(aclrtRecordEvent(block.event, stream_), "record async probe batch event");
            pending_.push_back(index);
            accepted_ += count;
            return true;
        } catch (...) {
            // A launch may already reference this mapped block. Keep it reserved.
            poisoned_ = true;
            throw;
        }
    }

    std::vector<Row> poll()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (pending_.empty()) return {};
        // The event is queried on a different Host thread after submit has
        // recorded it and published its queue entry under this mutex.
        const aclError context_status = aclrtSetCurrentContext(context_);
        if (context_status != ACL_SUCCESS) {
            poisoned_ = true;
            check(context_status, "bind async collector context");
        }
        const size_t index = pending_.front();
        auto& block = blocks_[index];
        if (block.handed_out)
            throw std::runtime_error("async probe batch must be released before polling again");
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        const aclError query_status = aclrtQueryEventStatus(block.event, &status);
        if (query_status != ACL_SUCCESS) {
            poisoned_ = true;
            check(query_status, "query async probe batch event");
        }
        if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) return {};
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            poisoned_ = true;
            throw std::runtime_error("unexpected async probe event status");
        }
        std::vector<Row> rows;
        rows.reserve(block.count);
        for (unsigned i = 0; i < block.count; ++i) {
            const uint64_t sequence = block.first_sequence + i;
            const uint64_t* record = records_->host + (index * size_t(batch_size_) + i) * 8;
            const auto& launch = block.launch_times[i];
            if (record[0] != sequence || record[1] == 0 || record[2] < record[1] ||
                record[3] != sequence + 1 + work_iters_ || record[4] != sequence + 1 ||
                record[5] != kProbeMagic || record[6] != work_iters_ || record[7] != ~sequence ||
                launch[1] < launch[0]) {
                poisoned_ = true;
                throw std::runtime_error("invalid async clock probe generation or record at sequence " +
                                         std::to_string(sequence));
            }
            rows.push_back({sequence, launch[0], launch[1], record[1], record[2], record[3]});
        }
        // The caller owns this copy, but the mapped block stays reserved until
        // it confirms that the copy has entered its CPU queue.
        block.handed_out = true;
        return rows;
    }

    void release(uint64_t first_sequence)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ensure_open();
        if (pending_.empty()) throw std::runtime_error("no async probe batch to release");
        auto& block = blocks_[pending_.front()];
        if (!block.handed_out || block.first_sequence != first_sequence) {
            poisoned_ = true;
            throw std::runtime_error("async probe release generation mismatch");
        }
        pending_.pop_front();
        block.handed_out = false;
        block.in_flight = false;
    }

    size_t pending_batches() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return pending_.size();
    }

    std::array<uint64_t, 3> counts() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return {attempted_, accepted_, dropped_};
    }

    void close()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) return;
        if (!pending_.empty()) throw std::runtime_error("async probe close requires drained batches");
        // The final synchronization protects mapped pages during normal shutdown.
        if (stream_) check(aclrtSynchronizeStream(stream_), "close async probe stream");
        cleanup_after_sync();
        closed_ = true;
    }

    void abandon() noexcept
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (closed_) return;
        // On an unknown device state, neither a timeout nor a failed query
        // proves that mapped pages and event resources can be reclaimed.
        records_.release();
        stream_ = nullptr;
        for (auto& block : blocks_) block.event = nullptr;
        closed_ = true;
    }

private:
    struct Block {
        aclrtEvent event = nullptr;
        bool in_flight = false;
        bool handed_out = false;
        unsigned count = 0;
        uint64_t first_sequence = 0;
        std::vector<std::array<uint64_t, 2>> launch_times;
    };

    void ensure_open() const
    {
        if (closed_ || poisoned_) throw std::runtime_error("async clock probe is closed or failed");
    }

    void cleanup_after_sync() noexcept
    {
        for (auto& block : blocks_) {
            if (block.event) { aclrtDestroyEvent(block.event); block.event = nullptr; }
        }
        if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        records_.reset();
    }

    void cleanup() noexcept
    {
        if (closed_) return;
        if (stream_ && aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
            // The device may still write to mapped pages. Leaking is safer than UAF.
            records_.release();
            return;
        }
        cleanup_after_sync();
        closed_ = true;
    }

    const unsigned batch_size_;
    const unsigned work_iters_;
    int32_t device_id_ = -1;
    std::string soc_;
    aclrtContext context_ = nullptr;
    std::unique_ptr<MappedRecords> records_;
    aclrtStream stream_ = nullptr;
    std::vector<Block> blocks_;
    std::deque<size_t> pending_;
    mutable std::mutex mutex_;
    uint64_t next_sequence_ = 0;
    uint64_t attempted_ = 0;
    uint64_t accepted_ = 0;
    uint64_t dropped_ = 0;
    bool poisoned_ = false;
    bool closed_ = false;
};
}  // namespace

void bind_clock_probe_async(pybind11::module_& module)
{
    namespace py = pybind11;
    py::class_<AsyncClockProbe>(module, "AsyncClockProbe")
        .def(py::init<unsigned, unsigned, unsigned>(), py::arg("batch_size"),
             py::arg("slot_batches"), py::arg("work_iters"))
        .def_property_readonly("device", &AsyncClockProbe::device)
        .def_property_readonly("soc", &AsyncClockProbe::soc)
        .def("submit", [](AsyncClockProbe& probe, unsigned count) {
            py::gil_scoped_release release;
            return probe.submit(count);
        })
        .def("poll", [](AsyncClockProbe& probe) {
            std::vector<AsyncClockProbe::Row> rows;
            { py::gil_scoped_release release; rows = probe.poll(); }
            return rows;
        })
        .def("release", &AsyncClockProbe::release, pybind11::arg("first_sequence"))
        .def("pending_batches", &AsyncClockProbe::pending_batches)
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
