#include <acl/acl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <time.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
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
constexpr uint32_t kMaxRecords = 16384;

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

std::pair<int32_t, std::string> device_info()
{
    int32_t device_id = -1;
    check(aclrtGetDevice(&device_id), "get probe device");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).rfind("Ascend950", 0) != 0)
        throw std::runtime_error("Mapped clock probe requires Ascend950");
    aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
    check(aclrtHostMemMapCapabilities(static_cast<uint32_t>(device_id), ACL_RT_HAC_TYPE_AIV, &capability),
          "query probe AIV mapped Host capability");
    if (capability != ACL_RT_HOST_MEM_MAP_SUPPORTED)
        throw std::runtime_error("AIV cannot access registered Host memory on this device");
    return {device_id, soc};
}

struct MappedRecords {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    explicit MappedRecords(size_t count)
    {
        const size_t bytes = (count * kRecordBytes + kPage - 1) & ~(kPage - 1);
        check(aclrtMallocHost(&allocation, bytes + kPage - 1), "allocate pinned probe records");
        const auto address = reinterpret_cast<uintptr_t>(allocation);
        host = reinterpret_cast<uint64_t*>((address + kPage - 1) & ~(uintptr_t(kPage - 1)));
        std::memset(host, 0, bytes);
        const aclError status = aclrtHostRegister(host, bytes, ACL_HOST_REGISTER_MAPPED, &device);
        if (status != ACL_SUCCESS) {
            try { check(status, "register mapped probe records"); }
            catch (...) { aclrtFreeHost(allocation); allocation = nullptr; throw; }
        }
        if (!device || (reinterpret_cast<uintptr_t>(device) & 31U) != 0) {
            aclrtHostUnregister(host);
            aclrtFreeHost(allocation);
            allocation = nullptr;
            device = nullptr;
            throw std::runtime_error("mapped probe pointer is null or not 32-byte aligned");
        }
    }
    MappedRecords(const MappedRecords&) = delete;
    MappedRecords& operator=(const MappedRecords&) = delete;
    ~MappedRecords()
    {
        if (device) aclrtHostUnregister(host);
        if (allocation) aclrtFreeHost(allocation);
    }
};

struct Resources {
    std::unique_ptr<MappedRecords> records;
    aclrtStream stream = nullptr;
    bool completed = false;
    ~Resources()
    {
        if (stream && !completed && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            // A failed stream may still reference the mapped pages.
            records.release();
            return;
        }
        if (stream) aclrtDestroyStream(stream);
    }
};

uint64_t expected_result(uint64_t sequence, uint32_t work_iters)
{
    return sequence + 1 + work_iters;
}
}  // namespace

void bind_clock_probe(pybind11::module_& module)
{
    module.def("clock_probe_batch", [](unsigned count, unsigned warmup, unsigned work_iters) {
        const uint64_t total = uint64_t(count) + warmup;
        if (count < 1 || total > kMaxRecords || work_iters < 1 || work_iters > 4096)
            throw std::invalid_argument("count + warmup must be 1..16384; work_iters must be 1..4096");
        const auto [device_id, soc] = device_info();
        std::vector<std::array<uint64_t, 6>> rows;
        rows.reserve(count);
        {
            pybind11::gil_scoped_release release;
            Resources resources;
            resources.records = std::make_unique<MappedRecords>(static_cast<size_t>(total));
            check(aclrtCreateStream(&resources.stream), "create clock probe stream");
            std::vector<std::array<uint64_t, 2>> launch_times(static_cast<size_t>(total));
            for (uint64_t i = 0; i < total; ++i) {
                const uint64_t before = monotonic_ns();
                auto* slot = static_cast<uint8_t*>(resources.records->device) + i * kRecordBytes;
                deepep_clock_probe_do(resources.stream, slot, i, work_iters);
                const uint64_t after = monotonic_ns();
                launch_times[static_cast<size_t>(i)] = {before, after};
            }
            // One stream synchronization for the entire batch, never per operator.
            check(aclrtSynchronizeStream(resources.stream), "complete clock probe batch");
            resources.completed = true;
            for (uint64_t i = 0; i < total; ++i) {
                const uint64_t* record = resources.records->host + i * (kRecordBytes / sizeof(uint64_t));
                if (record[0] != i || record[1] == 0 || record[2] < record[1] ||
                    record[3] != expected_result(i, work_iters) || record[4] != i + 1 ||
                    record[5] != kProbeMagic || record[6] != work_iters || record[7] != ~i ||
                    launch_times[static_cast<size_t>(i)][1] < launch_times[static_cast<size_t>(i)][0])
                    throw std::runtime_error("invalid clock probe record at sequence " + std::to_string(i));
                if (i >= warmup)
                    rows.push_back({i, launch_times[static_cast<size_t>(i)][0],
                                    launch_times[static_cast<size_t>(i)][1], record[1], record[2], record[3]});
            }
        }
        return pybind11::make_tuple(device_id, soc, rows);
    }, pybind11::arg("count") = 1000, pybind11::arg("warmup") = 20,
       pybind11::arg("work_iters") = 32);
}
}  // namespace deepep
