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

extern "C" void deepep_capture_clock_do(void* stream, void* mapped_record, uint64_t sequence);

namespace deepep {
namespace {
constexpr size_t kPage = 4096;
constexpr uint64_t kRecordMagic = 0x434c4f434b4d4150ULL;

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
    check(aclrtGetDevice(&device_id), "get device");
    const char* soc = aclrtGetSocName();
    if (!soc || std::string(soc).rfind("Ascend950", 0) != 0)
        throw std::runtime_error("Mapped clock calibration requires Ascend950");
    aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
    check(aclrtHostMemMapCapabilities(static_cast<uint32_t>(device_id), ACL_RT_HAC_TYPE_AIV, &capability),
          "query AIV mapped Host capability");
    if (capability != ACL_RT_HOST_MEM_MAP_SUPPORTED)
        throw std::runtime_error("AIV cannot access registered Host memory on this device");
    return {device_id, soc};
}

// aclrtMallocHost guarantees 64-byte alignment. Register an interior 4-KB page.
struct MappedRecord {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    MappedRecord()
    {
        check(aclrtMallocHost(&allocation, 2 * kPage), "allocate pinned Host record");
        const auto address = reinterpret_cast<uintptr_t>(allocation);
        host = reinterpret_cast<uint64_t*>((address + kPage - 1) & ~(uintptr_t(kPage - 1)));
        std::memset(host, 0, kPage);
        const aclError status = aclrtHostRegister(host, kPage, ACL_HOST_REGISTER_MAPPED, &device);
        if (status != ACL_SUCCESS) {
            try { check(status, "register mapped Host record"); }
            catch (...) { aclrtFreeHost(allocation); allocation = nullptr; throw; }
        }
        if (!device || (reinterpret_cast<uintptr_t>(device) & 31U) != 0) {
            aclrtHostUnregister(host);
            aclrtFreeHost(allocation);
            allocation = nullptr;
            device = nullptr;
            throw std::runtime_error("mapped device pointer is null or not 32-byte aligned");
        }
    }
    MappedRecord(const MappedRecord&) = delete;
    MappedRecord& operator=(const MappedRecord&) = delete;
    ~MappedRecord()
    {
        if (device) aclrtHostUnregister(host);
        if (allocation) aclrtFreeHost(allocation);
    }
};

struct Resources {
    std::unique_ptr<MappedRecord> record;
    aclrtStream stream = nullptr;
    ~Resources()
    {
        if (stream && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            // A faulted stream may still reference the registered Host page.
            record.release();
            return;
        }
        if (stream) aclrtDestroyStream(stream);
    }
};
}  // namespace

void bind_clock_calibration(pybind11::module_& module)
{
    module.def("clock_monotonic_ns", &monotonic_ns);
    module.def("clock_calibration_samples", [](unsigned count, unsigned warmup) {
        if (count < 1 || count > 4096 || warmup > 4096)
            throw std::invalid_argument("count must be 1..4096; warmup 0..4096");
        const auto [device_id, soc] = device_info();
        std::vector<std::array<uint64_t, 3>> samples;
        samples.reserve(count);
        {
            pybind11::gil_scoped_release release;
            Resources resources;
            resources.record = std::make_unique<MappedRecord>();
            check(aclrtCreateStream(&resources.stream), "create calibration stream");
            for (unsigned i = 0; i < warmup + count; ++i) {
                const uint64_t h0 = monotonic_ns();
                deepep_capture_clock_do(resources.stream, resources.record->device, i);
                const aclError status = aclrtSynchronizeStream(resources.stream);
                if (status != ACL_SUCCESS)
                    check(status, ("complete mapped calibration sample " + std::to_string(i)).c_str());
                const uint64_t h1 = monotonic_ns();
                const uint64_t* record = resources.record->host;
                if (record[0] != i || record[1] == 0 || record[2] != uint64_t(i) + 1 ||
                    record[3] != kRecordMagic || h1 < h0)
                    throw std::runtime_error("invalid mapped calibration record at sample " + std::to_string(i));
                if (i >= warmup) samples.push_back({h0, record[1], h1});
            }
        }
        return pybind11::make_tuple(device_id, soc, samples);
    }, pybind11::arg("count") = 32, pybind11::arg("warmup") = 5);
}
}  // namespace deepep
