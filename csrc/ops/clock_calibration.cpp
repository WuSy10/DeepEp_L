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

// 统一包装 ACL 错误，并附上运行时最近的诊断信息。
void check(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(operation) + ": " + std::to_string(status);
    const char* detail = aclGetRecentErrMsg();
    if (detail && *detail) message += " (" + std::string(detail) + ")";
    throw std::runtime_error(message);
}

// 读取与 Python 换算结果同属一条时间轴的 Host CLOCK_MONOTONIC 纳秒。
uint64_t monotonic_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// 获取当前设备及 SoC，并确认 AIV 支持访问映射 Host 内存。
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

// 每批样本复用同一个 32 字节记录位置。先申请 8 KB，再从中找 4 KB 对齐的一页注册映射；
// 这是为了满足注册地址的对齐要求，不是每条采样都需要 4 KB。
struct MappedRecord {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    MappedRecord()
    {
        // aclrtMallocHost 只保证较小的对齐；多申请一页，才能容纳完整的对齐区域。
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
    // 先取消设备映射，再释放原始 Host 分配。
    ~MappedRecord()
    {
        if (device) aclrtHostUnregister(host);
        if (allocation) aclrtFreeHost(allocation);
    }
};

struct Resources {
    std::unique_ptr<MappedRecord> record;
    aclrtStream stream = nullptr;
    // 正常退出先等流完成，防止仍在运行的 kernel 访问已释放的映射页。
    ~Resources()
    {
        if (stream && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            // 流故障时设备状态不明，保留映射页直到进程结束，避免悬空访问。
            record.release();
            return;
        }
        if (stream) aclrtDestroyStream(stream);
    }
};
}  // namespace

// 将 Host 采样能力暴露给 Python；一次调用对应一批样本和一条新校准流。
void bind_clock_calibration(pybind11::module_& module)
{
    module.def("clock_monotonic_ns", &monotonic_ns);
    module.def("clock_calibration_samples", [](unsigned count, unsigned warmup) {
        // 返回 (device_id, soc, [(H0, D, H1), ...])；预热样本不进入结果。
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
                // 逐点执行：H0 → 提交设备读钟 → 同步同一条流 → H1。
                // 因为同步后才读取映射记录，设备读钟时刻必落在 [H0, H1] 内。
                const uint64_t h0 = monotonic_ns();
                deepep_capture_clock_do(resources.stream, resources.record->device, i);
                const aclError status = aclrtSynchronizeStream(resources.stream);
                if (status != ACL_SUCCESS)
                    check(status, ("complete mapped calibration sample " + std::to_string(i)).c_str());
                const uint64_t h1 = monotonic_ns();
                const uint64_t* record = resources.record->host;
                // 同一记录位置会被下一点覆盖；先检查本轮序号和字段，再保存 D。
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
