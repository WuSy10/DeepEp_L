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

// 将 ACL 错误转成带操作名和运行时诊断信息的异常。
void check(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(operation) + ": " + std::to_string(status);
    const char* detail = aclGetRecentErrMsg();
    if (detail && *detail) message += " (" + std::string(detail) + ")";
    throw std::runtime_error(message);
}

// 记录 Host 底层 launch 调用前后的单调时钟纳秒，不代表 NPU 执行时间。
uint64_t monotonic_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// 限定当前设备为 Ascend 950，并检查 AIV 可写映射 Host 内存。
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

// 同步参考实验的记录区：预先给预热和正式算子各分配一个独立的 64 字节槽位。
struct MappedRecords {
    void* allocation = nullptr;
    uint64_t* host = nullptr;
    void* device = nullptr;

    explicit MappedRecords(size_t count)
    {
        // 注册区按页向上取整；额外预留空间以找到 4 KB 对齐的起始地址。
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
    // 流已经完成后，取消映射并释放锁页内存。
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
    // 异常退出也先尝试等设备停止访问，再销毁流和映射记录。
    ~Resources()
    {
        if (stream && !completed && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            // 同步失败无法证明设备已停止访问，故保留映射页直到进程结束。
            records.release();
            return;
        }
        if (stream) aclrtDestroyStream(stream);
    }
};

// CPU 独立复算向量第 0 个元素的预期结果，用于检查设备记录。
uint64_t expected_result(uint64_t sequence, uint32_t work_iters)
{
    return sequence + 1 + work_iters;
}
}  // namespace

// 暴露同步参考探针：逐个提交到同一流，全部提交后仅同步一次。
void bind_clock_probe(pybind11::module_& module)
{
    module.def("clock_probe_batch", [](unsigned count, unsigned warmup, unsigned work_iters) {
        // 每次 launch 独占槽位；返回的六元组同时保留 Host 提交窗和 NPU 起止 tick。
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
                // before/after 只包围 Host 提交调用，不能当作设备执行起止时间。
                const uint64_t before = monotonic_ns();
                auto* slot = static_cast<uint8_t*>(resources.records->device) + i * kRecordBytes;
                deepep_clock_probe_do(resources.stream, slot, i, work_iters);
                const uint64_t after = monotonic_ns();
                launch_times[static_cast<size_t>(i)] = {before, after};
            }
            // 整批只同步一次流；与校准时每个样本都同步的做法不同。
            check(aclrtSynchronizeStream(resources.stream), "complete clock probe batch");
            resources.completed = true;
            for (uint64_t i = 0; i < total; ++i) {
                // 同步完成后才读取并校验所有槽位，预热记录也要检查但不返回。
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
