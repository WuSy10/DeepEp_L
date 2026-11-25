#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ascend_deepep::trace {
struct TraceRecord;

// 正式日志按通信组和物理 PCI 设备分流，单设备共享文件锁与清理生命周期。
class TraceRecorder {
public:
    static std::shared_ptr<TraceRecorder> Open(int device);
    ~TraceRecorder();

    void Attach(const std::string& comm_id);
    void Detach(const std::string& comm_id);
    void Write(const std::vector<TraceRecord>& records);

private:
    // 测试可以使用临时目录和小容量验证同一套轮转逻辑。正式入口的容量始终固定。
    friend struct TraceRecorderTestAccess;
    static std::shared_ptr<TraceRecorder> OpenAt(const std::string& directory, uint32_t bus_id,
                                                 const std::string& device_uuid, uint64_t file_limit);

    struct Impl;
    explicit TraceRecorder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
