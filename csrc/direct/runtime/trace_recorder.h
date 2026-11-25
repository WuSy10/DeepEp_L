#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ascend_deepep::trace {
struct TraceRecord;

class TraceRecorder {
public:
    static std::shared_ptr<TraceRecorder> Open(int device);
    ~TraceRecorder();

    void Attach(const std::string& comm_id);
    void Detach(const std::string& comm_id);
    void Write(const std::vector<TraceRecord>& records);

private:
    friend struct TraceRecorderTestAccess;
    static std::shared_ptr<TraceRecorder> OpenAt(const std::string& directory, uint32_t bus_id,
                                                 const std::string& device_uuid, uint64_t file_limit);

    struct Impl;
    explicit TraceRecorder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
