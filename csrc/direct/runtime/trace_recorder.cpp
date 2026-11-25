#include "runtime/trace_recorder.h"

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>

#include "runtime/kernel_trace.h"

namespace ascend_deepep::trace {
namespace {
constexpr char kDefaultDirectory[] = "/dockerdata/.trmt/deepep/";
constexpr std::string_view kFilePrefix = "trmt-deepep-trace-";
constexpr std::string_view kFileSuffix = ".txt";
constexpr unsigned kFileCount = 8;
constexpr uint64_t kFileLimit = 1000000000;
constexpr unsigned kWriteRetries = 3;
constexpr std::chrono::milliseconds kWriteRetryDelay{3};
constexpr size_t kCommIdChars = 32;
static_assert(kFileCount <= 10);

// 所有打开的描述符由各自的对象管理，初始化中途失败也能释放目录和锁。
class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) : fd_(fd) {}
    ~FileDescriptor() { Reset(); }
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept
    {
        if (this != &other) {
            Reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }
    int Get() const { return fd_; }
    int Release() { return std::exchange(fd_, -1); }

private:
    void Reset()
    {
        if (fd_ >= 0) close(fd_);
    }
    int fd_;
};

[[noreturn]] void IoError(const std::string& action, int error = errno)
{
    throw std::system_error(error, std::generic_category(), action);
}

template <class Integer>
std::optional<Integer> ParseNumber(std::string_view text, int base = 10)
{
    if (text.empty()) return {};
    Integer value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return {};
    return value;
}

std::optional<uint32_t> PackPciBdf(std::string_view bdf)
{
    const auto colon1 = bdf.find(':');
    if (colon1 == std::string_view::npos) return {};
    const auto colon2 = bdf.find(':', colon1 + 1);
    if (colon2 == std::string_view::npos) return {};
    const auto dot = bdf.find('.', colon2 + 1);
    if (dot == std::string_view::npos) return {};
    const auto domain = ParseNumber<uint32_t>(bdf.substr(0, colon1), 16);
    const auto bus = ParseNumber<uint32_t>(bdf.substr(colon1 + 1, colon2 - colon1 - 1), 16);
    const auto slot = ParseNumber<uint32_t>(bdf.substr(colon2 + 1, dot - colon2 - 1), 16);
    const auto function = ParseNumber<uint32_t>(bdf.substr(dot + 1), 16);
    if (!domain || !bus || !slot || !function || *domain > 0xffff || *bus > 0xff || *slot > 0x1f ||
        *function > 7) return {};
    // 保留 PCI BDF 的全部位，区分同一条 PCI 总线上的设备和功能。
    return (*domain << 16) | (*bus << 8) | (*slot << 3) | *function;
}

uint32_t PciBusId(int device)
{
    // 动态解析接口，缺少新符号的 CANN 仍可加载扩展和运行未启用采集的任务。
    using Query = aclError (*)(int32_t, char*, int32_t);
    const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "aclrtDeviceGetPCIBusId"));
    char address[64]{};
    if (query && query(device, address, sizeof(address)) == ACL_SUCCESS) {
        const auto length = strnlen(address, sizeof(address));
        if (length < sizeof(address)) {
            if (const auto bus = PackPciBdf(std::string_view(address, length))) return *bus;
        }
    }

    // 若只有反查接口，用系统列出的 BDF 反查 ACL 设备号，拒绝不唯一的匹配。
    using ReverseQuery = aclError (*)(const char*, int32_t*);
    const auto reverse = reinterpret_cast<ReverseQuery>(dlsym(RTLD_DEFAULT, "aclrtDeviceGetByPCIBusId"));
    std::optional<uint32_t> bus_id;
    if (reverse) {
        std::error_code error;
        auto iter = std::filesystem::directory_iterator("/sys/bus/pci/devices", error);
        const auto end = std::filesystem::directory_iterator{};
        for (; !error && iter != end; iter.increment(error)) {
            const auto bdf = iter->path().filename().string();
            const auto candidate = PackPciBdf(bdf);
            int32_t matched = -1;
            if (!candidate || reverse(bdf.c_str(), &matched) != ACL_SUCCESS || matched != device) continue;
            if (bus_id && *bus_id != *candidate)
                throw std::runtime_error("ambiguous PCI BDF for ACL device " + std::to_string(device));
            bus_id = candidate;
        }
        if (bus_id && !error) return *bus_id;
    }
    // 不使用 ACL 逻辑编号或占位值，以免清理到其他物理设备的日志。
    throw std::runtime_error("PCI BDF unavailable for ACL device " + std::to_string(device));
}

std::string DeviceUuid(int device)
{
    // 容器中的 BDF 可能重复，设备归属用 CANN 返回的设备 UUID 区分。
    using Query = aclError (*)(int32_t, aclrtUuid*);
    const auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "aclrtDeviceGetUuid"));
    aclrtUuid uuid{};
    if (!query || query(device, &uuid) != ACL_SUCCESS)
        throw std::runtime_error("device UUID unavailable for ACL device " + std::to_string(device));
    constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    for (const unsigned char byte : uuid.bytes) {
        hex += digits[byte >> 4];
        hex += digits[byte & 0xf];
    }
    if (hex.find_first_not_of('0') == std::string::npos)
        throw std::runtime_error("empty device UUID for ACL device " + std::to_string(device));
    return hex;
}

FileDescriptor OpenDirectory(const std::string& directory)
{
    // 一次性解析用户选择的相对路径和符号链接。随后逐级打开结果，拒绝遍历中被替换成的链接。
    const auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(directory));
    constexpr int flags = O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    FileDescriptor dir(open("/", flags));
    if (dir.Get() < 0) IoError("open trace directory root");
    // O_PATH 遍历父目录只需搜索权限，不要求用户有权列出父目录内容。
    for (const auto& part : path.relative_path()) {
        if (part.empty() || part == ".") continue;
        if (part == "..") throw std::invalid_argument("invalid trace directory");
        FileDescriptor next(openat(dir.Get(), part.c_str(), flags));
        if (next.Get() < 0 && errno == ENOENT) {
            if (mkdirat(dir.Get(), part.c_str(), 0755) < 0 && errno != EEXIST)
                IoError("create trace directory " + part.string());
            next = FileDescriptor(openat(dir.Get(), part.c_str(), flags));
        }
        if (next.Get() < 0) IoError("open trace directory " + part.string());
        dir = std::move(next);
    }
    // 重新打开已定位的最终目录，清理历史文件时需要可读的目录描述符。
    FileDescriptor output(openat(dir.Get(), ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (output.Get() < 0) IoError("read trace directory");
    return output;
}

void CheckPrivateRegularFile(int fd, const std::string& name)
{
    struct stat info {};
    if (fstat(fd, &info) < 0) IoError("stat " + name);
    if (!S_ISREG(info.st_mode) || info.st_nlink != 1)
        throw std::runtime_error("expected a regular trace file with one link: " + name);
}

int Flock(int fd, int operation) noexcept
{
    int result;
    do {
        result = flock(fd, operation);
    } while (result < 0 && errno == EINTR);
    return result;
}

FileDescriptor LockDevice(int directory, const std::string& device_uuid)
{
    // 同一设备跨进程使用同一锁，虚拟 BDF 相同的不同设备可以分别采集。
    const auto name = ".trmt-deepep-trace-device-" + device_uuid + ".lock";
    // 本机 Linux 的 flock 支持只读描述符，换 UID 重跑时不必有旧锁文件的写权限。
    FileDescriptor lock(openat(directory, name.c_str(), O_RDONLY | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0644));
    if (lock.Get() < 0) IoError("open trace lock " + name);
    CheckPrivateRegularFile(lock.Get(), name);
    if (Flock(lock.Get(), LOCK_EX | LOCK_NB) < 0) IoError("lock trace recorder for device " + device_uuid);
    return lock;
}

bool ValidCommId(std::string_view uuid)
{
    if (uuid.size() != kCommIdChars) return false;
    for (const char c : uuid) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

std::string TraceFileStem(std::string_view comm_id, uint32_t bus_id)
{
    return std::string(kFilePrefix) + std::string(comm_id) + "-" + std::to_string(bus_id) + "-";
}

struct DirectoryCloser {
    void operator()(DIR* directory) const { closedir(directory); }
};

void CleanHistory(int directory, uint32_t bus_id, const std::string& owner_prefix)
{
    FileDescriptor scan(openat(directory, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (scan.Get() < 0) IoError("open trace history");
    std::unique_ptr<DIR, DirectoryCloser> entries(fdopendir(scan.Get()));
    if (!entries) IoError("read trace history");
    scan.Release();
    std::vector<std::string> candidates;
    while (true) {
        errno = 0;
        const auto* entry = readdir(entries.get());
        if (!entry) {
            if (errno) IoError("scan trace history");
            break;
        }
        const std::string_view name(entry->d_name);
        if (name.substr(0, owner_prefix.size()) == owner_prefix && ValidCommId(name.substr(owner_prefix.size())))
            candidates.emplace_back(name);
    }
    // 扫描后再删除，避免遍历同一目录时修改其内容造成漏扫。
    entries.reset();
    for (const auto& marker : candidates) {
        // 只有归属标记与设备 UUID 和 BDF 均匹配时，才删除该组的普通日志。
        struct stat info {};
        if (fstatat(directory, marker.c_str(), &info, AT_SYMLINK_NOFOLLOW) < 0) {
            if (errno == ENOENT) continue;
            IoError("stat trace ownership " + marker);
        }
        if (!S_ISREG(info.st_mode) || info.st_nlink != 1) continue;
        const auto stem = TraceFileStem(std::string_view(marker).substr(owner_prefix.size()), bus_id);
        for (unsigned index = 0; index < kFileCount; ++index) {
            const auto name = stem + std::to_string(index) + std::string(kFileSuffix);
            if (fstatat(directory, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) < 0) {
                if (errno == ENOENT) continue;
                IoError("stat old trace " + name);
            }
            if (S_ISREG(info.st_mode) && unlinkat(directory, name.c_str(), 0) < 0 && errno != ENOENT)
                IoError("remove old trace " + name);
        }
        if (unlinkat(directory, marker.c_str(), 0) < 0 && errno != ENOENT)
            IoError("remove trace ownership " + marker);
    }
}

std::string WallTime(int64_t ns)
{
    // 固定 UTC+8 和六位微秒，避免文件内容随本机 TZ 设置变化。
    constexpr int64_t kNsPerSecond = 1000000000;
    auto seconds = ns / kNsPerSecond;
    auto fraction = ns % kNsPerSecond;
    if (fraction < 0) {
        --seconds;
        fraction += kNsPerSecond;
    }
    const auto timestamp = static_cast<time_t>(seconds + 8 * 3600);
    std::tm calendar{};
    char result[32]{};
    if (!gmtime_r(&timestamp, &calendar)) throw std::runtime_error("cannot convert trace wall time");
    const auto size = std::strftime(result, sizeof(result), "%Y-%m-%d %H:%M:%S", &calendar);
    if (!size) throw std::runtime_error("cannot format trace wall time");
    std::snprintf(result + size, sizeof(result) - size, ".%06d", static_cast<int>(fraction / 1000));
    return std::string("\"") + result + '"';
}

std::string FormatRecord(const TraceRecord& record, std::string_view comm_id, uint32_t bus_id)
{
    const auto value = [&](const char* name) -> std::optional<int64_t> {
        const auto it = record.values.find(name);
        if (it == record.values.end()) return {};
        return it->second;
    };
    std::string line;
    line.reserve(512);
    line = "{\"type\":";
    const bool completed = record.event == "end";
    line += completed ? '2' : '1';
    const auto field = [&](const char* name, std::string_view json) {
        line += ",\"";
        line += name;
        line += "\":";
        line += json;
    };
    const auto number = [&](const char* name, std::optional<int64_t> n) {
        if (n) field(name, std::to_string(*n));
        else field(name, "null");
    };
    const auto time = [&](const char* name, std::optional<int64_t> ns) {
        if (ns) field(name, WallTime(*ns));
        else field(name, "null");
    };

    // 三个通知 kernel 和本地归约使用短格式，同一 op 的超时与完成记录字段一致。
    const auto op = value("op");
    const bool short_format = op == 0 || op == 1 || op == 3 || op == 6;
    number("gid", value("gid"));
    number("op", op);
    number("op_count", value("op_count"));
    number("dispatch_count", value("dispatch_count"));
    // 短格式直接省略这五个字段，长格式缺少字段值时仍写 null。
    if (!short_format) number("ep_size", value("ep_size"));
    number("bus_id", bus_id);
    const auto rank = value("rank");
    number("rank", rank);
    number("nvl_rank", rank ? std::optional<int64_t>{*rank % 16} : std::nullopt);
    if (!short_format) {
        number("hidden_size", value("hidden_size"));
        number("num_experts", value("num_experts"));
    }
    number("num_sms", value("aiv"));
    if (!short_format) {
        number("num_tokens", value("num_tokens"));
        number("topk", value("topk"));
    }
    number("pid", value("pid"));
    // 原生事件用十进制字符串携带 ACL 句柄，正式日志以完整 uint64 数字写出。
    const auto stream = record.labels.find("stream");
    const auto handle = stream == record.labels.end() ? std::nullopt : ParseNumber<uint64_t>(stream->second);
    field("stream", handle ? std::to_string(*handle) : "null");
    field("comm_id", "\"" + std::string(comm_id) + '"');
    time("launch_time", value("launch_ns"));
    // 超时记录保留 Host 提交与元数据，未确认的设备时间始终写 null。
    time("start_time", completed ? value("start_ns") : std::nullopt);
    time("end_time", completed ? value("end_ns") : std::nullopt);
    number("elapsed_ns", completed ? value("elapsed_ns") : std::nullopt);
    number("start_tick", completed ? value("start_tick") : std::nullopt);
    number("end_tick", completed ? value("end_tick") : std::nullopt);
    line += "}\n";
    return line;
}

void Truncate(int fd, off_t size)
{
    int result;
    do {
        result = ftruncate(fd, size);
    } while (result < 0 && errno == EINTR);
    if (result < 0) IoError("truncate trace file");
}

// 一个组只保持活跃文件打开，轮转覆盖下一个索引的同一普通文件。
struct LogRing {
    std::array<std::string, kFileCount> names;
    FileDescriptor active;
    unsigned index = 0;
    uint64_t size = 0;
    std::size_t references = 1;
    bool failed = false;

    void Attach(int directory, const std::string& comm_id, uint32_t bus_id, const std::string& owner_marker)
    {
        const auto stem = TraceFileStem(comm_id, bus_id);
        for (unsigned i = 0; i < kFileCount; ++i) names[i] = stem + std::to_string(i) + std::string(kFileSuffix);
        unsigned created = 0;
        try {
            while (created < kFileCount) {
                // O_EXCL 防止同名文件或符号链接被意外覆盖。
                const auto& name = names[created];
                FileDescriptor file(openat(directory, name.c_str(),
                                           O_WRONLY | O_APPEND | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644));
                if (file.Get() < 0) IoError("create trace file " + name);
                if (created++ == 0) active = std::move(file);
            }
            // 文件全部创建成功后才登记归属，避免清理到原本已存在的同名文件。
            FileDescriptor marker(openat(directory, owner_marker.c_str(),
                                         O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644));
            if (marker.Get() < 0) IoError("create trace ownership " + owner_marker);
        } catch (...) {
            while (created != 0) unlinkat(directory, names[--created].c_str(), 0);
            throw;
        }
    }

    void Append(int directory, const std::string& line, uint64_t file_limit)
    {
        if (line.size() > file_limit) throw std::runtime_error("single trace record exceeds file limit");
        if (line.size() > file_limit - size) {
            const auto next = (index + 1) % kFileCount;
            FileDescriptor file(openat(directory, names[next].c_str(),
                                       O_WRONLY | O_APPEND | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
            if (file.Get() < 0) IoError("open trace file " + names[next]);
            CheckPrivateRegularFile(file.Get(), names[next]);
            Truncate(file.Get(), 0);
            active = std::move(file);
            index = next;
            size = 0;
        }
        // 同一行共用重试次数，短写后从剩余字节继续。EINTR 立即再试，不消耗次数。
        size_t written = 0;
        unsigned retries = 0;
        while (written < line.size()) {
            const auto n = write(active.Get(), line.data() + written, line.size() - written);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                const auto error = n < 0 ? errno : EIO;
                if (retries < kWriteRetries) {
                    ++retries;
                    std::this_thread::sleep_for(kWriteRetryDelay);
                    continue;
                }
                // 重试耗尽时清掉半行，保留此前写完的 JSONL 记录。
                if (written) Truncate(active.Get(), static_cast<off_t>(size));
                IoError("write trace file " + names[index], error);
            }
            written += static_cast<size_t>(n);
        }
        size += written;
    }
};
}

struct TraceRecorder::Impl {
    FileDescriptor directory, lock;
    uint32_t bus_id;
    std::string owner_prefix;
    uint64_t file_limit;
    pid_t owner;
    std::mutex mutex;
    std::map<std::string, LogRing> groups;

    Impl(FileDescriptor dir, FileDescriptor device_lock, uint32_t bus, std::string prefix, uint64_t limit)
        : directory(std::move(dir)), lock(std::move(device_lock)), bus_id(bus), owner_prefix(std::move(prefix)),
          file_limit(limit), owner(getpid()) {}
};

TraceRecorder::TraceRecorder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TraceRecorder::~TraceRecorder()
{
    // 先关闭本进程的日志。仅拥有者显式解锁，避免 fork 子进程继承的描述符延长持锁时间。
    if (impl_->owner == getpid()) {
        impl_->groups.clear();
        (void)Flock(impl_->lock.Get(), LOCK_UN);
    }
}

std::shared_ptr<TraceRecorder> TraceRecorder::OpenAt(const std::string& directory, uint32_t bus_id,
                                                     const std::string& device_uuid, uint64_t file_limit)
{
    if (!file_limit || file_limit > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::invalid_argument("invalid trace file size limit");
    if (!ValidCommId(device_uuid)) throw std::invalid_argument("invalid trace device UUID");
    auto dir = OpenDirectory(directory);
    auto lock = LockDevice(dir.Get(), device_uuid);
    auto prefix = ".trmt-deepep-trace-owner-" + device_uuid + "-" + std::to_string(bus_id) + "-";
    CleanHistory(dir.Get(), bus_id, prefix);
    return std::shared_ptr<TraceRecorder>(
        new TraceRecorder(std::make_unique<Impl>(std::move(dir), std::move(lock), bus_id, std::move(prefix), file_limit)));
}

std::shared_ptr<TraceRecorder> TraceRecorder::Open(int device)
{
    static std::mutex mutex;
    static std::map<int, std::weak_ptr<TraceRecorder>> devices;
    static std::map<std::pair<uint32_t, std::string>, std::weak_ptr<TraceRecorder>> writers;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto old = devices[device].lock()) {
        if (old->impl_->owner != getpid()) throw std::runtime_error("cannot use inherited trace recorder");
        return old;
    }
    // 活跃的 ACL context 共享设备锁。最后一个使用者释放后，新采集器才会重新清理。
    const auto bus_id = PciBusId(device);
    const auto device_uuid = DeviceUuid(device);
    auto& cached = writers[{bus_id, device_uuid}];
    auto recorder = cached.lock();
    if (!recorder) {
        // 首次初始化时读取输出目录；未设置或为空时使用默认路径。
        // 设备锁、历史清理和轮转文件都位于选定目录中。
        const char* directory = std::getenv("DEEPEP_TRACE_DIR");
        recorder = OpenAt(directory && *directory ? directory : kDefaultDirectory, bus_id, device_uuid, kFileLimit);
        cached = recorder;
    }
    if (recorder->impl_->owner != getpid()) throw std::runtime_error("cannot use inherited trace recorder");
    devices[device] = recorder;
    return recorder;
}

void TraceRecorder::Attach(const std::string& comm_id)
{
    if (impl_->owner != getpid()) throw std::runtime_error("cannot use inherited trace recorder");
    if (!ValidCommId(comm_id)) throw std::invalid_argument("expected a lowercase hexadecimal trace communication UUID");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto [it, inserted] = impl_->groups.try_emplace(comm_id);
    if (!inserted) {
        // 同一 UUID 的多个注册共用这一组文件，最后一个注销时才关闭活跃文件。
        ++it->second.references;
        return;
    }
    try {
        it->second.Attach(impl_->directory.Get(), comm_id, impl_->bus_id, impl_->owner_prefix + comm_id);
    } catch (...) {
        impl_->groups.erase(it);
        throw;
    }
}

void TraceRecorder::Detach(const std::string& comm_id)
{
    if (impl_->owner != getpid()) throw std::runtime_error("cannot use inherited trace recorder");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->groups.find(comm_id);
    if (it == impl_->groups.end()) return;
    // 调用方先收集本组最后一批记录。注销只关闭文件描述符，日志由下次启动清理。
    if (--it->second.references == 0) impl_->groups.erase(it);
}

void TraceRecorder::Write(const std::vector<TraceRecord>& records)
{
    if (impl_->owner != getpid()) throw std::runtime_error("cannot use inherited trace recorder");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::exception_ptr failure;
    for (const auto& record : records) {
        if (record.event != "end" && record.event != "timeout") continue;
        const auto comm_id = record.labels.find("comm_id");
        if (comm_id == record.labels.end()) continue;
        const auto group = impl_->groups.find(comm_id->second);
        if (group == impl_->groups.end() || group->second.failed) continue;
        try {
            group->second.Append(impl_->directory.Get(), FormatRecord(record, comm_id->second, impl_->bus_id),
                                 impl_->file_limit);
        } catch (...) {
            // 批次里其他通信组仍可写出，已失败的组停止写入并向采集线程报告一次。
            group->second.failed = true;
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
}
}
