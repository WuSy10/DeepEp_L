"""验证生产采集入口、历史清理及磁盘轮转，不依赖 NPU。"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
GROUP_A = "1" * 32
GROUP_B = "2" * 32
OLD_GROUP = "3" * 32
BUS = 8413184


def _load_monitor():
    spec = importlib.util.spec_from_file_location("trace_recorder_test_monitor", ROOT / "deep_ep/utils/kernel_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_monitor_enablement_and_shared_source(monkeypatch):
    monitor = _load_monitor().KernelTrace
    polled = threading.Event()
    calls = []
    waits = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(monitor, "_stop", RecordingEvent())

    class Source:
        identity = 17
        poll_interval_ms = 7
        synchronized = False

        def __init__(self):
            self.pending = []

        def calibration(self):
            return {"device": 0}

        def synchronize(self):
            self.synchronized = True

    class Recorder:
        def __init__(self):
            self.groups = set()

        def attach(self, comm_id):
            self.groups.add(comm_id)
            calls.append(("attach", comm_id))

        def detach(self, comm_id):
            self.groups.remove(comm_id)
            calls.append(("detach", comm_id))

        def drain(self, source, closing):
            assert source is active_source
            for comm_id in source.pending:
                assert comm_id in self.groups
                calls.append(("record", comm_id))
            source.pending.clear()
            if closing:
                assert source.synchronized
                calls.append(("close", None))
            else:
                polled.set()

    active_source = Source()
    recorder = Recorder()

    class Native:
        @staticmethod
        def get_kernel_trace():
            calls.append(("get", None))
            return active_source

        @staticmethod
        def open_trace_recorder(source):
            assert source is active_source
            calls.append(("open", None))
            return recorder

    try:
        # 单独配置目录或使用其他真值字符串都不能打开生产采集。
        monkeypatch.setenv("DEEPEP_TRACE_DIR", "custom-dir")
        monkeypatch.delenv("DEEPEP_TRACE_RECORDER_ENABLE", raising=False)
        assert monitor.attach(Native, comm_id=GROUP_A) is None
        monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "0")
        assert monitor.attach(Native, comm_id=GROUP_A) is None
        monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "true")
        assert monitor.attach(Native, comm_id=GROUP_A) is None
        assert calls == []
        monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")

        first = monitor.attach(Native, comm_id=GROUP_A)
        second = monitor.attach(Native, comm_id=GROUP_B)
        assert first is not None
        assert calls.count(("open", None)) == 1
        # 共用一个原生采集源的两个通信组仍要分别注册文件组。
        assert [row for row in calls if row[0] == "attach"] == [("attach", GROUP_A), ("attach", GROUP_B)]
        assert polled.wait(2)
        # 验证后台线程真的按原生配置等待，而不是继续用写死的 2 ms。
        assert waits[0] == pytest.approx(0.007)
        # 模拟调用方先同步完设备；退出 A 时先写出剩余记录，B 不受影响。
        with monitor._lock:
            active_source.pending.append(GROUP_A)
            monitor.detach(first, GROUP_A, synchronized=True)
        assert not active_source.synchronized
        assert calls[-2:] == [("record", GROUP_A), ("detach", GROUP_A)]
        assert recorder.groups == {GROUP_B}
        with monitor._lock:
            active_source.pending.append(GROUP_B)
            monitor.detach(second, GROUP_B)
        assert calls[-3:] == [("record", GROUP_B), ("close", None), ("detach", GROUP_B)]
        assert calls.count(("close", None)) == 1
    finally:
        monitor.shutdown()


def test_idle_poller_does_not_hold_last_writer(monkeypatch):
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    monitor = _load_monitor().KernelTrace
    polled = threading.Event()
    writer_refs = []
    source = SimpleNamespace(
        identity=1, poll_interval_ms=2, calibration=lambda: {"device": 0}, synchronize=lambda: None
    )

    class Recorder:
        def attach(self, comm_id):
            pass

        def detach(self, comm_id):
            pass

        def drain(self, source, closing):
            if not closing:
                polled.set()

    def open_recorder(source):
        writer = Recorder()
        writer_refs.append(weakref.ref(writer))
        return writer

    native = SimpleNamespace(get_kernel_trace=lambda: source, open_trace_recorder=open_recorder)
    try:
        identity = monitor.attach(native, GROUP_A)
        assert identity is not None
        assert polled.wait(2)
        monitor.detach(identity)
        # 等待状态不能续持上次扫描的 writer，否则同一设备重跑时不会重新清理。
        assert monitor._thread.is_alive()
        assert writer_refs[0]() is None
    finally:
        monitor.shutdown()


def test_shutdown_drains_once_and_balances_each_group_reference(monkeypatch):
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    monitor = _load_monitor().KernelTrace
    calls = []
    source = SimpleNamespace(
        identity=1,
        poll_interval_ms=1000,
        calibration=lambda: {"device": 0},
        synchronize=lambda: calls.append(("synchronize", None)),
    )

    def drain(source, closing):
        if closing:
            calls.append(("close", None))

    writer = SimpleNamespace(
        attach=lambda group: calls.append(("attach", group)),
        detach=lambda group: calls.append(("detach", group)),
        drain=drain,
    )
    native = SimpleNamespace(get_kernel_trace=lambda: source, open_trace_recorder=lambda _: writer)
    try:
        for group in (GROUP_A, GROUP_B, GROUP_A):
            assert monitor.attach(native, group) == source.identity
    finally:
        monitor.shutdown()
    # 一个采集源只关闭一次，但共享同组的多个 Buffer 仍要逐个归还引用。
    assert calls[3:5] == [("synchronize", None), ("close", None)]
    assert calls[5:].count(("detach", GROUP_A)) == 2
    assert calls[5:].count(("detach", GROUP_B)) == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_production_comm_id_is_shared_from_rank_zero(monkeypatch, rank):
    trace = _load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    monkeypatch.setattr(trace.uuid, "uuid4", lambda: SimpleNamespace(hex=GROUP_A))
    monkeypatch.setattr(trace, "_global_rank", lambda: 10 + rank)
    attached = []
    native = SimpleNamespace(
        trace_recorder_version=3,
        kernel_trace_metadata_version=2,
        make_trace_metadata=lambda fields, *, recorder: fields,
    )

    def all_gather(group, tensor, ep_size, proposal):
        assert ep_size == 2 and proposal == (True, GROUP_A if rank == 0 else None, "", False)
        return [(True, GROUP_A, "", False), (True, None, "", False)]

    def attach(native_module, comm_id):
        assert native_module is native
        attached.append(comm_id)
        return 23

    # rank 0 的 UUID 必须同时用于 writer 文件组和快照中的通信组字段。
    monkeypatch.setattr(trace.KernelTrace, "attach", staticmethod(attach))
    identity, session = trace.attach_trace_session(native, None, rank, 2, all_gather)
    assert identity == 23 and attached == [GROUP_A]
    assert session.fields["comm_id"] == GROUP_A


def test_production_rejects_old_recorder_extension(monkeypatch):
    trace = _load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    native = SimpleNamespace(trace_recorder_version=2, kernel_trace_metadata_version=2)

    # 旧扩展缺少注销接口，创建采集会话时就应提示重新构建。
    with pytest.raises(RuntimeError, match=r"trace recorder requires .* \(version 3\)"):
        trace.attach_trace_session(native, None, 0, 1, lambda group, tensor, size, proposal: [proposal])


def test_dispatch_count_follows_python_call_and_origin_handle():
    trace = _load_monitor()
    passed = []

    class Native:
        kernel_trace_metadata_version = 2

        @staticmethod
        def make_trace_metadata(fields, parent=None, *, recorder=False):
            return fields

        @staticmethod
        def push_trace_metadata(template, api, handle, count):
            passed.append(count)

        @staticmethod
        def restore_trace_metadata(previous):
            pass

    session = trace.KernelTraceSession(Native, GROUP_A, 4, recorder=True)
    x = SimpleNamespace(shape=(8, 6144), dtype="bf16", element_size=lambda: 2)
    routes = SimpleNamespace(shape=(8, 4))
    handle = SimpleNamespace()
    with session.call("dispatch", x, routes, 256, handle) as meta:
        assert meta["dispatch_count"] == 0
        # 模拟 fresh dispatch 把本次调用编号交给返回的 handle。
        handle._trace_dispatch_count = meta["dispatch_count"]
    with session.call("dispatch", x, routes, 256, handle, cached=True) as meta:
        assert meta["dispatch_count"] == 1
    with session.call("combine", x, routes, 256, handle) as meta:
        assert meta["dispatch_count"] == 0
    with session.call("combine", x, routes, 256, SimpleNamespace()) as meta:
        assert "dispatch_count" not in meta
    with session.call("dispatch", x, routes, 256) as meta:
        assert meta["dispatch_count"] == 2
    assert passed == [0, 1, 0, -1, 2]


DEVICE_UUID = "22" * 16
FOREIGN_DEVICE_UUID = "33" * 16


@pytest.fixture(scope="module")
def recorder_executable(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not sys.platform.startswith("linux") or compiler is None:
        pytest.skip("生产文件轮转测试需要 Linux 和 C++17 编译器")
    build = tmp_path_factory.mktemp("trace_recorder")
    (build / "acl").mkdir()
    (build / "acl/acl.h").write_text(
        "#pragma once\n#include <cstdint>\nusing aclrtContext = void*;\n"
        "using aclError = int;\ninline constexpr aclError ACL_SUCCESS = 0;\n"
        "struct aclrtUuid { char bytes[16]; };\n",
        encoding="utf-8",
    )
    source = build / "test_recorder.cpp"
    source.write_text(
        r'''
#include "runtime/kernel_trace.h"
#include "runtime/trace_recorder.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>

namespace ascend_deepep::trace {
struct TraceRecorderTestAccess {
    static std::shared_ptr<TraceRecorder> Open(const std::string& directory, uint64_t limit) {
        // 使用真实写入器，只给测试提供临时目录和小容量。
        return TraceRecorder::OpenAt(directory, 8413184, std::string(32, '2'), limit);
    }
};
}

using namespace ascend_deepep::trace;

namespace {
enum class WriteAction { interrupt, error, zero, partial };
using Clock = std::chrono::steady_clock;
struct WriteAttempt {
    Clock::time_point time;
    size_t bytes;
};
struct WriteFault {
    std::vector<WriteAction> actions;
    std::vector<WriteAttempt> attempts;
    int fd = -1;
} fault;
bool readonly_existing_lock = false;
bool saw_device_lock = false;

void ExpectWrites(size_t count, std::initializer_list<size_t> delayed_after)
{
    if (fault.attempts.size() != count) throw std::runtime_error("unexpected number of write attempts");
    for (const auto index : delayed_after) {
        if (fault.attempts.at(index + 1).time - fault.attempts.at(index).time < std::chrono::milliseconds(3))
            throw std::runtime_error("retry did not wait 3 ms");
    }
}
}

extern "C" ssize_t __real_write(int fd, const void* data, size_t bytes);
extern "C" int __real_openat(int directory, const char* name, int flags, ...);

extern "C" int __wrap_openat(int directory, const char* name, int flags, ...)
{
    constexpr char lock_prefix[] = ".trmt-deepep-trace-device-";
    if (readonly_existing_lock && std::strncmp(name, lock_prefix, sizeof(lock_prefix) - 1) == 0) {
        // 模拟新 UID 只能读取旧锁，避免测试以 root 执行时绕过文件权限。
        saw_device_lock = true;
        if ((flags & O_ACCMODE) != O_RDONLY) {
            errno = EACCES;
            return -1;
        }
    }
    if (flags & O_CREAT) {
        va_list arguments;
        va_start(arguments, flags);
        const auto mode = va_arg(arguments, int);
        va_end(arguments);
        return __real_openat(directory, name, flags, mode);
    }
    return __real_openat(directory, name, flags);
}

extern "C" ssize_t __wrap_write(int fd, const void* data, size_t bytes)
{
    // 仅拦截写入器启用故障注入后遇到的第一个文件，其他通信组照常写入。
    if (fault.actions.empty()) return __real_write(fd, data, bytes);
    if (fault.fd < 0) fault.fd = fd;
    if (fault.fd != fd) return __real_write(fd, data, bytes);
    const auto index = fault.attempts.size();
    fault.attempts.push_back({Clock::now(), bytes});
    if (index >= fault.actions.size()) return __real_write(fd, data, bytes);
    switch (fault.actions[index]) {
    case WriteAction::interrupt: errno = EINTR; return -1;
    case WriteAction::error: errno = EIO; return -1;
    case WriteAction::zero: return 0;
    case WriteAction::partial: return __real_write(fd, data, std::min(bytes, size_t{17}));
    }
    return __real_write(fd, data, bytes);
}

// 让正式入口通过 dlsym 获取 PCI 地址，测试目录选择时不需要真实设备。
extern "C" aclError aclrtDeviceGetPCIBusId(int32_t device, char* address, int32_t length)
{
    const char bdf[] = "0080:60:00.0";
    if ((device != 0 && device != 1) || length < static_cast<int32_t>(sizeof(bdf))) return 1;
    std::memcpy(address, bdf, sizeof(bdf));
    return ACL_SUCCESS;
}

// 两张模拟设备使用相同 BDF，但有不同 UUID，覆盖共享目录下的设备区分。
extern "C" aclError aclrtDeviceGetUuid(int32_t device, aclrtUuid* uuid)
{
    if (device != 0 && device != 1) return 1;
    std::memset(uuid->bytes, device == 0 ? 0x22 : 0x33, sizeof(uuid->bytes));
    return ACL_SUCCESS;
}

TraceRecord Record(const std::string& group, int64_t gid, bool complete, int64_t op = 2)
{
    TraceRecord result;
    result.event = complete ? "end" : "timeout";
    result.values = {{"gid", gid}, {"op", op}, {"op_count", gid - 1}, {"dispatch_count", 7},
        {"ep_size", 32}, {"rank", 17}, {"hidden_size", 6144}, {"num_experts", 256},
        {"aiv", 32}, {"num_tokens", 2048}, {"topk", 8}, {"pid", 1864},
        {"launch_ns", 1794468544967346000LL}};
    if (complete) {
        result.values.insert({{"start_ns", 1794468544967558000LL}, {"end_ns", 1794468544967564000LL},
                              {"elapsed_ns", 6000}, {"start_tick", 120000}, {"end_tick", 126000}});
    }
    // 测试流句柄的无符号上界，超过有符号 64 位的数字不能被截断。
    result.labels = {{"comm_id", group}, {"stream", "18446744073709551615"}};
    return result;
}

void TestWriteRetry(TraceRecorder& writer, const std::string& mode)
{
    const std::string a(32, '1'), b(32, '2');
    writer.Attach(a);
    if (mode == "retry-recover") {
        // EINTR 不占重试额度；另三次失败后，第四次常规调用仍能完成该行。
        fault.actions = {WriteAction::interrupt, WriteAction::error, WriteAction::zero, WriteAction::error};
        writer.Write({Record(a, 1, true)});
        ExpectWrites(5, {1, 2, 3});
    } else if (mode == "retry-partial") {
        fault.actions = {WriteAction::partial, WriteAction::error, WriteAction::error};
        writer.Write({Record(a, 1, true)});
        ExpectWrites(4, {1, 2});
        const auto remaining = fault.attempts[0].bytes - 17;
        for (size_t i = 1; i < fault.attempts.size(); ++i) {
            if (fault.attempts[i].bytes != remaining) throw std::runtime_error("retry did not resume after partial write");
        }
    } else {
        writer.Attach(b);
        writer.Write({Record(a, 1, true)});
        // 短写前的错误也占本行重试额度。耗尽时应回退半行，保留先前完整的一行。
        fault.actions = {WriteAction::error, WriteAction::partial, WriteAction::error,
                         WriteAction::error, WriteAction::error};
        bool failed = false;
        try {
            writer.Write({Record(a, 2, true), Record(b, 3, true), Record(a, 4, true)});
        } catch (const std::system_error& error) {
            if (error.code().value() != EIO) throw;
            failed = true;
        }
        if (!failed) throw std::runtime_error("expected write retries to be exhausted");
        ExpectWrites(5, {0, 2, 3});
        writer.Write({Record(a, 5, true), Record(b, 6, true)});
        ExpectWrites(5, {0, 2, 3});
    }
}

void TestGroupDetach(TraceRecorder& writer)
{
    const auto fd_count = [] {
        return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                             std::filesystem::directory_iterator{});
    };
    const auto before = fd_count();
    const std::string group(32, '1');
    writer.Attach(group);
    writer.Attach(group);
    if (fd_count() != before + 1) throw std::runtime_error("group did not share a single log descriptor");
    writer.Write({Record(group, 1, true)});
    writer.Detach(group);
    if (fd_count() != before + 1) throw std::runtime_error("group descriptor closed while still referenced");
    writer.Write({Record(group, 2, true)});
    writer.Detach(group);
    // 测试仍持有写入器和设备锁，只允许该组的日志描述符减少。
    if (fd_count() != before) throw std::runtime_error("detached group retained a log descriptor");
    writer.Write({Record(group, 3, true)});
}

void WaitForChild(pid_t child)
{
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("recorder fork child failed");
}

void TestForkLock(std::shared_ptr<TraceRecorder>& writer, const std::string& directory)
{
    const std::string a(32, '1'), b(32, '2');
    writer->Attach(a);
    writer->Write({Record(a, 1, true)});

    const auto child_destroys = fork();
    if (child_destroys < 0) throw std::runtime_error("cannot fork recorder child");
    if (child_destroys == 0) {
        writer.reset();
        _exit(0);
    }
    WaitForChild(child_destroys);
    // 子进程关闭继承的写入器不能解除父进程仍持有的锁。
    bool still_locked = false;
    try {
        TraceRecorderTestAccess::Open(directory, 512);
    } catch (const std::system_error& error) {
        if (error.code().value() != EWOULDBLOCK) throw;
        still_locked = true;
    }
    if (!still_locked) throw std::runtime_error("fork child unlocked owner's recorder");

    int release[2];
    if (pipe(release) < 0) throw std::runtime_error("cannot create recorder child pipe");
    const auto child_stays_alive = fork();
    if (child_stays_alive < 0) {
        close(release[0]);
        close(release[1]);
        throw std::runtime_error("cannot fork recorder lock holder");
    }
    if (child_stays_alive == 0) {
        close(release[1]);
        char byte;
        ssize_t count;
        do {
            count = read(release[0], &byte, 1);
        } while (count < 0 && errno == EINTR);
        _exit(count == 0 ? 0 : 1);
    }
    close(release[0]);
    try {
        writer.reset();
        // 子进程保持存活并持有旧描述符。只有父进程主动解锁，新写入器才可以立即启动。
        auto next = TraceRecorderTestAccess::Open(directory, 512);
        next->Attach(b);
        next->Write({Record(b, 2, true)});
    } catch (...) {
        close(release[1]);
        WaitForChild(child_stays_alive);
        throw;
    }
    close(release[1]);
    WaitForChild(child_stays_alive);
}

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const std::string directory = argv[1], mode = argv[2];
    try {
        // 格式和重试测试写入同一文件，轮转测试独立使用小上限。
        const bool retry = mode.rfind("retry-", 0) == 0;
        const bool configured = mode == "configured" || mode == "foreign-device";
        const bool large_limit = mode == "schemas" || mode == "detach" || retry;
        readonly_existing_lock = mode == "readonly-lock";
        auto writer = configured ? TraceRecorder::Open(mode == "configured" ? 0 : 1)
            : TraceRecorderTestAccess::Open(directory, large_limit ? 16 * 1024 : 512);
        if (configured) {
            const std::string group(32, mode == "configured" ? '1' : '2');
            writer->Attach(group);
            writer->Write({Record(group, 1, true)});
        } else if (mode == "hold") {
            const std::string group(32, '1');
            writer->Attach(group);
            writer->Write({Record(group, 1, true)});
            std::ofstream(directory + "/test-lock-ready") << "ready\n";
            std::cin.get();
        } else if (mode == "emit") {
            const std::string a(32, '1'), b(32, '2');
            writer->Attach(a);
            writer->Attach(b);
            std::vector<TraceRecord> batch;
            for (int64_t gid = 1; gid <= 9; ++gid) batch.push_back(Record(a, gid, true));
            batch.push_back(Record(b, 1, false));
            writer->Write(batch);
        } else if (mode == "schemas") {
            const std::string group(32, '1');
            writer->Attach(group);
            std::vector<TraceRecord> batch;
            for (int64_t op = 0; op < 10; ++op) {
                batch.push_back(Record(group, op + 1, false, op));
                batch.push_back(Record(group, op + 1, true, op));
            }
            writer->Write(batch);
        } else if (mode == "detach") {
            TestGroupDetach(*writer);
        } else if (mode == "fork-lock") {
            TestForkLock(writer, directory);
        } else if (mode == "readonly-lock") {
            if (!saw_device_lock) throw std::runtime_error("expected opening the existing lock");
            const std::string group(32, '1');
            writer->Attach(group);
            writer->Write({Record(group, 1, true)});
        } else if (retry) {
            TestWriteRetry(*writer, mode);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
''',
        encoding="utf-8",
    )
    executable = build / "test_recorder"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-rdynamic",
            "-I",
            str(build),
            "-I",
            str(ROOT / "csrc/direct"),
            str(source),
            str(ROOT / "csrc/direct/runtime/trace_recorder.cpp"),
            "-Wl,--wrap=write",
            "-Wl,--wrap=openat",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        timeout=90,
    )
    return executable


def _filename(directory, comm_id, bus_id, index):
    return directory / f"trmt-deepep-trace-{comm_id}-{bus_id}-{index}.txt"


def _owner_marker(directory, comm_id, bus_id, device_uuid=DEVICE_UUID):
    return directory / f".trmt-deepep-trace-owner-{device_uuid}-{bus_id}-{comm_id}"


def test_production_directory_from_environment(recorder_executable, tmp_path):
    # 含有上级引用和符号链接的路径应指向选定的目录，写入和清理均在该目录内。
    directory = tmp_path / "configured logs"
    directory.mkdir()
    (tmp_path / "child").mkdir()
    (tmp_path / "link to logs").symlink_to(directory, target_is_directory=True)
    selected = tmp_path / "child" / ".." / "link to logs"
    old = _filename(directory, OLD_GROUP, BUS, 0)
    old.write_text("old", encoding="utf-8")
    _owner_marker(directory, OLD_GROUP, BUS).touch()
    outside = _filename(tmp_path, OLD_GROUP, BUS, 0)
    outside.write_text("keep outside file", encoding="utf-8")
    outside_marker = _owner_marker(tmp_path, OLD_GROUP, BUS)
    outside_marker.touch()
    subprocess.run(
        [recorder_executable, tmp_path, "configured"],
        env=dict(os.environ, DEEPEP_TRACE_DIR=str(selected)),
        check=True,
        timeout=15,
    )
    assert not old.exists()
    assert not _owner_marker(directory, OLD_GROUP, BUS).exists()
    assert outside.read_text(encoding="utf-8") == "keep outside file"
    assert outside_marker.is_file()
    files = [_filename(directory, GROUP_A, BUS, i) for i in range(8)]
    assert json.loads(files[0].read_bytes())["comm_id"] == GROUP_A
    assert all(path.read_bytes() == b"" for path in files[1:])
    assert _owner_marker(directory, GROUP_A, BUS).read_bytes() == b""


def test_native_cleanup_rotation_and_record_fields(recorder_executable, tmp_path):
    old = _filename(tmp_path, OLD_GROUP, BUS, 0)
    old.write_text("old", encoding="utf-8")
    old_marker = _owner_marker(tmp_path, OLD_GROUP, BUS)
    old_marker.touch()
    # 未登记归属或归属其他设备的日志即使匹配 BDF，也不能被清理。
    unmarked = _filename(tmp_path, "4" * 32, BUS, 0)
    unmarked.write_text("keep unmarked file", encoding="utf-8")
    foreign_group = "5" * 32
    foreign = _filename(tmp_path, foreign_group, BUS, 0)
    foreign.write_text("keep same BDF foreign file", encoding="utf-8")
    foreign_marker = _owner_marker(tmp_path, foreign_group, BUS, FOREIGN_DEVICE_UUID)
    foreign_marker.touch()
    other_bus = _filename(tmp_path, OLD_GROUP, BUS + 1, 0)
    other_bus.write_text("keep other device", encoding="utf-8")
    other_bus_marker = _owner_marker(tmp_path, OLD_GROUP, BUS + 1)
    other_bus_marker.touch()
    other_index = _filename(tmp_path, OLD_GROUP, BUS, 8)
    other_index.write_text("keep unrelated filename", encoding="utf-8")
    other_file = tmp_path / "user_archive.txt"
    other_file.write_text("keep archive", encoding="utf-8")
    symbolic_link = _filename(tmp_path, OLD_GROUP, BUS, 1)
    symbolic_link.symlink_to(other_file)
    matching_directory = _filename(tmp_path, OLD_GROUP, BUS, 2)
    matching_directory.mkdir()
    marked_by_symlink = _filename(tmp_path, "6" * 32, BUS, 0)
    marked_by_symlink.write_text("keep a file marked by a symlink", encoding="utf-8")
    symlink_marker = _owner_marker(tmp_path, "6" * 32, BUS)
    symlink_marker.symlink_to(other_file)

    subprocess.run([recorder_executable, tmp_path, "emit"], check=True, timeout=15)

    assert not old.exists()
    assert not old_marker.exists()
    assert unmarked.read_text(encoding="utf-8") == "keep unmarked file"
    assert foreign.read_text(encoding="utf-8") == "keep same BDF foreign file"
    assert foreign_marker.is_file()
    assert other_bus.read_text(encoding="utf-8") == "keep other device"
    assert other_bus_marker.is_file()
    assert other_index.read_text(encoding="utf-8") == "keep unrelated filename"
    assert other_file.read_text(encoding="utf-8") == "keep archive"
    assert symbolic_link.is_symlink()
    assert matching_directory.is_dir()
    assert marked_by_symlink.read_text(encoding="utf-8") == "keep a file marked by a symlink"
    assert symlink_marker.is_symlink()

    # 一条记录能写入 512 B，两条不能；第九条必须回到 0 并覆盖首条。
    group_a = [_filename(tmp_path, GROUP_A, BUS, i).read_bytes() for i in range(8)]
    assert all(data.endswith(b"\n") and 0 < len(data) <= 512 for data in group_a)
    assert len(group_a[0]) + len(group_a[1]) > 512
    records = [json.loads(data) for data in group_a]
    assert [r["gid"] for r in records] == [9, 2, 3, 4, 5, 6, 7, 8]
    assert records[0] == {
        "type": 2,
        "gid": 9,
        "op": 2,
        "op_count": 8,
        "dispatch_count": 7,
        "ep_size": 32,
        "bus_id": BUS,
        "rank": 17,
        "nvl_rank": 1,
        "hidden_size": 6144,
        "num_experts": 256,
        "num_sms": 32,
        "num_tokens": 2048,
        "topk": 8,
        "pid": 1864,
        "stream": 2**64 - 1,
        "comm_id": GROUP_A,
        "launch_time": "2026-11-12 15:29:04.967346",
        "start_time": "2026-11-12 15:29:04.967558",
        "end_time": "2026-11-12 15:29:04.967564",
        "elapsed_ns": 6000,
        "start_tick": 120000,
        "end_tick": 126000,
    }
    group_b = [_filename(tmp_path, GROUP_B, BUS, i).read_bytes() for i in range(8)]
    assert all(data == b"" for data in group_b[1:])
    timeout = json.loads(group_b[0])
    assert timeout["type"] == 1 and timeout["comm_id"] == GROUP_B
    assert timeout.keys() == records[0].keys()
    assert all(timeout[key] is None for key in ("start_time", "end_time", "elapsed_ns", "start_tick", "end_tick"))


def test_native_recorder_skips_cleanup_for_active_bus(recorder_executable, tmp_path):
    # 当另一个进程持有同设备锁时，竞争者不能删掉其写入的日志。
    with subprocess.Popen([recorder_executable, tmp_path, "hold"], stdin=subprocess.PIPE) as owner:
        try:
            ready = tmp_path / "test-lock-ready"
            deadline = time.monotonic() + 10
            while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.is_file()
            active = _filename(tmp_path, GROUP_A, BUS, 0)
            original = active.read_bytes()
            assert json.loads(original)["comm_id"] == GROUP_A
            own_marker = _owner_marker(tmp_path, GROUP_A, BUS)
            assert own_marker.is_file()
            contender = subprocess.run([recorder_executable, tmp_path, "contend"], check=False, timeout=10)
            assert contender.returncode == 1
            assert active.read_bytes() == original
            assert own_marker.is_file()
            # 同一目录的不同真实设备即使 BDF 相同，也可获取各自的锁且不互删日志。
            subprocess.run(
                [recorder_executable, tmp_path, "foreign-device"],
                env=dict(os.environ, DEEPEP_TRACE_DIR=str(tmp_path)),
                check=True,
                timeout=10,
            )
            assert active.read_bytes() == original
            assert own_marker.is_file()
            assert _owner_marker(tmp_path, GROUP_B, BUS, FOREIGN_DEVICE_UUID).is_file()
        finally:
            owner.communicate(input=b"x", timeout=10)


def test_native_recorder_detach_closes_group_without_deleting_logs(recorder_executable, tmp_path):
    subprocess.run([recorder_executable, tmp_path, "detach"], check=True, timeout=15)
    # 两个引用逐个注销后，只有注销前的记录仍在，且轮转文件和归属标记未被删除。
    files = [_filename(tmp_path, GROUP_A, BUS, i) for i in range(8)]
    assert [json.loads(line)["gid"] for line in files[0].read_bytes().splitlines()] == [1, 2]
    assert all(path.read_bytes() == b"" for path in files[1:])
    assert _owner_marker(tmp_path, GROUP_A, BUS).is_file()


def test_native_recorder_reopens_readonly_lock(recorder_executable, tmp_path):
    lock = tmp_path / f".trmt-deepep-trace-device-{DEVICE_UUID}.lock"
    lock.touch()
    lock.chmod(0o444)
    original = lock.stat()
    subprocess.run([recorder_executable, tmp_path, "readonly-lock"], check=True, timeout=15)
    assert lock.stat().st_ino == original.st_ino
    assert json.loads(_filename(tmp_path, GROUP_A, BUS, 0).read_bytes())["gid"] == 1


def test_native_recorder_owner_releases_forked_lock(recorder_executable, tmp_path):
    subprocess.run([recorder_executable, tmp_path, "fork-lock"], check=True, timeout=15)
    assert not _owner_marker(tmp_path, GROUP_A, BUS).exists()
    assert not _filename(tmp_path, GROUP_A, BUS, 0).exists()
    assert json.loads(_filename(tmp_path, GROUP_B, BUS, 0).read_bytes())["gid"] == 2


def test_native_recorder_long_and_short_schemas(recorder_executable, tmp_path):
    subprocess.run([recorder_executable, tmp_path, "schemas"], check=True, timeout=15)
    lines = _filename(tmp_path, GROUP_A, BUS, 0).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [(row["op"], row["type"]) for row in records] == [
        (op, record_type) for op in range(10) for record_type in (1, 2)
    ]
    shape_fields = {"ep_size": 32, "hidden_size": 6144, "num_experts": 256, "num_tokens": 2048, "topk": 8}
    time_fields = {
        "start_time": "2026-11-12 15:29:04.967558",
        "end_time": "2026-11-12 15:29:04.967564",
        "elapsed_ns": 6000,
        "start_tick": 120000,
        "end_tick": 126000,
    }
    for row in records:
        # 短格式即使收到形状字段也不写这些键；超时仍保留设备时间字段，其值为 null。
        expected = {
            "type": row["type"],
            "gid": row["op"] + 1,
            "op": row["op"],
            "op_count": row["op"],
            "dispatch_count": 7,
            "bus_id": BUS,
            "rank": 17,
            "nvl_rank": 1,
            "num_sms": 32,
            "pid": 1864,
            "stream": 2**64 - 1,
            "comm_id": GROUP_A,
            "launch_time": "2026-11-12 15:29:04.967346",
        }
        if row["op"] in (2, 4, 5, 7, 8, 9):
            expected.update(shape_fields)
        expected.update(time_fields if row["type"] == 2 else dict.fromkeys(time_fields))
        assert row == expected


@pytest.mark.parametrize("mode", ["retry-recover", "retry-partial", "retry-exhaust"])
def test_native_recorder_write_retry(recorder_executable, tmp_path, mode):
    subprocess.run([recorder_executable, tmp_path, mode], check=True, timeout=15)
    a = _filename(tmp_path, GROUP_A, BUS, 0).read_bytes()
    # 完整行可解析，且回退后没有留下半条记录。
    assert a.endswith(b"\n")
    assert [json.loads(line)["gid"] for line in a.splitlines()] == [1]
    if mode == "retry-exhaust":
        b = _filename(tmp_path, GROUP_B, BUS, 0).read_bytes()
        assert b.endswith(b"\n")
        assert [json.loads(line)["gid"] for line in b.splitlines()] == [3, 6]
