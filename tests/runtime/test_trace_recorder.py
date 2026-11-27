"""Test the production recorder entry point, stale file cleanup, and rotation without an NPU."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
GROUP_A = str(2**64 - 1)
GROUP_B = "0"
OLD_GROUP = "34567"
PHYSICAL_DEVICE_ID = 3
BUS = PHYSICAL_DEVICE_ID


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
        def open_trace_recorder(source, comm_id):
            assert source is active_source
            calls.append(("open", comm_id))
            return recorder

    try:
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
        assert [row for row in calls if row[0] == "open"] == [("open", GROUP_A)]
        assert [row for row in calls if row[0] == "attach"] == [("attach", GROUP_A), ("attach", GROUP_B)]
        assert polled.wait(2)
        assert waits[0] == pytest.approx(0.007)
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


def test_legacy_recorder_open_does_not_touch_history(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    monitor = _load_monitor().KernelTrace
    source = SimpleNamespace(identity=1, poll_interval_ms=2, calibration=lambda: {"device": 0})
    history = tmp_path / "old-trace.txt"
    history.write_bytes(b"previous trace\n")

    def open_legacy_recorder(source):
        history.unlink()

    native = SimpleNamespace(get_kernel_trace=lambda: source, open_trace_recorder=open_legacy_recorder)
    try:
        with pytest.warns(RuntimeWarning, match=r"open_legacy_recorder.*argument"):
            assert monitor.attach(native, GROUP_A) is None
        assert history.read_bytes() == b"previous trace\n"
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

    def open_recorder(trace_source, comm_id):
        assert trace_source is source and comm_id == GROUP_A
        writer = Recorder()
        writer_refs.append(weakref.ref(writer))
        return writer

    native = SimpleNamespace(get_kernel_trace=lambda: source, open_trace_recorder=open_recorder)
    try:
        identity = monitor.attach(native, GROUP_A)
        assert identity is not None
        assert polled.wait(2)
        monitor.detach(identity)
        assert monitor._thread.is_alive()
        assert writer_refs[0]() is None
    finally:
        monitor.shutdown()


@pytest.mark.parametrize("synchronized", [False, True])
def test_last_detach_honors_existing_stream_synchronization(monkeypatch, synchronized):
    monitor = _load_monitor().KernelTrace
    calls = []
    source = SimpleNamespace(synchronize=lambda: calls.append("synchronize"))
    recorder = SimpleNamespace(
        drain=lambda _, closing: calls.append(("drain", closing)),
        detach=lambda group: calls.append(("detach", group)),
    )
    entry = dict(source=source, recorder=recorder, groups={GROUP_A: 1}, references=1, stream=None, mode="")
    monkeypatch.setattr(monitor, "_sources", {17: entry})

    monitor.detach(17, GROUP_A, synchronized=synchronized)

    expected = [] if synchronized else ["synchronize"]
    assert calls == [*expected, ("drain", True), ("detach", GROUP_A)]
    assert monitor._sources == {}


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
    native = SimpleNamespace(get_kernel_trace=lambda: source, open_trace_recorder=lambda _, comm_id: writer)
    try:
        for group in (GROUP_A, GROUP_B, GROUP_A):
            assert monitor.attach(native, group) == source.identity
    finally:
        monitor.shutdown()
    assert calls[3:5] == [("synchronize", None), ("close", None)]
    assert calls[5:].count(("detach", GROUP_A)) == 2
    assert calls[5:].count(("detach", GROUP_B)) == 1


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("comm_id", [GROUP_A, GROUP_B])
@pytest.mark.parametrize("use_mpi", [False, True])
def test_production_comm_id_is_shared_from_rank_zero(monkeypatch, rank, comm_id, use_mpi):
    trace = _load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_RECORDER_ENABLE", "1")
    monkeypatch.setattr(trace, "_global_rank", lambda: 10 + rank)
    random_bits = []
    attached = []
    group = None if use_mpi else object()
    comm = object() if use_mpi else None
    native = SimpleNamespace(
        kernel_trace_metadata_version=2,
        make_trace_metadata=lambda fields, *, recorder: fields,
    )

    def randbits(bits):
        random_bits.append(bits)
        return int(comm_id)

    def all_gather(actual_group, actual_comm, ep_size, proposal):
        assert actual_group is group and actual_comm is comm
        assert ep_size == 2 and proposal == (True, comm_id if rank == 0 else None, "", False)
        return [(True, comm_id, "", False), (True, None, "", False)]

    def attach(native_module, comm_id):
        assert native_module is native
        attached.append(comm_id)
        return 23

    monkeypatch.setattr(trace.secrets, "randbits", randbits)
    monkeypatch.setattr(trace.KernelTrace, "attach", staticmethod(attach))
    identity, session = trace.attach_trace_session(native, group, rank, 2, all_gather, comm=comm)
    assert random_bits == ([64] if rank == 0 else [])
    assert identity == 23 and attached == [comm_id]
    assert session.fields["comm_id"] == comm_id


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


def test_trace_shape_omits_unknown_fields_and_restores_native_context():
    trace = _load_monitor()
    context = object()
    active = context
    passed = []

    def make_metadata(fields, parent=None, *, recorder=False):
        assert all(value is not None for value in fields.values())
        return fields

    def push_metadata(template, api, handle, count):
        nonlocal active
        previous, active = active, (template, handle, count)
        passed.append(active)
        return previous

    def restore_metadata(previous):
        nonlocal active
        active = previous

    native = SimpleNamespace(
        kernel_trace_metadata_version=2,
        make_trace_metadata=make_metadata,
        push_trace_metadata=push_metadata,
        restore_trace_metadata=restore_metadata,
    )
    session = trace.KernelTraceSession(native, GROUP_A, 4, recorder=True)
    x = SimpleNamespace(shape=(8, 6144), dtype="bf16", element_size=lambda: 2)
    with session.call_shape("dispatch", x, num_tokens=8, num_experts=256, topk=None, capacity=16) as meta:
        handle = SimpleNamespace(_trace_handle_id=meta["handle_id"], _trace_dispatch_count=meta["dispatch_count"])
        assert meta["num_tokens"] == 8 and meta["num_experts"] == 256
        assert "topk" not in meta
    assert active is context

    with session.call_shape("dispatch", x, num_tokens=8, num_experts=256, topk=None, handle=handle, cached=True):
        assert active[1:] == (handle._trace_handle_id, 1)

    with session.call_shape("combine", x, num_tokens=4, num_experts=256, topk=None, handle=handle) as meta:
        assert meta["num_tokens"] == 4 and meta["api_input_rows"] == 8
        assert meta["dispatch_count"] == 0

    with pytest.raises(RuntimeError, match="native call failed"):
        with session.call_shape("combine", x, num_tokens=4, num_experts=None, topk=None) as meta:
            assert "num_experts" not in meta and "dispatch_count" not in meta
            raise RuntimeError("native call failed")
    assert active is context
    assert [count for _, _, count in passed] == [0, 1, 0, -1]


@pytest.fixture
def v1_buffer_module(monkeypatch):
    import deep_ep.backends as backends

    direct = "deep_ep.backends.direct"
    if direct in sys.modules or "torch" in sys.modules:
        from deep_ep.backends.direct import buffer

        yield buffer
        return

    # Direct imports only need Torch's dtype names before the mocked calls.
    existing = set(sys.modules)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", SimpleNamespace(int32="torch.int32", int64="torch.int64"))
        try:
            from deep_ep.backends.direct import buffer

            yield buffer
        finally:
            for name in set(sys.modules) - existing:
                if name == direct or name.startswith(f"{direct}."):
                    sys.modules.pop(name, None)
            vars(backends).pop("direct", None)


@pytest.mark.parametrize("topk", [None, 4])
def test_v1_buffer_traces_owned_and_plain_handles(monkeypatch, v1_buffer_module, topk):
    module = v1_buffer_module
    trace = _load_monitor()
    active = None
    calls = []

    def tensor(*shape):
        return SimpleNamespace(
            shape=shape, ndim=len(shape), size=shape.__getitem__, dtype="bf16", element_size=lambda: 2
        )

    def make_metadata(fields, parent=None, *, recorder=False):
        assert all(value is not None for value in fields.values())
        return fields

    def push_metadata(template, api, handle, count):
        nonlocal active
        previous, active = active, (template, handle, count)
        return previous

    def restore_metadata(previous):
        nonlocal active
        active = previous

    x, received = tensor(8, 6144), tensor(12, 6144)
    send_head, recv_src_idx = tensor(8, 4), tensor(12)
    rank_prefix, channel_prefix, recv_channel_prefix, plan = (object() for _ in range(4))

    def dispatch(*args):
        calls.append(active)
        return (
            received, None, None, None, [12], rank_prefix, channel_prefix, recv_channel_prefix, recv_src_idx,
            send_head, plan,
        )

    def combine(*args):
        calls.append(active)
        return x, None

    metadata_native = SimpleNamespace(
        kernel_trace_metadata_version=2,
        make_trace_metadata=make_metadata,
        push_trace_metadata=push_metadata,
        restore_trace_metadata=restore_metadata,
    )
    buffer = object.__new__(module.Buffer)
    buffer._destroyed = True  # Keep the incomplete mock out of the finalizer path.
    buffer._ensure_alive = lambda: None
    buffer._native = SimpleNamespace(intranode_dispatch_with_plan=dispatch, intranode_combine=combine)
    buffer._trace_session = trace.KernelTraceSession(metadata_native, GROUP_A, 4, recorder=True)
    buffer._finish_comm_event = lambda async_finish, tensors: "event"
    buffer.transport, buffer.group_size = "udma", 4
    buffer._communication_stream = object()
    buffer._num_max_tokens_per_rank = 16
    buffer._udma_auto_policy = None
    buffer._udma_schedule = "peer"
    buffer._udma_dispatch_peer_order = buffer._udma_combine_peer_order = "rank"
    monkeypatch.setattr(module, "_get_allocation_stream_id", lambda *_: None)
    monkeypatch.setattr(module, "_record_recv_outputs_on_current_stream", lambda *_: None)
    monkeypatch.setattr(module, "_use_comm_stream_after_dependency", lambda *_: nullcontext())
    monkeypatch.setattr(module, "_is_capturing", lambda: False)
    config = module.Config(num_aiv=32, udma_peers_per_worker=4, udma_wqes_per_visit=8)

    result = buffer.dispatch(
        x,
        topk_idx=tensor(8, topk) if topk is not None else None,
        num_tokens_per_rank=object(),
        is_token_in_rank=tensor(8, 4),
        num_tokens_per_expert=SimpleNamespace(numel=lambda: 256),
        config=config,
    )
    handle = result[4]
    assert len(handle) == 6 and result[-1] == "event"
    buffer.dispatch(x, handle=handle, config=config)
    buffer.combine(received, handle, config=config)
    buffer.dispatch(x, handle=tuple(handle), config=config)
    buffer.combine(received, tuple(handle), config=config)
    assert active is None

    fresh, cached, combined, legacy_cached, legacy_combined = calls
    assert [call[2] for call in calls] == [0, 1, 0, 2, -1]
    assert fresh[1] == cached[1] == combined[1]
    for fields, _, _ in (fresh, cached, combined):
        assert fields["num_tokens"] == 8 and fields["num_experts"] == 256
        assert fields.get("topk") == topk
    assert combined[0]["api_input_rows"] == 12
    for fields, _, _ in (legacy_cached, legacy_combined):
        assert fields["num_tokens"] == 8
        assert "num_experts" not in fields and "topk" not in fields
    buffer.combine(received, tuple(handle), topk_weights=tensor(12, 2), config=config)
    weighted_fields, _, weighted_count = calls[-1]
    assert weighted_fields["topk"] == 2 and "num_experts" not in weighted_fields
    assert weighted_count == -1

    monkeypatch.setattr(module, "_is_capturing", lambda: True)
    buffer.dispatch(x, handle=handle, config=config)
    buffer.combine(received, handle, config=config)
    assert calls[-2:] == [None, None]


def test_v1_destroy_orders_sync_detach_and_native_cleanup(monkeypatch, v1_buffer_module):
    from deep_ep.utils import kernel_trace

    module = v1_buffer_module
    calls = []
    interrupted = True
    session = object()

    def synchronize():
        calls.append("sync")
        if interrupted:
            raise RuntimeError("async work not synchronized")

    def detach(identity, active_session, *, synchronized):
        assert identity == 23 and active_session is session and synchronized
        calls.append("detach")
        raise RuntimeError("recorder close failed")

    monkeypatch.setattr(module, "_is_capturing", lambda: False)
    monkeypatch.setattr(kernel_trace, "detach_trace_session", detach)
    buffer = object.__new__(module.Buffer)
    buffer._destroyed = False
    buffer._lock = threading.RLock()
    buffer._slot = SimpleNamespace(state=module.SlotState.FREE)
    buffer._failed_event_tensors = (object(),)
    buffer._communication_stream = SimpleNamespace(synchronize=synchronize)
    buffer._native = SimpleNamespace(destroy=lambda: calls.append("destroy"))
    buffer._kernel_trace, buffer._trace_session = 23, session

    with pytest.raises(RuntimeError, match="async work not synchronized"):
        buffer.destroy()
    assert calls == ["sync"]
    assert buffer._kernel_trace == 23 and buffer._trace_session is session
    interrupted = False
    with pytest.raises(RuntimeError, match="recorder close failed"):
        buffer.destroy()
    assert buffer._kernel_trace is None and buffer._trace_session is None
    buffer.destroy()
    assert calls == ["sync", "sync", "detach", "destroy"]


DEVICE_UUID = "ab" * 16
FOREIGN_DEVICE_UUID = "cd" * 16


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
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>

namespace {
constexpr uint32_t kPhysicalDeviceId = 3;
const std::string kDeviceUuid = "abababababababababababababababab";
const std::string kGroupA = "18446744073709551615";
const std::string kGroupB = "0";
}

namespace ascend_deepep::trace {
struct TraceRecorderTestAccess {
    static std::shared_ptr<TraceRecorder> Open(const std::string& directory, uint64_t limit,
                                               uint32_t bus_id = kPhysicalDeviceId) {
        return TraceRecorder::OpenAt(directory, bus_id, kDeviceUuid, limit, kPhysicalDeviceId);
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
bool use_logic_to_physical_query = false;

struct MockDevice {
    const char* mode;
    int32_t physical_id;
    unsigned char uuid_byte;
};

constexpr MockDevice devices[] = {
    {"configured", 3, 0xab},
    {"foreign-device", 3, 0xcd},
    {"physical-first", 5, 0xab},
    {"physical-second", 6, 0xab},
    {"invalid-physical-id", -1, 0xab},
    {"empty-device-uuid", 7, 0x00},
};

bool ValidDevice(int32_t device)
{
    return device >= 0 && static_cast<std::size_t>(device) < std::size(devices);
}

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
extern "C" void* __real_dlsym(void* handle, const char* name);

extern "C" void* __wrap_dlsym(void* handle, const char* name)
{
    if (use_logic_to_physical_query && std::strcmp(name, "aclrtGetPhyDevIdByUserDevId") == 0)
        return nullptr;
    return __real_dlsym(handle, name);
}

extern "C" int __wrap_openat(int directory, const char* name, int flags, ...)
{
    constexpr char lock_prefix[] = ".trmt-deepep-trace-device-";
    if (std::strncmp(name, lock_prefix, sizeof(lock_prefix) - 1) == 0) {
        saw_device_lock = true;
        if (readonly_existing_lock && (flags & O_ACCMODE) != O_RDONLY) {
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

extern "C" aclError aclrtGetPhyDevIdByUserDevId(int32_t device, int32_t* physical_id)
{
    if (!ValidDevice(device)) return 1;
    *physical_id = devices[device].physical_id;
    return ACL_SUCCESS;
}

extern "C" aclError aclrtGetPhyDevIdByLogicDevId(int32_t device, int32_t* physical_id)
{
    return aclrtGetPhyDevIdByUserDevId(device, physical_id);
}

extern "C" aclError aclrtDeviceGetUuid(int32_t device, aclrtUuid* uuid)
{
    if (!ValidDevice(device)) return 1;
    std::memset(uuid->bytes, devices[device].uuid_byte, sizeof(uuid->bytes));
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
    result.labels = {{"comm_id", group}, {"stream", "18446744073709551615"}};
    return result;
}

void TestWriteRetry(TraceRecorder& writer, const std::string& mode)
{
    const auto& a = kGroupA;
    const auto& b = kGroupB;
    writer.Attach(a);
    if (mode == "retry-recover") {
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
    const auto& group = kGroupA;
    writer.Attach(group);
    writer.Attach(group);
    if (fd_count() != before + 1) throw std::runtime_error("group did not share a single log descriptor");
    writer.Write({Record(group, 1, true)});
    writer.Detach(group);
    if (fd_count() != before + 1) throw std::runtime_error("group descriptor closed while still referenced");
    writer.Write({Record(group, 2, true)});
    writer.Detach(group);
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
    const auto& a = kGroupA;
    const auto& b = kGroupB;
    writer->Attach(a);
    writer->Write({Record(a, 1, true)});

    const auto child_destroys = fork();
    if (child_destroys < 0) throw std::runtime_error("cannot fork recorder child");
    if (child_destroys == 0) {
        writer.reset();
        _exit(0);
    }
    WaitForChild(child_destroys);
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

template <class Action>
void RejectNoncanonicalCommIds(Action action)
{
    for (const auto* invalid : {"", "00", "+1", "-1", "18446744073709551616", kDeviceUuid.c_str()}) {
        try {
            action(invalid);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("accepted invalid comm_id " + std::string(invalid));
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const std::string directory = argv[1], mode = argv[2];
    try {
        if (mode == "invalid-open-comm-id") {
            RejectNoncanonicalCommIds([](const std::string& comm_id) { TraceRecorder::Open(0, comm_id); });
            if (saw_device_lock) throw std::runtime_error("invalid comm_id opened a device lock");
            return 0;
        }
        if (mode == "physical-pair") {
            auto first = TraceRecorder::Open(2, kGroupA);
            auto second = TraceRecorder::Open(3, kGroupA);
            first->Attach(kGroupA);
            second->Attach(kGroupA);
            first->Write({Record(kGroupA, 1, true)});
            second->Write({Record(kGroupA, 1, true)});
            return 0;
        }
        const bool retry = mode.rfind("retry-", 0) == 0;
        use_logic_to_physical_query = mode == "legacy-physical-query";
        const auto device_mode = use_logic_to_physical_query ? "configured" : mode;
        const auto device = std::find_if(std::begin(devices), std::end(devices), [&](const MockDevice& value) {
            return device_mode == value.mode;
        });
        const bool configured = device != std::end(devices);
        const bool large_limit = mode == "schemas" || mode == "detach" || retry;
        readonly_existing_lock = mode == "readonly-lock";
        const auto& group = mode == "foreign-device" ? kGroupB : kGroupA;
        auto writer = configured ? TraceRecorder::Open(static_cast<int32_t>(device - std::begin(devices)), group)
            : TraceRecorderTestAccess::Open(directory, large_limit ? 16 * 1024 : 512,
                                            kPhysicalDeviceId + (mode == "contend-other-bus" ? 1 : 0));
        if (configured) {
            writer->Attach(group);
            writer->Write({Record(group, 1, true)});
        } else if (mode == "hold") {
            writer->Attach(kGroupA);
            writer->Write({Record(kGroupA, 1, true)});
            std::ofstream(directory + "/test-lock-ready") << "ready\n";
            std::cin.get();
        } else if (mode == "emit") {
            writer->Attach(kGroupA);
            writer->Attach(kGroupB);
            std::vector<TraceRecord> batch;
            for (int64_t gid = 1; gid <= 9; ++gid) batch.push_back(Record(kGroupA, gid, true));
            batch.push_back(Record(kGroupB, 1, false));
            writer->Write(batch);
        } else if (mode == "schemas") {
            writer->Attach(kGroupA);
            std::vector<TraceRecord> batch;
            for (int64_t op = 0; op < 10; ++op) {
                batch.push_back(Record(kGroupA, op + 1, false, op));
                batch.push_back(Record(kGroupA, op + 1, true, op));
            }
            writer->Write(batch);
        } else if (mode == "invalid-comm-id") {
            RejectNoncanonicalCommIds([&writer](const std::string& comm_id) { writer->Attach(comm_id); });
        } else if (mode == "detach") {
            TestGroupDetach(*writer);
        } else if (mode == "fork-lock") {
            TestForkLock(writer, directory);
        } else if (mode == "readonly-lock") {
            if (!saw_device_lock) throw std::runtime_error("expected opening the existing lock");
            writer->Attach(kGroupA);
            writer->Write({Record(kGroupA, 1, true)});
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
            "-Wl,--wrap=dlsym",
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


def _owner_marker(directory, comm_id, bus_id, device_uuid=DEVICE_UUID, physical_id=PHYSICAL_DEVICE_ID):
    return directory / f".trmt-deepep-trace-owner-v2-{device_uuid}-{physical_id}-{bus_id}-{comm_id}"


def _device_lock(directory, physical_id, device_uuid=DEVICE_UUID):
    return directory / f".trmt-deepep-trace-device-{device_uuid}-{physical_id}.lock"


@pytest.mark.parametrize("mode", ["configured", "legacy-physical-query"])
def test_production_directory_from_environment(recorder_executable, tmp_path, mode):
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
        [recorder_executable, tmp_path, mode],
        env=dict(os.environ, DEEPEP_TRACE_DIR=str(selected)),
        check=True,
        timeout=15,
    )
    assert not old.exists()
    assert not _owner_marker(directory, OLD_GROUP, BUS).exists()
    assert outside.read_text(encoding="utf-8") == "keep outside file"
    assert outside_marker.is_file()
    files = [_filename(directory, GROUP_A, BUS, i) for i in range(8)]
    assert json.loads(files[0].read_bytes())["comm_id"] == int(GROUP_A)
    assert all(path.read_bytes() == b"" for path in files[1:])
    assert _owner_marker(directory, GROUP_A, BUS).read_bytes() == b""


def test_native_recorder_uses_distinct_physical_ids_for_shared_uuid(recorder_executable, tmp_path):
    subprocess.run(
        [recorder_executable, tmp_path, "physical-pair"],
        env=dict(os.environ, DEEPEP_TRACE_DIR=str(tmp_path)),
        check=True,
        timeout=15,
    )

    for physical_id in (5, 6):
        files = [_filename(tmp_path, GROUP_A, physical_id, i) for i in range(8)]
        record = json.loads(files[0].read_bytes())
        assert record["comm_id"] == int(GROUP_A) and record["bus_id"] == physical_id
        assert all(path.read_bytes() == b"" for path in files[1:])
        assert _owner_marker(tmp_path, GROUP_A, physical_id, physical_id=physical_id).is_file()
        assert _device_lock(tmp_path, physical_id).is_file()


@pytest.mark.parametrize("mode", ["invalid-physical-id", "empty-device-uuid"])
def test_native_recorder_rejects_unresolved_device_identity(recorder_executable, tmp_path, mode):
    result = subprocess.run(
        [recorder_executable, tmp_path, mode],
        env=dict(os.environ, DEEPEP_TRACE_DIR=str(tmp_path)),
        check=False,
        timeout=15,
    )

    assert result.returncode == 1
    assert not list(tmp_path.iterdir())


def test_native_recorder_rejects_noncanonical_comm_ids(recorder_executable, tmp_path):
    subprocess.run([recorder_executable, tmp_path, "invalid-comm-id"], check=True, timeout=15)
    assert list(tmp_path.iterdir()) == [_device_lock(tmp_path, PHYSICAL_DEVICE_ID)]


def test_native_recorder_rejects_invalid_open_before_cleanup(recorder_executable, tmp_path):
    history = _filename(tmp_path, OLD_GROUP, BUS, 0)
    history.write_bytes(b"previous trace\n")
    marker = _owner_marker(tmp_path, OLD_GROUP, BUS)
    marker.touch()
    lock = _device_lock(tmp_path, PHYSICAL_DEVICE_ID)
    lock.touch()

    def snapshot():
        return {
            path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
            for path in tmp_path.iterdir()
        }

    previous = snapshot()
    subprocess.run(
        [recorder_executable, tmp_path, "invalid-open-comm-id"],
        env=dict(os.environ, DEEPEP_TRACE_DIR=str(tmp_path)),
        check=True,
        timeout=15,
    )
    assert snapshot() == previous


def test_native_cleanup_rotation_and_record_fields(recorder_executable, tmp_path):
    old = _filename(tmp_path, OLD_GROUP, BUS, 0)
    old.write_text("old", encoding="utf-8")
    old_marker = _owner_marker(tmp_path, OLD_GROUP, BUS)
    old_marker.touch()
    older_group = "a3" * 16
    older = _filename(tmp_path, older_group, BUS, 7)
    older.write_text("older", encoding="utf-8")
    older_marker = _owner_marker(tmp_path, older_group, BUS)
    older_marker.touch()
    unmarked = _filename(tmp_path, "444", BUS, 0)
    unmarked.write_text("keep unmarked file", encoding="utf-8")
    foreign_group = "555"
    foreign = _filename(tmp_path, foreign_group, BUS, 0)
    foreign.write_text("keep same bus ID foreign file", encoding="utf-8")
    foreign_marker = _owner_marker(tmp_path, foreign_group, BUS, FOREIGN_DEVICE_UUID)
    foreign_marker.touch()
    other_bus = _filename(tmp_path, OLD_GROUP, BUS + 1, 0)
    other_bus.write_text("keep other device", encoding="utf-8")
    other_bus_marker = _owner_marker(tmp_path, OLD_GROUP, BUS + 1)
    other_bus_marker.touch()
    other_device_group = "777"
    other_device = _filename(tmp_path, other_device_group, BUS, 0)
    other_device.write_text("keep same UUID other physical device", encoding="utf-8")
    other_device_marker = _owner_marker(tmp_path, other_device_group, BUS, physical_id=PHYSICAL_DEVICE_ID + 1)
    other_device_marker.touch()
    legacy_group = "a8" * 16
    legacy = _filename(tmp_path, legacy_group, BUS, 0)
    legacy.write_text("keep legacy history", encoding="utf-8")
    legacy_marker = tmp_path / f".trmt-deepep-trace-owner-{DEVICE_UUID}-{BUS}-{legacy_group}"
    legacy_marker.touch()
    previous_bus = (1 << 31) + PHYSICAL_DEVICE_ID
    previous = _filename(tmp_path, older_group, previous_bus, 0)
    previous.write_text("keep legacy history for a different bus ID", encoding="utf-8")
    previous_marker = _owner_marker(tmp_path, older_group, previous_bus)
    previous_marker.touch()
    invalid_group = str(2**64)
    invalid = _filename(tmp_path, invalid_group, BUS, 0)
    invalid.write_text("keep file with invalid comm_id", encoding="utf-8")
    invalid_marker = _owner_marker(tmp_path, invalid_group, BUS)
    invalid_marker.touch()
    other_index = _filename(tmp_path, OLD_GROUP, BUS, 8)
    other_index.write_text("keep unrelated filename", encoding="utf-8")
    other_file = tmp_path / "user_archive.txt"
    other_file.write_text("keep archive", encoding="utf-8")
    symbolic_link = _filename(tmp_path, OLD_GROUP, BUS, 1)
    symbolic_link.symlink_to(other_file)
    matching_directory = _filename(tmp_path, OLD_GROUP, BUS, 2)
    matching_directory.mkdir()
    marked_by_symlink = _filename(tmp_path, "666", BUS, 0)
    marked_by_symlink.write_text("keep a file marked by a symlink", encoding="utf-8")
    symlink_marker = _owner_marker(tmp_path, "666", BUS)
    symlink_marker.symlink_to(other_file)

    subprocess.run([recorder_executable, tmp_path, "emit"], check=True, timeout=15)

    assert not old.exists()
    assert not old_marker.exists()
    assert not older.exists()
    assert not older_marker.exists()
    assert unmarked.read_text(encoding="utf-8") == "keep unmarked file"
    assert foreign.read_text(encoding="utf-8") == "keep same bus ID foreign file"
    assert foreign_marker.is_file()
    assert other_bus.read_text(encoding="utf-8") == "keep other device"
    assert other_bus_marker.is_file()
    assert other_device.read_text(encoding="utf-8") == "keep same UUID other physical device"
    assert other_device_marker.is_file()
    assert legacy.read_text(encoding="utf-8") == "keep legacy history"
    assert legacy_marker.is_file()
    assert previous.read_text(encoding="utf-8") == "keep legacy history for a different bus ID"
    assert previous_marker.is_file()
    assert invalid.read_text(encoding="utf-8") == "keep file with invalid comm_id"
    assert invalid_marker.is_file()
    assert other_index.read_text(encoding="utf-8") == "keep unrelated filename"
    assert other_file.read_text(encoding="utf-8") == "keep archive"
    assert symbolic_link.is_symlink()
    assert matching_directory.is_dir()
    assert marked_by_symlink.read_text(encoding="utf-8") == "keep a file marked by a symlink"
    assert symlink_marker.is_symlink()

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
        "comm_id": int(GROUP_A),
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
    assert type(records[0]["comm_id"]) is int
    assert timeout["type"] == 1 and timeout["comm_id"] == int(GROUP_B)
    assert type(timeout["comm_id"]) is int
    assert timeout.keys() == records[0].keys()
    assert all(timeout[key] is None for key in ("start_time", "end_time", "elapsed_ns", "start_tick", "end_tick"))


def test_native_recorder_skips_cleanup_for_active_device(recorder_executable, tmp_path):
    with subprocess.Popen([recorder_executable, tmp_path, "hold"], stdin=subprocess.PIPE) as owner:
        try:
            ready = tmp_path / "test-lock-ready"
            deadline = time.monotonic() + 10
            while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.is_file()
            active = _filename(tmp_path, GROUP_A, BUS, 0)
            original = active.read_bytes()
            assert json.loads(original)["comm_id"] == int(GROUP_A)
            own_marker = _owner_marker(tmp_path, GROUP_A, BUS)
            assert own_marker.is_file()
            for mode in ("contend", "contend-other-bus"):
                contender = subprocess.run(
                    [recorder_executable, tmp_path, mode], capture_output=True, text=True, check=False, timeout=10
                )
                assert contender.returncode == 1
                assert "lock trace recorder" in contender.stderr
            assert active.read_bytes() == original
            assert own_marker.is_file()
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
    files = [_filename(tmp_path, GROUP_A, BUS, i) for i in range(8)]
    assert [json.loads(line)["gid"] for line in files[0].read_bytes().splitlines()] == [1, 2]
    assert all(path.read_bytes() == b"" for path in files[1:])
    assert _owner_marker(tmp_path, GROUP_A, BUS).is_file()


def test_native_recorder_reopens_readonly_lock(recorder_executable, tmp_path):
    lock = _device_lock(tmp_path, PHYSICAL_DEVICE_ID)
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
            "comm_id": int(GROUP_A),
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
    assert a.endswith(b"\n")
    assert [json.loads(line)["gid"] for line in a.splitlines()] == [1]
    if mode == "retry-exhaust":
        b = _filename(tmp_path, GROUP_B, BUS, 0).read_bytes()
        assert b.endswith(b"\n")
        assert [json.loads(line)["gid"] for line in b.splitlines()] == [3, 6]
