"""Python monitor lifecycle and native generation-reuse regression checks."""

import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_monitor():
    spec = importlib.util.spec_from_file_location("trace_monitor_test", ROOT / "deep_ep/utils/kernel_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KernelTrace


def test_python_monitor_shared_source_and_shutdown(tmp_path, monkeypatch):
    monitor = load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", str(tmp_path))

    class Source:
        identity = 123
        synchronized = False

        def __init__(self):
            self.queue = [{"event": "launch", "gid": 1}]
            self.calls = []
            self.closed = False

        def calibration(self):
            return {"event": "calibration", "device": 0}

        def drain_trace_json(self, closing=False):
            self.calls.append(threading.current_thread().name)
            result, self.queue = self.queue, []
            if closing:
                assert self.synchronized
                result.append({"event": "summary"})
                self.closed = True
            return "".join(json.dumps(row) + "\n" for row in result)

        def synchronize(self):
            self.synchronized = True

    source = Source()

    class Native:
        @staticmethod
        def get_kernel_trace():
            return source

    try:
        first = monitor.attach(Native)
        worker = monitor._thread
        second = monitor.attach(Native)
        assert first == second and monitor._thread is worker
        deadline = time.monotonic() + 2
        while not source.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        assert "deepep-kernel-trace" in source.calls
        monitor.detach(first)
        assert not source.closed
        source.queue.append({"event": "end", "gid": 1})
        monitor.detach(second)
        assert source.closed
        files = list(tmp_path.glob("kernel-*.jsonl"))
        assert len(files) == 1
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert [row["event"] for row in rows] == ["calibration", "launch", "end", "summary"]
    finally:
        monitor.shutdown()
    assert not worker.is_alive()


@pytest.mark.parametrize("windowed", [False, True])
def test_record_only_defers_all_collection_until_detach(tmp_path, monkeypatch, windowed):
    monitor = load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", "record_only")
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "0")

    class Source:
        identity = 987
        calls = 0
        synced = False

        def calibration(self):
            return dict(event="calibration", device=0, slots=4096)

        def synchronize(self):
            self.synced = True

        def drain_trace_json(self, closing=False):
            assert self.synced
            self.calls += 1
            return '{"event":"summary","completed":300,"dropped":0,"incomplete":0}\n' if closing else ""

    source = Source()
    native = type("Native", (), {"get_kernel_trace": staticmethod(lambda: source)})
    try:
        key = monitor.attach(native)
        entry = monitor._sources[key]
        assert monitor._thread is None
        for _ in range(256):
            monitor._drain(entry)  # Also protects deferred sources if another source owns a poller.
        assert source.calls == entry["deferred_drains"] == 0
        assert len(next(tmp_path.glob("kernel-*.jsonl")).read_text().splitlines()) == 1
        if windowed:
            for _ in range(25):
                source.synchronize()
                monitor.collect_boundary(key)
                monitor._drain(entry)  # A normal poll still cannot drain a deferred source.
            assert source.calls == entry["deferred_drains"] == 25
        monitor.detach(key)
        assert source.calls == entry["deferred_drains"] == (26 if windowed else 1)
        assert not list(tmp_path.glob("diagnostic-*.json"))
        assert json.loads(next(tmp_path.glob("kernel-*.jsonl")).read_text().splitlines()[-1])["completed"] == 300
    finally:
        monitor.shutdown()


def test_native_generation_reuse(tmp_path, monkeypatch):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required to execute native Host protocol test")
    acl = tmp_path / "acl"
    acl.mkdir()
    (acl / "acl.h").write_text("""
#pragma once
#include <cstdint>
#include <cstdlib>
using aclError=int;
using aclrtStream=void*;
using aclrtContext=void*;
constexpr int ACL_SUCCESS=0, ACL_HOST_REGISTER_MAPPED=1;
inline int aclrtGetCurrentContext(void** p){*p=(void*)1;return 0;}
inline int aclrtSetCurrentContext(void*){return 0;}
inline int aclrtGetDevice(int* p){*p=0;return 0;}
inline int aclrtMallocHost(void** p,size_t n){*p=calloc(1,n);return !*p;}
inline int aclrtFreeHost(void* p){free(p);return 0;}
inline int aclrtHostRegister(void* p,size_t,int,void** q){*q=p;return 0;}
inline int aclrtHostUnregister(void*){return 0;}
inline int aclrtCreateStream(void** p){*p=(void*)1;return 0;}
inline int aclrtDestroyStream(void*){return 0;}
inline int aclrtSynchronizeStream(void*){return 0;}
inline int aclrtSynchronizeDevice(){return 0;}
""")
    source = tmp_path / "test.cpp"
    source.write_text(r"""
#include "runtime/kernel_trace.h"
#include "common/kernel_trace_layout.h"
#include <cassert>
#include <thread>
#include <iostream>
#include <stdexcept>
using namespace ascend_deepep::trace;
int gil_calls=0;
int fake_gil(){++gil_calls; return 1;}
extern "C" void kernel_trace_clock_do(void*,uint8_t* p){*reinterpret_cast<uint64_t*>(p)=100;}
int main(){
    using P=DispatchProbePoint;
    ConfigureHostDispatchProbe(0, &fake_gil);
    {HostDispatchProbe p(0,false); p.Mark(P::WaitBegin); p.Finish();}
    assert(gil_calls==0 && TakeHostDispatchProbe().samples.empty());
    ConfigureHostDispatchProbe(2, &fake_gil);
    for (int i=0;i<3;++i) {
        HostDispatchProbe p(0,false);
        assert(CurrentHostDispatchProbe()==&p);
        p.SetWaitPath(1); p.Mark(P::WaitBegin); p.Mark(P::FirstPending); p.Mark(P::WaitEnd); p.Finish();
    }
    assert(!CurrentHostDispatchProbe());
    auto host=TakeHostDispatchProbe();
    assert(host.samples.size()==2 && host.dropped==1);
    assert(host.samples[0].gil[static_cast<unsigned>(P::FirstPending)]==1);
    assert(host.samples[0].ticks[static_cast<unsigned>(P::WaitEnd)]>=host.samples[0].ticks[static_cast<unsigned>(P::WaitBegin)]);
    assert(host.samples[0].success && host.samples[0].wait_path==1);
    ConfigureHostDispatchProbe(1, &fake_gil);
    try {HostDispatchProbe p(0,false); p.Finish(); throw std::runtime_error("test");} catch (...) {}
    assert(!TakeHostDispatchProbe().samples[0].success && !CurrentHostDispatchProbe());
    setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", "metadata", 1);
    assert(!KernelTrace::Open()); // No collector/calibration for Host-only metadata.
    unsetenv("DEEPEP_TRACE_DIAGNOSTIC_MODE");
    auto sink=KernelTrace::Open(); assert(sink);
    TraceMetadata outer;
    outer.values["api_call_id"] = 999;
    outer.labels["inherited"] = "parent";
    ExchangeTraceMetadata(std::make_shared<const TraceMetadata>(outer));
    std::thread other([] {
        assert(!CurrentTraceMetadata());
        TraceMetadata local; local.values["api_call_id"] = 123;
        ScopedTraceMetadata scope(std::make_shared<const TraceMetadata>(local));
        assert(CurrentTraceMetadata()->values.at("api_call_id") == 123);
    });
    other.join();
    assert(CurrentTraceMetadata()->values.at("api_call_id") == 999);
    const char* ops[] = {"notify_dispatch", "dispatch", "cached_notify_dispatch", "cached_notify_combine", "combine"};
    const char* kernels[] = {"notify_dispatch_kernel", "v1_dispatch_udma_kernel", "cached_notify_dispatch_kernel",
                             "cached_notify_combine_kernel", "v1_combine_udma_reduce_kernel"};
    for(int round=0;round<384;++round) {
        const int kind = round % 5;
        TraceMetadata call;
        call.parent = CurrentTraceMetadata();
        call.values["api_call_id"] = round + 1;
        call.labels["comm_id"] = "test-comm";
        KernelTrace::Ticket ticket;
        {
            ScopedTraceMetadata scope(std::make_shared<const TraceMetadata>(call));
            ticket=sink->Reserve(0,0,2,ops[kind],kernels[kind],12345); assert(ticket.address);
        }
        assert(CurrentTraceMetadata()->values.at("api_call_id") == 999);
        sink->Submit(ticket,1000,0);
        auto* bytes=reinterpret_cast<uint8_t*>(ticket.address);
        auto store=[&](int offset,uint64_t value){
            __atomic_store_n(reinterpret_cast<uint64_t*>(bytes+offset),value,__ATOMIC_RELEASE);
        };
        // Old nonzero values must never complete the new generation.
        for(int core=0;core<2;++core) {
            store(core*kCoreBytes,900);
            store(core*kCoreBytes+8,999);
            store(core*kCoreBytes+kLineBytes,ticket.sequence-1);
        }
        auto launch=sink->Drain(); assert(launch.size()==1 && launch[0].event=="launch");
        store(0,0); store(8,30); store(kLineBytes,ticket.sequence);
        assert(sink->Drain().empty()); // Second AIV still belongs to an old generation.
        store(kCoreBytes,5);store(kCoreBytes+8,40);
        store(kCoreBytes+kLineBytes,ticket.sequence);
        auto done=sink->Drain();assert(done.size()==1 && done[0].event=="end");
        assert(done[0].op_name==ops[kind] && done[0].kernel_name==kernels[kind]);
        assert(done[0].values.at("api_call_id")==round+1);
        assert(done[0].labels.at("comm_id")=="test-comm");
        assert(done[0].labels.at("inherited")=="parent");
        assert(done[0].labels.at("stream")=="12345");
        assert(done[0].values.at("gid")==static_cast<int64_t>(ticket.sequence));
        assert(done[0].values.at("elapsed_ns")==40); // Zero start tick is valid.
    }
    auto summary=sink->Drain(true);
    assert(summary.back().values.at("completed")==384);
    assert(summary.back().values.at("dropped")==0);
    sink.reset();
    sink=KernelTrace::Open(); assert(sink);
    assert(sink->Calibration().values.at("slots")==128);
    for (int window=0; window<5; ++window) {
        for (int i=0; i<120; ++i) {
            auto t=sink->Reserve(0,1,1,"dispatch","kernel"); assert(t.address);
            sink->Submit(t,1000,0);
            auto* p=reinterpret_cast<uint64_t*>(t.address);
            __atomic_store_n(p,uint64_t{10},__ATOMIC_RELEASE);
            __atomic_store_n(p+1,uint64_t{20},__ATOMIC_RELEASE);
            __atomic_store_n(reinterpret_cast<uint64_t*>(t.address+kLineBytes),t.sequence,__ATOMIC_RELEASE);
        }
        assert(sink->Drain().size()==240); // Collect only at the boundary; wrap the 128-slot ring next window.
    }
    auto window_summary=sink->Drain(true);
    assert(window_summary.back().values.at("completed")==600);
    assert(window_summary.back().values.at("dropped")==0);
    assert(window_summary.back().values.at("incomplete")==0);
    sink.reset();
    setenv("DEEPEP_TRACE_SLOTS", "512", 1);
    sink=KernelTrace::Open(); assert(sink);
    assert(sink->Calibration().values.at("slots")==512);
    // More than the default 128 tickets must survive with no intermediate Drain.
    for (int i=0; i<300; ++i) {
        auto t=sink->Reserve(0,1,1,"dispatch","kernel"); assert(t.address);
        sink->Submit(t,1000,0);
        auto* p=reinterpret_cast<uint64_t*>(t.address);
        __atomic_store_n(p,uint64_t{10},__ATOMIC_RELEASE);
        __atomic_store_n(p+1,uint64_t{20},__ATOMIC_RELEASE);
        __atomic_store_n(reinterpret_cast<uint64_t*>(t.address+kLineBytes),t.sequence,__ATOMIC_RELEASE);
    }
    auto deferred=sink->Drain(true);
    assert(deferred.size()==601);
    assert(deferred.back().values.at("completed")==300);
    assert(deferred.back().values.at("dropped")==0);
    assert(deferred.back().values.at("incomplete")==0);
    sink.reset();
    for (const char* invalid : {"", "127", "16385", "-1", "512oops"}) {
        setenv("DEEPEP_TRACE_SLOTS", invalid, 1);
        assert(!KernelTrace::Open());
    }
    TraceRecord encoded{"end", "dispatch", "kernel", {{"large", 9223372036854775807LL}},
        {{"escaped", "quote\"\\\n\t"}}};
    std::cout << EncodeTraceRecords({encoded});
}
""")
    binary = tmp_path / "test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-DASCEND_DEEPEP_ARCH=3510",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "csrc/direct"),
            str(ROOT / "csrc/direct/runtime/kernel_trace.cpp"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    monkeypatch.setenv("DEEPEP_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("DEEPEP_TRACE_SLOTS", raising=False)
    result = subprocess.run([str(binary)], check=True, timeout=20, capture_output=True, text=True)
    encoded = json.loads(result.stdout)
    assert encoded["large"] == 9223372036854775807
    assert encoded["escaped"] == 'quote"\\\n\t'


@pytest.fixture
def mixed_trace(tmp_path):
    spec = importlib.util.spec_from_file_location("trace_checker_test", ROOT / "scripts/check_kernel_trace.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    rows = [{"event": "calibration", "tick_hz": 1_000_000_000, "tick": 10, "offset_ns": 1000}]
    ends = []
    kernels = list(checker.KERNEL_OPS.items())
    # Multiple operation types, varying AIV counts, and interleaved completions.
    for i in range(130):
        name, op = kernels[i % len(kernels)]
        common = dict(
            gid=i + 1,
            rank=0,
            local_instance=0,
            launch_ns=1000 + i,
            op_name=op,
            kernel_name=name,
            aiv=16 if i % 2 else 32,
        )
        rows.append(dict(common, event="launch", status=0))
        ends.append(
            dict(
                common, event="end", start_tick=i, end_tick=i + 100, start_ns=i + 1000, end_ns=i + 1100, elapsed_ns=100
            )
        )
        if len(ends) == 2:
            rows.extend(reversed(ends))
            ends.clear()
    rows.append(dict(event="summary", completed=130, incomplete=0, dropped=0))
    path = tmp_path / "kernel-mixed.jsonl"
    return checker, path, rows


def test_python_monitor_mixed_kernels_and_checker(mixed_trace, monkeypatch):
    checker, path, rows = mixed_trace
    monitor = load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", str(path.parent))
    written = threading.Event()

    class Source:
        identity = 456
        pending = rows[1:-1]

        def calibration(self):
            return dict(rows[0], device=0)

        def drain_trace_json(self, closing=False):
            result, self.pending = self.pending, []
            if result:
                written.set()
            if closing:
                result.append(rows[-1])
            return "".join(json.dumps(row) + "\n" for row in result)

        def synchronize(self):
            pass

    class Native:
        @staticmethod
        def get_kernel_trace():
            return source

    source = Source()
    try:
        key = monitor.attach(Native)
        assert written.wait(2), "background monitor did not collect records"
        # Lock acquisition also waits for the current write/flush to finish.
        with monitor._lock:
            log = next(path.parent.glob("kernel-*.jsonl"))
            assert len(log.read_text().splitlines()) == len(rows) - 1
        monitor.detach(key)
        assert checker.check(path.parent, 1, set(checker.KERNEL_OPS.values()), require_reuse=True) == 130
    finally:
        monitor.shutdown()


def test_checker_uses_actual_capacity_for_reuse(mixed_trace):
    checker, path, rows = mixed_trace
    rows[0]["slots"] = 512
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert checker.check(path.parent, 1) == 130
    with pytest.raises(ValueError, match="more than 512"):
        checker.check(path.parent, 1, require_reuse=True)


@pytest.mark.parametrize("corruption", ["kernel", "aiv", "gid", "time", "dropped", "missing_op"])
def test_checker_rejects_mixed_trace_corruption(mixed_trace, corruption):
    checker, path, rows = mixed_trace
    end = next(row for row in rows if row["event"] == "end")
    if corruption == "kernel":
        end["kernel_name"] = "notify_dispatch_kernel"
    elif corruption == "aiv":
        end["aiv"] = 64
    elif corruption == "gid":
        end["gid"] = 10000
    elif corruption == "time":
        end["elapsed_ns"] += 1
    elif corruption == "dropped":
        rows[-1]["dropped"] = 1
    else:
        rows = [row for row in rows if row.get("op_name") != "combine"]
        rows[-1]["completed"] = sum(row["event"] == "end" for row in rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        checker.check(path.parent, 1, require_ops=["notify_dispatch", "dispatch", "combine"])


class MetadataTensor:
    dtype = "torch.bfloat16"

    def __init__(self, rows, cols):
        self.shape = (rows, cols)

    def size(self, dim):
        return self.shape[dim]

    def element_size(self):
        return 2

    def cpu(self):
        raise AssertionError("metadata collection must not read the device")


class MetadataNative:
    kernel_trace_diagnostic_version = 4

    def __init__(self):
        self.context = threading.local()
        self.templates = 0

    def make_trace_metadata(self, fields, parent=None):
        self.templates += 1
        return dict(parent or {}, **fields)

    def push_trace_metadata(self, template, api, handle):
        previous = getattr(self.context, "fields", {})
        self.context.fields = dict(template, api_call_id=api, handle_id=handle)
        return previous

    def restore_trace_metadata(self, previous):
        self.context.fields = previous


def load_trace_module():
    spec = importlib.util.spec_from_file_location("trace_metadata_test", ROOT / "deep_ep/utils/kernel_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["", "record_only"])
def test_call_and_handle_metadata_scopes(monkeypatch, mode):
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", mode)
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "0")
    module = load_trace_module()
    native = MetadataNative()
    session = module.KernelTraceSession(native, "comm-a", 4, 12)
    assert session.diagnostic_mode == "" and not session.diagnostic_timing
    routes = MetadataTensor(32, 8)
    x = MetadataTensor(32, 6144)
    handle = type("Handle", (), {})()
    with session.call("dispatch", x, routes, 128, capacity=64) as fresh:
        assert fresh["topk"] == 8 and fresh["global_rank"] == 12
        handle._trace_handle_id = fresh["handle_id"]
        snapshot = dict(native.context.fields)
        with pytest.raises(RuntimeError):
            with session.call("combine", MetadataTensor(90, 6144), routes, 128, handle, capacity=64) as combine:
                assert combine["api_call_id"] != fresh["api_call_id"]
                assert combine["handle_id"] == fresh["handle_id"]
                assert combine["num_tokens"] == 32 and combine["api_input_rows"] == 90
                assert combine["topk"] == 8  # No weights needed to preserve routing K.
                raise RuntimeError("launch failed")
        assert native.context.fields == snapshot
    assert native.context.fields == {}
    with session.call("dispatch", x, routes, 128, handle, cached=True) as cached:
        assert cached["handle_id"] == fresh["handle_id"] and cached["cached"] == 1
        assert cached["api_call_id"] != fresh["api_call_id"]
    with session.call("dispatch", x, routes, 128) as another:
        assert another["handle_id"] != fresh["handle_id"]
    assert snapshot["api_name"] == "dispatch"


def test_metadata_thread_isolation():
    module = load_trace_module()
    native = MetadataNative()
    session = module.KernelTraceSession(native, "comm", 4)
    barrier = threading.Barrier(2)
    results = []

    def run(rows):
        with session.call("dispatch", MetadataTensor(rows, 128), MetadataTensor(rows, 8), 128) as fields:
            barrier.wait(timeout=2)
            assert native.context.fields == fields
            results.append(dict(fields))
        assert native.context.fields == {}

    threads = [threading.Thread(target=run, args=(rows,)) for rows in (32, 64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert len(results) == 2
    assert results[0]["api_call_id"] != results[1]["api_call_id"]
    assert {row["num_tokens"] for row in results} == {32, 64}


def test_metadata_cache_shape_changes_and_eviction():
    module = load_trace_module()
    native = MetadataNative()
    session = module.KernelTraceSession(native, "comm", 4)
    routes = MetadataTensor(32, 8)
    with session.call("dispatch", MetadataTensor(32, 128), routes, 128):
        saved = native.context.fields
        for _ in range(20):
            with session.call("dispatch", MetadataTensor(32, 128), routes, 128):
                pass
        assert native.templates == 2  # One Buffer base and one reusable shape template.
        for size in range(65):
            with session.call("combine", MetadataTensor(size + 1, 256), routes, 128) as fields:
                assert fields["api_input_rows"] == size + 1
                assert fields["hidden_size"] == 256
        assert len(session._templates) <= 64
        assert native.context.fields is saved
        assert saved["hidden_size"] == 128
    assert native.context.fields == {}


def test_trace_group_initialization(monkeypatch):
    module = load_trace_module()
    native = MetadataNative()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", "test")
    monkeypatch.setattr(module.KernelTrace, "attach", lambda native: 456)
    monkeypatch.setattr(module, "_global_rank", lambda: 9)
    proposals = []

    def gather(group, comm, size, value):
        proposals.append(value)
        return [(True, "rank0-uuid", "", False), (True, None, "", False)]

    first = module.attach_trace_session(native, None, 0, 2, gather)[1]
    second = module.attach_trace_session(native, None, 1, 2, gather)[1]
    assert first.fields["comm_id"] == second.fields["comm_id"] == "rank0-uuid"
    assert first.fields["buffer_id"] != second.fields["buffer_id"]
    assert proposals[0][1] and proposals[1][1] is None
    monkeypatch.delenv("DEEPEP_TRACE_DIR")
    with pytest.raises(ValueError, match="all EP ranks"):
        module.attach_trace_session(native, None, 1, 2, gather)
    assert module.attach_trace_session(native, None, 0, 2, lambda *args: [(False, None, "", False)] * 2) == (None, None)


def test_metadata_only_and_mode_consensus(monkeypatch):
    module = load_trace_module()
    native = MetadataNative()
    native.kernel_trace_diagnostic_version = 4
    monkeypatch.setenv("DEEPEP_TRACE_DIR", "unused")
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", "metadata")
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "1")
    monkeypatch.setattr(module, "_global_rank", lambda: 0)
    monkeypatch.setattr(module.KernelTrace, "attach", lambda *_: pytest.fail("metadata mode started collector"))
    identity, session = module.attach_trace_session(native, None, 0, 1, lambda *items: [items[-1]])
    assert identity is None and session is not None
    with session.call("dispatch", MetadataTensor(32, 128), MetadataTensor(32, 8), 32):
        pass
    stats = session.diagnostic_stats["fresh_dispatch"]
    assert stats["calls"] == 1 and stats["enter_ns"] > 0 and stats["restore_ns"] > 0
    with pytest.raises(ValueError, match="all EP ranks"):
        module.attach_trace_session(native, None, 0, 1, lambda *items: [(True, "comm", "full", True)])
    native.kernel_trace_diagnostic_version = 3  # Old extension lacks configurable trace capacity.
    with pytest.raises(RuntimeError, match="matching rebuilt native"):
        module.attach_trace_session(native, None, 0, 1, lambda *items: [items[-1]])


@pytest.mark.parametrize("mode,timing", [("bogus", "0"), ("", "1"), ("full", "yes"), ("record_only", "1")])
def test_reject_invalid_diagnostic_settings(monkeypatch, mode, timing):
    module = load_trace_module()
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", mode)
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", timing)
    with pytest.raises(ValueError):
        module.diagnostic_config()


@pytest.mark.parametrize("mode", ["native", "json_discard", "json", "full"])
@pytest.mark.parametrize("timing", [False, True])
def test_diagnostic_monitor_recycling_and_output(tmp_path, monkeypatch, mode, timing):
    monitor = load_monitor()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", mode)
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", str(int(timing)))

    class Source:
        identity = 891
        recycled = 0

        def calibration(self):
            return dict(event="calibration", device=0)

        def synchronize(self):
            pass

        def drain_trace_diagnostic(self, include_records, timed, closing, return_text):
            assert include_records == (mode != "native") and timed == timing
            assert return_text == (mode != "json_discard")
            self.recycled += 1
            records = [dict(event="end", gid=self.recycled)]
            return dict(
                text="".join(json.dumps(row) + "\n" for row in records) if include_records and return_text else "",
                serialize_ns=1 if timed and include_records else 0,
                polls=1,
                records=1,
                ends=1,
                summaries=int(closing),
                completed=self.recycled if closing else 0,
                incomplete=0,
                dropped=0,
                drain_ns=1 if timing else 0,
                convert_ns=1 if timing else 0,
            )

    source = Source()
    native = type("Native", (), {"get_kernel_trace": staticmethod(lambda: source)})
    try:
        key = monitor.attach(native)
        with monitor._lock:
            for _ in range(129):
                monitor._drain(monitor._sources[key])
        monitor.detach(key)
        report = json.loads(next(tmp_path.glob("diagnostic-*.json")).read_text())
        assert not report["failed"] and report["mode"] == mode
        assert report["counters"]["completed"] == source.recycled > 128
        assert report["counters"]["summaries"] == 1
        assert bool(list(tmp_path.glob("kernel-*.jsonl"))) == (mode == "full")
        if timing and mode != "native":
            assert report["counters"]["serialize_ns"] > 0
        if timing and mode == "full":
            assert report["counters"]["write_flush_ns"] > 0
    finally:
        monitor.shutdown()


@pytest.mark.parametrize("corruption", [None, "missing", "end", "api", "handle"])
def test_checker_host_metadata(mixed_trace, corruption):
    checker, path, rows = mixed_trace
    for row in rows:
        if row["event"] not in ("launch", "end"):
            continue
        row.update(
            metadata_version=1,
            comm_id="comm",
            buffer_id=1,
            process_id="process",
            hostname="host",
            ep_size=1,
            api_name="combine" if "combine" in row["op_name"] else "dispatch",
            api_call_id=row["gid"],
            handle_id=1,
            hidden_size=6144,
            element_bytes=2,
            dtype="torch.bfloat16",
            num_experts=32,
            topk=8,
            num_tokens=2048,
            api_input_rows=2048,
            cached=0,
            do_expand=0,
            token_capacity=4096,
            pid=100,
            device=0,
            stream="12345",
        )
        if row["op_name"] in ("dispatch", "combine"):
            row.update(kernel_input_rows=2048, kernel_output_rows=2048)
    first = next(row for row in rows if row["event"] == "launch")
    if corruption == "missing":
        first.pop("topk")
    elif corruption == "end":
        next(row for row in rows if row["event"] == "end")["comm_id"] = "other"
    elif corruption in ("api", "handle"):
        for row in rows:
            if row.get("gid") == 2:
                if corruption == "api":
                    row["api_call_id"] = 1
                    row["api_input_rows"] = 100
                else:
                    row["topk"] = 4
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if corruption:
        with pytest.raises(ValueError):
            checker.check(path.parent, 1, require_metadata=True)
    else:
        assert checker.check(path.parent, 1, require_metadata=True) == 130


PERF_OPS = ("fresh_dispatch", "cached_dispatch", "combine_unweighted", "combine_weighted")

HOST_DISPATCH_POINTS = (
    "native_entry",
    "notify_begin",
    "notify_end",
    "wait_begin",
    "first_pending",
    "first_count_ready",
    "all_counts_ready",
    "wait_end",
    "outputs_begin",
    "outputs_end",
    "dispatch_begin",
    "dispatch_end",
    "native_exit",
)
HOST_DISPATCH_INTERVALS = {
    "native_prepare": ("native_entry", "notify_begin"),
    "notify_launch_call": ("notify_begin", "notify_end"),
    "counts_wait": ("wait_begin", "wait_end"),
    "first_count_wait": ("wait_begin", "first_count_ready"),
    "remaining_counts": ("first_count_ready", "all_counts_ready"),
    "counts_postprocess": ("wait_end", "outputs_begin"),
    "outputs_prepare": ("outputs_begin", "outputs_end"),
    "wait_end_to_dispatch": ("wait_end", "dispatch_begin"),
    "dispatch_launch_call": ("dispatch_begin", "dispatch_end"),
    "native_scope": ("native_entry", "native_exit"),
}


def validate_host_dispatch_report(report, args, rank):
    per_kind = 1 + args.warmup + args.windows * args.iterations
    rows = report["samples"]
    control = report.get("gil_control")
    if control not in (dict(held_control=1, released_control=0), dict(held_control=-1, released_control=-1)):
        raise ValueError("Host GIL probe controls failed")
    supported = control["held_control"] == 1
    if report.get("version") != 1 or report.get("dropped") != 0 or len(rows) != 2 * per_kind:
        raise ValueError("incomplete Host dispatch probe")
    seen = [0, 0]
    for ordinal, row in enumerate(rows, 1):
        cached = row["cached"]
        if row["ordinal"] != ordinal or row["rank"] != rank or cached not in (0, 1) or not row["success"]:
            raise ValueError("invalid Host dispatch sample identity")
        ticks, gil = row["steady_ns"], row["gil_held"]
        if set(ticks) != set(HOST_DISPATCH_POINTS) or set(gil) != set(HOST_DISPATCH_POINTS):
            raise ValueError("missing Host dispatch points")
        for point in HOST_DISPATCH_POINTS:
            if type(ticks[point]) is not int or ticks[point] < -1:
                raise ValueError("invalid Host dispatch timestamp")
            valid_gil = (0, 1) if supported and ticks[point] >= 0 else (-1,)
            if gil[point] not in valid_gil:
                raise ValueError("invalid Host GIL observation")
        path = row["wait_path"]
        if (cached and path != 4) or (not cached and path not in (1, 2)):
            raise ValueError("unexpected Host count wait path")
        ordered = ["native_entry", "notify_begin", "notify_end"]
        if not cached:
            ordered += ["wait_begin"]
            if path == 1:
                ordered += ["first_count_ready", "all_counts_ready"]
            ordered += ["wait_end"]
        ordered += ["outputs_begin", "outputs_end", "dispatch_begin", "dispatch_end", "native_exit"]
        values = [ticks[p] for p in ordered]
        if any(v < 0 for v in values) or values != sorted(values):
            raise ValueError("unordered Host dispatch phases")
        for point in ("wait_begin", "wait_end", "first_pending", "first_count_ready", "all_counts_ready"):
            if (cached or (path == 2 and point not in ("wait_begin", "wait_end"))) and ticks[point] != -1:
                raise ValueError("unexpected Host wait observation")
        if (
            ticks["first_pending"] >= 0
            and not ticks["wait_begin"] <= ticks["first_pending"] <= ticks["all_counts_ready"]
        ):
            raise ValueError("pending observation outside count wait")
        index = seen[cached]
        seen[cached] += 1
        row["kind"] = "cached_dispatch" if cached else "fresh_dispatch"
        row["phase"] = "precision" if index == 0 else "warmup" if index <= args.warmup else "steady"
        row["window"] = (index - 1 - args.warmup) // args.iterations if row["phase"] == "steady" else -1
        row["iteration"] = (index - 1 - args.warmup) % args.iterations if row["phase"] == "steady" else -1
        row["durations_ns"] = {
            name: ticks[b] - ticks[a]
            for name, (a, b) in HOST_DISPATCH_INTERVALS.items()
            if ticks[a] >= 0 and ticks[b] >= 0
        }
    if seen != [per_kind, per_kind]:
        raise ValueError("Host dispatch call count mismatch")
    summary = {}
    for kind in PERF_OPS[:2]:
        steady = [r for r in rows if r["kind"] == kind and r["phase"] == "steady"]
        durations = {}
        for name in HOST_DISPATCH_INTERVALS:
            values = sorted(r["durations_ns"][name] for r in steady if name in r["durations_ns"])
            if values:
                durations[name] = dict(
                    count=len(values), median_ns=statistics.median(values), min_ns=values[0], max_ns=values[-1]
                )
        summary[kind] = dict(
            durations=durations,
            gil_observations={
                point: {
                    str(state): sum(r["gil_held"][point] == state and r["steady_ns"][point] >= 0 for r in steady)
                    for state in (-1, 0, 1)
                }
                for point in HOST_DISPATCH_POINTS
            },
        )
    report["steady_summary"] = summary
    report["clock"] = "C++ steady_clock nanoseconds; not CPU wall time or NPU time"
    return report


def make_host_probe_report(args, rank=0, path=1, gil_state=1):
    per_kind = 1 + args.warmup + args.windows * args.iterations
    rows = []
    for cached in (0, 1):
        for _ in range(per_kind):
            ordinal = len(rows) + 1
            ticks = {name: ordinal * 1000 + i * 10 for i, name in enumerate(HOST_DISPATCH_POINTS)}
            for point in ("wait_begin", "wait_end", "first_pending", "first_count_ready", "all_counts_ready"):
                if cached or (path == 2 and point not in ("wait_begin", "wait_end")):
                    ticks[point] = -1
            rows.append(
                dict(
                    ordinal=ordinal,
                    rank=rank,
                    cached=cached,
                    success=True,
                    api_call_id=ordinal,
                    wait_path=4 if cached else path,
                    steady_ns=ticks,
                    gil_held={p: gil_state if ticks[p] >= 0 else -1 for p in HOST_DISPATCH_POINTS},
                )
            )
    return dict(
        version=1,
        dropped=0,
        samples=rows,
        gil_control=dict(held_control=1, released_control=0)
        if gil_state != -1
        else dict(held_control=-1, released_control=-1),
    )


@pytest.mark.parametrize("path", [1, 2])
@pytest.mark.parametrize("gil_state", [0, 1, -1])
def test_host_dispatch_report_phases_and_gil(path, gil_state):
    args = npu_parser().parse_args(["--output", "unused", "--windows", "5", "--iterations", "60"])
    report = validate_host_dispatch_report(make_host_probe_report(args, path=path, gil_state=gil_state), args, 0)
    fresh = report["steady_summary"]["fresh_dispatch"]
    assert fresh["durations"]["counts_wait"]["count"] == 300
    assert fresh["durations"]["counts_wait"]["median_ns"] == 40
    assert fresh["gil_observations"]["wait_begin"][str(gil_state)] == 300
    assert "counts_wait" not in report["steady_summary"]["cached_dispatch"]["durations"]
    steady = [r for r in report["samples"] if r["phase"] == "steady" and not r["cached"]]
    assert [(r["window"], r["iteration"]) for r in steady] == [(w, i) for w in range(5) for i in range(60)]


@pytest.mark.parametrize("corruption", ["control", "dropped", "missing", "order", "gil", "path", "cached_wait"])
def test_host_dispatch_report_rejects_invalid_evidence(corruption):
    args = npu_parser().parse_args(["--output", "unused", "--windows", "1", "--iterations", "32"])
    report = make_host_probe_report(args)
    row = report["samples"][0]
    if corruption == "control":
        report["gil_control"]["released_control"] = 1
    elif corruption == "dropped":
        report["dropped"] = 1
    elif corruption == "missing":
        report["samples"].pop()
    elif corruption == "order":
        row["steady_ns"]["wait_end"] = 0
    elif corruption == "gil":
        row["gil_held"]["wait_begin"] = 2
    elif corruption == "path":
        row["wait_path"] = 3
    else:
        row = report["samples"][-1]
        row["steady_ns"]["wait_begin"] = row["steady_ns"]["notify_end"]
        row["gil_held"]["wait_begin"] = 1
    with pytest.raises(ValueError):
        validate_host_dispatch_report(report, args, 0)


@pytest.mark.parametrize("pending", ["absent", "after_total"])
def test_host_dispatch_pending_is_an_observation(pending):
    args = npu_parser().parse_args(["--output", "unused", "--windows", "1", "--iterations", "32"])
    report = make_host_probe_report(args)
    for row in report["samples"]:
        if not row["cached"]:
            row["steady_ns"]["first_pending"] = -1 if pending == "absent" else row["steady_ns"]["first_count_ready"] + 1
            row["gil_held"]["first_pending"] = -1 if pending == "absent" else 1
    validate_host_dispatch_report(report, args, 0)


@pytest.mark.parametrize("extra", [[], ["--diagnostic", "--diagnostic-timing"]])
def test_host_dispatch_timing_requires_separate_diagnostic_run(tmp_path, extra):
    args = npu_parser().parse_args(["--host-dispatch-timing", "--output", str(tmp_path / "unused"), *extra])
    with pytest.raises(ValueError, match="--host-dispatch-timing requires"):
        run_npu_acceptance(args)
    assert not args.output.exists()


def overhead_result(off, on):
    """Paired, process-level samples; inner windows are not independent trials."""
    import math

    if len(off) != len(on) or len(off) < 6:
        raise ValueError("at least six matched process pairs are required")
    if any(not math.isfinite(v) or v <= 0 for v in (*off, *on)):
        raise ValueError("timings must be finite and positive")
    ratios = [100 * (b / a - 1) for a, b in zip(off, on)]
    rng = random.Random(20260823)
    estimates = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    low, high = estimates[249], estimates[9749]
    return dict(
        status="PASS" if high < 1 else "FAIL" if low >= 1 else "INCONCLUSIVE",
        paired_overhead_percent=ratios,
        median_percent=statistics.median(ratios),
        bootstrap_95_percent=[low, high],
    )


@pytest.mark.parametrize("on,status", [([99] * 6, "PASS"), ([103] * 6, "FAIL"), ([96, 104] * 3, "INCONCLUSIVE")])
def test_overhead_acceptance(on, status):
    assert overhead_result([100] * 6, on)["status"] == status


@pytest.mark.parametrize("off,on", [([100] * 5, [100] * 5), ([0] * 6, [100] * 6), ([100] * 6, [float("nan")] * 6)])
def test_overhead_rejects_invalid_samples(off, on):
    with pytest.raises(ValueError):
        overhead_result(off, on)


def validate_npu_metadata(directory, manifests, args):
    spec = importlib.util.spec_from_file_location("trace_checker_npu", ROOT / "scripts/check_kernel_trace.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.check(
        directory,
        args.ep_size,
        ("notify_dispatch", "cached_notify_dispatch", "dispatch", "combine"),
        8 * (1 + args.warmup + args.windows * args.iterations) > (args.trace_slots or 128),
        True,
    )
    calls = {}
    stages = {}
    for path in directory.glob("kernel-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "end":
                continue
            rank = row["rank"]
            manifest = manifests[rank]
            expected = dict(
                num_tokens=args.num_tokens,
                hidden_size=args.hidden,
                topk=args.num_topk,
                num_experts=args.ep_size * args.experts_per_rank,
                dtype="torch.bfloat16",
                element_bytes=2,
                do_expand=0,
                token_capacity=args.num_tokens,
                ep_size=args.ep_size,
                global_rank=rank,
                device=args.first_device + rank,
                pid=manifest["pid"],
            )
            is_combine = row["api_name"] == "combine"
            expected["api_input_rows"] = manifest["recv_rows"] if is_combine else args.num_tokens
            if row["op_name"] in ("dispatch", "combine"):
                expected["kernel_input_rows"] = expected["api_input_rows"]
                expected["kernel_output_rows"] = args.num_tokens if is_combine else manifest["recv_rows"]
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ValueError(f"rank {rank} gid {row['gid']}: {key}={row.get(key)!r}, expected {value!r}")
            if (is_combine or row["cached"]) and row["handle_id"] != manifest["handle_id"]:
                raise ValueError("cached dispatch/combine lost the original fresh handle association")
            calls[rank, row["api_call_id"]] = row
            stages.setdefault((rank, row["api_call_id"]), set()).add(row["op_name"])
    for key, row in calls.items():
        required = (
            {"combine"}
            if row["api_name"] == "combine"
            else {"dispatch", "cached_notify_dispatch" if row["cached"] else "notify_dispatch"}
        )
        if not required <= stages[key]:
            raise ValueError(f"rank/API {key}: missing kernel stages {required - stages[key]}")
    per_kind = 1 + args.warmup + args.windows * args.iterations
    for rank in range(args.ep_size):
        rows = [row for (r, _), row in calls.items() if r == rank]
        counts = {
            "fresh": sum(row["api_name"] == "dispatch" and not row["cached"] for row in rows),
            "cached": sum(row["api_name"] == "dispatch" and bool(row["cached"]) for row in rows),
            "combine": sum(row["api_name"] == "combine" for row in rows),
        }
        if counts != dict(fresh=per_kind, cached=per_kind, combine=2 * per_kind):
            raise ValueError(f"rank {rank}: missing/extra API records: {counts}")
        fresh_ids = [row["handle_id"] for row in rows if row["api_name"] == "dispatch" and not row["cached"]]
        if len(set(fresh_ids)) != per_kind or manifests[rank]["handle_id"] not in fresh_ids:
            raise ValueError("fresh calls did not produce distinct, associated handles")


@pytest.mark.parametrize("corruption", [None, "shape", "handle", "missing_call", "missing_stage"])
def test_npu_metadata_acceptance(tmp_path, corruption):
    from types import SimpleNamespace

    args = SimpleNamespace(
        ep_size=1,
        num_tokens=2048,
        hidden=6144,
        num_topk=8,
        experts_per_rank=32,
        first_device=0,
        warmup=1,
        windows=1,
        iterations=32,
        trace_slots=128,
    )
    per_kind = 1 + args.warmup + args.iterations
    rows = [dict(event="calibration", tick_hz=1_000_000_000, tick=10, offset_ns=1000)]
    gid = call_id = 0
    for kind in PERF_OPS:
        for i in range(per_kind):
            call_id += 1
            if corruption == "missing_call" and kind == "combine_weighted" and i == 0:
                continue
            combine = kind.startswith("combine")
            cached = kind == "cached_dispatch"
            stages = (
                [("combine", "v1_combine_udma_comm_kernel")]
                if combine
                else [
                    ("cached_notify_dispatch", "cached_notify_dispatch_kernel")
                    if cached
                    else ("notify_dispatch", "notify_dispatch_kernel"),
                    ("dispatch", "v1_dispatch_udma_kernel"),
                ]
            )
            for op, kernel in stages:
                if corruption == "missing_stage" and kind == "fresh_dispatch" and i == 0 and op == "dispatch":
                    continue
                gid += 1
                row = dict(
                    metadata_version=1,
                    comm_id="comm",
                    buffer_id=1,
                    process_id="process",
                    hostname="host",
                    ep_size=1,
                    api_name="combine" if combine else "dispatch",
                    api_call_id=call_id,
                    handle_id=1 if combine or cached else i + 1,
                    hidden_size=6144,
                    element_bytes=2,
                    dtype="torch.bfloat16",
                    num_experts=32,
                    topk=8,
                    num_tokens=2048,
                    api_input_rows=2048,
                    cached=int(cached),
                    do_expand=0,
                    token_capacity=2048,
                    pid=100,
                    device=0,
                    global_rank=0,
                    rank=0,
                    local_instance=0,
                    stream="12345",
                    gid=gid,
                    launch_ns=1000,
                    op_name=op,
                    kernel_name=kernel,
                    aiv=32,
                )
                if op in ("combine", "dispatch"):
                    row.update(kernel_input_rows=2048, kernel_output_rows=2048)
                if corruption == "shape":
                    row["hidden_size"] = 5120  # Internally consistent, but does not match the workload.
                if corruption == "handle" and cached:
                    row["handle_id"] = 999
                rows += [
                    dict(row, event="launch", status=0),
                    dict(row, event="end", start_tick=1, end_tick=101, start_ns=1001, end_ns=1101, elapsed_ns=100),
                ]
    rows.append(dict(event="summary", completed=gid, incomplete=0, dropped=0))
    (tmp_path / "kernel-fixture.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifests = [dict(pid=100, recv_rows=2048, handle_id=1)]
    if corruption:
        with pytest.raises(
            ValueError,
            match={
                "shape": "hidden_size",
                "handle": "original fresh handle",
                "missing_call": "API records",
                "missing_stage": "missing kernel stages",
            }[corruption],
        ):
            validate_npu_metadata(tmp_path, manifests, args)
    else:
        validate_npu_metadata(tmp_path, manifests, args)


def boundary_labels(args):
    return ["precision"] + [
        label for name in PERF_OPS for label in [f"{name}/warmup", *[f"{name}/window{w}" for w in range(args.windows)]]
    ]


def record_slot_requirement(args):
    # Precision has four APIs; each subsequent boundary contains only one API kind.
    if args.windowed_collection:
        return max(8, 2 * args.warmup, 2 * args.iterations)
    return 8 * (1 + args.warmup + args.windows * args.iterations)


def measure_api_windows(args, operations, consume, synchronize, barrier, collect=None, clock=time.perf_counter_ns):
    """Keep synchronization/collection at matched boundaries, never within the measured API window."""
    boundaries, timings = [], {}

    def boundary(label):
        if collect is None:
            return
        synchronize()
        begin = clock()
        collect(label)
        boundaries.append(dict(label=label, collect_ns=clock() - begin))

    boundary("precision")
    for name, operation in zip(PERF_OPS, operations):
        for _ in range(args.warmup):
            consume(operation())
        boundary(f"{name}/warmup")
        timings[name] = []
        for window in range(args.windows):
            synchronize()
            barrier()
            started = clock()
            for _ in range(args.iterations):
                consume(operation())
            synchronize()
            timings[name].append((clock() - started) / args.iterations)
            boundary(f"{name}/window{window}")
    return timings, boundaries


def test_window_collection_is_outside_timing_and_fits_128_slots():
    args = npu_parser().parse_args(
        ["--output", "unused", "--windowed-collection", "--windows", "5", "--iterations", "60"]
    )
    elapsed, pending = 0, 8  # Four precision APIs have already run.
    occupancies, labels = [], []

    def operation():
        nonlocal elapsed, pending
        elapsed += 10
        pending += 2
        assert pending <= 128

    def consume(_):
        nonlocal elapsed
        elapsed += 2

    def synchronize():
        nonlocal elapsed
        elapsed += 3

    def barrier():
        nonlocal elapsed
        elapsed += 5

    def collect(label):
        nonlocal elapsed, pending
        occupancies.append(pending)
        labels.append(label)
        pending = 0
        elapsed += 100000  # A deliberately expensive collector must not affect API timings.

    timings, boundaries = measure_api_windows(
        args, [operation] * 4, consume, synchronize, barrier, collect, lambda: elapsed
    )
    assert labels == boundary_labels(args)
    assert occupancies == [8] + [v for _ in PERF_OPS for v in [20, 120, 120, 120, 120, 120]]
    assert pending == 0 and len(boundaries) == 25
    assert all(b["collect_ns"] == 100000 for b in boundaries)
    assert all(values == pytest.approx([12.05] * 5) for values in timings.values())


@pytest.mark.parametrize(
    "options",
    [["--iterations", "65"], ["--warmup", "65", "--iterations", "60"], ["--trace-slots", "256", "--iterations", "60"]],
)
def test_window_collection_rejects_capacity_overflow(tmp_path, options):
    args = npu_parser().parse_args(
        [
            "--npu",
            "--diagnostic",
            "--windowed-collection",
            "--modes",
            "off",
            "record_only",
            "--output",
            str(tmp_path / "unused"),
            *options,
        ]
    )
    with pytest.raises(ValueError, match="windowed collection requires"):
        run_npu_acceptance(args)
    assert not args.output.exists()


def npu_worker(args):
    """Use the installed wheel. CPU references and assertions stay outside timing."""
    import datetime
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import deep_ep
    from deep_ep.backends.global_clos import ElasticBuffer
    from deep_ep.utils.kernel_trace import KernelTrace

    sys.path.insert(0, str(ROOT / "tests/clos"))
    import common as golden

    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == args.ep_size
    torch.set_num_threads(1)
    torch.npu.set_device(args.first_device + rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
    args.dtype, args.seed = "bf16", 20260823
    args.num_aiv, args.expert_alignment = args.aiv, 1
    routes = [golden._make_topk_idx(peer, args) for peer in range(args.ep_size)]
    source = golden._make_dispatch_x(rank, args)
    weights_cpu = golden._make_weights(rank, args)
    x, idx, weights = source.to("npu"), routes[rank].to("npu"), weights_cpu.to("npu")
    started = time.perf_counter_ns()
    buffer = ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        explicitly_destroy=True,
        transport="udma",
    )
    init_ns = time.perf_counter_ns() - started
    trace_entry = KernelTrace._sources.get(buffer._kernel_trace)
    deferred = os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_MODE") == "record_only"
    if deferred:
        assert trace_entry is not None and KernelTrace._thread is None
        assert trace_entry["slots"] >= record_slot_requirement(args)
    host_probe_native = None
    gil_control = None
    if args.host_dispatch_timing:
        from deep_ep.backends.direct._native import load_native_module

        host_probe_native = load_native_module()
        if getattr(host_probe_native, "host_dispatch_probe_version", 0) != 1:
            raise RuntimeError("Host dispatch timing requires the matching rebuilt native extension")
        gil_control = host_probe_native.check_host_dispatch_gil_probe()
        if gil_control not in (dict(held_control=1, released_control=0), dict(held_control=-1, released_control=-1)):
            raise RuntimeError(f"GIL observation controls failed: {gil_control}")
        host_probe_native.configure_host_dispatch_probe(2 * (1 + args.warmup + args.windows * args.iterations))

    def dispatch(handle=None):
        return buffer.dispatch(
            x,
            topk_idx=idx if handle is None else None,
            topk_weights=weights if handle is None else None,
            num_experts=args.ep_size * args.experts_per_rank,
            handle=handle,
            num_sms=args.aiv,
            expert_alignment=1,
            async_with_compute_stream=True,
        )

    def consume(result):
        # Preserve the normal consumer dependency; this is a stream wait, not CPU synchronization.
        result[-1].current_stream_wait()
        return result

    fresh = consume(dispatch())
    torch.npu.synchronize()
    ref = golden._dispatch_reference(rank, args)
    for actual, key in zip(fresh[:3], ("recv_x", "recv_topk_idx", "recv_topk_weights")):
        golden._assert_equal(key, actual, ref[key])
    handle = fresh[3]
    owned, recv_weights = fresh[0].clone(), fresh[2].clone()
    cached = consume(dispatch(handle))
    torch.npu.synchronize()
    golden._assert_equal("cached dispatch", cached[0], ref["recv_x"])

    def combine(weighted):
        return buffer.combine(
            owned,
            handle,
            topk_weights=recv_weights if weighted else None,
            num_sms=args.aiv,
            reduce_num_sms=args.reduce_aiv,
            async_with_compute_stream=True,
        )

    expected = torch.zeros_like(source, dtype=torch.float32)
    for peer in range(args.ep_size):
        mask = (routes[rank] >= peer * args.experts_per_rank) & (routes[rank] < (peer + 1) * args.experts_per_rank)
        expected[mask.any(1)] += source[mask.any(1)].float()
    for weighted in (False, True):
        result = consume(combine(weighted))
        torch.npu.synchronize()
        golden._assert_equal("combine payload", result[0], expected.to(torch.bfloat16))
        if weighted:
            expected_weights = weights_cpu.clone()
            expected_weights[routes[rank] == -1] = 0
            golden._assert_equal("combine weights", result[1], expected_weights)
        else:
            assert result[1] is None
    boundary_count = 0

    def collect(label):
        nonlocal boundary_count
        if deferred:
            assert KernelTrace._thread is None
            assert trace_entry["deferred_drains"] == boundary_count, "collection outside declared boundary"
        KernelTrace.collect_boundary(buffer._kernel_trace)
        boundary_count += 1
        if deferred:
            assert trace_entry["deferred_drains"] == boundary_count

    timings, boundaries = measure_api_windows(
        args,
        (dispatch, lambda: dispatch(handle), lambda: combine(False), lambda: combine(True)),
        consume,
        torch.npu.synchronize,
        dist.barrier,
        collect if args.windowed_collection else None,
    )
    if host_probe_native is not None:
        host_report = host_probe_native.take_host_dispatch_probe()  # Disable and export only after all timed windows.
        host_report.update(
            gil_control=gil_control,
            python_version=sys.version,
            python_gil_enabled=getattr(sys, "_is_gil_enabled", lambda: None)(),
        )
        validate_host_dispatch_report(host_report, args, rank)
        (args.output / f"host_dispatch_rank{rank}.json").write_text(json.dumps(host_report, indent=2))
    manifest = dict(
        rank=rank,
        pid=os.getpid(),
        wheel=deep_ep.__file__,
        init_ns=init_ns,
        timings_ns=timings,
        recv_rows=owned.size(0),
        handle_id=getattr(handle, "_trace_handle_id", None),
        precision_passed=True,
        trace_mode=os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_MODE", "production")
        if os.environ.get("DEEPEP_TRACE_DIR")
        else "off",
        trace_ring_active=buffer._kernel_trace is not None,
        metadata_statistics=getattr(buffer._trace_session, "diagnostic_stats", {}),
        trace_slots=trace_entry["slots"] if trace_entry else 0,
        windowed_collection=args.windowed_collection,
        collection_boundaries=boundaries,
        host_dispatch_timing=args.host_dispatch_timing,
    )
    if deferred:
        manifest["deferred_evidence"] = dict(
            thread_absent=KernelTrace._thread is None,
            drains_before_destroy=trace_entry["deferred_drains"],
        )
    started = time.perf_counter_ns()
    buffer.destroy()  # Flush/close trace before the controller validates logs.
    manifest["destroy_and_drain_ns"] = time.perf_counter_ns() - started
    if deferred:
        manifest["deferred_evidence"]["drains_after_destroy"] = trace_entry["deferred_drains"]
    (args.output / f"rank{rank}.json").write_text(json.dumps(manifest, indent=2))
    dist.destroy_process_group()


def npu_parser():
    import argparse

    parser = argparse.ArgumentParser(description="KernelTrace real-NPU metadata and trace off/on acceptance")
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--diagnostic", action="store_true", help="layered attribution, not performance acceptance")
    parser.add_argument("--diagnostic-timing", action="store_true", help="separate instrumented run only")
    parser.add_argument(
        "--host-dispatch-timing",
        action="store_true",
        help="separate diagnostic run: native dispatch phases and sampled GIL state",
    )
    parser.add_argument(
        "--windowed-collection",
        action="store_true",
        help="diagnostic: collect outside each window using 128 slots; use --iterations 60 --windows 5",
    )
    parser.add_argument(
        "--paired-baselines",
        action="store_true",
        help="diagnostic: run a separate adjacent off trial for each enabled mode",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "production", "record_only", "metadata", "native", "json_discard", "json", "full"),
        default=["off", "metadata", "native", "json", "full"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in dict(
        ep_size=4,
        first_device=0,
        num_tokens=2048,
        hidden=6144,
        num_topk=8,
        experts_per_rank=32,
        aiv=32,
        reduce_aiv=32,
        pairs=6,
        windows=3,
        iterations=100,
        warmup=10,
        timeout=1800,
        trace_slots=0,
    ).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    return parser


def run_npu_process(command, destination, env, log, timeout):
    import signal

    # torchrun owns the workers. On timeout, terminate this run's process group only.
    with subprocess.Popen(
        command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if code:
            raise RuntimeError(f"NPU worker exited {code}; see {destination / 'run.log'}")


def run_npu_acceptance(args):
    args._created_output = False
    try:
        if args.diagnostic_timing and not args.diagnostic:
            raise ValueError("--diagnostic-timing requires --diagnostic")
        if args.paired_baselines and not args.diagnostic:
            raise ValueError("--paired-baselines requires --diagnostic")
        if args.windowed_collection and not args.diagnostic:
            raise ValueError("--windowed-collection requires --diagnostic")
        if args.host_dispatch_timing and (not args.diagnostic or args.diagnostic_timing):
            raise ValueError("--host-dispatch-timing requires --diagnostic without --diagnostic-timing")
        if args.diagnostic and args.diagnostic_timing and any(m in args.modes for m in ("production", "record_only")):
            raise ValueError("production control and record_only must run without --diagnostic-timing")
        required_slots = record_slot_requirement(args)
        deferred = args.diagnostic and "record_only" in args.modes
        if args.trace_slots == 0:
            args.trace_slots = max(128, required_slots) if deferred and not args.windowed_collection else 128
        if args.windowed_collection and (args.trace_slots != 128 or required_slots > 128):
            raise ValueError("windowed collection requires 128 slots and at most 128 kernels between boundaries")
        if not 128 <= args.trace_slots <= 16384 or (deferred and args.trace_slots < required_slots):
            raise ValueError("trace slots must be in [128, 16384] and cover all record_only kernels")
        return _run_npu_diagnostic(args) if args.diagnostic else _run_npu_acceptance(args)
    except Exception as error:
        # Do not modify an existing output directory rejected by the initial mkdir.
        if getattr(args, "_created_output", False):
            name = "diagnosis.json" if args.diagnostic else "acceptance.json"
            (args.output / name).write_text(json.dumps(dict(passed=False, error=str(error)), indent=2))
        raise


def npu_run_trial(args, destination, mode, diagnostic=False):
    destination.mkdir()
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "DEEPEP_TRACE_DIR",
        "DEEPEP_TRACE_DIAGNOSTIC_MODE",
        "DEEPEP_TRACE_DIAGNOSTIC_TIMING",
        "DEEPEP_TRACE_SLOTS",
    ):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if mode != "off":
        env["DEEPEP_TRACE_DIR"] = str(destination / "trace")
        env["DEEPEP_TRACE_SLOTS"] = str(args.trace_slots)
        if diagnostic and mode != "production":
            env["DEEPEP_TRACE_DIAGNOSTIC_MODE"] = mode
            env["DEEPEP_TRACE_DIAGNOSTIC_TIMING"] = "1" if args.diagnostic_timing else "0"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.ep_size}",
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(destination),
    ]
    for key, value in vars(args).items():
        if type(value) is int and not key.startswith("_"):
            command += ["--" + key.replace("_", "-"), str(value)]
    if args.windowed_collection:
        command.append("--windowed-collection")
    if args.host_dispatch_timing:
        command.append("--host-dispatch-timing")
    print(f"RUN {destination.name}", flush=True)
    with (destination / "run.log").open("w") as log:
        run_npu_process(command, destination, env, log, args.timeout)
    manifests = [json.loads((destination / f"rank{rank}.json").read_text()) for rank in range(args.ep_size)]
    values = {
        op: statistics.median(max(m["timings_ns"][op][w] for m in manifests) for w in range(args.windows))
        for op in PERF_OPS
    }
    return manifests, values


def validate_diagnostic_trial(destination, manifests, args, mode):
    trace = destination / "trace"
    if mode in ("production", "record_only", "full"):
        validate_npu_metadata(trace, manifests, args)
    elif list(trace.glob("kernel-*.jsonl")):
        raise ValueError(f"{mode}: unexpected per-event log")
    per_kind = 1 + args.warmup + args.windows * args.iterations
    reports = list(trace.glob("diagnostic-*.json"))
    collector_modes = ("native", "json_discard", "json", "full")
    if mode in collector_modes:
        if len(reports) != args.ep_size:
            raise ValueError("missing diagnostic collector reports")
        by_pid = {json.loads(f.read_text())["pid"]: json.loads(f.read_text()) for f in reports}
    elif reports:
        raise ValueError("off/metadata/production unexpectedly produced diagnostic collector reports")
    for rank, manifest in enumerate(manifests):
        if manifest.get("host_dispatch_timing", False) != args.host_dispatch_timing:
            raise ValueError("Host dispatch timing option was not applied")
        if args.host_dispatch_timing:
            validate_host_dispatch_report(
                json.loads((destination / f"host_dispatch_rank{rank}.json").read_text()), args, rank
            )
        if manifest["trace_mode"] != mode or not manifest["precision_passed"]:
            raise ValueError("wrong mode or precision failure")
        if manifest["trace_ring_active"] != (mode in (*collector_modes, "production", "record_only")):
            raise ValueError("unexpected trace ring state")
        if manifest["trace_ring_active"] and manifest.get("trace_slots") != args.trace_slots:
            raise ValueError("trace capacity does not match the paired control")
        expected_boundaries = boundary_labels(args) if args.windowed_collection else []
        actual_boundaries = manifest.get("collection_boundaries", [])
        if (
            manifest.get("windowed_collection", False) != args.windowed_collection
            or [b["label"] for b in actual_boundaries] != expected_boundaries
            or any(type(b.get("collect_ns")) is not int or b["collect_ns"] < 0 for b in actual_boundaries)
        ):
            raise ValueError("mismatched collection boundaries")
        if mode == "record_only" and manifest.get("deferred_evidence") != dict(
            thread_absent=True,
            drains_before_destroy=len(expected_boundaries),
            drains_after_destroy=len(expected_boundaries) + 1,
        ):
            raise ValueError("record_only collected early or started a polling thread")
        metadata = manifest["metadata_statistics"]
        expected_calls = dict(fresh_dispatch=per_kind, cached_dispatch=per_kind, combine=2 * per_kind)
        if mode in ("off", "production", "record_only") and metadata:
            raise ValueError("off/production must not execute diagnostic metadata counters")
        if (
            mode not in ("off", "production", "record_only")
            and {k: v["calls"] for k, v in metadata.items()} != expected_calls
        ):
            raise ValueError("diagnostic metadata call count mismatch")
        if mode in collector_modes:
            report = by_pid[manifest["pid"]]
            stats = report["counters"]
            # Current eager fixture: notify+dispatch and communication+reduce, two kernels per API.
            expected = 8 * per_kind
            if (
                report["failed"]
                or report["mode"] != mode
                or report["timing"] != args.diagnostic_timing
                or report["device"] != args.first_device + rank
                or stats.get("summaries") != 1
                or stats.get("completed") != expected
                or stats.get("launches") != expected
                or stats.get("ends") != expected
                or any(stats.get(key) != 0 for key in ("dropped", "incomplete", "launch_errors"))
            ):
                raise ValueError(f"invalid diagnostic collection, rank {rank}: {report}")
            encoded, returned = stats.get("encoded_bytes"), stats.get("returned_text_bytes")
            if (
                not isinstance(encoded, int)
                or not isinstance(returned, int)
                or (mode == "native" and (encoded != 0 or returned != 0))
                or (mode != "native" and encoded <= 0)
                or (mode == "json_discard" and returned != 0)
                or (mode in ("json", "full") and returned != encoded)
            ):
                raise ValueError(f"invalid diagnostic text counters, rank {rank}: {stats}")


def diagnostic_schedule(args):
    """Counterbalance mode order and off/on order without changing the worker workload."""
    modes = [mode for mode in args.modes if mode != "off"] if args.paired_baselines else args.modes
    for round_id in range(args.rounds):
        # Reverse each rotation on the next round; rotating every round would cancel
        # reversal for two modes and keep the same mode first forever.
        offset = (round_id // 2) % len(modes)
        order = modes[offset:] + modes[:offset]
        if round_id % 2:
            order = list(reversed(order))
        for mode in order:
            trials = (mode,)
            pair = None
            if args.paired_baselines:
                pair = mode
                trials = ("off", mode) if round_id % 2 == 0 else (mode, "off")
            for trial in trials:
                name = f"round{round_id:02d}_" + (f"pair_{pair}_" if pair else "") + trial
                yield dict(round=round_id, pair=pair, mode=trial, directory=name)


def diagnostic_layer_increments(runs):
    """Report both raw differences and own-baseline-adjusted differences; neither is exclusive cost."""
    baseline = {(r["round"], r["pair"]): r["ns_per_api"] for r in runs if r["mode"] == "off"}
    selected = {(r["round"], r["mode"]): r for r in runs if r["mode"] != "off"}
    edges = (
        ("metadata", "record_only"),
        ("record_only", "production"),
        ("production", "full"),
        ("metadata", "native"),
        ("native", "json_discard"),
        ("json_discard", "json"),
        ("json", "full"),
    )
    result = {}
    for left, right in edges:
        rounds = sorted(r for r, mode in selected if mode == left and (r, right) in selected)
        if not rounds:
            continue
        result[f"{left} -> {right}"] = {}
        for op in PERF_OPS:
            raw, adjusted = [], []
            for round_id in rounds:
                a, b = selected[round_id, left], selected[round_id, right]
                delta = b["ns_per_api"][op] - a["ns_per_api"][op]
                raw.append(delta)
                adjusted.append(delta - baseline[round_id, b["pair"]][op] + baseline[round_id, a["pair"]][op])
            result[f"{left} -> {right}"][op] = dict(
                rounds=rounds,
                raw_ns=raw,
                baseline_adjusted_ns=adjusted,
                raw_median_ns=statistics.median(raw),
                adjusted_median_ns=statistics.median(adjusted),
                adjusted_range_ns=[min(adjusted), max(adjusted)],
                adjusted_positive_rounds=sum(v > 0 for v in adjusted),
            )
    return result


def _run_npu_diagnostic(args):
    if args.rounds < 1 or args.windows < 1 or args.iterations < 32 or args.warmup < 1:
        raise ValueError("require rounds/windows/warmup>=1, iterations>=32")
    if "off" not in args.modes or len(set(args.modes)) != len(args.modes) or len(args.modes) < 2:
        raise ValueError("diagnostic modes must be unique and include off plus at least one enabled mode")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    schedule = list(diagnostic_schedule(args))
    (args.output / "plan.json").write_text(json.dumps(schedule, indent=2))
    runs = []
    for trial in schedule:
        destination = args.output / trial["directory"]
        manifests, values = npu_run_trial(args, destination, trial["mode"], diagnostic=True)
        validate_diagnostic_trial(destination, manifests, args, trial["mode"])
        runs.append(dict(**trial, ns_per_api=values))
        (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
    baseline = {(r["round"], r["pair"]): r["ns_per_api"] for r in runs if r["mode"] == "off"}
    comparisons = {}
    for mode in args.modes:
        comparisons[mode] = {}
        for op in PERF_OPS:
            selected = [r for r in runs if r["mode"] == mode]
            deltas = [100 * (r["ns_per_api"][op] / baseline[r["round"], r["pair"]][op] - 1) for r in selected]
            comparisons[mode][op] = dict(
                median_ns=statistics.median(r["ns_per_api"][op] for r in selected),
                paired_percent=deltas,
                median_percent=statistics.median(deltas),
            )
    report = dict(
        diagnostic_valid=True,
        performance_acceptance=False,
        instrumented=args.diagnostic_timing or args.host_dispatch_timing,
        host_dispatch_timing=args.host_dispatch_timing,
        paired_baselines=args.paired_baselines,
        layer_increments=diagnostic_layer_increments(runs),
        attribution_note="Layer differences include overlap and scheduling; they are not exclusive component costs.",
        comparisons=comparisons,
    )
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return True  # Diagnostic success means valid evidence, never a <1% claim.


def _run_npu_acceptance(args):
    if args.pairs < 6 or args.windows < 1 or args.iterations < 32 or args.warmup < 1:
        raise ValueError("require pairs>=6, windows>=1, iterations>=32, warmup>=1")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    runs = []
    for pair in range(args.pairs):
        # Fresh processes avoid inherited native singleton/env state; counterbalance thermal/order effects.
        for enabled in (False, True) if pair % 2 == 0 else (True, False):
            destination = args.output / f"pair{pair:02d}_{'on' if enabled else 'off'}"
            manifests, values = npu_run_trial(args, destination, "full" if enabled else "off")
            if enabled:
                validate_npu_metadata(destination / "trace", manifests, args)
            elif list(destination.rglob("kernel-*.jsonl")):
                raise AssertionError("trace-off run unexpectedly produced logs")
            runs.append(dict(pair=pair, enabled=enabled, ns_per_api=values))
            (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
    results = {
        op: overhead_result(
            [r["ns_per_api"][op] for r in runs if not r["enabled"]],
            [r["ns_per_api"][op] for r in runs if r["enabled"]],
        )
        for op in PERF_OPS
    }
    passed = all(result["status"] == "PASS" for result in results.values())
    report = dict(passed=passed, metadata_passed=True, precision_passed=True, operations=results)
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return passed


def test_npu_controller_pairs_and_report(tmp_path, monkeypatch):
    args = npu_parser().parse_args(["--npu", "--output", str(tmp_path / "ab")])
    order = []
    checked = []
    monkeypatch.setenv("DEEPEP_TRACE_DIR", "must-not-leak")
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_MODE", "metadata")
    monkeypatch.setenv("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "1")

    def launch(command, destination, env, log, timeout):
        enabled = "DEEPEP_TRACE_DIR" in env
        order.append(enabled)
        assert "PYTHONPATH" not in env
        assert "DEEPEP_TRACE_DIAGNOSTIC_MODE" not in env and "DEEPEP_TRACE_DIAGNOSTIC_TIMING" not in env
        assert "--worker" in command and "--standalone" in command
        if enabled:
            assert env["DEEPEP_TRACE_DIR"] == str(destination / "trace")
        for rank in range(args.ep_size):
            manifest = dict(timings_ns={op: [99 if enabled else 100] * args.windows for op in PERF_OPS})
            (destination / f"rank{rank}.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(sys.modules[__name__], "run_npu_process", launch)
    monkeypatch.setattr(sys.modules[__name__], "validate_npu_metadata", lambda *items: checked.append(items[0]))
    assert run_npu_acceptance(args)
    assert order == [False, True, True, False] * 3
    assert len(checked) == 6
    report = json.loads((args.output / "acceptance.json").read_text())
    assert report["passed"] and set(report["operations"]) == set(PERF_OPS)
    for result in report["operations"].values():
        assert result["median_percent"] == pytest.approx(-1)


@pytest.mark.parametrize("paired,windowed", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "corruption", [None, "dropped", "completed", "summaries", "returned_text_bytes", "encoded_bytes"]
)
def test_diagnostic_controller_and_incomplete_rejection(tmp_path, monkeypatch, corruption, paired, windowed):
    args = npu_parser().parse_args(
        ["--npu", "--diagnostic", "--rounds", "2", "--output", str(tmp_path / "diagnosis")]
        + (
            [
                "--paired-baselines",
                "--modes",
                "off",
                "production",
                "record_only",
                "native",
                "json_discard",
                "json",
                "full",
            ]
            if paired
            else ["--diagnostic-timing"]
        )
        + (["--windowed-collection", "--windows", "5", "--iterations", "60"] if windowed else [])
    )
    order = []
    per_kind = 1 + args.warmup + args.windows * args.iterations

    def launch(command, destination, env, log, timeout):
        assert ("--windowed-collection" in command) == windowed
        mode = env.get("DEEPEP_TRACE_DIAGNOSTIC_MODE", "production" if "DEEPEP_TRACE_DIR" in env else "off")
        order.append(mode)
        if mode in ("off", "production"):
            assert "DEEPEP_TRACE_DIAGNOSTIC_MODE" not in env and "DEEPEP_TRACE_DIAGNOSTIC_TIMING" not in env
        else:
            assert env["DEEPEP_TRACE_DIAGNOSTIC_TIMING"] == str(int(args.diagnostic_timing))
        # Deliberately vary the baseline between pairs to catch accidental shared-off comparisons.
        # Use the planned pair label because mode names themselves may contain underscores.
        planned = next(t for t in diagnostic_schedule(args) if t["directory"] == destination.name)
        drift = 100 * args.modes.index(planned["pair"]) if paired else 0
        excess = dict(off=0, production=35, record_only=25, metadata=10, native=20, json_discard=30, json=40, full=45)[
            mode
        ]
        for rank in range(args.ep_size):
            manifest = dict(
                pid=100 + rank,
                trace_mode=mode,
                trace_ring_active=mode in ("production", "record_only", "native", "json_discard", "json", "full"),
                trace_slots=args.trace_slots,
                windowed_collection=windowed,
                collection_boundaries=[dict(label=label, collect_ns=10) for label in boundary_labels(args)]
                if windowed
                else [],
                deferred_evidence=dict(
                    thread_absent=True,
                    drains_before_destroy=25 if windowed else 0,
                    drains_after_destroy=26 if windowed else 1,
                ),
                precision_passed=True,
                metadata_statistics={}
                if mode in ("off", "production", "record_only")
                else {
                    k: dict(calls=v, enter_ns=1, restore_ns=1)
                    for k, v in dict(fresh_dispatch=per_kind, cached_dispatch=per_kind, combine=2 * per_kind).items()
                },
                timings_ns={op: [100 + drift + excess] * args.windows for op in PERF_OPS},
            )
            (destination / f"rank{rank}.json").write_text(json.dumps(manifest))
            if mode in ("native", "json_discard", "json", "full"):
                trace = destination / "trace"
                trace.mkdir(exist_ok=True)
                counters = dict(
                    summaries=1,
                    completed=8 * per_kind,
                    launches=8 * per_kind,
                    ends=8 * per_kind,
                    dropped=0,
                    incomplete=0,
                    launch_errors=0,
                    encoded_bytes=0 if mode == "native" else 1024,
                    returned_text_bytes=1024 if mode in ("json", "full") else 0,
                )
                if corruption in ("encoded_bytes", "returned_text_bytes"):
                    if mode == ("json_discard" if paired else "json"):
                        if corruption == "encoded_bytes":
                            counters[corruption] = 0
                        else:
                            counters[corruption] += 1
                elif corruption:
                    counters[corruption] += 1
                report = dict(
                    pid=100 + rank,
                    device=rank,
                    mode=mode,
                    timing=args.diagnostic_timing,
                    failed=False,
                    counters=counters,
                )
                (trace / f"diagnostic-{rank}.json").write_text(json.dumps(report))

    monkeypatch.setattr(sys.modules[__name__], "run_npu_process", launch)
    monkeypatch.setattr(sys.modules[__name__], "validate_npu_metadata", lambda *items: None)
    if corruption:
        with pytest.raises(ValueError, match="invalid diagnostic (collection|text counters)"):
            run_npu_acceptance(args)
        assert not json.loads((args.output / "diagnosis.json").read_text())["passed"]
    else:
        assert run_npu_acceptance(args)
        report = json.loads((args.output / "diagnosis.json").read_text())
        assert report["diagnostic_valid"] and not report["performance_acceptance"]
        assert report["instrumented"] == args.diagnostic_timing
        assert order == [t["mode"] for t in diagnostic_schedule(args)]
        if paired:
            assert len(order) == 24
            assert report["comparisons"]["json"]["fresh_dispatch"]["median_percent"] == pytest.approx(100 * 40 / 600)
            for edge in (
                "record_only -> production",
                "production -> full",
                "native -> json_discard",
                "json_discard -> json",
            ):
                assert report["layer_increments"][edge]["fresh_dispatch"]["baseline_adjusted_ns"] == [10, 10]
        else:
            assert report["comparisons"]["json"]["fresh_dispatch"]["median_percent"] == pytest.approx(40)


def test_host_dispatch_controller_exports_instrumented_evidence(tmp_path, monkeypatch):
    args = npu_parser().parse_args(
        [
            "--npu",
            "--diagnostic",
            "--host-dispatch-timing",
            "--windowed-collection",
            "--modes",
            "off",
            "record_only",
            "production",
            "--rounds",
            "1",
            "--warmup",
            "1",
            "--windows",
            "1",
            "--iterations",
            "32",
            "--output",
            str(tmp_path / "host"),
        ]
    )

    def launch(command, destination, env, log, timeout):
        assert "--host-dispatch-timing" in command and "--windowed-collection" in command
        mode = env.get("DEEPEP_TRACE_DIAGNOSTIC_MODE", "production" if "DEEPEP_TRACE_DIR" in env else "off")
        count = len(boundary_labels(args))
        for rank in range(args.ep_size):
            manifest = dict(
                trace_mode=mode,
                precision_passed=True,
                trace_ring_active=mode != "off",
                trace_slots=128 if mode != "off" else 0,
                metadata_statistics={},
                host_dispatch_timing=True,
                windowed_collection=True,
                collection_boundaries=[dict(label=label, collect_ns=1) for label in boundary_labels(args)],
                deferred_evidence=dict(thread_absent=True, drains_before_destroy=count, drains_after_destroy=count + 1),
                timings_ns={op: [100] for op in PERF_OPS},
            )
            (destination / f"rank{rank}.json").write_text(json.dumps(manifest))
            (destination / f"host_dispatch_rank{rank}.json").write_text(json.dumps(make_host_probe_report(args, rank)))

    monkeypatch.setattr(sys.modules[__name__], "run_npu_process", launch)
    monkeypatch.setattr(sys.modules[__name__], "validate_npu_metadata", lambda *items: None)
    assert run_npu_acceptance(args)
    report = json.loads((args.output / "diagnosis.json").read_text())
    assert report["instrumented"] and report["host_dispatch_timing"] and not report["performance_acceptance"]
    assert len(list(args.output.glob("round*/host_dispatch_rank*.json"))) == 3 * args.ep_size


def test_diagnostic_paired_schedule_and_production_guard(tmp_path):
    args = npu_parser().parse_args(
        [
            "--npu",
            "--diagnostic",
            "--paired-baselines",
            "--rounds",
            "6",
            "--modes",
            "off",
            "production",
            "full",
            "--output",
            str(tmp_path / "unused"),
        ]
    )
    plan = list(diagnostic_schedule(args))
    assert len(plan) == 24 and len({t["directory"] for t in plan}) == 24
    enabled_order = [[t["mode"] for t in plan if t["round"] == r and t["mode"] != "off"] for r in range(6)]
    assert enabled_order == [
        ["production", "full"],
        ["full", "production"],
        ["full", "production"],
        ["production", "full"],
        ["production", "full"],
        ["full", "production"],
    ]
    for i in range(0, len(plan), 2):
        a, b = plan[i : i + 2]
        assert a["round"] == b["round"] and a["pair"] == b["pair"]
        assert [a["mode"], b["mode"]] == (["off", a["pair"]] if a["round"] % 2 == 0 else [a["pair"], "off"])
    args.diagnostic_timing = True
    with pytest.raises(ValueError, match="production control"):
        run_npu_acceptance(args)
    assert not args.output.exists()


@pytest.mark.parametrize("slots", [127, 128, 16385])
def test_record_only_rejects_insufficient_capacity(tmp_path, slots):
    args = npu_parser().parse_args(
        [
            "--npu",
            "--diagnostic",
            "--modes",
            "off",
            "record_only",
            "--trace-slots",
            str(slots),
            "--output",
            str(tmp_path / "unused"),
        ]
    )
    with pytest.raises(ValueError, match="trace slots"):
        run_npu_acceptance(args)
    assert not args.output.exists()


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("corruption", ["capacity", "thread", "early", "missing_final", "boundaries"])
def test_record_only_rejects_invalid_collection_evidence(tmp_path, monkeypatch, corruption, windowed):
    args = npu_parser().parse_args(
        ["--output", str(tmp_path), "--trace-slots", "128" if windowed else "2488"]
        + (["--windowed-collection", "--windows", "5", "--iterations", "60"] if windowed else [])
    )
    count = 25 if windowed else 0
    evidence = dict(thread_absent=True, drains_before_destroy=count, drains_after_destroy=count + 1)
    manifest = dict(
        trace_mode="record_only",
        precision_passed=True,
        trace_ring_active=True,
        trace_slots=args.trace_slots,
        windowed_collection=windowed,
        collection_boundaries=[dict(label=label, collect_ns=10) for label in boundary_labels(args)] if windowed else [],
        metadata_statistics={},
        deferred_evidence=evidence,
    )
    if corruption == "capacity":
        manifest["trace_slots"] = 256
    elif corruption == "thread":
        evidence["thread_absent"] = False
    elif corruption == "early":
        evidence["drains_before_destroy"] += 1
    elif corruption == "boundaries":
        manifest["collection_boundaries"] = [] if windowed else [dict(label="unexpected", collect_ns=0)]
    else:
        evidence["drains_after_destroy"] = 0
    monkeypatch.setattr(sys.modules[__name__], "validate_npu_metadata", lambda *items: None)
    with pytest.raises(ValueError, match="trace capacity|record_only collected|mismatched collection boundaries"):
        validate_diagnostic_trial(tmp_path, [manifest], args, "record_only")


@pytest.mark.skipif(os.environ.get("DEEPEP_RUN_TRACE_NPU") != "1", reason="opt-in real NPU acceptance")
def test_npu_metadata_and_overhead(tmp_path):
    args = npu_parser().parse_args(["--npu", "--output", str(tmp_path / "acceptance")])
    assert run_npu_acceptance(args), "not all operations demonstrate <1% overhead; inspect acceptance.json"


if __name__ == "__main__":
    arguments = npu_parser().parse_args()
    if arguments.worker:
        npu_worker(arguments)
    elif arguments.npu:
        sys.exit(0 if run_npu_acceptance(arguments) else 1)
    else:
        raise SystemExit("use --npu, or run this file with pytest for CPU protocol tests")
