"""Python monitor lifecycle and native generation-reuse regression checks."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
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

        def drain_trace(self, closing=False):
            self.calls.append(threading.current_thread().name)
            result, self.queue = self.queue, []
            if closing:
                assert self.synchronized
                result.append({"event": "summary"})
                self.closed = True
            return result

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
    source.write_text("""
#include "runtime/kernel_trace.h"
#include "common/kernel_trace_layout.h"
#include <cassert>
#include <thread>
using namespace ascend_deepep::trace;
extern "C" void kernel_trace_clock_do(void*,uint8_t* p){*reinterpret_cast<uint64_t*>(p)=100;}
int main(){
    auto sink=KernelTrace::Open(); assert(sink);
    TraceMetadata outer;
    outer.values["api_call_id"] = 999;
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
        assert(done[0].labels.at("stream")=="12345");
        assert(done[0].values.at("gid")==static_cast<int64_t>(ticket.sequence));
        assert(done[0].values.at("elapsed_ns")==40); // Zero start tick is valid.
    }
    auto summary=sink->Drain(true);
    assert(summary.back().values.at("completed")==384);
    assert(summary.back().values.at("dropped")==0);
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
    subprocess.run([str(binary)], check=True, timeout=20)


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

        def drain_trace(self, closing=False):
            result, self.pending = self.pending, []
            if result:
                written.set()
            if closing:
                result.append(rows[-1])
            return result

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
    def __init__(self):
        self.context = threading.local()

    def exchange_kernel_trace_metadata(self, fields):
        previous = getattr(self.context, "fields", {})
        self.context.fields = dict(fields)
        return previous


def load_trace_module():
    spec = importlib.util.spec_from_file_location("trace_metadata_test", ROOT / "deep_ep/utils/kernel_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_call_and_handle_metadata_scopes():
    module = load_trace_module()
    native = MetadataNative()
    session = module.KernelTraceSession(native, "comm-a", 4, 12)
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


def test_trace_group_initialization(monkeypatch):
    module = load_trace_module()
    native = MetadataNative()
    monkeypatch.setenv("DEEPEP_TRACE_DIR", "test")
    monkeypatch.setattr(module.KernelTrace, "attach", lambda native: 456)
    monkeypatch.setattr(module, "_global_rank", lambda: 9)
    proposals = []

    def gather(group, comm, size, value):
        proposals.append(value)
        return [(True, "rank0-uuid"), (True, None)]

    first = module.attach_trace_session(native, None, 0, 2, gather)[1]
    second = module.attach_trace_session(native, None, 1, 2, gather)[1]
    assert first.fields["comm_id"] == second.fields["comm_id"] == "rank0-uuid"
    assert first.fields["buffer_id"] != second.fields["buffer_id"]
    assert proposals[0][1] and proposals[1][1] is None
    monkeypatch.delenv("DEEPEP_TRACE_DIR")
    with pytest.raises(ValueError, match="all EP ranks"):
        module.attach_trace_session(native, None, 1, 2, gather)
    assert module.attach_trace_session(native, None, 0, 2, lambda *args: [(False, None)] * 2) == (None, None)


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
