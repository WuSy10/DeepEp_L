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
using namespace ascend_deepep::trace;
extern "C" void kernel_trace_clock_do(void*,uint8_t* p){*reinterpret_cast<uint64_t*>(p)=100;}
int main(){
    auto sink=KernelTrace::Open(); assert(sink);
    const char* ops[] = {"notify_dispatch", "dispatch", "cached_notify_dispatch", "cached_notify_combine", "combine"};
    const char* kernels[] = {"notify_dispatch_kernel", "v1_dispatch_udma_kernel", "cached_notify_dispatch_kernel",
                             "cached_notify_combine_kernel", "v1_combine_udma_reduce_kernel"};
    for(int round=0;round<384;++round) {
        const int kind = round % 5;
        auto ticket=sink->Reserve(0,0,2,ops[kind],kernels[kind]); assert(ticket.address);
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
