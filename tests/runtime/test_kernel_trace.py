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


PERF_OPS = ("fresh_dispatch", "cached_dispatch", "combine_unweighted", "combine_weighted")


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
        directory, args.ep_size, ("notify_dispatch", "cached_notify_dispatch", "dispatch", "combine"), True, True
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


def npu_worker(args):
    """Use the installed wheel. CPU references and assertions stay outside timing."""
    import datetime
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import deep_ep
    from deep_ep.backends.global_clos import ElasticBuffer

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
    timings = {}
    for name, operation in zip(
        PERF_OPS, (dispatch, lambda: dispatch(handle), lambda: combine(False), lambda: combine(True))
    ):
        for _ in range(args.warmup):
            consume(operation())
        timings[name] = []
        for _ in range(args.windows):
            torch.npu.synchronize()
            dist.barrier()
            started = time.perf_counter_ns()
            for _ in range(args.iterations):
                consume(operation())
            torch.npu.synchronize()
            timings[name].append((time.perf_counter_ns() - started) / args.iterations)
    manifest = dict(
        rank=rank,
        pid=os.getpid(),
        wheel=deep_ep.__file__,
        init_ns=init_ns,
        timings_ns=timings,
        recv_rows=owned.size(0),
        handle_id=getattr(handle, "_trace_handle_id", None),
        precision_passed=True,
    )
    started = time.perf_counter_ns()
    buffer.destroy()  # Flush/close trace before the controller validates logs.
    manifest["destroy_and_drain_ns"] = time.perf_counter_ns() - started
    (args.output / f"rank{rank}.json").write_text(json.dumps(manifest, indent=2))
    dist.destroy_process_group()


def npu_parser():
    import argparse

    parser = argparse.ArgumentParser(description="KernelTrace real-NPU metadata and trace off/on acceptance")
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
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
        return _run_npu_acceptance(args)
    except Exception as error:
        # Do not modify an existing output directory rejected by the initial mkdir.
        if getattr(args, "_created_output", False):
            (args.output / "acceptance.json").write_text(json.dumps(dict(passed=False, error=str(error)), indent=2))
        raise


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
            destination.mkdir()
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("DEEPEP_TRACE_DIR", None)
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if enabled:
                env["DEEPEP_TRACE_DIR"] = str(destination / "trace")
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
                if key not in ("npu", "worker", "output") and not key.startswith("_"):
                    command += ["--" + key.replace("_", "-"), str(value)]
            print(f"RUN {destination.name}", flush=True)
            with (destination / "run.log").open("w") as log:
                run_npu_process(command, destination, env, log, args.timeout)
            manifests = [json.loads((destination / f"rank{rank}.json").read_text()) for rank in range(args.ep_size)]
            if enabled:
                validate_npu_metadata(destination / "trace", manifests, args)
            elif list(destination.rglob("kernel-*.jsonl")):
                raise AssertionError("trace-off run unexpectedly produced logs")
            # Slowest rank per window determines collective progress; median windows per process trial.
            values = {
                op: statistics.median(max(m["timings_ns"][op][w] for m in manifests) for w in range(args.windows))
                for op in PERF_OPS
            }
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

    def launch(command, destination, env, log, timeout):
        enabled = "DEEPEP_TRACE_DIR" in env
        order.append(enabled)
        assert "PYTHONPATH" not in env
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
