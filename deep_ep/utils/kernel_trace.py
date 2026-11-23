"""Process-local Python monitor for native kernel trace sources."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from collections.abc import Mapping
import itertools
import json
import os
from pathlib import Path
import socket
import threading
import time
import warnings
import uuid


def diagnostic_config():
    """Read once at initialization; an empty mode preserves the production path."""
    # 普通采集保持空模式；分层计时仅在明确选定兼容的诊断模式时开启。
    mode = os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_MODE", "")
    timing = os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "0")
    if mode not in ("", "record_only", "metadata", "native", "json_discard", "json", "full") or timing not in (
        "0",
        "1",
    ):
        raise ValueError("invalid KernelTrace diagnostic mode/timing")
    if timing == "1" and not mode:
        raise ValueError("diagnostic timing requires a diagnostic mode")
    if mode == "record_only" and timing == "1":
        raise ValueError("record_only must run without diagnostic timing")
    return mode, timing == "1"


class KernelTrace:
    """Own log files and a shared polling thread for active native sources."""

    _lock = threading.RLock()
    _sources = {}
    _thread = None
    _stop = threading.Event()

    @classmethod
    def attach(cls, native_module):
        directory = os.environ.get("DEEPEP_TRACE_DIR")
        if not directory:
            return None
        mode, timing = diagnostic_config()
        # 仅测元数据时不创建采集源、轮询线程或日志文件。
        if mode == "metadata":
            return None
        source = native_module.get_kernel_trace()
        if source is None:
            return None
        identity = source.identity
        with cls._lock:
            # 同一 ACL context 的多个 Buffer 共享采集器，按引用计数关闭。
            if identity in cls._sources:
                if (cls._sources[identity]["mode"], cls._sources[identity]["timing"]) != (mode, timing):
                    raise ValueError("cannot change diagnostic mode while a trace source is active")
                cls._sources[identity]["references"] += 1
                return identity
            stream = None
            try:
                # 用主机、进程、设备和创建时间区分文件，校准记录先于 kernel 事件写入。
                calibration = source.calibration()
                Path(directory).mkdir(parents=True, exist_ok=True)
                name = f"kernel-{socket.gethostname()}-{os.getpid()}-d{calibration['device']}-{time.time_ns()}.jsonl"
                if mode in ("", "record_only", "full"):
                    stream = (Path(directory) / name).open("x", encoding="utf-8")
                    stream.write(json.dumps(calibration) + "\n")
                    stream.flush()
                cls._sources[identity] = dict(
                    source=source,
                    stream=stream,
                    references=1,
                    failed=False,
                    mode=mode,
                    timing=timing,
                    statistics={},
                    diagnostic_path=Path(directory) / name.replace("kernel-", "diagnostic-").replace(".jsonl", ".json"),
                    device=calibration["device"],
                    slots=calibration.get("slots", 128),
                    deferred_drains=0,
                )
                # 进程内的活跃源共用一个线程；只记录模式无需为它启动轮询。
                if mode != "record_only" and cls._thread is None:
                    cls._stop.clear()
                    cls._thread = threading.Thread(target=cls._poll, name="deepep-kernel-trace", daemon=True)
                    cls._thread.start()
                return identity
            except Exception as error:
                cls._sources.pop(identity, None)
                if cls._thread is not None and not cls._thread.is_alive():
                    cls._thread = None
                if stream is not None:
                    stream.close()
                warnings.warn(f"KernelTrace logging unavailable: {error}", RuntimeWarning)
                return None

    @staticmethod
    def _drain(entry, closing=False, explicit=False):
        # record_only 将读取和写盘推迟到显式采集边界或关闭时，轮询线程会跳过它。
        if entry["mode"] == "record_only":
            if not closing and not explicit:
                return
            entry["deferred_drains"] += 1
        try:
            if entry["mode"] not in ("", "record_only"):
                KernelTrace._diagnostic_drain(entry, closing)
                return
            # 正常批量路径由 C++ 生成 JSONL，Python 只负责追加和刷新文件。
            text = entry["source"].drain_trace_json(closing)
            if text:
                entry["stream"].write(text)
                entry["stream"].flush()
        except Exception as error:
            if not entry["failed"]:
                warnings.warn(f"KernelTrace logging failed: {error}", RuntimeWarning)
                entry["failed"] = True

    @staticmethod
    def _diagnostic_drain(entry, closing):
        mode, timing = entry["mode"], entry["timing"]
        # native 跳过编码，json_discard 编码后在原生侧丢弃，full 才写盘。
        begin = time.perf_counter_ns() if timing else 0
        batch = entry["source"].drain_trace_diagnostic(mode != "native", timing, closing, mode != "json_discard")
        if timing:
            batch["drain_call_wall_ns"] = time.perf_counter_ns() - begin
        text = batch.pop("text")
        stats = entry["statistics"]
        for key, value in batch.items():
            stats[key] = stats.get(key, 0) + value
        if mode != "full" or not text:
            return
        begin = time.perf_counter_ns() if timing else 0
        entry["stream"].write(text)
        entry["stream"].flush()
        if timing:
            stats["write_flush_ns"] = stats.get("write_flush_ns", 0) + time.perf_counter_ns() - begin

    @classmethod
    def _poll(cls):
        # 用 Event.wait 轮询，使退出信号可以提前结束等待。
        while not cls._stop.wait(0.002):
            with cls._lock:
                for entry in cls._sources.values():
                    cls._drain(entry)

    @classmethod
    def collect_boundary(cls, identity):
        """Diagnostic-only collection after the caller synchronizes, outside API timing."""
        # 调用者已在计时窗口外同步设备，这里只负责收集并检查错误。
        if identity is None:
            return
        with cls._lock:
            entry = cls._sources[identity]
            cls._drain(entry, explicit=True)
            if entry["failed"]:
                raise RuntimeError("KernelTrace boundary collection failed")

    @classmethod
    def detach(cls, identity):
        with cls._lock:
            entry = cls._sources.get(identity)
            if entry is None:
                return
            entry["references"] -= 1
            if entry["references"]:
                return
            try:
                # 最后一个使用者退出时，等设备写完再读取剩余记录和汇总。
                entry["source"].synchronize()
                cls._drain(entry, closing=True)
            finally:
                try:
                    if entry["stream"] is not None:
                        entry["stream"].close()
                    if entry["mode"] not in ("", "record_only"):
                        report = dict(
                            mode=entry["mode"],
                            timing=entry["timing"],
                            pid=os.getpid(),
                            device=entry["device"],
                            failed=entry["failed"],
                            counters=entry["statistics"],
                        )
                        entry["diagnostic_path"].write_text(json.dumps(report, indent=2), encoding="utf-8")
                finally:
                    del cls._sources[identity]

    @classmethod
    def shutdown(cls):
        # 先停线程，再强制排空剩余采集源，避免关闭文件时仍有后台写入。
        cls._stop.set()
        thread = cls._thread
        if thread is not None:
            thread.join()
        with cls._lock:
            for identity in list(cls._sources):
                cls._sources[identity]["references"] = 1
                try:
                    cls.detach(identity)
                except Exception as error:
                    warnings.warn(f"KernelTrace shutdown failed: {error}", RuntimeWarning)
            cls._thread = None


atexit.register(KernelTrace.shutdown)


_identity_lock = threading.Lock()
_identities = itertools.count(1)
_process_id = uuid.uuid4().hex


class DispatchPythonProbe:
    """Opt-in single-thread diagnostic; retains only preallocated timestamp rows, never tensors."""

    version = 1
    # 时间点名称与验收脚本里的区间端点一致，-1 表示该时间点未经过。
    points = (
        "entry",
        "dependency_begin",
        "dependency_end",
        "trace_created",
        "trace_entered",
        "stream_entered",
        "layout_begin",
        "layout_end",
        "args_ready",
        "native_end",
        "epilogue_end",
        "stream_exited",
        "trace_exited",
        "handle_end",
        "outputs_recorded",
        "finish_event_begin",
        "finish_event_end",
        "return",
        "metadata_begin",
        "identity_done",
        "template_lookup_done",
        "template_done",
        "dynamic_ready",
        "push_done",
        "restore_begin",
        "restore_done",
    )

    def __init__(self, capacity, clock=time.perf_counter_ns):
        if type(capacity) is not int or not 0 < capacity <= 1000000:
            raise ValueError("invalid Python dispatch probe capacity")
        # 测量开始前分配好采样行，计时点只更新当前行的对应位置。
        self.rows = [[-1] * len(self.points) for _ in range(capacity)]
        self.indices = {name: i for i, name in enumerate(self.points)}
        self.clock = clock
        self.owner = threading.get_ident()
        self.used = 0
        self.active = None

    def begin(self):
        # 每次调用占一行；限制线程和嵌套，避免覆盖尚未结束的样本。
        if self.owner != threading.get_ident() or self.active is not None:
            raise RuntimeError("Python dispatch probe requires non-nested calls on its owner thread")
        if self.used == len(self.rows):
            raise RuntimeError("Python dispatch probe capacity exhausted")
        self.active = self.rows[self.used]
        self.used += 1
        self.mark("entry")

    def mark(self, point):
        self.active[self.indices[point]] = self.clock()

    def finish(self):
        self.mark("return")
        self.active = None

    def take(self):
        if self.owner != threading.get_ident() or self.active is not None:
            raise RuntimeError("incomplete or wrong-thread Python dispatch probe export")
        return dict(
            version=self.version,
            samples=[
                dict(ordinal=i + 1, ticks_ns=dict(zip(self.points, row)))
                for i, row in enumerate(self.rows[: self.used])
            ],
        )


def _next_identity():
    with _identity_lock:
        return next(_identities)


def _global_rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else None


class _CallMetadata(Mapping):
    """Lazy read-only mapping; no per-call dict/ChainMap and no tensor ownership."""

    __slots__ = ("api", "handle", "template", "base")

    def __init__(self, api, handle, template, base):
        self.api, self.handle, self.template, self.base = api, handle, template, base

    def __getitem__(self, key):
        # 按需查找：先取本次调用的 ID，再找形状模板和 Buffer 级字段。
        if key == "api_call_id":
            return self.api
        if key == "handle_id":
            return self.handle
        if key in self.template:
            return self.template[key]
        return self.base[key]

    def __iter__(self):
        yield "api_call_id"
        yield "handle_id"
        yield from (k for k in self.template if k not in ("api_call_id", "handle_id"))
        yield from (k for k in self.base if k not in self.template and k not in ("api_call_id", "handle_id"))

    def __len__(self):
        return sum(1 for _ in self)


class KernelTraceSession:
    """Host-only metadata for one Buffer; each call is snapshotted by native Reserve."""

    def __init__(self, native, comm_id, ep_size, global_rank=None):
        if getattr(native, "kernel_trace_metadata_version", 0) != 2:
            raise RuntimeError("KernelTrace metadata requires the matching rebuilt native extension (version 2)")
        self.native = native
        self.diagnostic_mode, self.diagnostic_timing = diagnostic_config()
        if self.diagnostic_mode == "record_only":
            self.diagnostic_mode = ""  # Match production metadata work, without diagnostic per-call counters.
        self.diagnostic_stats = {}
        # Buffer 固定字段只构造一次，供所有形状模板共享。
        self.fields = dict(
            metadata_version=1,
            comm_id=comm_id,
            buffer_id=_next_identity(),
            process_id=_process_id,
            hostname=socket.gethostname(),
            ep_size=ep_size,
        )
        if global_rank is not None:
            self.fields["global_rank"] = global_rank
        self._base = native.make_trace_metadata(self.fields)
        self._templates = {}
        self._element_bytes = {}

    @contextmanager
    def call(
        self, api_name, x, routes, num_experts, handle=None, *, cached=False, expanded=False, capacity=0, _probe=None
    ):
        if _probe is not None:
            _probe.mark("metadata_begin")
        started = time.perf_counter_ns() if self.diagnostic_timing else 0
        # handle ID 随路由对象复用；api_call_id 每次调用重新分配。
        with _identity_lock:
            handle_id = getattr(handle, "_trace_handle_id", None) if handle is not None else None
            if handle_id is None:
                handle_id = next(_identities)
                if handle is not None:
                    handle._trace_handle_id = handle_id
            api_call_id = next(_identities)
        # 缓存键只含形状、dtype 等标量信息，避免延长用户张量或 handle 的生命周期。
        if _probe is not None:
            _probe.mark("identity_done")
        x_shape, route_shape, dtype = x.shape, routes.shape, x.dtype
        # 同一 dtype 的元素字节数可复用，稳定形状也能命中元数据模板。
        element_bytes = self._element_bytes.get(dtype)
        if element_bytes is None:
            element_bytes = x.element_size()
            if len(self._element_bytes) >= 64:
                self._element_bytes.clear()
            self._element_bytes[dtype] = element_bytes
        key = (
            api_name,
            x_shape[1],
            element_bytes,
            dtype,
            int(num_experts),
            route_shape[1],
            route_shape[0],
            x_shape[0],
            int(cached),
            int(expanded),
            int(capacity),
        )
        template = self._templates.get(key)
        if _probe is not None:
            _probe.mark("template_lookup_done")
        if template is None:
            fields = dict(
                zip(
                    (
                        "api_name",
                        "hidden_size",
                        "element_bytes",
                        "dtype",
                        "num_experts",
                        "topk",
                        "num_tokens",
                        "api_input_rows",
                        "cached",
                        "do_expand",
                        "token_capacity",
                    ),
                    key,
                )
            )
            fields["dtype"] = str(fields["dtype"])
            template = (fields, self.native.make_trace_metadata(fields, self._base))
            if len(self._templates) >= 64:
                # 在途记录持有原生模板的引用，清空缓存后仍可读取。
                self._templates.clear()
            self._templates[key] = template
        if _probe is not None:
            _probe.mark("template_done")
        fields_view = _CallMetadata(api_call_id, handle_id, template[0], self.fields)
        if _probe is not None:
            _probe.mark("dynamic_ready")
        # 把调用元数据挂到原生线程局部变量，供启动时取快照；退出作用域时恢复。
        previous = self.native.push_trace_metadata(template[1], api_call_id, handle_id)
        if _probe is not None:
            _probe.mark("push_done")
        if self.diagnostic_mode:
            kind = ("cached_dispatch" if cached else "fresh_dispatch") if api_name == "dispatch" else api_name
            stats = self.diagnostic_stats.setdefault(kind, dict(calls=0, enter_ns=0, restore_ns=0))
            stats["calls"] += 1
            if self.diagnostic_timing:
                stats["enter_ns"] += time.perf_counter_ns() - started
        try:
            yield fields_view
        finally:
            # 无论调用成功与否，都恢复原线程上下文，避免后续 kernel 串用 ID。
            if _probe is not None:
                _probe.mark("restore_begin")
            started = time.perf_counter_ns() if self.diagnostic_timing else 0
            self.native.restore_trace_metadata(previous)
            if _probe is not None:
                _probe.mark("restore_done")
            if self.diagnostic_timing:
                stats["restore_ns"] += time.perf_counter_ns() - started


def attach_trace_session(native, group, rank, ep_size, all_gather):
    """Agree once at Buffer initialization, including disabled ranks; no per-call collective."""
    enabled = bool(os.environ.get("DEEPEP_TRACE_DIR"))
    mode, timing = diagnostic_config()
    proposal = uuid.uuid4().hex if enabled and rank == 0 else None
    # 所有 EP rank 在 Buffer 初始化时统一开关、诊断模式和通信组 ID。
    peers = all_gather(group, None, ep_size, (enabled, proposal, mode, timing))
    if any((peer[0], peer[2], peer[3]) != (enabled, mode, timing) for peer in peers):
        raise ValueError("all EP ranks must agree on DEEPEP_TRACE_DIR enablement and diagnostic settings")
    if not enabled:
        return None, None
    if getattr(native, "kernel_trace_diagnostic_version", 0) != 4:
        raise RuntimeError("KernelTrace requires the matching rebuilt native extension")
    identity = None if mode == "metadata" else KernelTrace.attach(native)
    if identity is None and mode != "metadata":
        return None, None
    try:
        global_rank = _global_rank()
    except (ImportError, RuntimeError):
        global_rank = None
    return identity, KernelTraceSession(native, peers[0][1], ep_size, global_rank)
