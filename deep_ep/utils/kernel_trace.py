"""Process-local Python monitor for native kernel trace sources."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from collections import ChainMap
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
        if mode == "metadata":
            return None
        source = native_module.get_kernel_trace()
        if source is None:
            return None
        identity = source.identity
        with cls._lock:
            if identity in cls._sources:
                if (cls._sources[identity]["mode"], cls._sources[identity]["timing"]) != (mode, timing):
                    raise ValueError("cannot change diagnostic mode while a trace source is active")
                cls._sources[identity]["references"] += 1
                return identity
            stream = None
            try:
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
        # Even if another source owns a polling thread, deferred sources are never collected early.
        if entry["mode"] == "record_only":
            if not closing and not explicit:
                return
            entry["deferred_drains"] += 1
        try:
            if entry["mode"] not in ("", "record_only"):
                KernelTrace._diagnostic_drain(entry, closing)
                return
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
        while not cls._stop.wait(0.002):
            with cls._lock:
                for entry in cls._sources.values():
                    cls._drain(entry)

    @classmethod
    def collect_boundary(cls, identity):
        """Diagnostic-only collection after the caller synchronizes, outside API timing."""
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


def _next_identity():
    with _identity_lock:
        return next(_identities)


def _global_rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else None


class KernelTraceSession:
    """Host-only metadata for one Buffer; each call is snapshotted by native Reserve."""

    def __init__(self, native, comm_id, ep_size, global_rank=None):
        self.native = native
        self.diagnostic_mode, self.diagnostic_timing = diagnostic_config()
        if self.diagnostic_mode == "record_only":
            self.diagnostic_mode = ""  # Match production metadata work, without diagnostic per-call counters.
        self.diagnostic_stats = {}
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

    @contextmanager
    def call(self, api_name, x, routes, num_experts, handle=None, *, cached=False, expanded=False, capacity=0):
        started = time.perf_counter_ns() if self.diagnostic_timing else 0
        with _identity_lock:
            handle_id = getattr(handle, "_trace_handle_id", None) if handle is not None else None
            if handle_id is None:
                handle_id = next(_identities)
                if handle is not None:
                    handle._trace_handle_id = handle_id
        # Cache by scalar shape/context only; never retain user tensors or handles.
        key = (
            api_name,
            x.size(1),
            x.element_size(),
            x.dtype,
            int(num_experts),
            routes.size(1),
            routes.size(0),
            x.size(0),
            int(cached),
            int(expanded),
            int(capacity),
        )
        template = self._templates.get(key)
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
                self._templates.clear()  # In-flight native snapshots own their parents independently.
            self._templates[key] = template
        dynamic = dict(api_call_id=_next_identity(), handle_id=handle_id)
        previous = self.native.push_trace_metadata(template[1], dynamic["api_call_id"], handle_id)
        if self.diagnostic_mode:
            kind = ("cached_dispatch" if cached else "fresh_dispatch") if api_name == "dispatch" else api_name
            stats = self.diagnostic_stats.setdefault(kind, dict(calls=0, enter_ns=0, restore_ns=0))
            stats["calls"] += 1
            if self.diagnostic_timing:
                stats["enter_ns"] += time.perf_counter_ns() - started
        try:
            yield ChainMap(dynamic, template[0], self.fields)
        finally:
            started = time.perf_counter_ns() if self.diagnostic_timing else 0
            self.native.restore_trace_metadata(previous)
            if self.diagnostic_timing:
                stats["restore_ns"] += time.perf_counter_ns() - started


def attach_trace_session(native, group, rank, ep_size, all_gather):
    """Agree once at Buffer initialization, including disabled ranks; no per-call collective."""
    enabled = bool(os.environ.get("DEEPEP_TRACE_DIR"))
    mode, timing = diagnostic_config()
    proposal = uuid.uuid4().hex if enabled and rank == 0 else None
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
