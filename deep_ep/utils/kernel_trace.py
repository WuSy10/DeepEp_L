"""Process-local Python monitor for native kernel trace sources."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import socket
import threading
import time
import warnings
import uuid


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
        source = native_module.get_kernel_trace()
        if source is None:
            return None
        identity = source.identity
        with cls._lock:
            if identity in cls._sources:
                cls._sources[identity]["references"] += 1
                return identity
            stream = None
            try:
                calibration = source.calibration()
                Path(directory).mkdir(parents=True, exist_ok=True)
                name = f"kernel-{socket.gethostname()}-{os.getpid()}-d{calibration['device']}-{time.time_ns()}.jsonl"
                stream = (Path(directory) / name).open("x", encoding="utf-8")
                stream.write(json.dumps(calibration) + "\n")
                stream.flush()
                cls._sources[identity] = dict(source=source, stream=stream, references=1, failed=False)
                if cls._thread is None:
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
    def _drain(entry, closing=False):
        try:
            records = entry["source"].drain_trace(closing)
            if records:
                entry["stream"].writelines(json.dumps(record) + "\n" for record in records)
                entry["stream"].flush()
        except Exception as error:
            if not entry["failed"]:
                warnings.warn(f"KernelTrace logging failed: {error}", RuntimeWarning)
                entry["failed"] = True

    @classmethod
    def _poll(cls):
        while not cls._stop.wait(0.002):
            with cls._lock:
                for entry in cls._sources.values():
                    cls._drain(entry)

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
                entry["stream"].close()
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

    @contextmanager
    def call(self, api_name, x, routes, num_experts, handle=None, *, cached=False, expanded=False, capacity=0):
        with _identity_lock:
            handle_id = getattr(handle, "_trace_handle_id", None) if handle is not None else None
            if handle_id is None:
                handle_id = next(_identities)
                if handle is not None:
                    handle._trace_handle_id = handle_id
        fields = dict(
            self.fields,
            api_name=api_name,
            api_call_id=_next_identity(),
            handle_id=handle_id,
            hidden_size=int(x.size(1)),
            element_bytes=x.element_size(),
            dtype=str(x.dtype),
            num_experts=int(num_experts),
            topk=int(routes.size(1)),
            num_tokens=int(routes.size(0)),
            api_input_rows=int(x.size(0)),
            cached=int(cached),
            do_expand=int(expanded),
            token_capacity=int(capacity),
        )
        previous = self.native.exchange_kernel_trace_metadata(fields)
        try:
            yield fields
        finally:
            self.native.exchange_kernel_trace_metadata(previous)


def attach_trace_session(native, group, rank, ep_size, all_gather):
    """Agree once at Buffer initialization, including disabled ranks; no per-call collective."""
    enabled = bool(os.environ.get("DEEPEP_TRACE_DIR"))
    proposal = uuid.uuid4().hex if enabled and rank == 0 else None
    peers = all_gather(group, None, ep_size, (enabled, proposal))
    if any(peer[0] != enabled for peer in peers):
        raise ValueError("all EP ranks must agree on whether DEEPEP_TRACE_DIR is enabled")
    if not enabled:
        return None, None
    identity = KernelTrace.attach(native)
    if identity is None:
        return None, None
    try:
        global_rank = _global_rank()
    except (ImportError, RuntimeError):
        global_rank = None
    return identity, KernelTraceSession(native, peers[0][1], ep_size, global_rank)
