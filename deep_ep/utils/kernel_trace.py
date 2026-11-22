"""Process-local Python monitor for native kernel trace sources."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import socket
import threading
import time
import warnings


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
