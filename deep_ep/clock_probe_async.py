"""Collect mapped-Host single-AIV timestamps while later batches are submitted.

This is a bounded experiment. Completed device records are copied into a Host
queue before their mapped slots are reused; a separate writer converts and
flushes JSONL. The existing clock_probe command remains the synchronous
reference run.
"""

import argparse
import json
import queue
import threading
import time
import uuid

from ._native import require_native
from .clock_calibration import calibrate, from_samples, validate_holdout
from .clock_probe import operator_row


def run_pipeline(probe, mapping, calibration_id, header, output_path, *, iterations,
                 batch_size, queue_batches, poll_ms, flush_ms, flush_records,
                 submit_interval_us, drain_timeout):
    """Submit, collect and write with bounded queues; return final counters."""
    ready = queue.Queue(maxsize=queue_batches)
    producer_done = threading.Event()
    stopped = threading.Event()
    failures = []
    sentinel = object()
    final = {}

    def fail(exc):
        failures.append(exc)
        stopped.set()

    def collect():
        deadline = None
        try:
            while not stopped.is_set():
                if producer_done.is_set() and deadline is None:
                    deadline = time.monotonic() + drain_timeout
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("async probe drain timed out; mapped slots were not released")
                if ready.full():
                    time.sleep(poll_ms / 1000)
                    continue
                rows = probe.poll()  # Empty means the oldest batch event is not ready.
                if rows:
                    # The mapped block remains reserved until the CPU copy is queued.
                    ready.put_nowait(rows)
                    probe.release(rows[0][0])
                    continue
                if producer_done.is_set() and probe.pending_batches() == 0:
                    return
                time.sleep(poll_ms / 1000)
        except BaseException as exc:
            fail(exc)

    def write(output):
        pending = []
        logged = 0
        last_flush = time.monotonic()

        def flush():
            nonlocal last_flush
            if pending:
                output.write("".join(pending))
                output.flush()
                pending.clear()
            last_flush = time.monotonic()

        try:
            while True:
                try:
                    item = ready.get(timeout=flush_ms / 1000)
                except queue.Empty:
                    item = None
                if item is sentinel:
                    flush()
                    return
                if isinstance(item, tuple) and item[0] == "summary":
                    attempted, accepted, dropped = item[1]
                    if attempted != accepted + dropped or accepted != logged:
                        raise RuntimeError("async probe counters do not match written records")
                    flush()
                    summary = dict(type="summary", schema_version=2,
                                   calibration_id=calibration_id, attempted=attempted,
                                   accepted=accepted, dropped=dropped, logged=logged)
                    output.write(json.dumps(summary) + "\n")
                    output.flush()
                    final.update(summary)
                    continue
                if item is not None:
                    for raw in item:
                        row = operator_row(mapping, calibration_id, header["probe"]["work_iters"], raw)
                        row["schema_version"] = 2
                        row["generation"] = row["sequence"] + 1
                        pending.append(json.dumps(row) + "\n")
                        logged += 1
                if pending and (len(pending) >= flush_records or
                                time.monotonic() - last_flush >= flush_ms / 1000):
                    flush()
        except BaseException as exc:
            fail(exc)

    with open(output_path, "w", encoding="utf-8") as output:
        output.write(json.dumps(header) + "\n")
        output.flush()
        writer = threading.Thread(target=write, args=(output,), name="clock-probe-writer", daemon=True)
        collector = threading.Thread(target=collect, name="clock-probe-collector", daemon=True)
        writer.start()
        collector.start()
        try:
            for offset in range(0, iterations, batch_size):
                if stopped.is_set():
                    raise RuntimeError("async probe background worker failed")
                probe.submit(min(batch_size, iterations - offset))
                if submit_interval_us:
                    time.sleep(submit_interval_us / 1_000_000)
        except BaseException as exc:
            fail(exc)
        finally:
            producer_done.set()

        collector.join(timeout=drain_timeout + 2)
        if collector.is_alive():
            fail(TimeoutError("async probe collector did not stop"))
        if not failures:
            try:
                probe.close()
            except BaseException as exc:
                fail(exc)
        if not failures:
            enqueue_deadline = time.monotonic() + drain_timeout
            while True:
                try:
                    ready.put(("summary", probe.counts()), timeout=0.1)
                    break
                except queue.Full:
                    if stopped.is_set() or not writer.is_alive() or time.monotonic() >= enqueue_deadline:
                        fail(RuntimeError("async probe writer stopped before summary"))
                        break
        if writer.is_alive():
            enqueue_deadline = time.monotonic() + drain_timeout
            while True:
                try:
                    ready.put(sentinel, timeout=0.1)
                    break
                except queue.Full:
                    if not writer.is_alive() or time.monotonic() >= enqueue_deadline:
                        fail(TimeoutError("async probe writer queue did not drain"))
                        break
            writer.join(timeout=drain_timeout + 2)
        if writer.is_alive():
            fail(TimeoutError("async probe writer did not stop"))
        if failures:
            probe.abandon()  # Unknown device state: do not reuse/free mapped pages.
            raise failures[0]
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--probe-warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--work-iters", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slot-batches", type=int, default=8)
    parser.add_argument("--queue-batches", type=int, default=8)
    parser.add_argument("--poll-ms", type=float, default=1.0)
    parser.add_argument("--flush-ms", type=float, default=100.0)
    parser.add_argument("--flush-records", type=int, default=256)
    parser.add_argument("--submit-interval-us", type=int, default=1000)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument("--output", default="clock-probe-async.jsonl")
    args = parser.parse_args()
    if (not 1 <= args.samples <= 4096 or not 0 <= args.warmup <= 4096 or
            not 0 < args.interval <= 3600 or not 0 <= args.probe_warmup <= 4096 or
            not 1 <= args.iterations <= 1_000_000 or not 1 <= args.work_iters <= 4096 or
            not 1 <= args.batch_size <= 1024 or not 1 <= args.slot_batches <= 64 or
            args.batch_size * args.slot_batches > 16384 or not 1 <= args.queue_batches <= 64 or
            not 0 < args.poll_ms <= 1000 or not 0 < args.flush_ms <= 10000 or
            not 1 <= args.flush_records <= 100000 or not 0 <= args.submit_interval_us <= 1_000_000 or
            not 0 < args.drain_timeout <= 3600):
        parser.error("invalid sample, capacity, interval, or probe arguments; see --help")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    mapping = calibrate(args.samples, args.warmup, args.interval)
    time.sleep(args.interval)
    native = require_native("async clock probe")
    device, soc, samples = native.clock_calibration_samples(args.samples, args.warmup)
    holdout = from_samples(device, soc, samples)
    verdict = validate_holdout(mapping, holdout)
    if verdict["holdout_violations"]:
        raise RuntimeError(f"clock calibration holdout failed: {verdict}")
    if args.probe_warmup:
        warm_device, warm_soc, rows = native.clock_probe_batch(args.probe_warmup, 0, args.work_iters)
        if warm_device != mapping.device or warm_soc != mapping.soc or len(rows) != args.probe_warmup:
            raise RuntimeError("async probe warmup differs from calibration")
    probe = native.AsyncClockProbe(args.batch_size, args.slot_batches, args.work_iters)
    if probe.device != mapping.device or probe.soc != mapping.soc:
        probe.close()
        raise RuntimeError("async probe device or SoC differs from calibration")
    calibration_id = uuid.uuid4().hex
    header = dict(type="calibration", schema_version=2, calibration_id=calibration_id,
                  transport="mapped_host_mte3", collection="event_gated_async",
                  clock=mapping.to_dict(), holdout=holdout.to_dict(), **verdict,
                  probe=dict(iterations=args.iterations, warmup=args.probe_warmup,
                             work_iters=args.work_iters, workload="vector_add_32_int32",
                             batch_size=args.batch_size, slot_batches=args.slot_batches,
                             queue_batches=args.queue_batches))
    result = run_pipeline(probe, mapping, calibration_id, header, args.output,
                          iterations=args.iterations, batch_size=args.batch_size,
                          queue_batches=args.queue_batches, poll_ms=args.poll_ms,
                          flush_ms=args.flush_ms, flush_records=args.flush_records,
                          submit_interval_us=args.submit_interval_us,
                          drain_timeout=args.drain_timeout)
    print(f"async clock probe passed: device={device} attempted={result['attempted']} "
          f"logged={result['logged']} dropped={result['dropped']} "
          f"holdout_violations=0 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
