"""后续算子继续提交时，后台收集单 AIV 探针写入映射 Host 的时间戳。

收集线程先复制完成的设备记录，再归还映射槽位；写入线程批量换算并落盘。
这是有界队列实验；同步参考命令在 clock_probe.py。
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
                 queue_records, poll_ms, flush_ms, flush_records,
                 submit_interval_us, drain_timeout):
    """运行提交、收集、写入三段流水线，返回最终计数。

    映射槽位属于 NPU/原生层；ready 队列只存已复制的 CPU 数据。
    队列满时收集暂停，槽位不会提前复用，背压最终会造成日志丢弃。
    """
    ready = queue.Queue(maxsize=queue_records)
    producer_done = threading.Event()
    stopped = threading.Event()
    failures = []
    sentinel = object()
    final = {}

    def fail(exc):
        """记录后台异常并通知其余线程停止正常工作。"""
        failures.append(exc)
        stopped.set()

    def collect():
        """等 Event 完成后取 CPU 副本；成功入队才释放原映射槽位。"""
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
                rows = probe.poll()  # 空结果表示队首槽位的 Event 尚未完成。
                if rows:
                    # 先让 CPU 副本进入有界队列，再允许设备复用映射槽位。
                    ready.put_nowait(rows[0])
                    probe.release(rows[0][0])
                    continue
                if producer_done.is_set() and probe.pending_records() == 0:
                    return
                time.sleep(poll_ms / 1000)
        except BaseException as exc:
            fail(exc)

    def write(output):
        """从 CPU 队列取记录，批量换算 tick 并写出 JSONL。"""
        pending = []  # 等待批量换算和写盘的原始 CPU 副本。
        logged = 0
        last_flush = time.monotonic()

        def flush():
            """把 pending 中的原始 tick 换算为 Host 纳秒并一次写出。"""
            nonlocal last_flush, logged
            if pending:
                lines = []
                for raw in pending:
                    row = operator_row(mapping, calibration_id, header["probe"]["work_iters"], raw)
                    row["schema_version"] = 2
                    row["generation"] = row["sequence"] + 1
                    lines.append(json.dumps(row) + "\n")
                output.write("".join(lines))
                output.flush()
                logged += len(pending)
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
                    flush()
                    if attempted != accepted + dropped or accepted != logged:
                        raise RuntimeError("async probe counters do not match written records")
                    summary = dict(type="summary", schema_version=2,
                                   calibration_id=calibration_id, attempted=attempted,
                                   accepted=accepted, dropped=dropped, logged=logged)
                    output.write(json.dumps(summary) + "\n")
                    output.flush()
                    final.update(summary)
                    continue
                if item is not None:
                    pending.append(item)
                if pending and (len(pending) >= flush_records or
                                time.monotonic() - last_flush >= flush_ms / 1000):
                    flush()
        except BaseException as exc:
            fail(exc)

    with open(output_path, "w", encoding="utf-8") as output:
        # 校准 header 先于算子提交落盘；正常收尾时 summary 最后写入。
        output.write(json.dumps(header) + "\n")
        output.flush()
        writer = threading.Thread(target=write, args=(output,), name="clock-probe-writer", daemon=True)
        collector = threading.Thread(target=collect, name="clock-probe-collector", daemon=True)
        writer.start()
        collector.start()
        try:
            for _ in range(iterations):
                if stopped.is_set():
                    raise RuntimeError("async probe background worker failed")
                # submit() 返回 false 只表示该算子的日志丢弃，算子仍会运行。
                probe.submit()
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
            probe.abandon()  # 设备状态不明时保留映射页，避免设备继续写已释放内存。
            raise failures[0]
    return final


def main():
    """单卡异步实验入口：先校准和留出验收，再启动事件门控采集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--probe-warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--work-iters", type=int, default=32)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--queue-records", type=int, default=128)
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
            not 1 <= args.slots <= 16384 or not 1 <= args.queue_records <= 65536 or
            not 0 < args.poll_ms <= 1000 or not 0 < args.flush_ms <= 10000 or
            not 1 <= args.flush_records <= 100000 or not 0 <= args.submit_interval_us <= 1_000_000 or
            not 0 < args.drain_timeout <= 3600):
        parser.error("invalid sample, capacity, interval, or probe arguments; see --help")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    mapping = calibrate(args.samples, args.warmup, args.interval)
    # 前两批决定映射；第三批只验证，失败时不会创建异步探针。
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
    probe = native.AsyncClockProbe(args.slots, args.work_iters)
    if probe.device != mapping.device or probe.soc != mapping.soc:
        probe.close()
        raise RuntimeError("async probe device or SoC differs from calibration")
    calibration_id = uuid.uuid4().hex
    header = dict(type="calibration", schema_version=2, calibration_id=calibration_id,
                  transport="mapped_host_mte3", collection="event_gated_async",
                  clock=mapping.to_dict(), holdout=holdout.to_dict(), **verdict,
                  probe=dict(iterations=args.iterations, warmup=args.probe_warmup,
                             work_iters=args.work_iters, workload="vector_add_32_int32",
                             slots=args.slots, queue_records=args.queue_records))
    result = run_pipeline(probe, mapping, calibration_id, header, args.output,
                          iterations=args.iterations, queue_records=args.queue_records,
                          poll_ms=args.poll_ms,
                          flush_ms=args.flush_ms, flush_records=args.flush_records,
                          submit_interval_us=args.submit_interval_us,
                          drain_timeout=args.drain_timeout)
    print(f"async clock probe passed: device={device} attempted={result['attempted']} "
          f"logged={result['logged']} dropped={result['dropped']} "
          f"holdout_violations=0 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
