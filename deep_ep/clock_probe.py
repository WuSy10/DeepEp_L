"""先校准 Ascend 950 时钟，再批量记录单 AIV 探针算子的起止 tick。

每个算子使用独立的映射 Host 记录位置；全部提交后只同步一次探针流。
这是同步批次实验，异步后台收集见 clock_probe_async.py。
"""

import argparse
import json
import time
import uuid

from ._native import require_native
from .clock_calibration import calibrate, from_samples, validate_holdout


def operator_row(mapping, calibration_id, work_iters, native_row):
    """将一条原始记录换算为日志行，并同时保留原始 tick 与 Host 提交时间。

    launch_begin/end 是 CPU 调用底层 launch 前后的时间；start/end 是
    NPU 执行探针负载前后的 tick，二者不能当作同一段耗时。
    """
    sequence, launch_begin, launch_end, start, end, result = native_row
    if launch_end < launch_begin or start <= 0 or end < start:
        raise ValueError(f"invalid probe times at sequence {sequence}")
    start_host = mapping.to_host_ns(start)
    end_host = mapping.to_host_ns(end)
    return dict(type="operator", schema_version=1, calibration_id=calibration_id,
                device=mapping.device, sequence=sequence, work_iters=work_iters,
                cpu_launch_begin_ns=launch_begin, cpu_launch_end_ns=launch_end,
                cpu_launch_duration_ns=launch_end - launch_begin,
                npu_start_tick=start, npu_end_tick=end, npu_duration_ticks=end - start,
                npu_start_host_ns=start_host, npu_end_host_ns=end_host,
                npu_start_uncertainty_ns=mapping.mapping_uncertainty_ns(start),
                npu_end_uncertainty_ns=mapping.mapping_uncertainty_ns(end),
                npu_duration_ns=end_host - start_host, result=result)


def main():
    """同步参考实验：两批拟合、第三批验收、批量探针、统一写 JSONL。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--probe-warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--work-iters", type=int, default=32)
    parser.add_argument("--output", default="clock-probe.jsonl")
    args = parser.parse_args()
    if not 1 <= args.samples <= 4096 or not 0 <= args.warmup <= 4096 or not 0 < args.interval <= 3600:
        parser.error("samples 1..4096, warmup 0..4096, interval (0, 3600] seconds required")
    if (not 1 <= args.iterations <= 16384 or not 0 <= args.probe_warmup <= 4096 or
            args.iterations + args.probe_warmup > 16384 or not 1 <= args.work_iters <= 4096):
        parser.error("iterations + probe-warmup must be 1..16384; work-iters must be 1..4096")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    mapping = calibrate(args.samples, args.warmup, args.interval)
    # 留出批次只验收既有映射；失败时不启动正式探针。
    time.sleep(args.interval)
    native = require_native("clock probe")
    device, soc, samples = native.clock_calibration_samples(args.samples, args.warmup)
    holdout = from_samples(device, soc, samples)
    verdict = validate_holdout(mapping, holdout)
    if verdict["holdout_violations"]:
        raise RuntimeError(f"clock calibration holdout failed: {verdict}")

    # 同步探针为每次 launch 预留独立槽位，全部提交后才同步一次流。
    device, soc, raw_rows = native.clock_probe_batch(args.iterations, args.probe_warmup, args.work_iters)
    if device != mapping.device or soc != mapping.soc or len(raw_rows) != args.iterations:
        raise RuntimeError("probe device, SoC, or record count differs from calibration")
    calibration_id = uuid.uuid4().hex
    header = dict(type="calibration", schema_version=1, calibration_id=calibration_id,
                  transport="mapped_host_mte3", clock=mapping.to_dict(),
                  holdout=holdout.to_dict(), **verdict,
                  probe=dict(iterations=args.iterations, warmup=args.probe_warmup,
                             work_iters=args.work_iters, workload="vector_add_32_int32"))
    entries = [operator_row(mapping, calibration_id, args.work_iters, row) for row in raw_rows]
    with open(args.output, "w", encoding="utf-8") as output:
        output.write(json.dumps(header) + "\n")
        for entry in entries:
            output.write(json.dumps(entry) + "\n")
    print(f"clock probe passed: device={device} operators={len(raw_rows)} "
          f"holdout_violations=0 calibration_id={calibration_id} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
