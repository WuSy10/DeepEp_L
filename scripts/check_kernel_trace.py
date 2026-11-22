"""Verify a completed single-node kernel trace run, including every expected rank."""

import argparse
import json
from pathlib import Path


KERNEL_OPS = {
    "notify_dispatch_kernel": "notify_dispatch",
    "cached_notify_dispatch_kernel": "cached_notify_dispatch",
    "cached_notify_combine_kernel": "cached_notify_combine",
    "v1_dispatch_udma_kernel": "dispatch",
    **{
        f"v1_combine_udma{suffix}_kernel": "combine"
        for suffix in ("", "_comm", "_reduce", "_early", "_remaining", "_pipeline")
    },
}


def identity(row):
    return tuple(row[key] for key in ("rank", "local_instance", "launch_ns", "op_name", "kernel_name", "aiv"))


def check(directory, ranks, require_ops=(), require_reuse=False):
    observed = set()
    completed = 0
    rank_ops = {}
    paths = sorted(directory.glob("kernel-*.jsonl"))
    if not paths:
        raise ValueError(f"no trace files in {directory}")
    for path in paths:
        offset = None
        launches = {}
        ended = set()
        finished = False
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                kind = row["event"]
                where = f"{path.name} line {number}"
                if finished or (offset is None and kind != "calibration"):
                    raise ValueError(f"{where}: unexpected event order")
                if kind == "calibration":
                    if offset is not None or row["tick_hz"] != 1_000_000_000 or row["tick"] <= 0:
                        raise ValueError(f"{where}: invalid calibration")
                    offset = row["offset_ns"]
                elif kind == "launch":
                    gid = row["gid"]
                    if (
                        gid <= 0
                        or gid in launches
                        or row["status"]
                        or KERNEL_OPS.get(row["kernel_name"]) != row["op_name"]
                        or not 1 <= row["aiv"] <= 64
                    ):
                        raise ValueError(f"{where}: invalid launch")
                    launches[gid] = identity(row)
                elif kind == "end":
                    gid = row["gid"]
                    if launches.get(gid) != identity(row) or gid in ended:
                        raise ValueError(f"{where}: unmatched end")
                    if not (
                        0 <= row["start_tick"] <= row["end_tick"]
                        and row["start_ns"] == row["start_tick"] + offset
                        and row["end_ns"] == row["end_tick"] + offset
                        and row["elapsed_ns"] == row["end_tick"] - row["start_tick"]
                    ):
                        raise ValueError(f"{where}: inconsistent device timing")
                    observed.add(row["rank"])
                    rank_ops.setdefault(row["rank"], set()).add(row["op_name"])
                    ended.add(gid)
                elif kind == "summary":
                    if (
                        row["dropped"]
                        or row["incomplete"]
                        or launches.keys() != ended
                        or row["completed"] != len(ended)
                    ):
                        raise ValueError(f"{where}: incomplete trace: {row}")
                    finished = True
                else:
                    raise ValueError(f"{where}: {kind}")
        if not finished:
            raise ValueError(f"{path}: missing final summary")
        if require_reuse and len(ended) <= 128:
            raise ValueError(f"{path}: need more than 128 completed records to exercise slot reuse")
        completed += len(ended)
    if observed != set(range(ranks)):
        raise ValueError(f"expected ranks 0..{ranks - 1}, got {sorted(observed)}")
    for rank, ops in rank_ops.items():
        missing = set(require_ops) - ops
        if missing:
            raise ValueError(f"rank {rank}: missing operations {sorted(missing)}")
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--require-ops", nargs="+", choices=sorted(set(KERNEL_OPS.values())), default=[])
    parser.add_argument("--require-reuse", action="store_true")
    args = parser.parse_args()
    print(
        f"verified {check(args.directory, args.ranks, args.require_ops, args.require_reuse)} completed kernel records"
    )
