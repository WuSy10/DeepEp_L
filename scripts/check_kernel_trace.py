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


METADATA_FIELDS = (
    "metadata_version",
    "comm_id",
    "buffer_id",
    "process_id",
    "hostname",
    "ep_size",
    "api_name",
    "api_call_id",
    "handle_id",
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
    "pid",
    "device",
    "stream",
)


def metadata_identity(row):
    keys = METADATA_FIELDS + ("global_rank", "kernel_input_rows", "kernel_output_rows")
    return tuple((key, row.get(key)) for key in keys)


def validate_metadata(row, ranks, where):
    if any(key not in row for key in METADATA_FIELDS):
        raise ValueError(f"{where}: missing Host metadata")
    if (
        row["metadata_version"] != 1
        or row["ep_size"] != ranks
        or row["api_name"] not in ("dispatch", "combine")
        or any(not row[key] for key in ("comm_id", "process_id", "hostname", "dtype"))
        or any(
            row[key] <= 0
            for key in ("buffer_id", "api_call_id", "handle_id", "hidden_size", "element_bytes", "num_experts", "topk")
        )
        or row["num_tokens"] < 0
        or row["api_input_rows"] < 0
        or row["cached"] not in (0, 1)
        or row["do_expand"] not in (0, 1)
        or not str(row["stream"]).isdigit()
        or int(row["stream"]) <= 0
    ):
        raise ValueError(f"{where}: invalid Host metadata")
    if row["op_name"] in ("dispatch", "combine"):
        if any(key not in row or row[key] < 0 for key in ("kernel_input_rows", "kernel_output_rows")):
            raise ValueError(f"{where}: missing kernel row counts")
    expected = "combine" if "combine" in row["op_name"] else "dispatch"
    if row["api_name"] != expected:
        raise ValueError(f"{where}: wrong API association")


def identity(row):
    return (
        tuple(row[key] for key in ("rank", "local_instance", "launch_ns", "op_name", "kernel_name", "aiv")),
        metadata_identity(row),
    )


def check(directory, ranks, require_ops=(), require_reuse=False, require_metadata=False):
    observed = set()
    completed = 0
    rank_ops = {}
    calls = {}
    handles = {}
    comm_ranks = {}
    buffer_comms = {}
    paths = sorted(directory.glob("kernel-*.jsonl"))
    if not paths:
        raise ValueError(f"no trace files in {directory}")
    for path in paths:
        offset = None
        slots = 128  # Older calibration records used the fixed-size ring.
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
                    slots = row.get("slots", 128)
                    if type(slots) is not int or not 128 <= slots <= 16384:
                        raise ValueError(f"{where}: invalid trace capacity")
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
                    if require_metadata or "metadata_version" in row:
                        validate_metadata(row, ranks, where)
                        comm_ranks.setdefault(row["comm_id"], set()).add(row["rank"])
                        buffer_key = (row["process_id"], row["buffer_id"])
                        if buffer_key in buffer_comms and buffer_comms[buffer_key] != row["comm_id"]:
                            raise ValueError(f"{where}: Buffer communication identity changed")
                        buffer_comms[buffer_key] = row["comm_id"]
                        call_key = (row["process_id"], row["api_call_id"])
                        call_fields = tuple((key, row.get(key)) for key in METADATA_FIELDS if key not in ("stream",))
                        if call_key in calls and calls[call_key] != call_fields:
                            raise ValueError(f"{where}: API metadata changed within one call")
                        calls[call_key] = call_fields
                        handle_key = (row["process_id"], row["handle_id"])
                        route = tuple(
                            row[key]
                            for key in ("comm_id", "buffer_id", "num_experts", "topk", "num_tokens", "do_expand")
                        )
                        if handle_key in handles and handles[handle_key] != route:
                            raise ValueError(f"{where}: handle routing metadata changed")
                        handles[handle_key] = route
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
        if require_reuse and len(ended) <= slots:
            raise ValueError(f"{path}: need more than {slots} completed records to exercise slot reuse")
        completed += len(ended)
    if observed != set(range(ranks)):
        raise ValueError(f"expected ranks 0..{ranks - 1}, got {sorted(observed)}")
    for rank, ops in rank_ops.items():
        missing = set(require_ops) - ops
        if missing:
            raise ValueError(f"rank {rank}: missing operations {sorted(missing)}")
    if require_metadata:
        for comm_id, members in comm_ranks.items():
            if members != set(range(ranks)):
                raise ValueError(f"communication group {comm_id}: missing ranks, got {sorted(members)}")
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--require-ops", nargs="+", choices=sorted(set(KERNEL_OPS.values())), default=[])
    parser.add_argument("--require-reuse", action="store_true")
    parser.add_argument("--require-metadata", action="store_true")
    args = parser.parse_args()
    print(
        f"verified {check(args.directory, args.ranks, args.require_ops, args.require_reuse, args.require_metadata)} completed kernel records"
    )
