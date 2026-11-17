import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from deep_ep import clock_probe
from deep_ep.clock_calibration import ClockCalibration, from_samples


def mapping():
    first = from_samples(0, "Ascend950PR", [(999_999_998, 1_000_000_000, 1_000_000_002)])
    second = from_samples(0, "Ascend950PR", [(1_999_999_998, 2_000_000_000, 2_000_000_002)])
    return ClockCalibration(first, second)


def test_operator_row_keeps_launch_and_device_durations_separate():
    row = clock_probe.operator_row(mapping(), "cal-1", 32,
                                   (7, 3_000_000_100, 3_000_000_200,
                                    3_000_000_120, 3_000_000_150, 40))
    assert row["calibration_id"] == "cal-1"
    assert row["cpu_launch_duration_ns"] == 100
    assert row["npu_duration_ticks"] == 30
    assert row["npu_duration_ns"] == 30
    assert row["npu_start_tick"] == 3_000_000_120
    assert row["npu_start_host_ns"] == 3_000_000_120
    with pytest.raises(ValueError):
        clock_probe.operator_row(mapping(), "cal-1", 32, (7, 200, 100, 120, 150, 40))


def test_cli_writes_calibration_then_one_row_per_retained_operator(tmp_path, monkeypatch):
    npu_calls = []
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(set_device=lambda device: npu_calls.append(("device", device)),
                                synchronize=lambda: npu_calls.append(("sync",)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(clock_probe, "calibrate", lambda *args: mapping())
    monkeypatch.setattr(clock_probe.time, "sleep", lambda seconds: None)
    native_calls = []
    def probe_batch(count, warmup, work_iters):
        native_calls.append((count, warmup, work_iters))
        return (0, "Ascend950PR", [(1, 3_000_000_100, 3_000_000_200,
                                    3_000_000_120, 3_000_000_150, 34)])

    native = SimpleNamespace(
        clock_calibration_samples=lambda count, warmup: (
            0, "Ascend950PR", [(2_999_999_998, 3_000_000_000, 3_000_000_002)]),
        clock_probe_batch=probe_batch)
    monkeypatch.setattr(clock_probe, "require_native", lambda operation: native)
    output = tmp_path / "probe.jsonl"
    monkeypatch.setattr(sys, "argv", ["clock_probe", "--samples", "1", "--warmup", "0",
                                      "--interval", "1", "--probe-warmup", "1",
                                      "--iterations", "1", "--work-iters", "32",
                                      "--output", str(output)])

    assert clock_probe.main() == 0
    header, operator = [json.loads(line) for line in output.read_text().splitlines()]
    assert npu_calls == [("device", 0), ("sync",)]
    assert native_calls == [(1, 1, 32)]
    assert header["type"] == "calibration"
    assert header["holdout_violations"] == 0
    assert header["probe"]["workload"] == "vector_add_32_int32"
    assert operator["type"] == "operator"
    assert operator["calibration_id"] == header["calibration_id"]
    assert operator["sequence"] == 1
    assert operator["cpu_launch_duration_ns"] == 100
    assert operator["npu_duration_ns"] == 30
