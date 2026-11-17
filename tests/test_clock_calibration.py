from types import SimpleNamespace

import pytest

from deep_ep.clock_calibration import ClockCalibration, calibrate, from_samples, host_time_ns, validate_holdout


def batch(host_ns, device_cycle, width=4, *, device=0, soc="Ascend950PR"):
    return from_samples(device, soc, [(host_ns - width // 2, device_cycle, host_ns + width // 2)])


def test_fitted_rate_predicts_independent_batch_with_large_timestamps():
    host = 10**18
    first = batch(host, 10**15)
    second = batch(host + 1_000_032_000, 10**15 + 1_000_000_000)
    third = batch(host + 2_000_064_000, 10**15 + 2_000_000_000)
    mapping = ClockCalibration(first, second)
    assert mapping.schema_version == 2
    assert mapping.rate_numerator_ns == 1_000_032_000
    assert mapping.rate_denominator_ticks == 1_000_000_000
    assert mapping.to_host_ns(third.anchor[1]) == third.midpoint_ns
    assert validate_holdout(mapping, third) == {"holdout_max_violation_ns": 0, "holdout_violations": 0}
    assert mapping.to_dict()["observed_ticks_per_host_ns"] < 1


def test_holdout_detects_rate_change():
    first = batch(1_000_000, 1_000_000)
    second = batch(1_001_000_000, 1_001_000_000)
    third = batch(2_002_100_000, 2_001_000_000)
    result = validate_holdout(ClockCalibration(first, second), third)
    assert result["holdout_violations"] == 1
    assert result["holdout_max_violation_ns"] > 100_000


def test_shortest_bracket_and_negative_extrapolation():
    first = from_samples(0, "Ascend950PR", [(1000, 50, 1100), (1199, 250, 1201)])
    second = batch(2200, 1250)
    mapping = ClockCalibration(first, second)
    assert first.selected == 1
    assert mapping.to_host_ns(249) == 1199
    assert mapping.mapping_uncertainty_ns(2250) >= second.uncertainty_ns


@pytest.mark.parametrize("samples", [[], [(1, 2)], [(3, 1, 2)], [(1, 0, 2)], [(1.0, 2, 3)]])
def test_reject_bad_samples(samples):
    with pytest.raises(ValueError):
        from_samples(0, "Ascend950PR", samples)


def test_reject_cross_device_and_counter_reset():
    first = batch(1000, 1000)
    with pytest.raises(ValueError):
        ClockCalibration(first, batch(2000, 2000, device=1))
    with pytest.raises(ValueError):
        ClockCalibration(first, batch(2000, 900))


def test_calibrate_uses_bundled_native_loader(monkeypatch):
    calls = []
    samples = iter([(0, "Ascend950PR", [(998, 1000, 1002)]),
                    (0, "Ascend950PR", [(1_000_998, 1_001_000, 1_001_002)])])
    native = SimpleNamespace(clock_calibration_samples=lambda count, warmup: next(samples),
                             clock_monotonic_ns=lambda: 1234)

    def load(operation):
        calls.append(operation)
        return native

    monkeypatch.setattr("deep_ep.clock_calibration.require_native", load)
    monkeypatch.setattr("deep_ep.clock_calibration.time.sleep", lambda interval: None)
    assert calibrate(count=1, warmup=0, interval=1).to_host_ns(1000) == 1000
    assert host_time_ns() == 1234
    assert calls == ["clock calibration", "clock calibration host time"]
