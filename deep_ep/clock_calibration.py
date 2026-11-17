"""Map Ascend950 GetSystemCycle ticks to Host CLOCK_MONOTONIC nanoseconds.

Importing this module requires neither Torch nor CANN. Hardware is accessed only
by calibrate() / host_time_ns(), after the caller initializes its NPU context.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import time

from ._native import require_native


@dataclass(frozen=True)
class ClockSampleBatch:
    device: int
    soc: str
    samples: tuple
    selected: int

    @property
    def anchor(self):
        return self.samples[self.selected]

    @property
    def midpoint_ns(self):
        h0, _, h1 = self.anchor
        return h0 + (h1 - h0) // 2

    @property
    def bracket_ns(self):
        h0, _, h1 = self.anchor
        return h1 - h0

    @property
    def uncertainty_ns(self):
        return (self.bracket_ns + 1) // 2

    def to_dict(self):
        result = asdict(self)
        result.update(bracket_ns=self.bracket_ns, uncertainty_ns=self.uncertainty_ns)
        return result


def from_samples(device, soc, samples):
    """Keep raw samples and choose the shortest Host bracket as one anchor."""
    samples = tuple(tuple(row) for row in samples)
    if not samples:
        raise ValueError("at least one sample is required")
    for row in samples:
        if len(row) != 3 or any(type(v) is not int for v in row):
            raise ValueError("samples must contain integer (host_before_ns, device_cycle, host_after_ns)")
        h0, d, h1 = row
        if h0 < 0 or d <= 0 or h1 < h0:
            raise ValueError("invalid clock sample")
    selected = min(range(len(samples)), key=lambda i: samples[i][2] - samples[i][0])
    return ClockSampleBatch(device, soc, samples, selected)


def _round_ratio(numerator, denominator):
    """Round an integer ratio to nearest, including negative extrapolation."""
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


@dataclass(frozen=True)
class ClockCalibration:
    first: ClockSampleBatch
    second: ClockSampleBatch
    schema_version: int = 2
    host_clock: str = "CLOCK_MONOTONIC"
    device_clock: str = "GetSystemCycle"

    def __post_init__(self):
        if self.first.device != self.second.device or self.first.soc != self.second.soc:
            raise ValueError("calibration batches must come from the same device and SoC")
        if self.rate_numerator_ns <= 0 or self.rate_denominator_ticks <= 0:
            raise ValueError("clock anchors must advance in both clock domains")
        # Reject a reset or clearly incompatible timer, without claiming that
        # this wide guard is a precision threshold.
        if abs(self.rate_numerator_ns - self.rate_denominator_ticks) * 1_000_000 > (
            self.rate_denominator_ticks * 5_000
        ):
            raise ValueError("clock rate differs from the Ascend950 nominal 1 GHz by over 5000 ppm")

    @property
    def device(self):
        return self.first.device

    @property
    def soc(self):
        return self.first.soc

    @property
    def rate_numerator_ns(self):
        return self.second.midpoint_ns - self.first.midpoint_ns

    @property
    def rate_denominator_ticks(self):
        return self.second.anchor[1] - self.first.anchor[1]

    def to_host_ns(self, device_cycle):
        """Convert ticks using two measured anchors; return integer Host ns."""
        if type(device_cycle) is not int or device_cycle < 0:
            raise ValueError("device_cycle must be a nonnegative integer")
        delta = device_cycle - self.first.anchor[1]
        return self.first.midpoint_ns + _round_ratio(delta * self.rate_numerator_ns, self.rate_denominator_ticks)

    def mapping_uncertainty_ns(self, device_cycle):
        """Anchor bracket error, including extrapolation beyond either anchor."""
        d0, d1 = self.first.anchor[1], self.second.anchor[1]
        weighted = abs(d1 - device_cycle) * self.first.uncertainty_ns
        weighted += abs(device_cycle - d0) * self.second.uncertainty_ns
        return (weighted + self.rate_denominator_ticks - 1) // self.rate_denominator_ticks + 1

    def to_dict(self):
        return dict(schema_version=self.schema_version, device=self.device, soc=self.soc,
                    host_clock=self.host_clock, device_clock=self.device_clock,
                    first=self.first.to_dict(), second=self.second.to_dict(),
                    rate_numerator_ns=self.rate_numerator_ns,
                    rate_denominator_ticks=self.rate_denominator_ticks,
                    host_ns_per_tick=self.rate_numerator_ns / self.rate_denominator_ticks,
                    observed_ticks_per_host_ns=self.rate_denominator_ticks / self.rate_numerator_ns,
                    nominal_rate_error_ppm=(self.rate_denominator_ticks / self.rate_numerator_ns - 1) * 1_000_000)


def _read_batch(native, count, warmup):
    device, soc, samples = native.clock_calibration_samples(count, warmup)
    return from_samples(device, soc, samples)


def _validate_args(count, warmup, interval):
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("count must be 1..4096")
    if type(warmup) is not int or not 0 <= warmup <= 4096:
        raise ValueError("warmup must be 0..4096")
    if type(interval) not in (int, float) or not 0 < interval <= 3600:
        raise ValueError("interval must be in (0, 3600] seconds")


def calibrate(count=32, warmup=5, interval=10.0):
    """Blocking two-batch calibration on the current device."""
    _validate_args(count, warmup, interval)
    native = require_native("clock calibration")
    first = _read_batch(native, count, warmup)
    time.sleep(interval)
    second = _read_batch(native, count, warmup)
    return ClockCalibration(first, second)


def host_time_ns():
    """Read the same Host clock used by native calibration."""
    return require_native("clock calibration host time").clock_monotonic_ns()


def validate_holdout(mapping, batch):
    """Check an independent batch against the fitted mapping and bracket bounds."""
    if mapping.device != batch.device or mapping.soc != batch.soc:
        raise ValueError("holdout batch is from another device or SoC")
    violations = []
    for h0, d, h1 in batch.samples:
        estimate = mapping.to_host_ns(d)
        uncertainty = mapping.mapping_uncertainty_ns(d)
        violations.append(max(h0 - estimate - uncertainty, estimate - h1 - uncertainty, 0))
    return dict(holdout_max_violation_ns=max(violations), holdout_violations=sum(value > 0 for value in violations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--interval", type=float, default=10.0,
                        help="seconds between calibration batches; holdout follows after the same interval")
    parser.add_argument("--output", default="clock-calibration.json")
    args = parser.parse_args()
    try:
        _validate_args(args.samples, args.warmup, args.interval)
    except ValueError as exc:
        parser.error(str(exc))
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    mapping = calibrate(args.samples, args.warmup, args.interval)
    time.sleep(args.interval)
    third = _read_batch(require_native("clock calibration"), args.samples, args.warmup)
    verdict = validate_holdout(mapping, third)
    report = dict(mapping.to_dict(), transport="mapped_host_mte3", holdout=third.to_dict(), **verdict)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    return 1 if verdict["holdout_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
