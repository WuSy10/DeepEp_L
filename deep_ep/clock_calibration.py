"""把 Ascend 950 的 GetSystemCycle 计数映射到本机 CLOCK_MONOTONIC 纳秒。

导入本模块不需要 Torch 或 CANN；调用方建立 NPU 上下文后，calibrate()
和 host_time_ns() 才会访问硬件。映射只针对当前设备，不是墙上日期时间。
"""

import argparse
from dataclasses import asdict, dataclass
import json
import time

from ._native import require_native


@dataclass(frozen=True)
class ClockSampleBatch:
    """保存一批 (H0, D, H1) 样本及其中包围区间最短的锚点。"""
    device: int
    soc: str
    samples: tuple
    selected: int

    @property
    def anchor(self):
        """返回选中的完整三元组；H0、D、H1 必须来自同一次采样。"""
        return self.samples[self.selected]

    @property
    def midpoint_ns(self):
        """用 Host 包围区间中点估计设备读钟 D 发生的 Host 时刻。"""
        h0, _, h1 = self.anchor
        return h0 + (h1 - h0) // 2

    @property
    def bracket_ns(self):
        """返回 H1-H0；区间越窄，锚点的定位误差越小。"""
        h0, _, h1 = self.anchor
        return h1 - h0

    @property
    def uncertainty_ns(self):
        """取包围区间宽度的一半并向上取整，作为锚点定位误差。"""
        return (self.bracket_ns + 1) // 2

    def to_dict(self):
        """连同原始样本、选中下标和误差一起输出，便于复核。"""
        result = asdict(self)
        result.update(bracket_ns=self.bracket_ns, uncertainty_ns=self.uncertainty_ns)
        return result


def from_samples(device, soc, samples):
    """校验并保存整批样本，选 H1-H0 最小的一整条样本作为锚点。"""
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
    """整数除法四舍五入；负数表示向第一锚点之前外推。"""
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


@dataclass(frozen=True)
class ClockCalibration:
    """用同一设备的两批锚点建立 D → Host 纳秒的线性映射。"""
    first: ClockSampleBatch
    second: ClockSampleBatch
    schema_version: int = 2
    host_clock: str = "CLOCK_MONOTONIC"
    device_clock: str = "GetSystemCycle"

    def __post_init__(self):
        """排除跨设备、时钟倒退和明显偏离名义 1 GHz 的锚点组合。"""
        if self.first.device != self.second.device or self.first.soc != self.second.soc:
            raise ValueError("calibration batches must come from the same device and SoC")
        if self.rate_numerator_ns <= 0 or self.rate_denominator_ticks <= 0:
            raise ValueError("clock anchors must advance in both clock domains")
        # 5000 ppm 是排除复位或明显异常走速的宽松门槛，不是精度保证。
        if abs(self.rate_numerator_ns - self.rate_denominator_ticks) * 1_000_000 > (
            self.rate_denominator_ticks * 5_000
        ):
            raise ValueError("clock rate differs from the Ascend950 nominal 1 GHz by over 5000 ppm")

    @property
    def device(self):
        """返回两批共同对应的设备编号。"""
        return self.first.device

    @property
    def soc(self):
        """返回两批共同对应的 SoC 型号。"""
        return self.first.soc

    @property
    def rate_numerator_ns(self):
        """两锚点 Host 中点之差，即斜率的纳秒分子。"""
        return self.second.midpoint_ns - self.first.midpoint_ns

    @property
    def rate_denominator_ticks(self):
        """两锚点设备计数之差，即斜率的 tick 分母。"""
        return self.second.anchor[1] - self.first.anchor[1]

    def to_host_ns(self, device_cycle):
        """按 H(D)=Ha+round((D-Da)×ΔH/ΔD) 换算为 Host 单调纳秒。"""
        if type(device_cycle) is not int or device_cycle < 0:
            raise ValueError("device_cycle must be a nonnegative integer")
        delta = device_cycle - self.first.anchor[1]
        return self.first.midpoint_ns + _round_ratio(delta * self.rate_numerator_ns, self.rate_denominator_ticks)

    def mapping_uncertainty_ns(self, device_cycle):
        """按距两锚点的 tick 距离传播包围误差；外推时误差也会增长。

        U(D)=ceil((|Db-D|×Ua + |D-Da|×Ub)/(Db-Da))+1 ns。
        这里只涵盖 H0/H1 对锚点位置的不确定性，不包括后续时钟漂移。
        """
        d0, d1 = self.first.anchor[1], self.second.anchor[1]
        weighted = abs(d1 - device_cycle) * self.first.uncertainty_ns
        weighted += abs(device_cycle - d0) * self.second.uncertainty_ns
        return (weighted + self.rate_denominator_ticks - 1) // self.rate_denominator_ticks + 1

    def to_dict(self):
        """导出原始锚点、整数斜率及便于阅读的浮点走速摘要。"""
        return dict(schema_version=self.schema_version, device=self.device, soc=self.soc,
                    host_clock=self.host_clock, device_clock=self.device_clock,
                    first=self.first.to_dict(), second=self.second.to_dict(),
                    rate_numerator_ns=self.rate_numerator_ns,
                    rate_denominator_ticks=self.rate_denominator_ticks,
                    host_ns_per_tick=self.rate_numerator_ns / self.rate_denominator_ticks,
                    observed_ticks_per_host_ns=self.rate_denominator_ticks / self.rate_numerator_ns,
                    nominal_rate_error_ppm=(self.rate_denominator_ticks / self.rate_numerator_ns - 1) * 1_000_000)


def _read_batch(native, count, warmup):
    """让原生层采一批样本，再由 Python 从这一批选择锚点。"""
    device, soc, samples = native.clock_calibration_samples(count, warmup)
    return from_samples(device, soc, samples)


def _validate_args(count, warmup, interval):
    """检查每批样本数、预热次数和批间等待秒数。"""
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("count must be 1..4096")
    if type(warmup) is not int or not 0 <= warmup <= 4096:
        raise ValueError("warmup must be 0..4096")
    if type(interval) not in (int, float) or not 0 < interval <= 3600:
        raise ValueError("interval must be in (0, 3600] seconds")


def calibrate(count=32, warmup=5, interval=10.0):
    """在当前设备上先后采两批；每批一锚点，两锚点确定映射比例。"""
    _validate_args(count, warmup, interval)
    native = require_native("clock calibration")
    first = _read_batch(native, count, warmup)
    time.sleep(interval)
    second = _read_batch(native, count, warmup)
    return ClockCalibration(first, second)


def host_time_ns():
    """读取校准所用的同一 Host CLOCK_MONOTONIC 时钟。"""
    return require_native("clock calibration host time").clock_monotonic_ns()


def validate_holdout(mapping, batch):
    """用独立第三批验收前两批的映射，不用第三批重新拟合。

    每条样本比较预测区间 [estimate-U, estimate+U] 与实测 [H0,H1]：
    两区间不相交时，violation 是它们之间的纳秒间隙。
    """
    if mapping.device != batch.device or mapping.soc != batch.soc:
        raise ValueError("holdout batch is from another device or SoC")
    violations = []
    for h0, d, h1 in batch.samples:
        estimate = mapping.to_host_ns(d)
        uncertainty = mapping.mapping_uncertainty_ns(d)
        violations.append(max(h0 - estimate - uncertainty, estimate - h1 - uncertainty, 0))
    return dict(holdout_max_violation_ns=max(violations), holdout_violations=sum(value > 0 for value in violations))


def main():
    """独立校准命令：选设备、拟合两批、验收第三批并写 JSON 报告。"""
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

    # 先清空当前设备已有工作；随后三批各自创建校准流并逐样本同步。
    torch.npu.set_device(args.device)
    torch.npu.synchronize()
    mapping = calibrate(args.samples, args.warmup, args.interval)
    # 第三批只验收映射；即使失败，也保留原始结果到 JSON 供分析。
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
