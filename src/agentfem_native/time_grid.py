# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""固定步长的有限时间准入；不替代积分稳定性、预算或误差控制。"""

from numbers import Real

import numpy as np


def validate_time_interval(time_step, steps, *, start_time=0.0) -> float:
    """无历史分配的预检；返回终点，拒绝溢出或明显无法区分的相邻时间。"""
    for value, name in ((time_step, "时间步"), (start_time, "开始时间")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name}必须为有限实数。")
        if not np.isfinite(value):
            raise ValueError(f"{name}必须有限。")
    dt, start = float(time_step), float(start_time)
    if dt <= 0 or start < 0:
        raise ValueError("时间步必须为正，开始时间必须非负。")
    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 0
    ):
        raise ValueError("时间步数必须为非负整数。")
    if steps > min(2**53 - 1, np.iinfo(np.intp).max - 1):
        raise ValueError("时间步数超出精确时间索引或平台数组范围。")
    end = start + int(steps) * dt
    if not np.isfinite(end):
        raise ValueError("终点时间溢出，请缩短区间或调整时间单位。")
    if steps and (start + dt <= start or start + (int(steps) - 1) * dt >= end):
        raise ValueError("相邻时间在当前浮点尺度无法区分，请调整时间原点或单位。")
    return end


def time_grid(time_step, steps, *, start_time=0.0) -> np.ndarray:
    """延续既有算术顺序以保持重启语义；完整时间网格须严格递增。"""
    validate_time_interval(time_step, steps, start_time=start_time)
    values = float(start_time) + float(time_step) * np.arange(
        int(steps) + 1, dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError("时间网格必须有限且严格递增，请调整时间尺度。")
    return values
