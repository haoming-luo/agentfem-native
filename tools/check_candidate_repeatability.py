# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""定位合成候选重复计算的字段差异，不修改容差或把摘要差异掩盖掉。"""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path

import numpy as np

from agentfem_native.multiphysics import run_multiphysics_request
from tests.test_multiphysics import request_for


def differences(left, right, path="result"):
    if isinstance(left, dict):
        return [
            item
            for key in left
            for item in differences(left[key], right[key], f"{path}.{key}")
        ]
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "lengths": [len(left), len(right)]}]
        return [
            item
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for item in differences(a, b, f"{path}[{i}]")
        ]
    if left != right or (
        isinstance(left, float)
        and isinstance(right, float)
        and np.signbit(left) != np.signbit(right)
    ):
        item = {"path": path, "first": left, "repeated": right}
        if isinstance(left, (float, int)) and isinstance(right, (float, int)):
            item["absolute_difference"] = abs(left - right)
        return [item]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for dimension in (2, 3):
        for kind in (
            "thermal_steady",
            "thermal_transient",
            "thermoelastic",
            "finite_strain",
            "at2_fracture",
        ):
            request = request_for(kind, dimension)
            first = run_multiphysics_request(request)
            if first["status"] != "success":
                raise RuntimeError(first)
            first_fields = first["result"]
            changes = []
            for repetition in range(30):
                # 小幅改变分配器状态，避免只采样一个偶然的缓冲布局。
                scratch = np.empty((repetition % 31 + 1, 57))
                repeated = run_multiphysics_request(request)
                if repeated["status"] != "success":
                    raise RuntimeError(repeated)
                fields = repeated["result"]
                digest = sha256(
                    json.dumps(fields, allow_nan=False, sort_keys=True).encode()
                ).hexdigest()
                assert digest == repeated["evidence"]["result_digest"]
                if digest != first["evidence"]["result_digest"]:
                    changes.append(
                        {
                            "repetition": repetition,
                            "differences": differences(first_fields, fields),
                        }
                    )
                del scratch
            cases.append(
                {
                    "kind": kind,
                    "dimension": dimension,
                    "changed_repeats": len(changes),
                    "changes": changes[:3],
                }
            )
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
