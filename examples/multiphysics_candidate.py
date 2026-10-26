# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""生成并执行二维/三维候选算例；输入和完整证据可交给 AgentFEM 重放。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentfem_native import (
    MULTIPHYSICS_VERSION,
    plan_multiphysics_request,
    run_multiphysics_request,
    unit_cube_tetrahedra,
    unit_square_triangles,
)


def candidate_requests():
    """只使用合成单位方形/立方体，不包含任何客户模型。"""
    for dimension in (2, 3):
        mesh = unit_square_triangles(4) if dimension == 2 else unit_cube_tetrahedra(2)
        payload = {
            "points": mesh.points.tolist(),
            "cells": mesh.cells.tolist(),
            "boundary_sets": {k: v.tolist() for k, v in mesh.boundary_sets.items()},
        }
        left, right = mesh.nodes("left"), mesh.nodes("right")
        heat = {
            "conductivity": 2,
            "capacity": 3,
            "constraints": {
                "dofs": left.tolist(),
                "values": np.full(left.size, 20).tolist(),
            },
            "boundaries": [
                {"boundary_set": "right", "convection": 4, "ambient_temperature": 40}
            ],
        }
        for kind in (
            "thermal_steady",
            "thermal_transient",
            "thermoelastic",
            "finite_strain",
            "at2_fracture",
        ):
            request = {
                "contract_version": MULTIPHYSICS_VERSION,
                "request_id": f"{kind}-{dimension}d",
                "kind": kind,
                "mesh": payload,
            }
            if kind in ("thermal_steady", "thermal_transient", "thermoelastic"):
                request["thermal"] = dict(heat)
            if kind == "thermal_transient":
                request["thermal"]["initial"] = np.full(mesh.node_count, 20).tolist()
                request["procedure"] = {"time_step": 0.02, "steps": 10}
            if kind == "thermoelastic":
                dofs = (dimension * left[:, None] + np.arange(dimension)).ravel()
                request["material"] = {
                    "young_modulus": 100,
                    "poisson_ratio": 0.25,
                    "expansion": 0.001,
                    "reference_temperature": 20,
                }
                request["mechanical"] = {
                    "constraints": {
                        "dofs": dofs.tolist(),
                        "values": np.zeros(dofs.size).tolist(),
                    }
                }
            if kind == "finite_strain":
                request["experimental"] = True
                request["material"] = {"young_modulus": 100, "poisson_ratio": 0.25}
                transverse = (
                    dimension * np.arange(mesh.node_count)[:, None]
                    + np.arange(1, dimension)
                ).ravel()
                dofs = np.concatenate((dimension * left, transverse))
                stretch = 1.5
                piola = 40 * (stretch - 1 / stretch) + 40 * np.log(stretch) / stretch
                request["mechanical"] = {
                    "constraints": {
                        "dofs": dofs.tolist(),
                        "values": np.zeros(dofs.size).tolist(),
                    },
                    "tractions": {"right": [piola] + [0] * (dimension - 1)},
                }
                request["procedure"] = {"steps": 4, "maximum_accepted_steps": 32}
            if kind == "at2_fracture":
                request["experimental"] = True
                request["material"] = {
                    "young_modulus": 100,
                    "poisson_ratio": 0.25,
                    "toughness": 1,
                    "length_scale": 0.2,
                }
                dofs = np.concatenate(
                    (
                        (dimension * left[:, None] + np.arange(dimension)).ravel(),
                        dimension * right,
                    )
                )
                values = np.concatenate(
                    (np.zeros(dimension * left.size), np.full(right.size, 0.01))
                )
                initial = np.zeros(mesh.node_count)
                initial[(mesh.points[:, 0] <= 0.25) & (mesh.points[:, 1] == 0.5)] = 1
                request["mechanical"] = {
                    "constraints": {"dofs": dofs.tolist(), "values": values.tolist()},
                    "initial_damage": initial.tolist(),
                }
                request["procedure"] = {"load_factors": [0.5, 1.0, 0.25]}
            yield request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="保存可重放请求及完整 JSON 结果")
    args = parser.parse_args()
    summaries = []
    for request in candidate_requests():
        plan = plan_multiphysics_request(request)
        if plan["status"] != "planned":
            raise RuntimeError(plan)
        result = run_multiphysics_request(request)
        if result["status"] != "success":
            raise RuntimeError(result)
        summary = {
            "case": request["request_id"],
            "status": result["status"],
            "maturity": result["evidence"]["maturity"],
            "result_digest": result["evidence"]["result_digest"],
        }
        fields = result["result"]
        if "states" in fields:
            final = fields["states"][-1]
            summary["free_residual_norm"] = final["free_residual_norm"]
            if "cell_jacobian" in final:
                summary["minimum_jacobian"] = min(final["cell_jacobian"])
            else:
                summary["damage_kkt_residual"] = final["damage_kkt_residual"]
        summaries.append(summary)
        if args.directory:
            args.directory.mkdir(parents=True, exist_ok=True)
            for suffix, content in (("request", request), ("result", result)):
                (args.directory / f"{request['request_id']}-{suffix}.json").write_text(
                    json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
