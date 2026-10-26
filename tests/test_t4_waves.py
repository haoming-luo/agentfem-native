# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""横向约束三维条带：连续纵波解析值与 T4 空间细化证据。"""

import unittest
from dataclasses import replace
from itertools import pairwise

import numpy as np

from agentfem_native.dynamics import (
    central_difference_safe_time_step,
    integrate_constrained_linear_dynamics,
)
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
)
from agentfem_native.solid_dynamics import build_t4_linear_dynamics
from agentfem_native.volume_mesh import unit_cube_tetrahedra

WAVE_SPEED = np.sqrt(60.0)  # E=100、nu=.25、rho=2，lambda=mu=40。


def bar_problem(nx):
    mesh = unit_cube_tetrahedra(nx, 1, 1)
    return LinearElastic3DProblem(
        mesh,
        SolidElasticMaterial(100, 0.25),
        dirichlet=(
            SolidDisplacementCondition("left", "x", 0),
            SolidDisplacementCondition("boundary", "y", 0),
            SolidDisplacementCondition("boundary", "z", 0),
        ),
    )


def lowest_frequency(nx):
    system, fixed = build_t4_linear_dynamics(
        bar_problem(nx), 2, time_step=0.001, steps=1, lumped_mass=False
    )
    active = np.setdiff1d(np.arange(system.mass.shape[0]), fixed)
    # 仅在小型验证 oracle 中稠密求特征值；真实动力程序仍走 CSR。
    mass = system.mass.to_dense()[np.ix_(active, active)]
    stiffness = system.stiffness.to_dense()[np.ix_(active, active)]
    cholesky = np.linalg.cholesky(mass)
    inverse = np.linalg.solve(cholesky, np.eye(len(active)))
    return float(np.sqrt(np.linalg.eigvalsh(inverse @ stiffness @ inverse.T)[0]))


def pulse_evidence(nx):
    problem = bar_problem(nx)
    mesh = problem.mesh
    x = mesh.points[:, 0]
    initial = np.zeros_like(mesh.points)
    initial[:, 0] = 0.001 * np.exp(-0.5 * ((x - 0.2) / 0.04) ** 2)
    initial[mesh.nodes("left"), 0] = 0
    system, fixed = build_t4_linear_dynamics(
        problem,
        2,
        time_step=1e-4,
        steps=1,
        initial_displacement=initial,
        lumped_mass=True,
    )
    end_time = 0.08  # 右端反射尚未返回任一传感截面。
    steps = int(np.ceil(end_time / (0.2 * central_difference_safe_time_step(system))))
    result = integrate_constrained_linear_dynamics(
        replace(system, time_step=end_time / steps, steps=steps), fixed
    )
    arrivals = []
    for station in (0.5, 0.75):
        nodes = np.flatnonzero(np.isclose(x, station))
        signal = np.mean(result.displacement[:, 3 * nodes], axis=1)
        expected = (station - 0.2) / WAVE_SPEED
        window = np.flatnonzero(
            (result.times > 0.65 * expected) & (result.times < 1.35 * expected)
        )
        arrivals.append(float(result.times[window[np.argmax(signal[window])]]))
    speed = 0.25 / (arrivals[1] - arrivals[0])
    energy_drift = float(
        np.max(np.abs(result.total_energy - result.total_energy[0]))
        / result.total_energy[0]
    )
    return speed, energy_drift


class T4WaveTests(unittest.TestCase):
    def test_longitudinal_frequency_has_second_order_spatial_trend(self):
        exact = np.pi * WAVE_SPEED / 2
        errors = [abs(lowest_frequency(nx) / exact - 1) for nx in (8, 16, 32)]
        self.assertLess(errors[-1], 0.01)
        for coarse, fine in pairwise(errors):
            self.assertGreater(coarse / fine, 3)
            self.assertLess(coarse / fine, 5)

    def test_pulse_speed_and_energy_improve_under_refinement(self):
        coarse, coarse_energy = pulse_evidence(32)
        fine, fine_energy = pulse_evidence(64)
        self.assertLess(abs(fine / WAVE_SPEED - 1), abs(coarse / WAVE_SPEED - 1))
        self.assertLess(abs(fine / WAVE_SPEED - 1), 0.03)
        self.assertLess(fine_energy, coarse_energy)
        self.assertLess(fine_energy, 0.01)
