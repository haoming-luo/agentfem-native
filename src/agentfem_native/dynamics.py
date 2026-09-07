# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned linear second-order time integration and restart semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import AffineTriangleMap
from .mesh import TriangularMesh
from .runtime import ExecutionContext
from .sparse import CSRMatrix, conjugate_gradient

FloatArray: TypeAlias = NDArray[np.float64]
Integrator = Literal["central_difference", "newmark_average_acceleration"]
TimeLoad = ArrayLike | Callable[[float], ArrayLike]


def assemble_t3_mass(
    mesh: TriangularMesh,
    density: float,
    *,
    thickness: float = 1.0,
    lumped: bool = False,
) -> CSRMatrix:
    """Assemble a two-component T3 consistent or row-sum-lumped mass."""

    if isinstance(density, (bool, np.bool_)) or isinstance(thickness, (bool, np.bool_)):
        raise TypeError("Density and thickness must be real scalars.")
    density = float(density)
    thickness = float(thickness)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Density must be finite and positive.")
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("Thickness must be finite and positive.")
    entries_per_cell = 6 if lumped else 18
    rows = np.empty(mesh.cell_count * entries_per_cell, dtype=np.int64)
    columns = np.empty_like(rows)
    data = np.empty(rows.size, dtype=np.float64)
    cursor = 0
    for cell in mesh.cells:
        area = AffineTriangleMap(mesh.points[cell]).area
        if lumped:
            value = density * thickness * area / 3.0
            for node in cell:
                for component in range(2):
                    dof = 2 * int(node) + component
                    rows[cursor] = dof
                    columns[cursor] = dof
                    data[cursor] = value
                    cursor += 1
        else:
            scale = density * thickness * area / 12.0
            for local_row, row_node in enumerate(cell):
                for local_column, column_node in enumerate(cell):
                    value = scale * (2.0 if local_row == local_column else 1.0)
                    for component in range(2):
                        rows[cursor] = 2 * int(row_node) + component
                        columns[cursor] = 2 * int(column_node) + component
                        data[cursor] = value
                        cursor += 1
    return CSRMatrix.from_coo(
        (2 * mesh.node_count, 2 * mesh.node_count), rows, columns, data
    )


@dataclass(frozen=True, slots=True)
class LinearSecondOrderSystem:
    stiffness: CSRMatrix
    mass: CSRMatrix
    load: TimeLoad
    time_step: float
    steps: int
    initial_displacement: ArrayLike
    initial_velocity: ArrayLike

    def __post_init__(self) -> None:
        if self.stiffness.shape[0] != self.stiffness.shape[1]:
            raise ValueError("Dynamic stiffness must be square.")
        if self.mass.shape != self.stiffness.shape:
            raise ValueError("Dynamic mass and stiffness shapes must match.")
        if not np.isfinite(self.time_step) or self.time_step <= 0.0:
            raise ValueError("Dynamic time step must be finite and positive.")
        if (
            isinstance(self.steps, (bool, np.bool_))
            or not isinstance(self.steps, (int, np.integer))
            or int(self.steps) < 0
        ):
            raise ValueError("Dynamic step count must be a nonnegative integer.")
        size = self.stiffness.shape[0]
        for name in ("initial_displacement", "initial_velocity"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain one finite value per DOF.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "time_step", float(self.time_step))
        object.__setattr__(self, "steps", int(self.steps))


@dataclass(frozen=True, slots=True)
class DynamicCheckpoint:
    method: Integrator
    step: int
    time: float
    displacement: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    digest: str

    def __post_init__(self) -> None:
        for name in ("displacement", "velocity", "acceleration"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Checkpoint state arrays must be finite vectors.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.digest != _checkpoint_digest(
            self.method,
            self.step,
            self.time,
            self.displacement,
            self.velocity,
            self.acceleration,
        ):
            raise ValueError("Dynamic checkpoint digest does not match its state.")


@dataclass(frozen=True, slots=True)
class LinearDynamicsResult:
    method: Integrator
    start_step: int
    times: FloatArray
    displacement: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    kinetic_energy: FloatArray
    strain_energy: FloatArray
    total_energy: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "times",
            "displacement",
            "velocity",
            "acceleration",
            "kinetic_energy",
            "strain_energy",
            "total_energy",
        ):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError("Dynamic result history must be finite.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def checkpoint(self, index: int = -1) -> DynamicCheckpoint:
        normalized = index if index >= 0 else self.times.size + index
        if normalized < 0 or normalized >= self.times.size:
            raise IndexError("Dynamic checkpoint index is outside the history.")
        step = self.start_step + normalized
        time = float(self.times[normalized])
        displacement = self.displacement[normalized]
        velocity = self.velocity[normalized]
        acceleration = self.acceleration[normalized]
        return DynamicCheckpoint(
            self.method,
            step,
            time,
            displacement,
            velocity,
            acceleration,
            _checkpoint_digest(
                self.method, step, time, displacement, velocity, acceleration
            ),
        )


def _checkpoint_digest(
    method: Integrator,
    step: int,
    time: float,
    displacement: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
) -> str:
    digest = sha256()
    digest.update(method.encode("ascii"))
    digest.update(np.asarray((step,), dtype="<i8").tobytes())
    digest.update(np.asarray((time,), dtype="<f8").tobytes())
    for values in (displacement, velocity, acceleration):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_at(load: TimeLoad, time: float, size: int) -> FloatArray:
    raw = load(time) if callable(load) else load
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError("Dynamic load must return one finite value per DOF.")
    return values


def _linear_combination(
    left: CSRMatrix, right: CSRMatrix, right_scale: float
) -> CSRMatrix:
    return CSRMatrix.from_coo(
        left.shape,
        np.concatenate((left.row_indices(), right.row_indices())),
        np.concatenate((left.indices, right.indices)),
        np.concatenate((left.data, right_scale * right.data)),
    )


def _solve_spd(matrix: CSRMatrix, right_hand_side: FloatArray) -> FloatArray:
    result = conjugate_gradient(
        matrix,
        right_hand_side,
        relative_tolerance=1.0e-13,
        absolute_tolerance=1.0e-14,
    )
    if not result.report.converged:
        raise ValueError(
            f"Dynamic SPD solve failed with {result.report.reason} after "
            f"{result.report.iterations} iterations."
        )
    return result.solution


def _energies(
    mass: CSRMatrix,
    stiffness: CSRMatrix,
    displacement: FloatArray,
    velocity: FloatArray,
) -> tuple[float, float, float]:
    kinetic = 0.5 * float(velocity @ mass.matvec(velocity))
    strain = 0.5 * float(displacement @ stiffness.matvec(displacement))
    return kinetic, strain, kinetic + strain


def integrate_linear_dynamics(
    system: LinearSecondOrderSystem,
    *,
    method: Integrator = "central_difference",
    restart: DynamicCheckpoint | None = None,
    context: ExecutionContext | None = None,
) -> LinearDynamicsResult:
    """Integrate accepted states and return immutable energy/restart evidence."""

    if method not in {"central_difference", "newmark_average_acceleration"}:
        raise ValueError(f"Unknown linear dynamics method {method!r}.")
    size = system.stiffness.shape[0]
    if restart is not None:
        if restart.method != method:
            raise ValueError("Checkpoint integrator does not match requested method.")
        if restart.displacement.shape != (size,):
            raise ValueError("Checkpoint size does not match the dynamic system.")
        start_step = restart.step
        start_time = restart.time
        displacement = np.array(restart.displacement, copy=True)
        velocity = np.array(restart.velocity, copy=True)
        acceleration = np.array(restart.acceleration, copy=True)
    else:
        start_step = 0
        start_time = 0.0
        displacement = np.array(system.initial_displacement, copy=True)
        velocity = np.array(system.initial_velocity, copy=True)
        acceleration = _solve_spd(
            system.mass,
            _load_at(system.load, 0.0, size) - system.stiffness.matvec(displacement),
        )
    count = system.steps + 1
    times = start_time + system.time_step * np.arange(count, dtype=np.float64)
    displacements = np.empty((count, size), dtype=np.float64)
    velocities = np.empty_like(displacements)
    accelerations = np.empty_like(displacements)
    kinetic = np.empty(count, dtype=np.float64)
    strain = np.empty(count, dtype=np.float64)
    total = np.empty(count, dtype=np.float64)
    displacements[0] = displacement
    velocities[0] = velocity
    accelerations[0] = acceleration
    kinetic[0], strain[0], total[0] = _energies(
        system.mass, system.stiffness, displacement, velocity
    )
    dt = system.time_step
    if method == "central_difference":
        diagonal = system.mass.diagonal()
        if system.mass.nnz != size or np.any(diagonal <= 0.0):
            raise ValueError(
                "Central difference requires a positive diagonal mass matrix."
            )
        inverse_mass = 1.0 / diagonal
        for index in range(1, count):
            if context is not None:
                context.check("linear_dynamics", index - 1, system.steps)
            next_displacement = (
                displacement + dt * velocity + 0.5 * dt * dt * acceleration
            )
            next_acceleration = inverse_mass * (
                _load_at(system.load, float(times[index]), size)
                - system.stiffness.matvec(next_displacement)
            )
            next_velocity = velocity + 0.5 * dt * (acceleration + next_acceleration)
            displacement, velocity, acceleration = (
                next_displacement,
                next_velocity,
                next_acceleration,
            )
            displacements[index] = displacement
            velocities[index] = velocity
            accelerations[index] = acceleration
            kinetic[index], strain[index], total[index] = _energies(
                system.mass, system.stiffness, displacement, velocity
            )
    else:
        beta = 0.25
        gamma = 0.5
        effective = _linear_combination(system.mass, system.stiffness, beta * dt * dt)
        for index in range(1, count):
            if context is not None:
                context.check("linear_dynamics", index - 1, system.steps)
            displacement_predictor = (
                displacement + dt * velocity + dt * dt * (0.5 - beta) * acceleration
            )
            velocity_predictor = velocity + dt * (1.0 - gamma) * acceleration
            next_acceleration = _solve_spd(
                effective,
                _load_at(system.load, float(times[index]), size)
                - system.stiffness.matvec(displacement_predictor),
            )
            displacement = displacement_predictor + beta * dt * dt * next_acceleration
            velocity = velocity_predictor + gamma * dt * next_acceleration
            acceleration = next_acceleration
            displacements[index] = displacement
            velocities[index] = velocity
            accelerations[index] = acceleration
            kinetic[index], strain[index], total[index] = _energies(
                system.mass, system.stiffness, displacement, velocity
            )
    if context is not None:
        context.check("linear_dynamics", system.steps, system.steps)
    return LinearDynamicsResult(
        method,
        start_step,
        times,
        displacements,
        velocities,
        accelerations,
        kinetic,
        strain,
        total,
    )
