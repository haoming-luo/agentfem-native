# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自主线性二阶时间积分、约束、能量账本与重启语义。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elasticity import LinearElasticProblem
from .geometry import AffineTriangleMap
from .mesh import TriangularMesh
from .runtime import ExecutionContext
from .sparse import CSRAssemblyPlan, CSRMatrix, conjugate_gradient

FloatArray: TypeAlias = NDArray[np.float64]
Integrator = Literal["central_difference", "newmark_average_acceleration"]
TimeLoad = ArrayLike | Callable[[float], ArrayLike]
TimeScale = float | Callable[[float], float]


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
    external_work: FloatArray
    energy_balance_error: FloatArray
    constrained_dofs: NDArray[np.int64]
    reaction: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "times",
            "displacement",
            "velocity",
            "acceleration",
            "kinetic_energy",
            "strain_energy",
            "total_energy",
            "external_work",
            "energy_balance_error",
            "reaction",
        ):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError("Dynamic result history must be finite.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        constrained = np.array(self.constrained_dofs, dtype=np.int64, copy=True)
        if constrained.ndim != 1:
            raise ValueError("Dynamic constrained DOFs must be a vector.")
        constrained.setflags(write=False)
        object.__setattr__(self, "constrained_dofs", constrained)

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


def _principal_submatrix(matrix: CSRMatrix, active: NDArray[np.int64]) -> CSRMatrix:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Constrained dynamics requires square operators.")
    inverse = np.full(matrix.shape[0], -1, dtype=np.int64)
    inverse[active] = np.arange(active.size, dtype=np.int64)
    rows = matrix.row_indices()
    selected = (inverse[rows] >= 0) & (inverse[matrix.indices] >= 0)
    return CSRMatrix.from_coo(
        (active.size, active.size),
        inverse[rows[selected]],
        inverse[matrix.indices[selected]],
        matrix.data[selected],
    )


def central_difference_safe_time_step(system: LinearSecondOrderSystem) -> float:
    """返回中心差分的保守稳定步长上限，不执行特征值求解。"""

    size = system.stiffness.shape[0]
    diagonal = system.mass.diagonal()
    if system.mass.nnz != size or np.any(diagonal <= 0.0):
        raise ValueError("中心差分要求正的对角质量矩阵（diagonal mass）。")
    row_bounds = np.empty(size, dtype=np.float64)
    for row in range(size):
        start = int(system.stiffness.indptr[row])
        stop = int(system.stiffness.indptr[row + 1])
        row_bounds[row] = (
            np.sum(np.abs(system.stiffness.data[start:stop])) / diagonal[row]
        )
    spectral_bound = float(np.max(row_bounds, initial=0.0))
    if not np.isfinite(spectral_bound):
        raise ValueError("中心差分稳定上界计算得到非有限值。")
    if spectral_bound == 0.0:
        return float("inf")
    return 2.0 / np.sqrt(spectral_bound)


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
    dt = system.time_step
    if method == "central_difference":
        safe_time_step = central_difference_safe_time_step(system)
        if dt > safe_time_step:
            ratio = dt / safe_time_step
            raise ValueError(
                "中心差分时间步超过保守稳定上限："
                f"请求 {dt:.17g}，上限 {safe_time_step:.17g}，比值 {ratio:.6g}。"
            )
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
    external_work = np.zeros(count, dtype=np.float64)
    displacements[0] = displacement
    velocities[0] = velocity
    accelerations[0] = acceleration
    kinetic[0], strain[0], total[0] = _energies(
        system.mass, system.stiffness, displacement, velocity
    )
    previous_load = _load_at(system.load, start_time, size)
    if method == "central_difference":
        diagonal = system.mass.diagonal()
        inverse_mass = 1.0 / diagonal
        for index in range(1, count):
            if context is not None:
                context.check("linear_dynamics", index - 1, system.steps)
            next_displacement = (
                displacement + dt * velocity + 0.5 * dt * dt * acceleration
            )
            next_load = _load_at(system.load, float(times[index]), size)
            next_acceleration = inverse_mass * (
                next_load - system.stiffness.matvec(next_displacement)
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
            external_work[index] = external_work[index - 1] + 0.5 * float(
                (previous_load + next_load)
                @ (displacements[index] - displacements[index - 1])
            )
            previous_load = next_load
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
            next_load = _load_at(system.load, float(times[index]), size)
            next_acceleration = _solve_spd(
                effective,
                next_load - system.stiffness.matvec(displacement_predictor),
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
            external_work[index] = external_work[index - 1] + 0.5 * float(
                (previous_load + next_load)
                @ (displacements[index] - displacements[index - 1])
            )
            previous_load = next_load
    if context is not None:
        context.check("linear_dynamics", system.steps, system.steps)
    return LinearDynamicsResult(
        method=method,
        start_step=start_step,
        times=times,
        displacement=displacements,
        velocity=velocities,
        acceleration=accelerations,
        kinetic_energy=kinetic,
        strain_energy=strain,
        total_energy=total,
        external_work=external_work,
        energy_balance_error=total - total[0] - external_work,
        constrained_dofs=np.empty(0, dtype=np.int64),
        reaction=np.empty((count, 0), dtype=np.float64),
    )


def integrate_constrained_linear_dynamics(
    system: LinearSecondOrderSystem,
    constrained_dofs: ArrayLike,
    *,
    method: Integrator = "central_difference",
    restart: DynamicCheckpoint | None = None,
    context: ExecutionContext | None = None,
) -> LinearDynamicsResult:
    """Eliminate zero fixed DOFs, integrate the free system, and recover reactions."""

    constrained = np.asarray(constrained_dofs)
    if constrained.ndim != 1 or not np.issubdtype(constrained.dtype, np.integer):
        raise ValueError("Constrained dynamic DOFs must be an integer vector.")
    constrained = np.array(constrained, dtype=np.int64, copy=True)
    size = system.stiffness.shape[0]
    if (
        constrained.size == 0
        or np.any(constrained < 0)
        or np.any(constrained >= size)
        or np.unique(constrained).size != constrained.size
    ):
        raise ValueError("Constrained dynamic DOFs must be unique and in range.")
    constrained.sort()
    if np.any(system.initial_displacement[constrained] != 0.0) or np.any(
        system.initial_velocity[constrained] != 0.0
    ):
        raise ValueError(
            "Fixed dynamic DOFs require zero initial displacement/velocity."
        )
    active_mask = np.ones(size, dtype=bool)
    active_mask[constrained] = False
    active = np.flatnonzero(active_mask).astype(np.int64)
    if active.size == 0:
        raise ValueError("Constrained dynamics requires at least one active DOF.")

    def reduced_load(time: float) -> FloatArray:
        return _load_at(system.load, time, size)[active]

    reduced = LinearSecondOrderSystem(
        stiffness=_principal_submatrix(system.stiffness, active),
        mass=_principal_submatrix(system.mass, active),
        load=reduced_load,
        time_step=system.time_step,
        steps=system.steps,
        initial_displacement=system.initial_displacement[active],
        initial_velocity=system.initial_velocity[active],
    )
    reduced_restart = None
    if restart is not None:
        if restart.displacement.shape != (size,):
            raise ValueError("Checkpoint size does not match constrained dynamics.")
        if np.any(restart.displacement[constrained] != 0.0) or np.any(
            restart.velocity[constrained] != 0.0
        ):
            raise ValueError("Checkpoint violates zero fixed dynamic DOFs.")
        reduced_restart = DynamicCheckpoint(
            restart.method,
            restart.step,
            restart.time,
            restart.displacement[active],
            restart.velocity[active],
            restart.acceleration[active],
            _checkpoint_digest(
                restart.method,
                restart.step,
                restart.time,
                restart.displacement[active],
                restart.velocity[active],
                restart.acceleration[active],
            ),
        )
    result = integrate_linear_dynamics(
        reduced, method=method, restart=reduced_restart, context=context
    )
    count = result.times.size
    displacement = np.zeros((count, size), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    displacement[:, active] = result.displacement
    velocity[:, active] = result.velocity
    acceleration[:, active] = result.acceleration
    reaction = np.empty((count, constrained.size), dtype=np.float64)
    for index, time in enumerate(result.times):
        residual = (
            system.mass.matvec(acceleration[index])
            + system.stiffness.matvec(displacement[index])
            - _load_at(system.load, float(time), size)
        )
        reaction[index] = residual[constrained]
    return LinearDynamicsResult(
        method=result.method,
        start_step=result.start_step,
        times=result.times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        kinetic_energy=result.kinetic_energy,
        strain_energy=result.strain_energy,
        total_energy=result.total_energy,
        external_work=result.external_work,
        energy_balance_error=result.energy_balance_error,
        constrained_dofs=constrained,
        reaction=reaction,
    )


def build_t3_linear_dynamics(
    problem: LinearElasticProblem,
    density: float,
    *,
    time_step: float,
    steps: int,
    initial_displacement: ArrayLike | None = None,
    initial_velocity: ArrayLike | None = None,
    load_scale: TimeScale = 1.0,
    lumped_mass: bool = True,
    assembly_plan: CSRAssemblyPlan | None = None,
) -> tuple[LinearSecondOrderSystem, NDArray[np.int64]]:
    """从自有有限元语义建立零位移约束的 T3 结构动力系统。"""

    from .elasticity import (
        _collect_displacements,
        assemble_linear_elasticity,
        assemble_linear_elasticity_prepared,
    )

    if not isinstance(problem, LinearElasticProblem):
        raise TypeError("T3 dynamics requires a LinearElasticProblem.")
    if assembly_plan is None:
        matrix, static_load, dofs = assemble_linear_elasticity(problem)
        stiffness = CSRMatrix.from_coo(
            matrix.shape, matrix.rows, matrix.columns, matrix.data
        )
    else:
        stiffness, static_load, dofs = assemble_linear_elasticity_prepared(
            problem, assembly_plan
        )
    constrained, prescribed = _collect_displacements(problem, dofs)
    if np.any(prescribed != 0.0):
        raise ValueError("T3 dynamics currently admits only zero fixed displacements.")
    size = dofs.size

    def state_values(value: ArrayLike | None, *, name: str) -> FloatArray:
        if value is None:
            return np.zeros(size, dtype=np.float64)
        values = np.asarray(value, dtype=np.float64)
        if values.shape == (problem.mesh.node_count, 2):
            values = values.ravel()
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite nodal vector field.")
        return np.array(values, copy=True)

    displacement = state_values(initial_displacement, name="Initial displacement")
    velocity = state_values(initial_velocity, name="Initial velocity")
    if np.any(displacement[constrained] != 0.0) or np.any(velocity[constrained] != 0.0):
        raise ValueError("T3 initial state violates fixed displacement conditions.")

    def scaled_load(time: float) -> FloatArray:
        raw = load_scale(time) if callable(load_scale) else load_scale
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError("Dynamic load scale must be a real scalar.")
        scale = float(raw)
        if not np.isfinite(scale):
            raise ValueError("Dynamic load scale must be finite.")
        return scale * static_load

    mass = assemble_t3_mass(
        problem.mesh,
        density,
        thickness=problem.thickness,
        lumped=lumped_mass,
    )
    return (
        LinearSecondOrderSystem(
            stiffness,
            mass,
            scaled_load,
            time_step,
            steps,
            displacement,
            velocity,
        ),
        constrained,
    )
