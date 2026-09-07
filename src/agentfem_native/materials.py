# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independently specified nonlinear material-point candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _elastic_parameters(
    young_modulus: float, poisson_ratio: float
) -> tuple[float, float]:
    if isinstance(young_modulus, (bool, np.bool_)) or isinstance(
        poisson_ratio, (bool, np.bool_)
    ):
        raise TypeError("Elastic parameters must be real scalars.")
    modulus = float(young_modulus)
    ratio = float(poisson_ratio)
    if not np.isfinite(modulus) or modulus <= 0.0:
        raise ValueError("Young's modulus must be finite and positive.")
    if not np.isfinite(ratio) or ratio <= -1.0 or ratio >= 0.5:
        raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")
    shear = modulus / (2.0 * (1.0 + ratio))
    lame = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
    return shear, lame


@dataclass(frozen=True, slots=True)
class NeoHookeanResponse:
    energy_density: float
    first_piola: FloatArray
    cauchy_stress: FloatArray

    def __post_init__(self) -> None:
        for name in ("first_piola", "cauchy_stress"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class NeoHookeanMaterial:
    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        _elastic_parameters(self.young_modulus, self.poisson_ratio)
        object.__setattr__(self, "young_modulus", float(self.young_modulus))
        object.__setattr__(self, "poisson_ratio", float(self.poisson_ratio))

    def response(self, deformation_gradient: ArrayLike) -> NeoHookeanResponse:
        shear, lame = _elastic_parameters(self.young_modulus, self.poisson_ratio)
        gradient = np.asarray(deformation_gradient, dtype=np.float64)
        if gradient.shape != (3, 3) or not np.all(np.isfinite(gradient)):
            raise ValueError("Deformation gradient must be a finite 3x3 matrix.")
        jacobian = float(np.linalg.det(gradient))
        if jacobian <= 0.0 or not np.isfinite(jacobian):
            raise ValueError("Neo-Hookean deformation requires positive finite J.")
        inverse_transpose = np.linalg.inv(gradient).T
        log_j = float(np.log(jacobian))
        first_piola = (
            shear * (gradient - inverse_transpose) + lame * log_j * inverse_transpose
        )
        energy = (
            0.5 * shear * (float(np.sum(gradient * gradient)) - 3.0)
            - shear * log_j
            + 0.5 * lame * log_j * log_j
        )
        cauchy = first_piola @ gradient.T / jacobian
        return NeoHookeanResponse(float(energy), first_piola, cauchy)


@dataclass(frozen=True, slots=True)
class J2State:
    plastic_strain: FloatArray
    equivalent_plastic_strain: float = 0.0

    def __post_init__(self) -> None:
        strain = np.array(self.plastic_strain, dtype=np.float64, copy=True)
        if strain.shape != (3, 3) or not np.all(np.isfinite(strain)):
            raise ValueError("J2 plastic strain must be a finite 3x3 tensor.")
        if not np.allclose(strain, strain.T, rtol=0.0, atol=1.0e-14):
            raise ValueError("J2 plastic strain must be symmetric.")
        alpha = float(self.equivalent_plastic_strain)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError(
                "Equivalent plastic strain must be finite and nonnegative."
            )
        strain.setflags(write=False)
        object.__setattr__(self, "plastic_strain", strain)
        object.__setattr__(self, "equivalent_plastic_strain", alpha)

    @classmethod
    def zero(cls) -> J2State:
        return cls(np.zeros((3, 3)), 0.0)


@dataclass(frozen=True, slots=True)
class J2Response:
    stress: FloatArray
    state: J2State
    yielded: bool
    plastic_multiplier: float

    def __post_init__(self) -> None:
        stress = np.array(self.stress, dtype=np.float64, copy=True)
        stress.setflags(write=False)
        object.__setattr__(self, "stress", stress)


@dataclass(frozen=True, slots=True)
class SmallStrainJ2Material:
    young_modulus: float
    poisson_ratio: float
    initial_yield_stress: float
    hardening_modulus: float = 0.0

    def __post_init__(self) -> None:
        _elastic_parameters(self.young_modulus, self.poisson_ratio)
        yield_stress = float(self.initial_yield_stress)
        hardening = float(self.hardening_modulus)
        if not np.isfinite(yield_stress) or yield_stress <= 0.0:
            raise ValueError("Initial yield stress must be finite and positive.")
        if not np.isfinite(hardening) or hardening < 0.0:
            raise ValueError("Hardening modulus must be finite and nonnegative.")
        object.__setattr__(self, "young_modulus", float(self.young_modulus))
        object.__setattr__(self, "poisson_ratio", float(self.poisson_ratio))
        object.__setattr__(self, "initial_yield_stress", yield_stress)
        object.__setattr__(self, "hardening_modulus", hardening)

    def update(self, total_strain: ArrayLike, state: J2State) -> J2Response:
        shear, lame = _elastic_parameters(self.young_modulus, self.poisson_ratio)
        yield_stress = float(self.initial_yield_stress)
        hardening = float(self.hardening_modulus)
        strain = np.asarray(total_strain, dtype=np.float64)
        if (
            strain.shape != (3, 3)
            or not np.all(np.isfinite(strain))
            or not np.allclose(strain, strain.T, rtol=0.0, atol=1e-14)
        ):
            raise ValueError("J2 total strain must be a finite symmetric 3x3 tensor.")
        elastic = strain - state.plastic_strain
        trial = 2.0 * shear * elastic + lame * np.trace(elastic) * np.eye(3)
        pressure = np.trace(trial) / 3.0
        deviator = trial - pressure * np.eye(3)
        equivalent = float(np.sqrt(1.5 * np.sum(deviator * deviator)))
        current_yield = yield_stress + hardening * state.equivalent_plastic_strain
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, current_yield)
        if equivalent <= current_yield + tolerance:
            return J2Response(trial, state, False, 0.0)
        increment = (equivalent - current_yield) / (3.0 * shear + hardening)
        plastic_increment = 1.5 * increment * deviator / equivalent
        next_state = J2State(
            state.plastic_strain + plastic_increment,
            state.equivalent_plastic_strain + increment,
        )
        corrected_deviator = deviator * (1.0 - 3.0 * shear * increment / equivalent)
        stress = corrected_deviator + pressure * np.eye(3)
        return J2Response(stress, next_state, True, float(increment))


class J2MaterialPoint:
    """One committed material state with explicit trial/commit/rollback."""

    def __init__(
        self, material: SmallStrainJ2Material, state: J2State | None = None
    ) -> None:
        self.material = material
        self._committed = J2State.zero() if state is None else state
        self._trial: J2Response | None = None

    @property
    def committed_state(self) -> J2State:
        return self._committed

    @property
    def has_trial(self) -> bool:
        return self._trial is not None

    def begin(self, total_strain: ArrayLike) -> J2Response:
        if self._trial is not None:
            raise RuntimeError("Material point already has an active trial state.")
        self._trial = self.material.update(total_strain, self._committed)
        return self._trial

    def commit(self) -> J2State:
        if self._trial is None:
            raise RuntimeError("Material point has no trial state to commit.")
        self._committed = self._trial.state
        self._trial = None
        return self._committed

    def rollback(self) -> J2State:
        if self._trial is None:
            raise RuntimeError("Material point has no trial state to roll back.")
        self._trial = None
        return self._committed
