"""Circuit DSL for Clifford+T with measurement and classical feedforward."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class GateName(str, Enum):
    H = "h"
    S = "s"
    SDG = "sdg"
    X = "x"
    Z = "z"
    T = "t"
    TDG = "tdg"
    CNOT = "cnot"
    CZ = "cz"
    YSQRT = "ysqrt"
    YSQRT_DAG = "ysqrt_dag"


@dataclass(frozen=True)
class Gate:
    name: GateName
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class Measure:
    qubit: int
    cbit: int


@dataclass(frozen=True)
class ConditionalGate:
    cbit: int
    value: int
    gate: Gate


@dataclass(frozen=True)
class Unitary1Q:
    qubit: int
    matrix: tuple[tuple[complex, complex], tuple[complex, complex]]


@dataclass(frozen=True)
class PauliRotation:
    """Single T-count Pauli rotation R(P)=exp(-i*pi/8 * P) on n qubits."""

    label: str  # signed label, e.g. "+1IZZ"


@dataclass(frozen=True)
class CliffordUnitary:
    """Opaque Clifford (zero T-cost); used to apply a synthesized Clifford prefix."""

    flat: tuple[complex, ...]  # row-major 2^n x 2^n


Op = Union[Gate, Measure, ConditionalGate, Unitary1Q, PauliRotation, CliffordUnitary]


@dataclass
class Circuit:
    n_qubits: int
    ops: list[Op] = field(default_factory=list)
    n_cbits: int = 0

    def copy(self) -> Circuit:
        return Circuit(self.n_qubits, list(self.ops), self.n_cbits)

    def gate(self, name: GateName, *qubits: int) -> Circuit:
        self.ops.append(Gate(name, qubits))
        return self

    def h(self, q: int) -> Circuit:
        return self.gate(GateName.H, q)

    def s(self, q: int) -> Circuit:
        return self.gate(GateName.S, q)

    def sdg(self, q: int) -> Circuit:
        return self.gate(GateName.SDG, q)

    def x(self, q: int) -> Circuit:
        return self.gate(GateName.X, q)

    def z(self, q: int) -> Circuit:
        return self.gate(GateName.Z, q)

    def t(self, q: int) -> Circuit:
        return self.gate(GateName.T, q)

    def tdg(self, q: int) -> Circuit:
        return self.gate(GateName.TDG, q)

    def cnot(self, control: int, target: int) -> Circuit:
        return self.gate(GateName.CNOT, control, target)

    def cz(self, control: int, target: int) -> Circuit:
        return self.gate(GateName.CZ, control, target)

    def measure(self, qubit: int, cbit: int | None = None) -> Circuit:
        bit = cbit if cbit is not None else self.n_cbits
        if bit >= self.n_cbits:
            self.n_cbits = bit + 1
        self.ops.append(Measure(qubit, bit))
        return self

    def if_gate(self, cbit: int, value: int, gate: Gate) -> Circuit:
        self.ops.append(ConditionalGate(cbit, value, gate))
        return self

    def if_x(self, cbit: int, value: int, qubit: int) -> Circuit:
        return self.if_gate(cbit, value, Gate(GateName.X, (qubit,)))

    def if_z(self, cbit: int, value: int, qubit: int) -> Circuit:
        return self.if_gate(cbit, value, Gate(GateName.Z, (qubit,)))

    def if_cz(self, cbit: int, value: int, control: int, target: int) -> Circuit:
        return self.if_gate(cbit, value, Gate(GateName.CZ, (control, target)))

    def extend(self, other: Circuit) -> Circuit:
        if other.n_qubits != self.n_qubits:
            raise ValueError("Qubit count mismatch")
        self.ops.extend(other.ops)
        self.n_cbits = max(self.n_cbits, other.n_cbits)
        return self
