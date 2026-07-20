"""Statevector simulation with mid-circuit measurement and classical control."""

from __future__ import annotations

import math

import numpy as np

from tgates_mitm.circuit import Circuit, ConditionalGate, Gate, GateName, Measure, PauliRotation, CliffordUnitary, Unitary1Q
from tgates_mitm.gates import cnot, cz, single_on_qubit
from tgates_mitm.synthesis import pauli_matrix, rotation_for


def _apply_unitary1q(state: np.ndarray, u: np.ndarray, qubit: int, n: int) -> np.ndarray:
    full = single_on_qubit("i", 0, n)  # identity placeholder
    # Build full unitary via kron
    from tgates_mitm.gates import I, kron_all

    mats = [I] * n
    mats[qubit] = u
    op = kron_all(mats)
    return op @ state


def _apply_clifford_unitary(state: np.ndarray, flat: tuple[complex, ...], n: int) -> np.ndarray:
    dim = 2**n
    u = np.array(flat, dtype=complex).reshape(dim, dim)
    return u @ state


def _apply_pauli_rotation(state: np.ndarray, label: str, n: int) -> np.ndarray:
    from tgates_mitm.synthesis import SignedPauli, parse_signed_label, rotation_for

    sign, plab = parse_signed_label(label)
    return rotation_for(SignedPauli(plab, sign)) @ state


def _apply_gate(state: np.ndarray, gate: Gate, n: int) -> np.ndarray:
    name = gate.name
    if name in (
        GateName.H,
        GateName.S,
        GateName.SDG,
        GateName.X,
        GateName.Z,
        GateName.T,
        GateName.TDG,
        GateName.YSQRT,
        GateName.YSQRT_DAG,
    ):
        u = single_on_qubit(name.value, gate.qubits[0], n)
        return u @ state
    if name == GateName.CNOT:
        return cnot(gate.qubits[0], gate.qubits[1], n) @ state
    if name == GateName.CZ:
        return cz(gate.qubits[0], gate.qubits[1], n) @ state
    raise ValueError(f"Unsupported gate {name}")


def _basis_state(index: int, n: int) -> np.ndarray:
    dim = 2**n
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def initial_state(n_qubits: int, target_bits: int, n_target: int) -> np.ndarray:
    """|target_bits> on first n_target wires, |0> on ancilla."""
    full_index = target_bits << (n_qubits - n_target)
    return _basis_state(full_index, n_qubits)


def simulate_branches(
    circuit: Circuit,
    init_state: np.ndarray,
) -> list[tuple[float, np.ndarray, tuple[int, ...]]]:
    stack: list[tuple[np.ndarray, tuple[int, ...], float]] = [
        (init_state, tuple(), 1.0)
    ]

    for op in circuit.ops:
        next_stack: list[tuple[np.ndarray, tuple[int, ...], float]] = []
        if isinstance(op, Gate):
            for state, cbits, weight in stack:
                next_stack.append((_apply_gate(state, op, circuit.n_qubits), cbits, weight))
        elif isinstance(op, PauliRotation):
            for state, cbits, weight in stack:
                next_stack.append(
                    (_apply_pauli_rotation(state, op.label, circuit.n_qubits), cbits, weight)
                )
        elif isinstance(op, CliffordUnitary):
            for state, cbits, weight in stack:
                next_stack.append(
                    (_apply_clifford_unitary(state, op.flat, circuit.n_qubits), cbits, weight)
                )
        elif isinstance(op, Unitary1Q):
            u = np.array(op.matrix, dtype=complex)
            for state, cbits, weight in stack:
                next_stack.append(
                    (_apply_unitary1q(state, u, op.qubit, circuit.n_qubits), cbits, weight)
                )
        elif isinstance(op, Measure):
            dim = 2**circuit.n_qubits
            bit_pos = circuit.n_qubits - 1 - op.qubit
            for state, cbits, weight in stack:
                s0 = np.zeros(dim, dtype=complex)
                s1 = np.zeros(dim, dtype=complex)
                for i in range(dim):
                    amp = state[i]
                    if (i >> bit_pos) & 1:
                        s1[i] += amp
                    else:
                        s0[i] += amp
                p0 = float(np.vdot(s0, s0).real)
                p1 = float(np.vdot(s1, s1).real)
                ncb = max(circuit.n_cbits, op.cbit + 1)
                if p0 > 1e-12:
                    cb_list = list(cbits) + [0] * (ncb - len(cbits))
                    cb_list[op.cbit] = 0
                    next_stack.append(
                        (s0 / math.sqrt(p0), tuple(cb_list[:ncb]), weight * p0)
                    )
                if p1 > 1e-12:
                    cb_list = list(cbits) + [0] * (ncb - len(cbits))
                    cb_list[op.cbit] = 1
                    next_stack.append(
                        (s1 / math.sqrt(p1), tuple(cb_list[:ncb]), weight * p1)
                    )
        elif isinstance(op, ConditionalGate):
            for state, cbits, weight in stack:
                if op.cbit < len(cbits) and cbits[op.cbit] == op.value:
                    next_stack.append(
                        (_apply_gate(state, op.gate, circuit.n_qubits), cbits, weight)
                    )
                else:
                    next_stack.append((state, cbits, weight))
        else:
            raise TypeError(type(op))
        stack = next_stack

    merged: dict[tuple[int, ...], np.ndarray] = {}
    for state, cbits, weight in stack:
        amps = merged.get(cbits)
        if amps is None:
            amps = np.zeros_like(state)
        merged[cbits] = amps + math.sqrt(weight) * state

    out: list[tuple[float, np.ndarray, tuple[int, ...]]] = []
    for cbits, amps in merged.items():
        prob = float(np.vdot(amps, amps).real)
        if prob > 1e-12:
            out.append((prob, amps / math.sqrt(prob), cbits))
    return out


def simulate_unitary(circuit: Circuit) -> np.ndarray:
    if circuit.n_cbits:
        raise ValueError("Circuit has classical bits")
    n = circuit.n_qubits
    dim = 2**n
    u = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        state = _basis_state(j, n)
        for op in circuit.ops:
            if isinstance(op, Unitary1Q):
                u1 = np.array(op.matrix, dtype=complex)
                state = _apply_unitary1q(state, u1, op.qubit, n)
            elif isinstance(op, PauliRotation):
                state = _apply_pauli_rotation(state, op.label, n)
            elif isinstance(op, CliffordUnitary):
                state = _apply_clifford_unitary(state, op.flat, n)
            elif not isinstance(op, Gate):
                raise ValueError("Non-unitary op in simulate_unitary")
            else:
                state = _apply_gate(state, op, n)
        u[:, j] = state
    return u


def unitary_from_ops(n_qubits: int, ops: list[Gate]) -> np.ndarray:
    return simulate_unitary(Circuit(n_qubits, list(ops)))
