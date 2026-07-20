"""Verify circuits implement target gates."""

from __future__ import annotations

import math

import numpy as np

from tgates_mitm.circuit import Circuit, Gate, GateName, Measure, PauliRotation
from tgates_mitm.simulator import initial_state, simulate_branches, simulate_unitary
from tgates_mitm.targets import GateTarget


def count_t_gates(circuit: Circuit) -> int:
    n = 0
    for op in circuit.ops:
        if isinstance(op, PauliRotation):
            n += 1
        elif isinstance(op, Gate) and op.name in (GateName.T, GateName.TDG):
            n += 1
        elif hasattr(op, "gate") and isinstance(op.gate, Gate):
            if op.gate.name in (GateName.T, GateName.TDG):
                n += 1
    return n


def verify_unitary(circuit: Circuit, target: np.ndarray, atol: float = 1e-8) -> bool:
    u = simulate_unitary(circuit)
    global_phase: complex | None = None
    for j in range(target.shape[1]):
        if np.linalg.norm(target[:, j]) < atol:
            continue
        if np.linalg.norm(u[:, j]) < atol:
            return False
        mask = np.abs(target[:, j]) > atol
        ratio = u[mask, j] / target[mask, j]
        col_phase = ratio[0]
        if not np.allclose(ratio, col_phase, atol=atol):
            return False
        if global_phase is None:
            global_phase = col_phase
        elif not np.allclose(col_phase, global_phase, atol=atol):
            return False
    if global_phase is None:
        return False
    return np.allclose(u, global_phase * target, atol=atol)


def _target_density(state: np.ndarray, n_qubits: int, n_target: int) -> np.ndarray:
    dim_t = 2**n_target
    dim_a = 2 ** (n_qubits - n_target)
    tensor = state.reshape(dim_t, dim_a)
    rho = np.zeros((dim_t, dim_t), dtype=complex)
    for a in range(dim_a):
        v = tensor[:, a]
        rho += np.outer(v, v.conj())
    return rho


def _measured_qubits(circuit: Circuit) -> set[int]:
    return {op.qubit for op in circuit.ops if isinstance(op, Measure)}


def _fingerprint_bytes(rho: np.ndarray, decimals: int = 6) -> bytes:
    """Round and serialize a density matrix, normalizing -0.0 -> 0.0 so that
    numerically-equal matrices always produce identical bytes."""
    r = np.round(rho, decimals)
    # collapse -0.0 -> +0.0 in real and imag parts independently (adding 0.0 to
    # -0.0 yields +0.0 under IEEE), so numerically-equal matrices serialize equally
    r.real[r.real == 0] = 0.0
    r.imag[r.imag == 0] = 0.0
    return r.tobytes()


def _embed_target_state(psi: np.ndarray, n_qubits: int, n_target: int) -> np.ndarray:
    """Full-system state with target-wire amplitudes ``psi`` and ancilla in |0>."""
    dim = 2**n_qubits
    dim_a = 2 ** (n_qubits - n_target)
    full = np.zeros(dim, dtype=complex)
    for x in range(psi.shape[0]):
        full[x * dim_a] = psi[x]
    return full


def tomographic_inputs(n_target: int) -> list[np.ndarray]:
    """Tomographically complete set of target-wire pure states.

    The density matrices {|psi><psi|} of these states span the full Hermitian
    operator space, so matching a channel's output density matrix on every
    member is necessary and sufficient for channel equality (process tomography).
    Captures phases that computational-basis inputs alone cannot (e.g. diagonal
    gates such as CCZ/CCCZ).
    """
    dim_t = 2**n_target
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    states: list[np.ndarray] = []
    for x in range(dim_t):
        v = np.zeros(dim_t, dtype=complex)
        v[x] = 1.0
        states.append(v)
    for x in range(dim_t):
        for y in range(x + 1, dim_t):
            v = np.zeros(dim_t, dtype=complex)
            v[x] = inv_sqrt2
            v[y] = inv_sqrt2
            states.append(v)
            w = np.zeros(dim_t, dtype=complex)
            w[x] = inv_sqrt2
            w[y] = 1j * inv_sqrt2
            states.append(w)
    return states


def _channel_output_density(
    circuit: Circuit,
    psi: np.ndarray,
    n_qubits: int,
    n_target: int,
    measured: set[int],
    *,
    require_ancilla_zero: bool,
    atol: float,
) -> np.ndarray | None:
    """Full target-wire output density rho_out = sum_b p_b * Tr_anc(|b><b|).

    Returns None if any branch leaves an unmeasured ancilla off |0> (so the
    channel does not act on the target wires alone), or total probability != 1.
    """
    dim_t = 2**n_target
    inp = _embed_target_state(psi, n_qubits, n_target)
    branches = simulate_branches(circuit, inp)
    rho = np.zeros((dim_t, dim_t), dtype=complex)
    total_p = 0.0
    for prob, state, _cbits in branches:
        if require_ancilla_zero:
            p0 = _ancilla_zero_probability(state, n_qubits, n_target, measured)
            if not math.isclose(p0, 1.0, abs_tol=atol):
                return None
        rho += prob * _target_density(state, n_qubits, n_target)
        total_p += prob
    if not math.isclose(total_p, 1.0, abs_tol=1e-8):
        return None
    return rho


def _ancilla_zero_probability(
    state: np.ndarray,
    n_qubits: int,
    n_target: int,
    measured: set[int],
) -> float:
    """Probability that every *unmeasured* ancilla wire is |0>."""
    dim_a = 2 ** (n_qubits - n_target)
    dim_t = 2**n_target
    tensor = state.reshape(dim_t, dim_a)
    prob = 0.0
    for a in range(dim_a):
        ok = True
        for wire in range(n_target, n_qubits):
            if wire in measured:
                continue
            anc_bit = wire - n_target
            if (a >> anc_bit) & 1:
                ok = False
                break
        if ok:
            prob += float(np.vdot(tensor[:, a], tensor[:, a]).real)
    return prob


def verify_adaptive_channel(
    circuit: Circuit,
    target: GateTarget,
    atol: float = 1e-8,
    *,
    require_ancilla_zero: bool = True,
) -> bool:
    if target.unitary is None:
        raise ValueError(f"Target {target.name} has no unitary; use verify_and_compute")
    n_q = target.n_qubits
    n_t = target.n_target
    u = target.unitary

    measured = _measured_qubits(circuit)
    for psi in tomographic_inputs(n_t):
        rho = _channel_output_density(
            circuit, psi, n_q, n_t, measured,
            require_ancilla_zero=require_ancilla_zero, atol=atol,
        )
        if rho is None:
            return False
        out = u @ psi
        expected = np.outer(out, out.conj())  # phase-faithful: U|psi><psi|U^dag
        if not np.allclose(rho, expected, atol=atol):
            return False
    return True


def verify_and_compute(circuit: Circuit, atol: float = 1e-8) -> bool:
    """
    Gidney AND compute: valid on inputs |a,b,0>.

    Output target wire holds a AND b; relative phases differ from a true unitary extension.
  """
    for a in (0, 1):
        for b in (0, 1):
            x = (a << 2) | (b << 1) | 0
            inp = initial_state(3, x, 3)
            branches = simulate_branches(circuit, inp)
            acc = np.zeros(8, dtype=complex)
            for prob, state, _ in branches:
                acc += prob * state
            expected_index = (a << 2) | (b << 1) | (a & b)
            if abs(acc[expected_index]) < 1.0 - atol:
                return False
            wrong = sum(abs(acc[i]) ** 2 for i in range(8) if i != expected_index)
            if wrong > atol:
                return False
    return True


def verify_and_uncompute(circuit: Circuit, atol: float = 1e-8) -> bool:
    """
    Gidney AND uncompute: valid on inputs |a,b,a&b>, output is |a,b,0>.
    """
    for a in (0, 1):
        for b in (0, 1):
            t = a & b
            x = (a << 2) | (b << 1) | t
            inp = initial_state(3, x, 3)
            branches = simulate_branches(circuit, inp)
            acc = np.zeros(8, dtype=complex)
            for prob, state, _ in branches:
                acc += prob * state
            expected_index = (a << 2) | (b << 1) | 0
            if abs(acc[expected_index]) < 1.0 - atol:
                return False
            wrong = sum(abs(acc[i]) ** 2 for i in range(8) if i != expected_index)
            if wrong > atol:
                return False
    return True


def channel_fingerprint(
    circuit: Circuit,
    n_target: int,
    *,
    n_qubits: int | None = None,
    require_ancilla_zero: bool = True,
) -> bytes:
    """Exact adaptive channel fingerprint for MITMS matching (small n only).

    Concatenates per-input target-wire diagonal action (same basis as
    ``verify_adaptive_channel``).  Used to deduplicate candidates and for
    collision tables in the adaptive search.
    """
    n_q = n_qubits if n_qubits is not None else circuit.n_qubits
    measured = _measured_qubits(circuit)
    dim_t = 2**n_target
    parts: list[bytes] = []
    for psi in tomographic_inputs(n_target):
        rho = _channel_output_density(
            circuit, psi, n_q, n_target, measured,
            require_ancilla_zero=require_ancilla_zero, atol=1e-8,
        )
        if rho is None:
            rho = np.full((dim_t, dim_t), np.nan, dtype=complex)
        parts.append(_fingerprint_bytes(rho))
    return b"".join(parts)


def target_channel_fingerprint(target: GateTarget) -> bytes:
    """Reference channel fingerprint for a unitary target (adaptive model).

    Computed identically to ``channel_fingerprint`` (phase-faithful output
    density U|psi><psi|U^dag over the tomographic input set) so the two are
    directly comparable.
    """
    if target.unitary is None:
        raise ValueError(f"Target {target.name} has no unitary")
    n_t = target.n_target
    u = target.unitary
    parts: list[bytes] = []
    for psi in tomographic_inputs(n_t):
        out = u @ psi
        rho = np.outer(out, out.conj())
        parts.append(_fingerprint_bytes(rho))
    return b"".join(parts)


def verify_and_uncompute_pair(compute: Circuit, uncompute: Circuit, atol: float = 1e-8) -> bool:
    """After compute then uncompute, controls are unchanged and target returns to |0>."""
    combined = compute.copy()
    combined.extend(uncompute)
    for a in (0, 1):
        for b in (0, 1):
            x = (a << 2) | (b << 1) | 0
            inp = initial_state(3, x, 3)
            branches = simulate_branches(combined, inp)
            acc = np.zeros(8, dtype=complex)
            for prob, state, _ in branches:
                acc += prob * state
            if not np.allclose(acc[x], 1.0, atol=atol):
                return False
            other = sum(abs(acc[i]) ** 2 for i in range(8) if i != x)
            if other > atol:
                return False
    return True
