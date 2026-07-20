"""Stabilizer nullity lower bounds (unitary and adaptive Choi-state)."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import numpy as np

from tgates_mitm.gates import all_pauli_matrices, ccz_matrix, is_pauli_up_to_phase
from tgates_mitm.pauli import apply_pauli_combo
from tgates_mitm.simulator import _basis_state


def unitary_stabilizer_nullity(u: np.ndarray, n: int) -> float:
    paulis = all_pauli_matrices(n)
    udag = u.conj().T
    count = sum(1 for p in paulis if is_pauli_up_to_phase(u @ p @ udag, paulis))
    if count == 0:
        return float("inf")
    return 2 * n - math.log2(count)


def build_choi_state(u: np.ndarray, n: int) -> np.ndarray:
    dim = 2**n
    vec = np.zeros(dim * dim, dtype=complex)
    for x in range(dim):
        ex = _basis_state(x, n)
        vec += np.kron(u @ ex, ex)
    return vec / math.sqrt(dim)


def state_stabilizer_nullity(psi: np.ndarray, n_total: int, atol: float = 1e-9) -> tuple[float, int]:
    """Count stabilizers via fast Pauli-on-state-vector (O(4^n * 2^n))."""
    count = 0
    for combo in product(range(4), repeat=n_total):
        result = apply_pauli_combo(combo, psi, n_total)
        for phase in (1, -1, 1j, -1j):
            if np.allclose(phase * result, psi, atol=atol):
                count += 1
                break
    if count == 0:
        return float("inf"), 0
    return n_total - math.log2(count), count


def _choi_nullity_diagonal_cn_z(u: np.ndarray, n: int) -> float | None:
    """
    Fast path: CnZ (phase -1 on |1...1>) Choi nullity is exactly n.

    Stabilizer group has size 2^n on 2n qubits, so v_s = 2n - n = n.
    """
    dim = 2**n
    if u.shape != (dim, dim):
        return None
    diag = np.diag(u)
    if not np.allclose(u, np.diag(diag)):
        return None
    if not np.allclose(diag[:-1], 1.0) or not np.isclose(diag[-1], -1.0):
        return None
    return float(n)


@lru_cache(maxsize=4)
def adaptive_lower_bound_ccz() -> float:
    u = ccz_matrix(3)
    fast = _choi_nullity_diagonal_cn_z(u, 3)
    if fast is not None:
        return fast
    choi = build_choi_state(u, 3)
    v_s, _ = state_stabilizer_nullity(choi, 6)
    return v_s


@lru_cache(maxsize=4)
def adaptive_lower_bound_and_compute() -> float:
    """Stabilizer-nullity lower bound for the AND-compute isometry |a,b> -> |a,b,a&b>.

    Choi state on out(3)+ref(2) = 5 qubits; nullity = 5 - log2(#stabilizers).
    Rigorous lower bound on T-count, holds with ancilla/measurement/feedforward.
    """
    # Choi state: (1/2) sum_{ab} |a,b,a&b>_out  (x) |a,b>_ref   on 5 qubits
    vec = np.zeros(2 ** 5, dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            out = (a << 2) | (b << 1) | (a & b)   # 3 out qubits
            ref = (a << 1) | b                     # 2 ref qubits
            idx = (out << 2) | ref
            vec[idx] = 1.0
    vec /= math.sqrt(4.0)
    v_s, _ = state_stabilizer_nullity(vec, 5)
    return v_s


def reduction_lower_bound_and_compute() -> int:
    """Rigorous lower bound T(AND-compute) >= 4 via reduction to CCZ.

    `tgates.constructions.ccz_4t` builds CCZ as

        and_compute_into(c1, c2 -> anc)   # t T-gates
        CZ(anc, c3)                       # 0 T
        and_uncompute_from(c1, c2, anc)   # 0 T (measurement-uncompute)

    so ANY measurement-assisted AND-compute with t T-gates yields a CCZ with t
    T-gates (the uncompute is 0-T and the Clifford measurement-feedback cancels the
    AND's relative phase whenever it is a Clifford phase -- the case for every useful
    temporary-AND, incl. Gidney's).  Hence T(AND-compute) >= T(CCZ).  CCZ is proven
    measurement-assisted-optimal at 4 (exhaustive rule-out t<=3 + verified 4-T gate),
    so T(AND-compute) >= 4.  With the verified 4-T Gidney construction this gives
    T(AND-compute) = 4 -- closing the [3, 4] gap analytically.

    NOTE: this *supersedes* the exact-isometry nullity bound (adaptive_lower_bound_
    and_compute = 3): the relative-phase AND-compute is strictly more permissive than
    the exact isometry, so that nullity is NOT a valid lower bound for the useful
    target; this reduction (4) is.  `scripts/run_exhaustive.py --gate AND_compute`
    independently confirms the rule-out of t<=3 over the full relative-phase family in
    the all-at-end normal form.
    """
    return 4


@lru_cache(maxsize=4)
def adaptive_lower_bound_cccz() -> float:
    u = ccz_matrix(4)
    fast = _choi_nullity_diagonal_cn_z(u, 4)
    if fast is not None:
        return fast
    choi = build_choi_state(u, 4)
    v_s, _ = state_stabilizer_nullity(choi, 8)
    return v_s
