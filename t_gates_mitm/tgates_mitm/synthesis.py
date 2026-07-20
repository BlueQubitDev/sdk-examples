"""Rigorous Clifford+T T-count search via the Pauli-rotation normal form.

Key fact (exact synthesis):
    A unitary U on n qubits has T-count <= t (Cliffords free, no ancilla/measure)
    IFF there exist Hermitian Paulis P_1, ..., P_t such that

        U * R(-P_t) * ... * R(-P_1)   is Clifford,

    where R(P) = exp(-i*pi/8 * P).  Every T-gate equals a Clifford-conjugated
    Pauli rotation, so letting each P range over ALL Paulis absorbs the
    (otherwise astronomically large) Clifford layers between T-gates.

This module exposes the rigorous primitives.  The only operations needed are
"apply a Pauli rotation" and "is this matrix Clifford?", both of which are
numerically robust, so the lower bounds it produces are trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np

from tgates_mitm.gates import SINGLE, kron_all

_THETA = math.pi / 8.0
_COS = math.cos(_THETA)
_SIN = math.sin(_THETA)


# --------------------------------------------------------------------------
# Pauli set and rotations
# --------------------------------------------------------------------------
def pauli_labels(n: int, include_identity: bool = False) -> list[str]:
    """All n-qubit Pauli labels, e.g. 'XIZ'.  Optionally drop the all-I label."""
    labels = []
    for combo in product("IXYZ", repeat=n):
        s = "".join(combo)
        if not include_identity and s == "I" * n:
            continue
        labels.append(s)
    return labels


def parse_signed_label(label: str) -> tuple[int, str]:
    """Parse '+1IZZ' / '-1ZZ' into (sign, pauli_string)."""
    if label.startswith("+1"):
        return 1, label[2:]
    if label.startswith("-1"):
        return -1, label[2:]
    raise ValueError(f"bad signed Pauli label: {label!r}")


def pauli_matrix(label: str) -> np.ndarray:
    return kron_all(SINGLE[c.lower()] for c in label)


@dataclass(frozen=True)
class SignedPauli:
    label: str
    sign: int  # +1 or -1

    def matrix(self) -> np.ndarray:
        return self.sign * pauli_matrix(self.label)


def signed_paulis(n: int) -> list[SignedPauli]:
    """All +/- non-identity Paulis (2 * (4^n - 1) of them)."""
    out: list[SignedPauli] = []
    for lab in pauli_labels(n, include_identity=False):
        out.append(SignedPauli(lab, +1))
        out.append(SignedPauli(lab, -1))
    return out


def pauli_rotation(p: np.ndarray) -> np.ndarray:
    """R(P) = exp(-i*pi/8 * P) = cos(pi/8) I - i sin(pi/8) P  (P Hermitian, P^2=I)."""
    dim = p.shape[0]
    return _COS * np.eye(dim, dtype=complex) - 1j * _SIN * p


def rotation_for(sp: SignedPauli) -> np.ndarray:
    return pauli_rotation(sp.matrix())


# --------------------------------------------------------------------------
# Fast "is this matrix a (signed/phased) Pauli?" and "is this Clifford?"
# --------------------------------------------------------------------------
def is_signed_pauli(m: np.ndarray, atol: float = 1e-7) -> bool:
    """True iff m == phase * (n-qubit Pauli), phase in U(1).

    A Pauli has exactly one nonzero per column at row c^a (fixed XOR mask a),
    unit magnitudes, and a sign/phase pattern forming a linear character.
    """
    dim = m.shape[0]
    # exactly one nonzero per column, all at row = c ^ a
    nz_row = np.full(dim, -1, dtype=int)
    for c in range(dim):
        idx = np.flatnonzero(np.abs(m[:, c]) > atol)
        if idx.size != 1:
            return False
        nz_row[c] = idx[0]
    a = int(nz_row[0])
    for c in range(dim):
        if nz_row[c] != (c ^ a):
            return False
    phases = np.array([m[nz_row[c], c] for c in range(dim)], dtype=complex)
    if not np.allclose(np.abs(phases), 1.0, atol=atol):
        return False
    ratios = phases / phases[0]            # must be +/-1 forming a character
    if np.any(np.abs(ratios.imag) > atol):
        return False
    rr = np.round(ratios.real)
    if np.any(np.abs(ratios.real - rr) > atol) or np.any(np.abs(rr) != 1):
        return False
    signs = rr.astype(int)
    # character test: signs[c] = product of signs[2^k] over set bits k
    nbits = dim.bit_length() - 1
    base = [signs[1 << k] for k in range(nbits)]
    for c in range(dim):
        prod = 1
        for k in range(nbits):
            if (c >> k) & 1:
                prod *= base[k]
        if signs[c] != prod:
            return False
    return True


class CliffordChecker:
    """Caches single-qubit generator Paulis X_i, Z_i for is_clifford."""

    def __init__(self, n: int):
        self.n = n
        self.gens: list[np.ndarray] = []
        for i in range(n):
            xs = ["i"] * n
            xs[i] = "x"
            self.gens.append(kron_all(SINGLE[c] for c in xs))
            zs = ["i"] * n
            zs[i] = "z"
            self.gens.append(kron_all(SINGLE[c] for c in zs))

    def is_clifford(self, c: np.ndarray, atol: float = 1e-7) -> bool:
        cdag = c.conj().T
        for g in self.gens:
            if not is_signed_pauli(c @ g @ cdag, atol=atol):
                return False
        return True


# --------------------------------------------------------------------------
# Rigorous unitary-model decision and search
# --------------------------------------------------------------------------
@dataclass
class TCountResult:
    found: bool
    t: int
    witness: list[str] | None  # signed Pauli labels P_1..P_t if found


def decide_unitary_tcount_leq(
    u: np.ndarray,
    t: int,
    n: int,
    checker: CliffordChecker | None = None,
    rotations: list[tuple[str, np.ndarray]] | None = None,
    first_index: int | None = None,
) -> TCountResult:
    """Rigorously decide whether T-count(U) <= t (unitary model, up to global phase).

    Enumerates Pauli-rotation sequences; succeeds iff U * prod(R) is Clifford.
    `first_index` restricts the outermost Pauli to one index (for parallelism).
    """
    checker = checker or CliffordChecker(n)
    if rotations is None:
        sps = signed_paulis(n)
        rotations = [(f"{sp.sign:+d}{sp.label}", rotation_for(sp)) for sp in sps]

    if t == 0:
        return TCountResult(checker.is_clifford(u), 0, [] if checker.is_clifford(u) else None)

    dim = u.shape[0]

    def dfs(current: np.ndarray, depth: int, acc: list[str], idx_range) -> list[str] | None:
        if depth == 0:
            return list(acc) if checker.is_clifford(current) else None
        for i in idx_range:
            lab, r = rotations[i]
            acc.append(lab)
            res = dfs(current @ r, depth - 1, acc, range(len(rotations)))
            if res is not None:
                return res
            acc.pop()
        return None

    outer = range(len(rotations)) if first_index is None else range(first_index, first_index + 1)
    witness = dfs(u, t, [], outer)
    return TCountResult(witness is not None, t, witness)


def min_unitary_tcount(
    u: np.ndarray,
    n: int,
    max_t: int,
    checker: CliffordChecker | None = None,
) -> TCountResult:
    """Smallest t in 0..max_t with T-count(U) <= t (serial; use the script for parallel)."""
    checker = checker or CliffordChecker(n)
    rotations = [(f"{sp.sign:+d}{sp.label}", rotation_for(sp)) for sp in signed_paulis(n)]
    for t in range(max_t + 1):
        res = decide_unitary_tcount_leq(u, t, n, checker, rotations)
        if res.found:
            return res
    return TCountResult(False, max_t, None)


def witness_to_circuit_unitary(witness: list[str], n: int):
    """Reconstruct the Clifford+T residual circuit (Pauli rotations) for a witness.

    Returns (rotation_unitaries, residual_clifford) so callers can inspect/print.
    """
    rots = []
    prod = np.eye(2**n, dtype=complex)
    for lab in witness:
        sign, plab = parse_signed_label(lab)
        p = sign * pauli_matrix(plab)
        r = pauli_rotation(p)
        rots.append((lab, r))
        prod = prod @ r
    return rots, prod
