"""Elementary gate matrices and n-qubit builders."""

from __future__ import annotations

import math
from functools import reduce
from itertools import product
from typing import Iterable

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

import scipy.linalg as sla

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
YSQRT = sla.sqrtm(Y)
YSQRT_DAG = YSQRT.conj().T

SINGLE = {
    "i": I,
    "x": X,
    "y": Y,
    "ysqrt": YSQRT,
    "ysqrt_dag": YSQRT_DAG,
    "z": Z,
    "h": H,
    "s": S,
    "sdg": S.conj().T,
    "t": T,
    "tdg": T.conj().T,
}


def kron_all(mats: Iterable[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, mats)


def single_on_qubit(name: str, qubit: int, n: int) -> np.ndarray:
    mats = [I] * n
    mats[qubit] = SINGLE[name]
    return kron_all(mats)


def cnot(control: int, target: int, n: int) -> np.ndarray:
    dim = 2**n
    op = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = sum(b << (n - 1 - q) for q, b in enumerate(bits))
        op[j, i] = 1.0
    return op


def cz(control: int, target: int, n: int) -> np.ndarray:
    """Controlled-Z: phase -1 only when BOTH qubits are 1 (diag(1,1,1,-1))."""
    dim = 2**n
    op = np.eye(dim, dtype=complex)
    for i in range(dim):
        cbit = (i >> (n - 1 - control)) & 1
        tbit = (i >> (n - 1 - target)) & 1
        if cbit and tbit:
            op[i, i] = -1.0
    return op


def ccz_matrix(n: int = 3) -> np.ndarray:
    dim = 2**n
    u = np.eye(dim, dtype=complex)
    u[dim - 1, dim - 1] = -1.0
    return u


def toffoli_matrix() -> np.ndarray:
    dim = 8
    u = np.eye(dim, dtype=complex)
    for i in range(dim):
        c1, c2, t = (i >> 2) & 1, (i >> 1) & 1, i & 1
        if c1 and c2:
            u[i ^ 1, i] = 1.0
            u[i, i] = 0.0
    return u


def fredkin_matrix() -> np.ndarray:
    dim = 8
    u = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        c, t1, t2 = (i >> 2) & 1, (i >> 1) & 1, i & 1
        j = (c << 2) | (t2 << 1) | t1 if c else i
        u[j, i] = 1.0
    return u


def all_pauli_matrices(n: int) -> list[np.ndarray]:
    labels = ["i", "x", "y", "z"]
    out: list[np.ndarray] = []
    for combo in product(range(4), repeat=n):
        mats = [SINGLE[labels[c]] for c in combo]
        out.append(kron_all(mats))
    return out


def is_pauli_up_to_phase(matrix: np.ndarray, paulis: list[np.ndarray], atol: float = 1e-9) -> bool:
    for p in paulis:
        for phase in (1, -1, 1j, -1j):
            if np.allclose(matrix, phase * p, atol=atol):
                return True
    return False
