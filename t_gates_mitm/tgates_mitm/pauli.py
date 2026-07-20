"""Fast Pauli action on state vectors (no dense matrix build)."""

from __future__ import annotations

import numpy as np


def apply_pauli_combo(combo: tuple[int, ...], state: np.ndarray, n: int) -> np.ndarray:
    """Apply n-qubit Pauli (0=I,1=X,2=Y,3=Z) to state vector."""
    result = state
    for qubit, p in enumerate(combo):
        if p == 1:
            result = _apply_x(result, qubit, n)
        elif p == 2:
            result = _apply_y(result, qubit, n)
        elif p == 3:
            result = _apply_z(result, qubit, n)
    return result


def _bit_pos(qubit: int, n: int) -> int:
    return n - 1 - qubit


def _apply_x(state: np.ndarray, qubit: int, n: int) -> np.ndarray:
    result = np.zeros_like(state)
    bp = _bit_pos(qubit, n)
    flip = 1 << bp
    for basis in range(2**n):
        result[basis ^ flip] = state[basis]
    return result


def _apply_z(state: np.ndarray, qubit: int, n: int) -> np.ndarray:
    result = state.copy()
    bp = _bit_pos(qubit, n)
    for basis in range(2**n):
        if (basis >> bp) & 1:
            result[basis] *= -1
    return result


def _apply_y(state: np.ndarray, qubit: int, n: int) -> np.ndarray:
    return _apply_x(_apply_z(state, qubit, n), qubit, n) * 1j
