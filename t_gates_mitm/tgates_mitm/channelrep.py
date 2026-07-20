"""Channel (Pauli-transfer) representation and canonical form modulo left-Clifford.

This is the device that makes meet-in-the-middle work.

For an n-qubit unitary U, its channel matrix is the 4^n x 4^n real matrix
    M(U)[i, j] = Tr(P_i U P_j U^dag) / 2^n,
where P_0..P_{4^n-1} are the Hermitian Paulis. Facts we use:

  * M(U V) = M(U) M(V)            (homomorphism)
  * M(U) determines U up to global phase   (faithful)
  * U Clifford  <=>  M(U) is a signed permutation matrix.

Therefore two unitaries V, W lie in the same *left* Clifford coset (V = C W,
C Clifford) iff M(V) = S M(W) with S a signed permutation, i.e. M(V) and M(W)
agree after permuting rows and flipping row signs. The canonical form below
(sign-normalize each row, then sort rows) is invariant under that action, so
equal cosets always produce equal keys (no missed collisions). A larger group
of signed permutations may occasionally merge distinct Clifford cosets, so any
key collision is re-checked exactly with CliffordChecker before being accepted.
"""

from __future__ import annotations

import numpy as np

from tgates_mitm.gates import all_pauli_matrices


class ChannelRep:
    """Precomputes Pauli data for fast channel matrices on n qubits."""

    def __init__(self, n: int):
        self.n = n
        self.dim = 2**n
        self.paulis = all_pauli_matrices(n)  # 4^n Hermitian Paulis, P_0 = I
        self.npauli = len(self.paulis)
        self._P = np.stack(self.paulis)                       # (npauli, dim, dim)
        self._Pflat = self._P.reshape(self.npauli, -1)        # row-major flatten

    def channel_matrix(self, u: np.ndarray) -> np.ndarray:
        """M[i, j] = Tr(P_i U P_j U^dag) / 2^n   (real-valued), fully vectorized.

        Tr(P_i V_j) = sum_{a,b} P_i[a,b] V_j[b,a] = <flat(P_i), flat(V_j^T)>.
        """
        udag = u.conj().T
        up = np.einsum("ab,kbc->kac", u, self._P, optimize=True)     # U P_j
        v = np.einsum("kac,cd->kad", up, udag, optimize=True)        # U P_j U^dag
        vt = np.transpose(v, (0, 2, 1)).reshape(self.npauli, -1)     # flat(V_j^T)
        m = (self._Pflat @ vt.T).real / self.dim
        return m


def canonical_left_coset_key(m: np.ndarray, decimals: int = 6) -> bytes:
    """Key invariant under M -> (signed permutation) M (left action).

    Sign-normalize each row (first significant entry made positive), then sort
    the rows lexicographically. Equal left-Clifford cosets -> equal keys.
    """
    mr = np.round(m, decimals)
    rows = []
    for r in mr:
        nz = np.flatnonzero(np.abs(r) > 10.0 ** (-decimals) / 2)
        if nz.size and r[nz[0]] < 0:
            r = -r
        r = r + 0.0  # normalize -0.0 -> 0.0 so byte keys match
        rows.append(r)
    rows.sort(key=lambda v: v.tobytes())
    return np.stack(rows).tobytes()
