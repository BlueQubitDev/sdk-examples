"""Rigorous meet-in-the-middle T-count search (unitary model).

Decides T-count(U) <= t by splitting a length-t Pauli-rotation sequence into a
forward half (length a = ceil(t/2)) and a backward half (length b = t - a):

    U = C * R(P_1)..R(P_a) * R(P_{a+1})..R(P_t),   C Clifford
      <=>  M_left( R(P_1)..R(P_a) ) == M_left( U * (R(P_{a+1})..R(P_t))^dag )

where M_left is the channel-matrix canonical form modulo left-Clifford
(channelrep.canonical_left_coset_key). Forward keys go in a hash table; the
backward half looks for collisions. Every hash collision is verified exactly
(is the residual genuinely Clifford?) so the result is rigorous: no missed
collisions (Clifford => signed permutation => same key) and no false accepts
(exact check). This is the Amy/Gosset MITM specialized to T-rotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tgates_mitm.channelrep import ChannelRep, canonical_left_coset_key
from tgates_mitm.synthesis import CliffordChecker, rotation_for, signed_paulis


def _progress(iterable: Iterable, total: int | None, desc: str, show: bool):
    if not show:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:  # noqa: BLE001
        return iterable


@dataclass
class MitmResult:
    found: bool
    t: int
    witness: list[str] | None
    forward_table_size: int = 0


def _rotation_library(n: int):
    sps = signed_paulis(n)
    labels = [f"{sp.sign:+d}{sp.label}" for sp in sps]
    mats = [rotation_for(sp) for sp in sps]
    return labels, mats


def _enumerate_products(mats, depth: int, base: np.ndarray):
    """Yield (index_tuple, product_matrix) for all length-`depth` sequences."""
    n = len(mats)

    def rec(prefix_mat, prefix_idx, d):
        if d == 0:
            yield prefix_idx, prefix_mat
            return
        for i in range(n):
            yield from rec(prefix_mat @ mats[i], prefix_idx + (i,), d - 1)

    yield from rec(base, (), depth)


def mitm_decide_leq(
    u: np.ndarray,
    t: int,
    n: int,
    checker: CliffordChecker | None = None,
    crep: ChannelRep | None = None,
    show_progress: bool = True,
) -> MitmResult:
    checker = checker or CliffordChecker(n)
    crep = crep or ChannelRep(n)

    if t == 0:
        return MitmResult(checker.is_clifford(u), 0, [] if checker.is_clifford(u) else None)

    labels, mats = _rotation_library(n)
    nrot = len(mats)
    a = math.ceil(t / 2)
    b = t - a
    dim = 2**n
    eye = np.eye(dim, dtype=complex)

    # ---- forward: build table key -> list of forward index tuples ----
    table: dict[bytes, list[tuple[int, ...]]] = {}
    fwd_total = nrot**a
    for idx, fmat in _progress(
        _enumerate_products(mats, a, eye), fwd_total, f"fwd t={t} (a={a})", show_progress
    ):
        key = canonical_left_coset_key(crep.channel_matrix(fmat))
        table.setdefault(key, []).append(idx)

    # ---- backward: enumerate length-b, look up U * Bmat^dag ----
    bwd_total = nrot**b
    for bidx, bmat in _progress(
        _enumerate_products(mats, b, eye), bwd_total, f"bwd t={t} (b={b})", show_progress
    ):
        key = canonical_left_coset_key(crep.channel_matrix(u @ bmat.conj().T))
        cand = table.get(key)
        if not cand:
            continue
        for fidx in cand:
            fmat = eye.copy()
            for i in fidx:
                fmat = fmat @ mats[i]
            pmat = fmat @ bmat
            c = u @ pmat.conj().T
            if checker.is_clifford(c):
                witness = [labels[i] for i in fidx] + [labels[i] for i in bidx]
                return MitmResult(True, t, witness, len(table))
    return MitmResult(False, t, None, len(table))


def mitm_min_tcount(
    u: np.ndarray,
    n: int,
    max_t: int,
    show_progress: bool = True,
) -> MitmResult:
    checker = CliffordChecker(n)
    crep = ChannelRep(n)
    for t in range(max_t + 1):
        res = mitm_decide_leq(u, t, n, checker, crep, show_progress)
        if res.found:
            return res
    return MitmResult(False, max_t, None)
