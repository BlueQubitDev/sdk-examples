"""Provably-complete measurement-assisted T-count search (all-measurements-at-end form).

See docs/completeness.md. The normal form searched here is:

    R(P_1) ... R(P_t)            # t pi/8 Pauli rotations on w = n_t + n_a wires
    [final Clifford, folded in]  # absorbed: see below
    measure the n_a ancilla wires in a Pauli basis Q_1..Q_{n_a}
    per-outcome Clifford correction on the data wires
    trace out ancilla; output data

This is complete for every measurement-assisted circuit in which **no T-gate
follows a measurement** (which includes all known optimal constructions for
CCZ/CCCZ/Toffoli/Fredkin/AND: Jones, Gidney, Gidney-Jones). A null result at t
is therefore a rigorous lower bound *within that class*.

Matching test (the crux): a rotation product U_R realizes target unitary G via
this form iff there is a set of n_a commuting independent measured Paulis such
that, for every measurement outcome o, the induced data map V_o is a unitary
with `V_o G^dag` Clifford (a 0-T correction). We never enumerate the Clifford
group: the final/correction Cliffords are quotiented out by the `is_clifford`
check, exactly as the unitary engine quotients them with `is_clifford` on the
residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from tgates_mitm.gates import all_pauli_matrices, single_on_qubit, kron_all, I as I1
from tgates_mitm.synthesis import CliffordChecker, rotation_for, signed_paulis, SignedPauli, parse_signed_label


# ---------------------------------------------------------------------------
# Branch / measurement machinery
# ---------------------------------------------------------------------------
def _ancilla_zero_input(n_t: int, n_a: int) -> np.ndarray:
    """Columns = data basis states tensored with ancilla |0>: a (2^w, 2^n_t) isometry."""
    w = n_t + n_a
    dim_w = 2 ** w
    dim_t = 2 ** n_t
    dim_a = 2 ** n_a
    iso = np.zeros((dim_w, dim_t), dtype=complex)
    for x in range(dim_t):
        iso[x * dim_a, x] = 1.0  # |x>_data |0>_anc  (data are high bits)
    return iso


def _commuting(p: np.ndarray, q: np.ndarray, atol: float = 1e-9) -> bool:
    return np.allclose(p @ q, q @ p, atol=atol)


def measurement_pauli_sets(n_t: int, n_a: int, max_sets: int | None = None):
    """Yield sets of n_a commuting, independent Hermitian Paulis on w=n_t+n_a wires.

    These are the ancilla-readout bases (a final Clifford rotates Z_anc into an
    arbitrary such set; see docs/completeness.md). For n_a=0 yields the empty set.
    """
    w = n_t + n_a
    if n_a == 0:
        yield []
        return
    paulis = all_pauli_matrices(w)[1:]  # drop identity
    if n_a == 1:
        for p in paulis:
            yield [p]
        return
    if n_a == 2:
        count = 0
        for i in range(len(paulis)):
            for j in range(i + 1, len(paulis)):
                if _commuting(paulis[i], paulis[j]):
                    # independence: p_i != p_j and neither is +-I*the other (Paulis distinct already)
                    yield [paulis[i], paulis[j]]
                    count += 1
                    if max_sets is not None and count >= max_sets:
                        return
        return
    raise NotImplementedError("n_a > 2 not supported yet")


def _projectors(measured: list[np.ndarray], dim_w: int):
    """All 2^k joint eigen-projectors of commuting Hermitian Paulis `measured`."""
    k = len(measured)
    eye = np.eye(dim_w, dtype=complex)
    if k == 0:
        return [(tuple(), eye)]
    out = []
    for signs in product((+1, -1), repeat=k):
        proj = eye.copy()
        for s, P in zip(signs, measured):
            proj = proj @ ((eye + s * P) / 2.0)
        out.append((signs, proj))
    return out


def induced_data_map(branch_state_cols: np.ndarray, n_t: int, n_a: int, atol: float = 1e-7):
    """Given a (2^w, 2^n_t) matrix whose column x = (projected) state for input |x,0>,
    return (prob, V) where V is the data unitary (up to global phase) if every column
    factorizes as (same ancilla state) x (data vector) and the data map is unitary;
    else return (prob, None). prob is the per-branch probability (must be x-independent).
    """
    dim_t = 2 ** n_t
    dim_a = 2 ** n_a
    # reshape each column to (data, anc); require rank-1 with a COMMON ancilla vector
    data_cols = np.zeros((dim_t, dim_t), dtype=complex)
    probs = []
    anc_ref = None
    for x in range(dim_t):
        col = branch_state_cols[:, x].reshape(dim_t, dim_a)  # [data, anc]
        nrm2 = float(np.vdot(col, col).real)
        probs.append(nrm2)
        if nrm2 < atol:
            data_cols[:, x] = 0.0
            continue
        u, s, vh = np.linalg.svd(col)
        if s.size > 1 and s[1] > atol * (1 + s[0]):
            return 0.0, None  # data-ancilla entangled in this branch -> not coherent
        anc_vec = vh[0].conj()
        data_vec = u[:, 0] * s[0]
        if anc_ref is None:
            anc_ref = anc_vec
        else:
            # ancilla must collapse to the SAME state across inputs (else not a clean unitary channel)
            phase = np.vdot(anc_ref, anc_vec)
            if abs(abs(phase) - 1.0) > 1e-6:
                return 0.0, None
            data_vec = data_vec * phase  # reabsorb relative phase into data
        data_cols[:, x] = data_vec
    prob0 = probs[0]
    if any(abs(p - prob0) > 1e-6 for p in probs):
        return 0.0, None  # outcome probability depends on input -> not a unitary channel
    if prob0 < atol:
        return 0.0, None
    V = data_cols / np.sqrt(prob0)
    # V must be unitary (up to global phase)
    if not np.allclose(V.conj().T @ V, np.eye(dim_t), atol=1e-6):
        return 0.0, None
    return prob0, V


def realizes_target(
    U_R: np.ndarray,
    G: np.ndarray,
    n_t: int,
    n_a: int,
    measured: list[np.ndarray],
    checker: CliffordChecker,
    atol: float = 1e-7,
) -> bool:
    """True iff [U_R; measure `measured`; per-outcome Clifford correction] == G as a channel."""
    w = n_t + n_a
    dim_w = 2 ** w
    iso = _ancilla_zero_input(n_t, n_a)        # (2^w, 2^n_t)
    evolved = U_R @ iso                         # column x = U_R |x,0>
    total = 0.0
    for _signs, proj in _projectors(measured, dim_w):
        branch = proj @ evolved
        prob, V = induced_data_map(branch, n_t, n_a, atol=atol)
        if V is None:
            # branch may be zero-probability (allowed) -> skip; else fail
            colnorm = float(np.vdot(branch, branch).real)
            if colnorm < atol:
                continue
            return False
        total += prob * (2 ** 0)  # prob already per-branch (x-independent)
        # correction must be a 0-T Clifford: V_o G^dag in Clifford  <=> G V_o^dag in Clifford
        if not checker.is_clifford(V @ G.conj().T):
            return False
    # probabilities of accepted branches must sum to 1 (channel is trace preserving on data)
    if abs(total - 1.0) > 1e-6:
        return False
    return True


# ---------------------------------------------------------------------------
# AND-compute matcher (isometry target; the unitary matcher above does not apply)
# ---------------------------------------------------------------------------
# AND-compute is the Gidney temporary-AND isometry  |a,b,0> -> |a,b, a&b>  with a
# FREE relative phase per input (the phase is garbage, cancelled by the matching
# uncompute; see tgates/verify.verify_and_compute).  It is NOT a unitary on the 3
# output wires, so `realizes_target` cannot be used: only 4 of the 8 computational
# inputs (those with the out wire = 0) are populated, and the output is required only
# up to a per-input phase.
#
# Normal form decided here (a sound, explicitly-scoped restriction of the general
# all-measurements-at-end form -- the same restriction the CCCZ family search uses):
#
#     R(P_1) ... R(P_t)                 # t pi/8 rotations on w = n_t + n_a wires
#     measure a commuting Pauli set Q   # the n_a ancilla read-outs
#     per-outcome correction on the 3 output wires (a, b, out):
#         * an out-wire bit-fix  out -> out XOR (alpha*a XOR beta*b XOR c)
#           (every control-preserving computational-basis Clifford byproduct;
#            realised by CNOT(a->out), CNOT(b->out), X(out) conditioned on Q)
#         * any DIAGONAL Clifford (Z, S, CZ): only injects phases, which are free
#           for AND-compute, so it is absorbed and never needs to be solved
#
# Completeness scope: this covers every measurement-assisted AND-compute whose
# feedforward is a (control-preserving) Clifford -- which includes Gidney's 4-T
# construction and any phase-gadget variant.  A rule-out is therefore a rigorous
# lower bound within this normal form (stated explicitly in docs/and_compute_bound.md);
# it is the same intellectual-honesty scope as docs/completeness.md / cccz_bound.md.
# Soundness is exact algebra on the channel (no Clifford enumeration); the matcher is
# cross-checked against verify_and_compute on the known 4-T gate in tests/test_and_search.py.

# the four AND-compute inputs |a,b,0> and their nominal (uncorrected) output index
_AND_INPUTS = [(a, b) for a in (0, 1) for b in (0, 1)]


def _and_input_isometry(n_t: int, n_a: int) -> np.ndarray:
    """Columns = the 4 inputs |a,b,0>|0_anc>: a (2^w, 4) isometry.

    Free data wires are 0 (a) and 1 (b); the out wire (2) and the n_a ancillas are
    initialised to |0>.  Wire 0 is the most-significant bit (matches simulator/_ancilla_zero_input)."""
    w = n_t + n_a
    iso = np.zeros((2 ** w, 4), dtype=complex)
    for col, (a, b) in enumerate(_AND_INPUTS):
        idx = (a << (w - 1)) | (b << (w - 2))   # out wire and all ancillas = 0
        iso[idx, col] = 1.0
    return iso


def induced_and_map(branch_cols: np.ndarray, n_t: int, n_a: int, atol: float = 1e-7):
    """Given a (2^w, 4) matrix whose column (a,b) = (projected) state for input |a,b,0>,
    return (prob, outs) where outs[col] is the length-2^n_t output-wire vector if every
    column factorises as (output-wire vector) x (common ancilla vector) and the per-branch
    probability is input-independent; else (0.0, None).  Mirrors `induced_data_map` but for
    the 4-column AND isometry instead of a full 2^n_t-column unitary."""
    dim_t = 2 ** n_t
    dim_a = 2 ** n_a
    outs: list = [None, None, None, None]
    probs = []
    anc_ref = None
    for col in range(4):
        v = branch_cols[:, col].reshape(dim_t, dim_a)   # [output wires, ancilla]
        nrm2 = float(np.vdot(v, v).real)
        probs.append(nrm2)
        if nrm2 < atol:
            # AND is deterministic on every input; a branch that vanishes on one input
            # but not another has input-dependent outcome probability -> not a clean channel
            return 0.0, None
        u, s, vh = np.linalg.svd(v)
        if s.size > 1 and s[1] > atol * (1 + s[0]):
            return 0.0, None      # output-ancilla entangled in this branch -> not coherent
        anc_vec = vh[0].conj()
        out_vec = u[:, 0] * s[0]
        if anc_ref is None:
            anc_ref = anc_vec
        else:
            phase = np.vdot(anc_ref, anc_vec)
            if abs(abs(phase) - 1.0) > 1e-6:
                return 0.0, None  # ancilla must collapse to the SAME state across inputs
            out_vec = out_vec * phase
        outs[col] = out_vec
    prob0 = probs[0]
    if any(abs(p - prob0) > 1e-6 for p in probs):
        return 0.0, None          # outcome probability depends on input -> not a channel
    if prob0 < atol:
        return 0.0, None
    outs = [o / np.sqrt(prob0) for o in outs]
    return prob0, outs


def _accepts_and(outs: list, atol: float = 1e-5) -> bool:
    """True iff some control-preserving Clifford out-wire fix turns the 4 output vectors
    into the AND truth table: exists (alpha,beta,c) in {0,1}^3 s.t. for every input (a,b)
    the output is the single computational state |a, b, (a&b) XOR (alpha*a ^ beta*b ^ c)>
    (unit modulus, no leakage; the per-input phase is free)."""
    for alpha in (0, 1):
        for beta in (0, 1):
            for c in (0, 1):
                ok = True
                for col, (a, b) in enumerate(_AND_INPUTS):
                    obit = ((a & b) ^ (alpha * a) ^ (beta * b) ^ c) & 1
                    tgt = (a << 2) | (b << 1) | obit
                    vec = outs[col]
                    amp = abs(vec[tgt])
                    leak = float(np.vdot(vec, vec).real) - amp * amp
                    if abs(amp - 1.0) > atol or leak > 1e-7:
                        ok = False
                        break
                if ok:
                    return True
    return False


def realizes_and_compute(
    U_R: np.ndarray,
    n_t: int,
    n_a: int,
    measured: list[np.ndarray],
    atol: float = 1e-7,
) -> bool:
    """True iff [U_R; measure `measured`; per-outcome control-preserving Clifford fix]
    realises the AND-compute isometry |a,b,0> -> (phase) |a,b,a&b>.  Sound (exact channel
    algebra) and complete within the normal form documented above."""
    w = n_t + n_a
    dim_w = 2 ** w
    iso = _and_input_isometry(n_t, n_a)
    evolved = U_R @ iso
    total = 0.0
    for _signs, proj in _projectors(measured, dim_w):
        branch = proj @ evolved
        prob, outs = induced_and_map(branch, n_t, n_a, atol=atol)
        if outs is None:
            colnorm = float(np.vdot(branch, branch).real)
            if colnorm < atol:        # zero-probability branch -> allowed, skip
                continue
            return False
        if not _accepts_and(outs):
            return False
        total += prob
    if abs(total - 1.0) > 1e-6:
        return False
    return True


def and_compute_matcher(n_t: int, n_a: int, atol: float = 1e-7):
    """Return a `matcher(U_R, Q) -> bool` closure for `decide_meas_tcount_leq`."""
    def _match(U_R: np.ndarray, Q: list) -> bool:
        return realizes_and_compute(U_R, n_t, n_a, Q, atol=atol)
    return _match


# ---------------------------------------------------------------------------
# Enumeration of rotation products (deduped by left-Clifford coset) + decider
# ---------------------------------------------------------------------------
@dataclass
class MeasResult:
    found: bool
    t: int
    witness_rotation: np.ndarray | None
    witness_measure: list | None
    n_cosets: int


import hashlib


def _phase_canon_key(u: np.ndarray, decimals: int = 6) -> bytes:
    """Compact 32-byte key for a unitary up to global phase (divide out the phase
    of the largest-magnitude entry, round, then hash). Hashing keeps the dedup set
    tiny (32 B/state) so the search is memory-bounded even for large tables."""
    flat = u.reshape(-1)
    k = int(np.argmax(np.abs(flat)))
    ph = flat[k] / abs(flat[k])
    v = np.round((u / ph), decimals)
    v.real[v.real == 0] = 0.0
    v.imag[v.imag == 0] = 0.0
    return hashlib.sha256(v.tobytes()).digest()


def _coset_hash(crep, u: np.ndarray) -> bytes:
    """32-byte hash of the left-Clifford coset canonical key (the full key is the
    4^w x 4^w channel matrix -> too big to store; we only need it for dedup)."""
    from tgates_mitm.channelrep import canonical_left_coset_key

    return hashlib.sha256(canonical_left_coset_key(crep.channel_matrix(u))).digest()


def decide_meas_tcount_leq(
    G: np.ndarray,
    n_t: int,
    n_a: int,
    t: int,
    *,
    measure_sets: list | None = None,
    prefix: tuple = (),
    seen: set | None = None,
    matcher=None,
    rotations=None,
    crep=None,
    checker=None,
) -> MeasResult:
    """Decide whether `G` has measurement-assisted T-count <= t (all-at-end form).

    Memory-bounded: depth-first over length-`t` signed-Pauli-rotation products, each
    deduped by a 32-byte left-Clifford-coset hash (sound: U and C*U realize the same
    measurement-assisted channels, docs/completeness.md) and tested on the fly. Only
    the set of seen hashes is retained (~32 B per distinct coset), so there is no
    260k-unitary blow-up. `prefix` fixes the first len(prefix) outermost rotation
    indices (for parallelism / checkpointing — each prefix is one independent task).
    `matcher(U_R, Q)` overrides the unitary matcher (used for AND).
    """
    from tgates_mitm.channelrep import ChannelRep

    w = n_t + n_a
    chk_data = checker if checker is not None else CliffordChecker(n_t)
    crep = crep if crep is not None else ChannelRep(w)
    rotations = rotations if rotations is not None else [rotation_for(sp) for sp in signed_paulis(w)]
    qsets = measure_sets if measure_sets is not None else list(measurement_pauli_sets(n_t, n_a))
    eye = np.eye(2 ** w, dtype=complex)
    if seen is None:
        seen = set()

    def _matches(U_R) -> tuple | None:
        if matcher is not None:
            for Q in qsets:
                if matcher(U_R, Q):
                    return (U_R, Q)
            return None
        for Q in qsets:
            if realizes_target(U_R, G, n_t, n_a, Q, chk_data):
                return (U_R, Q)
        return None

    if t == 0:
        h = _coset_hash(crep, eye)
        seen.add(h)
        m = _matches(eye)
        if m:
            return MeasResult(True, 0, m[0], m[1], len(seen))
        return MeasResult(False, 0, None, None, len(seen))

    witness: list = [None]

    def rec(cur: np.ndarray, depth: int) -> bool:
        if depth == 0:
            h = _coset_hash(crep, cur)
            if h in seen:
                return False
            seen.add(h)
            m = _matches(cur)
            if m:
                witness[0] = m
                return True
            return False
        pos = t - depth  # 0 = outermost applied rotation
        idxs = [prefix[pos]] if pos < len(prefix) else range(len(rotations))
        for i in idxs:
            if rec(rotations[i] @ cur, depth - 1):
                return True
        return False

    found = rec(eye, t)
    if found:
        return MeasResult(True, t, witness[0][0], witness[0][1], len(seen))
    return MeasResult(False, t, None, None, len(seen))


def min_meas_tcount(G: np.ndarray, n_t: int, n_a: int, max_t: int) -> MeasResult:
    for t in range(max_t + 1):
        res = decide_meas_tcount_leq(G, n_t, n_a, t)
        if res.found:
            return res
    return MeasResult(False, max_t, None, None, 0)
