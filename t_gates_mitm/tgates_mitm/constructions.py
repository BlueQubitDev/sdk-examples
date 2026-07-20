"""Verified measurement-assisted achievability circuits (upper bounds).

Every circuit here is checked by ``tgates.verify.verify_adaptive_channel`` /
``verify_and_*`` in ``tests/test_constructions.py`` as an exact quantum channel
(full process verification over a tomographically complete input set, phases
included).  These are the trustworthy achievability witnesses for the T-count
table; the historical ``tgates.known`` circuits were transcribed incorrectly
(uncorrected byproducts) and must not be used.

Pattern: temporary-AND (Gidney 2018).  Compute AND(controls)->ancilla (4 T),
apply a controlled Clifford to the target, then measurement-uncompute the AND
(0 T); the relative phases of compute and uncompute cancel.
"""

from __future__ import annotations

from tgates_mitm.circuit import Circuit


def and_compute_into(c: Circuit, a: int, b: int, anc: int) -> None:
    """Append Gidney AND: anc (in |0>) <- a AND b, using 4 T-gates."""
    c.h(anc).t(anc)
    c.cnot(a, anc).cnot(b, anc)
    c.cnot(anc, a).cnot(anc, b)
    c.tdg(a).tdg(b).t(anc)
    c.cnot(anc, a).cnot(anc, b)
    c.h(anc).s(anc)


def and_uncompute_from(c: Circuit, a: int, b: int, anc: int, cbit: int) -> None:
    """Append measurement-uncompute of AND(a,b) held in ``anc`` (0 T-gates).

    Measures ``anc`` in the X basis; on outcome 1 resets it to |0> (if_x) and
    cancels the phase kickback with CZ on the controls (if_cz).
    """
    c.h(anc)
    c.measure(anc, cbit)
    c.if_x(cbit, 1, anc)
    c.if_cz(cbit, 1, a, b)


def and_compute() -> Circuit:
    """Gidney AND compute on (0,1)->2, 4 T.  |a,b,0> -> |a,b,a&b>."""
    c = Circuit(3, n_cbits=0)
    and_compute_into(c, 0, 1, 2)
    return c


def and_uncompute() -> Circuit:
    """Gidney AND uncompute on (0,1) with garbage in 2, 0 T.  |a,b,a&b> -> |a,b,0>."""
    c = Circuit(3, n_cbits=1)
    and_uncompute_from(c, 0, 1, 2, 0)
    return c


def ccz_4t() -> Circuit:
    """CCZ on qubits 0,1,2 with one ancilla (wire 3).  4 T-gates."""
    c = Circuit(4, n_cbits=1)
    and_compute_into(c, 0, 1, 3)   # anc <- (0 AND 1)
    c.cz(3, 2)                     # phase -1 iff (0&1) and 2  ==  0&1&2
    and_uncompute_from(c, 0, 1, 3, 0)
    return c


def toffoli_4t() -> Circuit:
    """Toffoli (controls 0,1; target 2) with one ancilla (wire 3).  4 T-gates."""
    c = Circuit(4, n_cbits=1)
    c.h(2)
    and_compute_into(c, 0, 1, 3)
    c.cz(3, 2)
    and_uncompute_from(c, 0, 1, 3, 0)
    c.h(2)
    return c


def fredkin_4t() -> Circuit:
    """Fredkin / controlled-SWAP (control 0; targets 1,2) with ancilla wire 3.  4 T."""
    c = Circuit(4, n_cbits=1)
    c.cnot(2, 1)
    c.h(2)
    and_compute_into(c, 0, 1, 3)
    c.cz(3, 2)
    and_uncompute_from(c, 0, 1, 3, 0)
    c.h(2)
    c.cnot(2, 1)
    return c


def cccz_8t() -> Circuit:
    """CCCZ on qubits 0,1,2,3 with two ancillas (wires 4,5).  8 T-gates (naive ladder).

    Superseded as an upper bound by ``cccz_6t``; retained as the And-ladder baseline
    (= Qualtran's 4(n-1) construction for n=3) and for regression tests.
    """
    c = Circuit(6, n_cbits=2)
    and_compute_into(c, 0, 1, 4)   # anc4 <- 0&1   (4T)
    and_compute_into(c, 4, 2, 5)   # anc5 <- (0&1)&2  (4T)
    c.cz(5, 3)                     # phase iff 0&1&2&3
    and_uncompute_from(c, 4, 2, 5, 1)
    and_uncompute_from(c, 0, 1, 4, 0)
    return c


def cccz_6t() -> Circuit:
    """CCCZ on qubits 0,1,2,3 with one ancilla (wire 4).  6 T-gates.

    Gidney & Jones, "A CCCZ gate performed with 6 T gates" (arXiv:2106.11513, Fig. 1).
    Two relative-phase Toffolis (phase kickback onto the controls) write
    ab XOR cd onto the ancilla; the ancilla is phased and removed by a
    measurement in the Y basis (H S-dag H == X^{-1/2}, then measure).  The
    kickback cancels the i^(ab) i^(cd) factors from the identity

        i^(ab XOR cd) = i^(ab) * i^(cd) * (-1)^(ab cd)

    leaving (-1)^(0&1&2&3) = CCCZ.  The two measurement outcomes need different
    Clifford fix-ups: CZ(2,3) on outcome 0, CZ(0,1) on outcome 1.
    """
    c = Circuit(5, n_cbits=1)
    a, b, cc, d, m = 0, 1, 2, 3, 4
    c.h(m).t(m)
    c.cnot(b, m).tdg(m)
    c.cnot(a, m).t(m)
    c.cnot(b, m).cnot(cc, m).tdg(m)
    c.cnot(d, m).t(m)
    c.cnot(cc, m).tdg(m)
    c.cnot(d, m)
    c.h(m).sdg(m).h(m)            # X^{-1/2}: rotate ancilla into the Y measurement basis
    c.measure(m, 0)
    c.if_cz(0, 0, cc, d)          # outcome 0 -> CZ on controls (2,3)
    c.if_cz(0, 1, a, b)          # outcome 1 -> CZ on controls (0,1)
    return c
