"""Smoke tests for the minimal public API (controlled-S + CCZ verify)."""

import numpy as np

from tgates_mitm import (
    adaptive_lower_bound_ccz,
    ccz_4t,
    ccz_target,
    count_t_gates,
    decide_meas_tcount_leq,
    mitm_decide_leq,
    verify_adaptive_channel,
)


def test_controlled_s_unitary_mitm():
    cs = np.diag([1, 1, 1, 1j]).astype(complex)
    assert not mitm_decide_leq(cs, 2, 2, show_progress=False).found
    assert mitm_decide_leq(cs, 3, 2, show_progress=False).found


def test_controlled_s_meas_exhaustive():
    cs = np.diag([1, 1, 1, 1j]).astype(complex)
    assert not decide_meas_tcount_leq(cs, 2, 0, 2).found
    assert decide_meas_tcount_leq(cs, 2, 0, 3).found


def test_ccz_construction_verified():
    circ = ccz_4t()
    assert count_t_gates(circ) == 4
    assert verify_adaptive_channel(circ, ccz_target(1))
    assert adaptive_lower_bound_ccz() >= 3
