"""Minimal T-count MITM library for BlueQubit integration.

Public surface for tutorials and SDK embedding: unitary MITM, measurement-assisted
exhaustive search in the all-measurements-at-end family, stabilizer nullity, and
verified constructions.
"""

__version__ = "0.1.0"

from tgates_mitm.mitm import MitmResult, mitm_decide_leq, mitm_min_tcount
from tgates_mitm.exhaustive import MeasResult, decide_meas_tcount_leq
from tgates_mitm.nullity import (
    adaptive_lower_bound_ccz,
    build_choi_state,
    state_stabilizer_nullity,
    unitary_stabilizer_nullity,
)
from tgates_mitm.constructions import ccz_4t
from tgates_mitm.verify import count_t_gates, verify_adaptive_channel
from tgates_mitm.targets import GateTarget, ccz_target

__all__ = [
    "MitmResult",
    "mitm_decide_leq",
    "mitm_min_tcount",
    "MeasResult",
    "decide_meas_tcount_leq",
    "unitary_stabilizer_nullity",
    "build_choi_state",
    "state_stabilizer_nullity",
    "adaptive_lower_bound_ccz",
    "ccz_4t",
    "count_t_gates",
    "verify_adaptive_channel",
    "GateTarget",
    "ccz_target",
]
