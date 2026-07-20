"""Target gates for T-count search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tgates_mitm.gates import ccz_matrix, fredkin_matrix, toffoli_matrix


class TargetKind(str, Enum):
    UNITARY = "unitary"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class GateTarget:
    name: str
    n_target: int
    n_ancilla: int
    kind: TargetKind
    unitary: np.ndarray | None = None
    description: str = ""

    @property
    def n_qubits(self) -> int:
        return self.n_target + self.n_ancilla


def ccz_target(n_ancilla: int = 1) -> GateTarget:
    return GateTarget(
        name="CCZ",
        n_target=3,
        n_ancilla=n_ancilla,
        kind=TargetKind.ADAPTIVE,
        unitary=ccz_matrix(3),
        description="3-qubit CCZ with ancilla initialized in |0>",
    )


def cccz_target(n_ancilla: int = 2) -> GateTarget:
    return GateTarget(
        name="CCCZ",
        n_target=4,
        n_ancilla=n_ancilla,
        kind=TargetKind.ADAPTIVE,
        unitary=ccz_matrix(4),
        description="4-qubit CCCZ with ancilla initialized in |0>",
    )


def toffoli_target(n_ancilla: int = 1) -> GateTarget:
    return GateTarget(
        name="Toffoli",
        n_target=3,
        n_ancilla=n_ancilla,
        kind=TargetKind.ADAPTIVE,
        unitary=toffoli_matrix(),
        description="Toffoli (X on target if both controls are 1)",
    )


def fredkin_target(n_ancilla: int = 0) -> GateTarget:
    return GateTarget(
        name="Fredkin",
        n_target=3,
        n_ancilla=n_ancilla,
        kind=TargetKind.ADAPTIVE,
        unitary=fredkin_matrix(),
        description="Controlled-SWAP (Fredkin)",
    )


def and_compute_target() -> GateTarget:
    return GateTarget(
        name="AND_compute",
        n_target=3,
        n_ancilla=0,
        kind=TargetKind.ADAPTIVE,
        unitary=None,
        description="Gidney AND compute: |x1,x2,0> -> |x1,x2,x1 AND x2> (relative phase)",
    )


def and_uncompute_target() -> GateTarget:
    return GateTarget(
        name="AND_uncompute",
        n_target=3,
        n_ancilla=0,
        kind=TargetKind.ADAPTIVE,
        unitary=None,
        description="Gidney AND uncompute (0 T; measurement-assisted)",
    )


def all_targets(n_ancilla: int = 1) -> list[GateTarget]:
    return [
        ccz_target(n_ancilla),
        cccz_target(max(2, n_ancilla)),
        toffoli_target(n_ancilla),
        fredkin_target(n_ancilla),
        and_compute_target(),
        and_uncompute_target(),
    ]
