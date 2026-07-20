# t_gates_mitm

Meet-in-the-middle T-count search and certifying lower bounds, as a small installable
package plus a full tutorial notebook.

Designed for BlueQubit SDK / platform notebooks (`import tgates_mitm`).

## Layout

```text
t_gates_mitm/
  tgates_mitm/                                      # Python package
  tests/                                            # pytest suite
  pyproject.toml
  tutorial_certifying_tcount_lower_bounds.ipynb     # full tutorial
  README.md
```

## Install

From this directory:

```bash
pip install -e ".[dev]"
pip install qiskit matplotlib   # for notebook circuit drawings
```

## Run tests

```bash
pytest
```

## Tutorial notebook

Open `tutorial_certifying_tcount_lower_bounds.ipynb` after installing. It covers:

- Clifford vs T / magic states
- Unitary vs measurement-assisted T-count
- Stabilizer nullity
- Meet-in-the-middle lower bounds
- A CCZ example (unitary rule-out + measurement-assisted construction)

## Public API

```python
from tgates_mitm import (
    mitm_decide_leq,
    mitm_min_tcount,
    decide_meas_tcount_leq,
    unitary_stabilizer_nullity,
    build_choi_state,
    state_stabilizer_nullity,
    adaptive_lower_bound_ccz,
    ccz_4t,
    verify_adaptive_channel,
    count_t_gates,
    ccz_target,
)
```
