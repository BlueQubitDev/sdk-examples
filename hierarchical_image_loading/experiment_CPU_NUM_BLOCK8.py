from functools import partial

import pennylane as qml
from bluequbit.library.hierarchical_circuit import ConnectivityAllToAllBuilder

schedule = [[4, 2], [10, 2], [14, 2], [18, 2]]
max_num_qubits = schedule[-1][0]

num_blocks_h = 2
num_blocks_w = 4
num_steps = len(schedule) * 1000
rotation_operations = [
    qml.RZ,
    qml.RY,
]
entanglement_operations = [qml.IsingZZ]
connectivity_builder = partial(
    ConnectivityAllToAllBuilder,
    num_qubits=max_num_qubits,
)
connectivity_type = "ALL_TO_ALL"
experiment_images = None
num_images_per_class_max = 10

device = "default.qubit"
gradient_method = "pennylane"
pennylane_diff_method = "best"
loss_type = "amplitude-l2"

num_gpus = 1
