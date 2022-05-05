#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.
    n = 3

    def ops(theta):
        qml.RY(theta,wires=3)
    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))
    
    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        for wire in range(3):
            qml.Hadamard(wires=wire)
        
        for i, theta in enumerate(thetas):
            wire_vals = np.array(list(np.binary_repr(i,width=n)),dtype=int)
            for wire, wire_val in enumerate(wire_vals):
                if wire_val == 0:
                    qml.PauliX(wires=wire)

            ops1 = qml.ctrl(ops, control=[0,1,2])
            ops1(theta)
            for wire, wire_val in enumerate(wire_vals):
                if wire_val == 0:
                    qml.PauliX(wires=wire)

            qml.Barrier(wires=range(4))
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
