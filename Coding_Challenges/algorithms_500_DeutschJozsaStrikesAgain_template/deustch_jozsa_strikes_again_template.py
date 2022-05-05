#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """
    
    # QHACK #
    all_bits = 5
    oracle_wires = range(2,5)
    dev = qml.device("default.qubit", wires=all_bits)

    ctrl_f1 = qml.ctrl(f1, control=[0,1])
    ctrl_f2 = qml.ctrl(f2, control=[0,1])
    ctrl_f3 = qml.ctrl(f3, control=[0,1])
    ctrl_f4 = qml.ctrl(f4, control=[0,1])

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=all_bits-1)
        qml.Barrier(wires=range(all_bits))

        for wire in range(all_bits):
            qml.Hadamard(wires=wire)

        qml.Barrier(wires=range(all_bits))
        ctrl_f1(oracle_wires)

        qml.Barrier(wires=range(all_bits))
        qml.PauliX(wires=0)
        ctrl_f2(oracle_wires)
        qml.PauliX(wires=0)

        qml.Barrier(wires=range(all_bits))
        qml.PauliX(wires=1)
        ctrl_f3(oracle_wires)
        qml.PauliX(wires=1)

        qml.Barrier(wires=range(all_bits))
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        ctrl_f4(oracle_wires)
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)

        qml.Barrier(wires=range(all_bits))
        for wire in range(0,all_bits-1):
            qml.Hadamard(wires=wire)

        return qml.probs(wires=range(2,4))
    
    probs = circuit()
    if np.isclose(probs[0],1) or np.isclose(probs[3],1):
        return '4 same'
    else:
        return '2 and 2'
    
    return circuit()
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
