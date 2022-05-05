import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    num_wires = 4
    
    dev = qml.device("default.qubit", wires=num_wires)
    hf_state = np.array([1, 1, 0, 0])
    
    def circuit(param, wires):
        qml.BasisState(hf_state, wires=range(num_wires))
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    
    
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(num_wires))
        return qml.expval(H)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    
    # store the values of the cost function
    energy = [cost_fn(theta)]

    # store the values of the circuit parameter
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        #if n % 2 == 0:
        #    print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break
    
    
    fin_state = np.zeros(2**4)
    fin_state[12] = np.cos(angle[-1]/2)
    fin_state[3] = -np.sin(angle[-1]/2)
    
    #print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    #print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
    return energy[-1], fin_state
    
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #
    gnd_matrix = np.outer(ground_state,ground_state)
    Hmat = qml.utils.sparse_hamiltonian(H).real.toarray()
    
    H1_matrix = Hmat + beta*gnd_matrix
    return qml.Hermitian(H1_matrix, wires=range(4))
    
    # QHACK #

def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    num_wires = 4
    
    dev = qml.device("default.qubit", wires=num_wires)
    hf_state = np.array([1, 0, 1, 0])
    
    def circuit(param, wires):
        qml.BasisState(hf_state, wires=range(num_wires))
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    
    
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(num_wires))
        return qml.expval(H1)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    
    # store the values of the cost function
    energy = [cost_fn(theta)]

    # store the values of the circuit parameter
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        #if n % 2 == 0:
        #    print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break
    
    #print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    #print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
    return energy[-1]
    
    # QHACK #

if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
