from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator, Pauli

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

theta_x = Parameter("theta_x")
theta_y = Parameter("theta_y")
backend = BasicAer.get_backend("qasm_simulator")
shots = 1

# Example error probabilities
p_reset = 0.03
p_meas = 0.1
p_gate1 = 0.05

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])


qc = QuantumCircuit(2)

def cost(p_dict):
    try:
        p_01 = p_dict["01"]/shots
    except:
        p_01 = 0
    try:
        p_10 = p_dict["10"]/shots
    except:
        p_10 = 0 
    c = (p_01-0.5)**2 + (p_10-0.5)**2
    return c

def run_circuit(angles):
    p_dict = execute(qc.bind_parameters({theta_x:angles[0], theta_y:angles[1]}), backend = backend, shots = shots, basis_gates=noise_bit_flip.basis_gates).result().get_counts()
    return cost(p_dict)

qc.ry(theta_y, 0)
qc.cx(0, 1)
qc.rx(theta_x, 0)

qc.measure_all()
print(qc.draw())

tol = 1e-2 
for i in np.logspace(0, 3, 4):  
    shots  = i
    optimum_value = minimize(run_circuit, [0,0], method="powell", tol=tol)
    print("Shots = ", shots, "Final x angle = ",optimum_value.x[0], "Final y angle = ",optimum_value.x[1], "Least cost = ",optimum_value.fun)


