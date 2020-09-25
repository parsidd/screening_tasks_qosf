from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator, Pauli

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

theta_x = Parameter("theta_x")
theta_y = Parameter("theta_y")
backend = BasicAer.get_backend("qasm_simulator")
shots = 1000

qc = QuantumCircuit(2)

def cost(p_dict):
    try:
        c = 1 - p_dict["10"]/shots
    except:
        c = 1
    return c

def run_circuit(angles):
    p_dict = execute(qc.bind_parameters({theta_x:angles[0], theta_y:angles[1]}), backend = backend, shots = shots).result().get_counts()
    return cost(p_dict)

qc.ry(theta_y, 0)
qc.cx(0, 1)
qc.rx(theta_x, 0)
qc.cx(0, 1)
qc.ry(np.pi/2, 0)

qc.measure_all()
print(qc.draw())

tol = 1e-2 
optimum_value = minimize(run_circuit, [0,0], method="powell", tol=tol)
print("Final x angle = ",optimum_value.x[0], "Final y angle = ",optimum_value.x[1], "Least cost = ",optimum_value.fun)


