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
shots = 1

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
    p_dict = execute(qc.bind_parameters({theta_x:angles[0], theta_y:angles[1]}), backend = backend, shots = shots).result().get_counts()
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


