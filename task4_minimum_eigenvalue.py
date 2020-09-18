from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator, Pauli

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


backend = BasicAer.get_backend("qasm_simulator")
shots = 4096
theta = Parameter('theta')

eigenvalues =  {0:{"00":1,"01":-1,"10":-1,"11":1},
                1:{"00":1,"01":-1,"10":1,"11":-1},
                2:{"00":-1,"01":1,"10":1,"11":-1}}

energy_values = []
angle_values = []
qc = []

def add_ansatz1():
    ansatz = QuantumCircuit(2)
    ansatz.ry(theta, 0)
    ansatz.cx(0, 1)
    ansatz.x(0)
    ansatz = ansatz.to_gate()
    ansatz.label = "ANSATZ(theta)"
    return ansatz
def add_ansatz2():
    ansatz = QuantumCircuit(2)
    ansatz.h(0)
    ansatz.cx(0, 1)
    ansatz.rx(theta, 0)
    ansatz = ansatz.to_gate()
    ansatz.label = "ANSATZ(theta)"
    return ansatz

def measure_H_expectation(angle):
    energy = 0
    results_z = execute(qc[0].bind_parameters({theta: angle[0]}), backend = backend, shots = shots).result().get_counts()
    results_xy = execute(qc[1].bind_parameters({theta: angle[0]}), backend = backend, shots = shots).result().get_counts()
    z_exp = measure_expectation(0, results_z)
    x_exp = measure_expectation(1, results_xy)
    y_exp = measure_expectation(2, results_xy)
    energy = (z_exp + 1 - x_exp - y_exp)/2
    angle_values.append(angle[0])
    energy_values.append(energy)
    # print("{} {}".format(angle[0], energy))
    return energy


def measure_expectation(basis, p_dict):
    expectation = 0
    for state,counts in p_dict.items():
        expectation += eigenvalues[basis][state]*counts/shots
    return expectation

for i in range(2):
    qc.append(QuantumCircuit(2))
    qc[i].append(add_ansatz1(), [0,1])
    # qc[i] = qc[i].bind_parameters({theta: 0})

qc[1].cx(0,1)
qc[1].h(0)

for i in qc:
    i.measure_all()
    print(i.decompose().draw())

# for angle in np.linspace(0,2*np.pi,10):
#     measure_H_expectation([angle])

tol = 1e-2 
optimised_result = minimize(measure_H_expectation, 0, method="powell", tol=tol)
print("Number of iterations taken = ",len(energy_values))
print("Least eigenvalue found = ", optimised_result.fun,"At theta = ",optimised_result.x[0])

fig = plt.figure()
plt.scatter(angle_values, energy_values)
plt.show()
# print(angle_values)
# print(energy_values)

