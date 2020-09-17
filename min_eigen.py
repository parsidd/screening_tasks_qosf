from qiskit import BasicAer, IBMQ, execute, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator, Pauli

from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit_textbook.tools import simon_oracle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
Task 4:
Find the lowest eigenvalue of the following matrix:

[1 0 0 0; 
0 0 -1 0;
0 -1 0 0; 
0 0 0 1]

using VQE-like circuits, created by yourself from scratch.
"""

""" Considering the Tensor products:
XX =    [0 0 0 1; 
         0 0 1 0;
         0 1 0 0; 
         1 0 0 0]

YY =    [0 0 0 -1; 
         0 0 1  0;
         0 1 0  0; 
        -1 0 0  0]

ZZ =    [1  0  0  0; 
         0 -1  0  0;
         0  0 -1  0; 
         0  0  0  1]

II =    [1 0 0 0;  (this is just I_4)
         0 1 0 0;
         0 0 1 0; 
         0 0 0 1]
It is easily shown that:
H = 1/2*(ZZ+II) -1/2*(XX+YY)

Can also be found by taking the trace with each of the Pauli operators in the required dimension as done in https://www.osti.gov/servlets/purl/1619265

The rot_x and rot_y operators are matrices that perform rotations in the 2-qubit Hilbert space, so that we can easily measure in the Z basis.

Try running the code with different starting points. Maybe choose them randomly.

"""

backend = BasicAer.get_backend("qasm_simulator")
shots = 4096
theta = Parameter('θ')
energy_values = []
angle_values = []
qc = []

rot_x = np.zeros([4,4])
rot_x[0][1] = 1
rot_x[0][2] = 1
rot_x[1][1] = -1
rot_x[1][2] = 1
rot_x[2][0] = -1
rot_x[2][3] = 1
rot_x[3][0] = 1
rot_x[3][3] = 1

rot_x = Operator(rot_x/np.sqrt(2))
# print(rot_x)

rot_y = np.zeros([4,4])
rot_y[0][0] = -1
rot_y[0][3] = 1
rot_y[1][0] = -1
rot_y[1][3] = -1
rot_y[2][1] = -1
rot_y[2][2] = 1
rot_y[3][1] = 1
rot_y[3][2] = 1

rot_y = Operator(rot_y/np.sqrt(2))
# print(rot)

# ansatz = QuantumCircuit(2)
# ansatz.h(0)
# ansatz.cx(0, 1)
# ansatz.rx(theta, 0)
# ansatz.ry(theta, 0)
# ansatz.cx(0, 1)
# ansatz.x(0)
# print(ansatz.decompose().draw())


def create_ansatz():
    ansatz = QuantumCircuit(2)
    ansatz.h(0)
    ansatz.cx(0, 1)
    ansatz.rx(theta, 0)
    # ansatz.ry(theta, 0)
    # ansatz.cx(0, 1)
    # ansatz.x(0)
    ansatz = ansatz.to_gate()
    ansatz.label = "ANSATZ(theta)"
    return ansatz

def measure_H_expectation(angle):
    energy = 0
    for basis, circuit in enumerate(qc):
        circuit = circuit.bind_parameters({theta: angle[0]})
        t = measure_expectation(circuit)
        if(basis == 1 or basis == 2):
            energy -= t
        else:
            energy += t
    energy += 1
    energy /= 2
    angle_values.append(angle[0])
    energy_values.append(energy)
    print("{} {}".format(angle[0], energy))
    return energy


def measure_expectation(qc):
    p_dict = execute(qc, backend=backend, shots=shots).result().get_counts()
    expectation = 0
    for state,counts in p_dict.items():
        sign = 1
        if(state == '01' or state == '10'):
            sign = -1
        expectation += sign*counts/shots
    return expectation

XX = Operator(Pauli(label='XX'))
YY = Operator(Pauli(label='YY'))
ZZ = Operator(Pauli(label='ZZ'))
II = Operator(Pauli(label='II'))

H_created = 0.5*(ZZ+II) - 0.5*(XX+YY)
for i in range(3):
    qc.append(QuantumCircuit(2))
    qc[i].append(create_ansatz(), [0,1])

qc[1].append(rot_x, [0,1])
qc[2].append(rot_y, [0,1])

for i in qc:
    i.measure_all()
    # print(i.decompose().draw())

for angle in np.linspace(0,2*np.pi,100):
    measure_H_expectation([angle])

# tol = 1e-1 
# optimised_result = minimize(measure_H_expectation, np.pi/2, method="powell", tol=tol)
# print("Number of iterations taken = ",len(energy_values))
# print("Least eigenvalue found = ", optimised_result.fun,"At theta = ",optimised_result.x[0])

fig = plt.figure()
plt.scatter(angle_values, energy_values)
plt.show()
# print(angle_values)
# print(energy_values)

