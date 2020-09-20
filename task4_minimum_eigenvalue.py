from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Pauli

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

backend = BasicAer.get_backend("qasm_simulator")

qubit_2_eigenvalues = {"Z":{"00":1,"01":-1,"10":-1,"11":1},
                       "X":{"00":1,"01":-1,"10":1,"11":-1},
                       "Y":{"00":-1,"01":1,"10":1,"11":-1}}


class VQE_2_qubit:
    def __init__(self, theta = Parameter("theta"), shots = 2048):
        self.energy_values = []
        self.angle_values = []
        self.qc = []
        self.theta = theta
        self.shots = shots

    # This is an ansatz I used first to run the program.
    def add_ansatz1(self):
        ansatz = QuantumCircuit(2)
        ansatz.ry(self.theta, 0)
        ansatz.cx(0, 1)
        ansatz.x(0)
        ansatz = ansatz.to_gate()
        ansatz.label = "ANSATZ1(theta)"
        return ansatz

    # Ansatz given in hint
    def add_ansatz2(self):
        ansatz = QuantumCircuit(2)
        ansatz.h(0)
        ansatz.cx(0, 1)
        ansatz.rx(self.theta, 0)
        ansatz = ansatz.to_gate()
        ansatz.label = "ANSATZ2(theta)"
        return ansatz

    # Need to add this part as well. Want to see if it converges faster. Probably wont.
    # def add_ansatz3():
    #     ansatz = QuantumCircuit(2)
    #     ansatz.0)
    #     ansatz.cx(0, 1)
    #     ansatz.rx(theta, 0)
    #     ansatz = ansatz.to_gate()
    #     ansatz.label = "ANSATZ(theta)"
    #     return ansatz

    # Measure the energy expectation value of the given operator given a variational parameter
    def measure_H_expectation(self, angle):
        energy = 0
        results_z = execute(self.qc[0].bind_parameters({self.theta: angle[0]}), backend = backend, shots = self.shots).result().get_counts()
        results_xy = execute(self.qc[1].bind_parameters({self.theta: angle[0]}), backend = backend, shots = self.shots).result().get_counts()
        # Find expectation of each Pauli operator
        z_exp = self.measure_expectation("Z", results_z)
        x_exp = self.measure_expectation("X", results_xy)
        y_exp = self.measure_expectation("Y", results_xy)
        # Energy in terms of expectation value in each basis
        energy = (z_exp + 1 - x_exp - y_exp)/2
        self.angle_values.append(angle[0])
        self.energy_values.append(energy)
        # print("{} {}".format(angle[0], energy))
        return energy

    # Measure the energy expectation value in the specified basis given the counts distribution from a circuit execution in Qiskit
    def measure_expectation(self, basis, p_dict):
        expectation = 0
        for state,counts in p_dict.items():
            expectation += qubit_2_eigenvalues[basis][state]*counts/self.shots
        return expectation

    def create_circuits(self):
        # Create the required circuits
        for i in range(2):
            self.qc.append(QuantumCircuit(2))
            # Use the 1st ansatz. Can also use the 2nd one instead with add_ansatz2()
            self.qc[i].append(self.add_ansatz1(), [0,1])
            # self.qc[i] = self.qc[i].bind_parameters({theta: 0})

        # Want to measure in the Z basis. Hence need to rotate so that the eigenvectors of the XX operator rotate to the eigenvectors of the ZZ operator. Refer to report for more details.
        self.qc[1].cx(0,1)
        self.qc[1].h(0)

        # Add the measurements
        for i in self.qc:
            i.measure_all()
            # Draw the circuit with each gate explicitly
            print(i.decompose().draw())

task4 = VQE_2_qubit()
task4.create_circuits()
# Could just search through all the possible values of the variational parameter instead of using the optimiser
# for angle in np.linspace(0,2*np.pi,100):
#     measure_H_expectation([angle])

# Define tolerance and use the optimiser
tol = 1e-3
optimised_result = minimize(task4.measure_H_expectation, 0, method="powell", tol=tol)
print("Number of iterations taken = ",len(task4.energy_values))
print("Least eigenvalue found = ", optimised_result.fun,"At theta = ",optimised_result.x[0])

# Plot the value of energy obtained against the variational parameter
fig = plt.figure()
plt.scatter(task4.angle_values, task4.energy_values)
plt.show()
# print(angle_values)
# print(energy_values)

