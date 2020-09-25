# Author: Parth S. Shah

# Import the necessary libraries
from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Pauli

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d

# for command line arguments
import sys

# Use the QASM simulator of qiskit
backend = BasicAer.get_backend("qasm_simulator")

# This is a dictionary that stores the eigenvectors(in ZZ basis) and the 
# corresponding eigenvalues for the Pauli decomposition of the given operator. 
# Refer to report for more details.
qubit_2_eigenvalues = {"Z":{"00":1,"01":-1,"10":-1,"11":1},
                       "X":{"00":1,"01":-1,"10":1,"11":-1},
                       "Y":{"00":-1,"01":1,"10":1,"11":-1}}


class Task4:
    def __init__(self, theta = Parameter("theta"), phi = Parameter("phi"), \
        shots = 2048):
        self.energy_values = []
        self.angle_values = []
        self.qc = []
        self.theta = theta
        self.phi = phi
        self.shots = shots

    # Ansatz with 2 variational parameters
    def add_ansatz3(self):
        ansatz = QuantumCircuit(2)
        ansatz.ry(self.theta, 0)
        ansatz.cx(0, 1)
        ansatz.rx(self.phi, 0)
        ansatz.draw("mpl")
        plt.show()
        ansatz = ansatz.to_gate()
        ansatz.label = "ANSATZ1(theta)"
        return ansatz

    # Function to create the entire circuits required for the given operator
    def create_circuits(self):
        # Create the required circuits
        for i in range(2):
            self.qc.append(QuantumCircuit(2))
            self.qc[i].append(self.add_ansatz3(), [0,1])

        # Want to measure in the Z basis. Hence need to rotate so that the 
        # eigenvectors of the XX operator rotate to the eigenvectors of the ZZ 
        # operator. Refer to report for more details.
        self.qc[1].cx(0,1)
        self.qc[1].h(0)

        # Add the measurements
        for i in self.qc:
            i.measure_all()
            # Draw the circuit with each gate explicitly
            print(i.decompose().draw())

    # Measure the energy expectation value of the given operator given a 
    # variational parameter
    def measure_H_expectation(self, angle):
        energy = 0
        results_z = execute(self.qc[0].bind_parameters({self.theta: angle[0],\
            self.phi: angle[1]}), backend = backend, shots = self.shots).result().get_counts()
        results_xy = execute(self.qc[1].bind_parameters({self.theta: angle[0],\
            self.phi: angle[1]}), backend = backend, shots = self.shots).result().get_counts()
        # Find expectation of each Pauli operator
        z_exp = self.measure_expectation("Z", results_z)
        x_exp = self.measure_expectation("X", results_xy)
        y_exp = self.measure_expectation("Y", results_xy)
        # Energy in terms of expectation value in each basis
        energy = (z_exp + 1 - x_exp - y_exp)/2
        self.angle_values.append(angle)
        self.energy_values.append(energy)
        # print("{} {}".format(angle[0], energy))
        return energy

    # Measure the energy expectation value in the specified basis given the 
    # counts distribution from a circuit execution in Qiskit
    def measure_expectation(self, basis, p_dict):
        expectation = 0
        for state,counts in p_dict.items():
            # counts/shots gives the probability of each measurement
            # Multiply by eigenvalue and sum over all measurements
            expectation += qubit_2_eigenvalues[basis][state]*counts/self.shots
        return expectation
    
    # Function for linear search throughout the range of theta
    def use_search(self, N = 25):
        for theta in np.linspace(0, 2*np.pi, N):
            for phi in np.linspace(0, 2*np.pi, N):
                self.measure_H_expectation([theta, phi])
        min_eigen =  min(task4.energy_values)
        min_angle = task4.angle_values[task4.energy_values.index(min_eigen)]
        print("Minimum eigenvalue found  = ",min_eigen," at theta = ",min_angle)
    
    # Function to use the optimiser for going through the search space
    def use_optimiser(self, tol = 1e-3):
        optimised_result = minimize(self.measure_H_expectation, [0,0],\
            method="powell", tol=tol)
        print("Number of iterations taken = ",len(self.energy_values))
        print("Least eigenvalue found = ", optimised_result.fun,"At theta = ",\
            optimised_result.x)

# If called directly. Just so that the file can be called externally as well.
if __name__ == "__main__":
    task4 = Task4()

    # Create the circuits
    task4.create_circuits()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    task4.use_search()
    x = np.array(task4.angle_values)[:,0]
    y = np.array(task4.angle_values)[:,1]
    z = task4.energy_values

    ax.plot_trisurf(x, y, z, cmap="jet")
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Energy')
    ax.set_title("Energy vs. Theta and Phi")
    plt.show()

    task4.energy_values = []
    task4.angle_values = []  

    task4.use_optimiser()

    # print(angle_values)
    # print(energy_values)

