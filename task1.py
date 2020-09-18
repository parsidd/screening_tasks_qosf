from qiskit import BasicAer, execute, QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram, plot_bloch_multivector

from qiskit.aqua.components.optimizers import COBYLA

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations

backend = BasicAer.get_backend("statevector_simulator")

N = 4
MAX_DEPTH = 10
params = []

complete_circuit = QuantumCircuit(N)
phi = random_statevector(2**N)
tol = 1e-3

def even_block(theta):
    eb = QuantumCircuit(N)
    for i in range(N):
        eb.rz(theta[i], i)
    for i in combinations(range(N), 2):
        eb.cz(i[0], i[1])
    return eb

def odd_block(theta):
    ob = QuantumCircuit(N)
    for i in range(N):
        ob.rx(theta[i], i)
    return ob

def evaluate_cost(angles):
    params_dict = {}
    for i, angle in enumerate(angles):
        params_dict[params[i]] = angle
    final_state = execute(complete_circuit.bind_parameters(params_dict), backend = backend).result().get_statevector()
    print(np.linalg.norm((Statevector(final_state)-phi).data))
    return np.linalg.norm((Statevector(final_state)-phi).data)

optimizer = COBYLA(maxiter=500, tol=tol)

for L in range(1, MAX_DEPTH+1):
    odd_params = [Parameter("theta_"+str(2*L-1)+"_"+str(i)) for i in range(N)]
    params += odd_params
    even_params = [Parameter("theta_"+str(2*L)+"_"+str(i)) for i in range(N)]
    params += even_params
    complete_circuit.append(odd_block(odd_params), range(N))
    complete_circuit.barrier()
    complete_circuit.append(even_block(even_params), range(N))
    complete_circuit.barrier()
    # print(complete_circuit.decompose().draw())
    # print(len(params))
    results = optimizer.optimize(len(params), objective_function=evaluate_cost, initial_point=np.random.random(len(params)))
    # print("Phi = ", phi.data)
    # print(results)
    print("Number of layers = ", L, "Minimal cost = ",results[1], "Number of iterations = ", results[2])

# angles = np.random.random(N)
# print(np.linalg.norm(evaluate_cost(angles)))