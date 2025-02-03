# Quantum-Geometry
Program to calculate quantum geometry properties of parametrized quantum circuits. Determines number of redundant parameters, effective quantum dimension and quantum natural gradient. Prunes redundant parameters from the circuit.

Companion code for http://arxiv.org/abs/2102.01659, https://doi.org/10.1103/PRXQuantum.2.040309
"Capacity and quantum geometry of parametrized quantum circuits" by T. Haug, K. Bharti, M.S. Kim, PRX QUANTUM 2, 040309 (2021)


Requirements:

Numpy

Scipy

Qutip (http://qutip.org/)
NOTE: Requires older version of qutip, namely <=4.7.5, install via "pip install qutip==4.7.5"

Additional Julia Code based on Yao to calculate effective dimension and gradients provided in QuantumCircuitCapacity.jl.
https://yaoquantum.org/


@author: Tobias Haug, Imperial College London
