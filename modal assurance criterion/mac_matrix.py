# This code computes the MAC between the current function and the benchmark function.
# It generates a 100-point grid between 0 and 1 and evaluates the current function 
# and the benchmark function on the grid. It then computes the MAC between the two 
# functions. 

import numpy as np

# Define the current function
def current(x):
    return x

# Define the benchmark function
def benchmark(x):
    return -3.195 * x + 4.195 * x ** 3 * (10/3 -10/3 * x + x ** 2)

# Generate a 100-point grid between 0 and 1
inter = np.linspace(0, 1, 100)

# Evaluate the current function on the grid
y_current = current(inter).T

# Evaluate the benchmark function on the grid
y_benchmark = benchmark(inter).T

# Define the function to compute the MAC
def mac(Phi1, Phi2):
    mAc = (abs(np.dot(Phi1.T, Phi2))) ** 2 / ((np.dot(Phi1.T, Phi1)) * (np.dot(Phi2.T, Phi2)))
    return mAc

# Compute the MAC
mac_matrix = mac(y_current, y_benchmark)
