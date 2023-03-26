import numpy as np

def current(x):
    return x

def benchmark(x):
    return -3.195 * x + 4.195 * x ** 3 * (10/3 -10/3 * x + x ** 2)

inter = np.linspace(0, 1, 100)
y_current = current(inter).T
y_benchmark = benchmark(inter).T

def mac(Phi1, Phi2):
    """
    > The function takes two matrices as input and returns the matrix of cosine similarities between the
    columns of the two matrices
    
    :param Phi1: The first feature vector
    :param Phi2: The feature matrix of the test data
    :return: the value of the MAC.
    """
    mAc = (abs(np.dot(Phi1.T, Phi2))) ** 2 / ((np.dot(Phi1.T, Phi1)) * (np.dot(Phi2.T, Phi2)))
    return mAc

mac_matrix = mac(y_current, y_benchmark)
