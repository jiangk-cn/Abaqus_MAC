import numpy as np

def current(x):
    return x

def benchmark(x):
    return -3.195 * x + 4.195 * x ** 3 * (10/3 -10/3 * x + x ** 2)

inter = np.linspace(0, 1, 100)
y_current = current(inter).T
y_benchmark = benchmark(inter).T

# 输入：当前模态形状和基准模态形状
# MAC_MATRIX - 计算给定基准的MAC矩阵
def mac(Phi1, Phi2):
    # 该函数计算phi1和phi2之间的MAC
    mAc = (abs(np.dot(Phi1.T, Phi2))) ** 2 / ((np.dot(Phi1.T, Phi1)) * (np.dot(Phi2.T, Phi2)))
    return mAc

mac_matrix = mac(y_current, y_benchmark)
