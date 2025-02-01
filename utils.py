import numpy as np

def running_average(x, N):
    return np.convolve(x, np.ones(N) / N, mode='valid')
