import numpy as np
import time
import matplotlib.pyplot as plt

from Cuda import cuda_func
from MultiProcessing import mp_shmm_func
from Jit import njit_func
from Parallel import parallel_func

if __name__ == '__main__':

    shape = (20000, 20000)
    dtype = np.bool8
    dtype_nbytes = np.ndarray(1, dtype).nbytes
    low, high = 0, 2

    mat = np.random.randint(low, high, shape, dtype)
    out_mat = np.ndarray(mat.shape, mat.dtype)

    outs = []
    data = {}

    for func in (cuda_func, mp_shmm_func, njit_func, parallel_func):
        _mat, _out_mat = mat.copy(), out_mat.copy()
        print('\n')
        print(func.__name__)
        time_data = []
        t = time.time()
        for _ in range(10):
            func(_mat, _out_mat)
            tmp = time.time() - t
            print(tmp)
            time_data.append(tmp)
        
        data[func.__name__] = time_data
        outs.append(_out_mat)

    print('\n\n')
    for i in range(len(outs) - 1):
        for j in range(i + 1, len(outs)):
            print((i, j), np.array_equiv(outs[i], outs[j]))

    for k, v in data.items():
        plt.plot(v, label = k)
    plt.legend()
    plt.show()