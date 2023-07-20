import numba
import numpy as np
import time

@numba.njit(parallel = True)
def parallel_func(mat: np.ndarray, out_mat: np.ndarray):
    for i in numba.prange(mat.shape[0]):
        for j in range(mat.shape[1]):
            neibours = mat[max(i-1, 0):min(i+2, mat.shape[0]), max(j-1, 0):min(j+2, mat.shape[1])].sum() - mat[i, j]
            if neibours < 2 or neibours > 3: out_mat[i, j] = 0
            elif neibours == 3: out_mat[i, j] = 1
            else: out_mat[i, j] = mat[i, j]

if __name__ == '__main__':

    shape = (20000, 20000)
    dtype = np.bool8
    dtype_nbytes = np.ndarray(1, dtype).nbytes
    low, high = 0, 2

    mat = np.random.randint(low, high, shape, dtype)
    out_mat = np.ndarray(mat.shape, mat.dtype)

    _mat, _out_mat = mat.copy(), out_mat.copy()
    t = time.time()
    parallel_func(_mat, _out_mat)
    print(time.time() - t)