import numpy as np
import numba
import numba.cuda
import time
import math

@numba.cuda.jit
def _cuda_func(mat: np.ndarray, out_mat: np.ndarray):
    _i, _j = numba.cuda.grid(2)
    ti, tj = numba.cuda.gridsize(2)
    for i in range(_i, mat.shape[0], ti):
        for j in range(_j, mat.shape[1], tj):
            neibours = 0
            for ii in range(max(0, i-1), min(i+2, mat.shape[0])):
                for jj in range(max(j-1, 0), min(j+2, mat.shape[1])):
                    if not (ii == i and jj == j): neibours += mat[ii, jj]
            if neibours < 2 or neibours > 3: out_mat[i, j] = 0
            elif neibours == 3: out_mat[i, j] = 1
            else: out_mat[i, j] = mat[i, j]

def cuda_func(mat: np.ndarray, out_mat: np.ndarray):
    device_mat = numba.cuda.to_device(mat)
    device_out_mat = numba.cuda.device_array_like(out_mat)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(device_mat.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(device_mat.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _cuda_func[blockspergrid, threadsperblock](device_mat, device_out_mat)
    numba.cuda.synchronize()

    device_out_mat.copy_to_host(out_mat)

if __name__ == '__main__':


    shape = (20000, 20000)
    dtype = np.bool8
    dtype_nbytes = np.ndarray(1, dtype).nbytes
    low, high = 0, 2

    mat = np.random.randint(low, high, shape, dtype)
    out_mat = np.ndarray(mat.shape, mat.dtype)

    _mat, _out_mat = mat.copy(), out_mat.copy()
    t = time.time()
    cuda_func(_mat, _out_mat)
    print(time.time() - t)

    print(_mat)
    print(_out_mat)