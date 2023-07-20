from typing import List
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import cpu_count
import numba
import time

@numba.njit
def _sub_process_func(sm_mat: np.ndarray, out_sm_mat: np.ndarray, idx: int, cpu_num: int):
    for i in range(idx, sm_mat.shape[0], cpu_num):
        for j in range(sm_mat.shape[1]):
            neibours = sm_mat[max(i-1, 0):min(i+2, sm_mat.shape[0]), max(j-1, 0):min(j+2, sm_mat.shape[1])].sum() - sm_mat[i, j]
            if neibours < 2 or neibours > 3: out_sm_mat[i, j] = 0
            elif neibours == 3: out_sm_mat[i, j] = 1
            else: out_sm_mat[i, j] = sm_mat[i, j]

def _process_func(name: str, out_name: str, idx: int, shape, dtype, cpu_num: int):
    sm = SharedMemory(name)
    out_sm = SharedMemory(out_name)
    sm_mat = np.ndarray(shape, dtype, sm.buf)
    out_sm_mat = np.ndarray(shape, dtype, out_sm.buf)

    _sub_process_func(sm_mat, out_sm_mat, idx, cpu_num)

def mp_shmm_func(mat: np.ndarray, out_mat: np.ndarray):
    smm = SharedMemoryManager()
    smm.start()

    sm = smm.SharedMemory(mat.nbytes)
    sm_mat = np.ndarray(mat.shape, mat.dtype, sm.buf)
    sm_mat[:] = mat
    out_sm = smm.SharedMemory(out_mat.nbytes)
    out_sm_mat = np.ndarray(out_mat.shape, out_mat.dtype, out_sm.buf)

    ps: List[Process] = []
    for idx in range(cpu_count()):
        ps.append(Process(target=_process_func, args=(sm.name, out_sm.name, idx, mat.shape, mat.dtype, cpu_count())))
        ps[-1].start()
    for p in ps: p.join()

    out_mat[:] = out_sm_mat

    smm.shutdown()

if __name__ == '__main__':

    shape = (20000, 20000)
    dtype = np.bool8
    dtype_nbytes = np.ndarray(1, dtype).nbytes
    low, high = 0, 2

    mat = np.random.randint(low, high, shape, dtype)
    out_mat = np.ndarray(mat.shape, mat.dtype)

    _mat, _out_mat = mat.copy(), out_mat.copy()
    t = time.time()
    mp_shmm_func(_mat, _out_mat)
    print(time.time() - t)