import numpy as np
from multiprocessing import Pool
import MatrixProductFunctions as mp
import time

def process_single_tensor(args):
    mpo_i, mps_i = args
    result = np.einsum("ijnl, mno -> imjlo", mpo_i, mps_i)
    return result.reshape(mpo_i.shape[0] * mps_i.shape[0], 2, mpo_i.shape[-1] * mps_i.shape[-1])

def mpo_x_mps_parallel(mpo, mps, num_workers=2):
    with Pool(num_workers) as pool:
        return pool.map(process_single_tensor, zip(mpo, mps))

if __name__ == '__main__':
    random_vector = np.random.randn(2**18)
    start = time.time()
    mpo_x_mps_parallel(mp.der_mpo(18), mp.vec_to_mps(random_vector))
    print(time.time()-start)