import numpy as np

class MatrixProduct():
    def __init__(self, vector):
        self.mps = self.transform_vec_to_mps(vector)

    def transform_vec_to_mps(self, vector, eps = 1e-6):
        psi_vec = vector
        n = int(np.log2(len(psi_vec)))
        tensors = []
        shape = (2, 2 ** (n - 1))
        psi_matrix = psi_vec.reshape(shape)

        for i in range(n - 1):
            psi_svd = np.linalg.svd(psi_matrix, full_matrices=False)
            bond = (psi_svd.S / psi_svd.S.sum() > eps).sum()

            psi_svd_U = psi_svd.U.reshape(int(psi_svd.U.shape[0] / 2), 2, int(psi_svd.U.shape[1]))[:, :, :bond]
            tensors.append(psi_svd_U)

            pmatrix = np.diag(psi_svd.S[:bond]) @ psi_svd.Vh[:bond, :]
            psi_matrix = pmatrix.reshape(int(pmatrix.shape[0] * 2), int(pmatrix.shape[1] / 2))

        last_tensor = pmatrix.reshape(int(pmatrix.shape[0]), 2, int(pmatrix.shape[1] / 2))
        tensors.append(last_tensor)
        return tensors
    print("Hello!")

a = MatrixProduct(np.array((1, 2, 4, 5, 6, 3, 8, 5)))
print(a)
print(a.mps)
