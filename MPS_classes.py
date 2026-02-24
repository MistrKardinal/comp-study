import numpy as np

class MatrixProductState():
    
    def __init__(self, vector, eps=1e-6, bond_limit=1000):
        if isinstance(vector, np.ndarray):
            self.mps = self.transform_vec_to_mps(vector, eps, bond_limit)
            self.length = np.log2(len(vector))
        else:
            self.mps = vector
            self.length = len(vector)
            
        self.norm = self.__abs__()
        self.bond = self.get_bond()

    def transform_vec_to_mps(self, vector, eps, bond_limit):
        psi_vec = np.asarray(vector)
        n = int(np.log2(len(psi_vec)))
        if 2**n != len(psi_vec):
            raise ValueError("Длина вектора должна быть степенью 2.")
            
        mps = []
        psi_matrix = psi_vec.reshape((2,-1))

        for i in range(n - 1):
            psi_svd = np.linalg.svd(psi_matrix, full_matrices=False)
            bond = min((psi_svd.S / psi_svd.S.sum() > eps).sum(), bond_limit)

            psi_svd_U = psi_svd.U.reshape(int(psi_svd.U.shape[0] / 2), 2, int(psi_svd.U.shape[1]))[:, :, :bond]
            mps.append(psi_svd_U)

            pmatrix = np.diag(psi_svd.S[:bond]) @ psi_svd.Vh[:bond, :]
            psi_matrix = pmatrix.reshape(int(pmatrix.shape[0] * 2), int(pmatrix.shape[1] / 2))

        last_tensor = pmatrix.reshape(int(pmatrix.shape[0]), 2, int(pmatrix.shape[1] / 2))
        mps.append(last_tensor)
        return mps
    
    def turn_to_lc_form(self):
        if not isinstance(self.mps, list):
            raise TypeError("Аргументы должны быть списками")

        mps_copy = self.mps.copy()
        left_canonical_mps = [] # Создаём лист для хранения тензоров
        temporary_element = [mps_copy[0]]
        for i in range(len(mps_copy)-1):
            sh = temporary_element[0].shape
            element = temporary_element[0].reshape(sh[0] * 2, sh[2]) # Переводим тензор в матрицу
            element_q, element_r = np.linalg.qr(element) # Делаем сингулярное разложение матрицы
            tensor = element_q.reshape(sh[0], 2, -1) # Переводим матрицу Vh в тензор
            left_canonical_mps.append(tensor) # Вносим тензор в лист хранения тензоров   
            newpmatrx = element_r
            pmatrix = np.tensordot(newpmatrx, mps_copy[i+1], axes = 1) # Перемножаем более левый тензор с перемноженными матрицами U и S
            temporary_element = [pmatrix]
        left_canonical_mps.append(pmatrix) # Добавляем последний тензор в лист хранения тензоров
        return left_canonical_mps
    
    def __abs__(self):
        left_canonical_mps = self.turn_to_lc_form()
        return np.sqrt(np.trace(np.conj(left_canonical_mps[-1])[:, 0, :].T @ left_canonical_mps[-1][:, 0, :]) + np.trace(np.conj(left_canonical_mps[-1])[:, 1, :].T @ left_canonical_mps[-1][:, 1, :]))

    def get_bond(self):
        dim1 = []
        dim2 = []
        for i in self.mps:
            dim1.append(i.shape[0])
            dim2.append(i.shape[-1])
        return max(max(dim1), max(dim2))
    
    def __add__(self, other):

        if not isinstance(other, MatrixProductState):
            raise TypeError("Операнд должен быть MatrixProductState")
        if len(self.mps) != len(other.mps):
            raise ValueError("Длины двух MPS должны быть одинаковы")

        sum_mps = []
        sum_mps.append(np.concatenate([self.mps[0], other.mps[0]], axis = 2).reshape(1, 2, -1))
        for i in range(1, len(self.mps)-1):
            zero_arr1 = np.zeros((other.mps[i].shape[0], 2, self.mps[i].shape[2]))
            zero_arr2 = np.zeros((self.mps[i].shape[0], 2, other.mps[i].shape[2]))
            ff1 = np.concatenate([self.mps[i], zero_arr1], axis=0).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, self.mps[i].shape[2])
            ff2 = np.concatenate([zero_arr2, other.mps[i]], axis=0).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, other.mps[i].shape[2])
            new_element = np.concatenate([ff1, ff2], axis = 2).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, other.mps[i].shape[2] + self.mps[i].shape[2])
            sum_mps.append(new_element)
        sum_mps.append(np.concatenate([self.mps[-1], other.mps[-1]], axis = 0).reshape(self.mps[-1].shape[0] + other.mps[-1].shape[0], 2, 1))
        return MatrixProductState(sum_mps)
    
    def __sub__(self, other):

        if not isinstance(other, MatrixProductState):
            raise TypeError("Операнд должен быть MatrixProductState")
        if len(self.mps) != len(other.mps):
            raise ValueError("Длины двух MPS должны быть одинаковы")

        sub_mps = []
        sub_mps.append(np.concatenate([self.mps[0], -other.mps[0]], axis = 2).reshape(1, 2, -1))
        for i in range(1, len(self.mps)-1):
            zero_arr1 = np.zeros((other.mps[i].shape[0], 2, self.mps[i].shape[2]))
            zero_arr2 = np.zeros((self.mps[i].shape[0], 2, other.mps[i].shape[2]))
            ff1 = np.concatenate([self.mps[i], zero_arr1], axis=0).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, self.mps[i].shape[2])
            ff2 = np.concatenate([zero_arr2, other.mps[i]], axis=0).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, other.mps[i].shape[2])
            new_element = np.concatenate([ff1, ff2], axis = 2).reshape(other.mps[i].shape[0] + self.mps[i].shape[0], 2, other.mps[i].shape[2] + self.mps[i].shape[2])
            sub_mps.append(new_element)
        sub_mps.append(np.concatenate([self.mps[-1], other.mps[-1]], axis = 0).reshape(self.mps[-1].shape[0] + other.mps[-1].shape[0], 2, 1))
        return MatrixProductState(sub_mps)
    
    def __mul__(self, number):
        new_mps = self.mps.copy()
        new_mps[0] = new_mps[0]*number
        return MatrixProductState(new_mps)
    
    def __truediv__(self, number):
        new_mps = self.mps.copy()
        new_mps[0] = new_mps[0]/number
        return MatrixProductState(new_mps)
    
    def __matmul__(self, other):
        
        if self.length != other.length:
            raise ValueError("Длины двух mps должны быть одинаковы")
        
        if isinstance(other, MatrixProductState):   
            mps1 = self.mps
            mps2 = other.mps
            sh1 = mps1[0].shape
            sh2 = mps2[0].shape
            result = np.einsum('ijk,mjl->imkl', np.conj(mps1[0]), mps2[0]).reshape(sh1[0] * sh2[0], sh1[2] * sh2[2])
            for i in range(1, len(mps1)):
                sh1 = mps1[i].shape
                sh2 = mps2[i].shape
                result = result @ np.einsum('ijk,mjl->imkl', np.conj(mps1[i]), mps2[i]).reshape(sh1[0]*sh2[0], sh1[2]*sh2[2])
            result = result.reshape(-1)[0]
            return result
        
        elif isinstance(other, MatrixProductOperator):   
            new_mps = []
            for i in range(int(self.length)):
                new_mps.append(np.einsum("ijnl, mjo -> imnlo", other.mpo[i], self.mps[i]).reshape(other.mpo[i].shape[0] * self.mps[i].shape[0], 2, other.mpo[i].shape[-1] * self.mps[i].shape[-1]))
            return MatrixProductState(new_mps)
        
        else:
            raise TypeError("Неправильное использование аргументов")
    
    def __str__(self):
        return (f"MPS(n_tensors={self.length}, "
                f"max_bond={self.bond})")
    
    def __len__(self):
        return len(self.mps)
    
    def __neg__(self):
        neg_mps = self.mps.copy()
        neg_mps[0] = -neg_mps[0]
        return MatrixProductState(neg_mps)
    
class MatrixProductOperator():
    def __init__(self, list_data):
        if not isinstance(list_data, list):
            raise TypeError("Некорректная инициализация MPO")
        else:
            self.mpo = list_data
            
        self.length = len(list_data)
        self.bond = self.get_bond()
        
    def get_bond(self):
        dim1 = []
        dim2 = []
        for i in self.mpo:
            dim1.append(i.shape[0])
            dim2.append(i.shape[-1])
        return max(max(dim1), max(dim2))
    
    def __matmul__(self, other):
        if self.length != other.length:
            raise ValueError("Длины объектов должны быть одинаковыми")
        
        if isinstance(other, MatrixProductOperator):
            mult_mpo = []
            for i in range(self.length):
                sh1 = self.mpo[i].shape
                sh2 = other.mpo[i].shape
                mult_mpo.append(np.einsum('ijkl, mkop -> imjolp', self.mpo[i], other.mpo[i]).reshape(sh1[0]*sh2[0], 2, 2, sh1[3]*sh2[3]))
            return MatrixProductOperator(mult_mpo)
        
        elif isinstance(other, MatrixProductState):
            new_mps = []
            for i in range(self.length):
                new_mps.append(np.einsum("ijnl, mno -> imjlo", self.mpo[i], other.mps[i]).reshape(self.mpo[i].shape[0] * other.mps[i].shape[0], 2, self.mpo[i].shape[-1] * other.mps[i].shape[-1]))
            return MatrixProductState(new_mps)
        
        else:
            raise TypeError("Неправильное использование аргументов")
            
    def __add__(self, other):
        common_tensor = []
        common_tensor.append(np.concatenate([self.mpo[0], other.mpo[0]], axis = 3).reshape(1, 2, 2, -1))
        for i in range(1, self.length-1):
            zero_arr1 = np.zeros((other.mpo[i].shape[0], 2, 2, self.mpo[i].shape[3]))
            zero_arr2 = np.zeros((self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3]))
            ff1 = np.concatenate([self.mpo[i], zero_arr1], axis=0).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, self.mpo[i].shape[3])
            ff2 = np.concatenate([zero_arr2, other.mpo[i]], axis=0).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3])
            new_element = np.concatenate([ff1, ff2], axis = 3).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3] + self.mpo[i].shape[3])
            common_tensor.append(new_element)
        common_tensor.append(np.concatenate([self.mpo[-1], other.mpo[-1]], axis = 0).reshape(self.mpo[-1].shape[0] + other.mpo[-1].shape[0], 2, 2, 1))
        return MatrixProductOperator(common_tensor)
    
    def __sub__(self, other):
        common_tensor = []
        common_tensor.append(np.concatenate([self.mpo[0], -other.mpo[0]], axis = 3).reshape(1, 2, 2, -1))
        for i in range(1, self.length-1):
            zero_arr1 = np.zeros((other.mpo[i].shape[0], 2, 2, self.mpo[i].shape[3]))
            zero_arr2 = np.zeros((self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3]))
            ff1 = np.concatenate([self.mpo[i], zero_arr1], axis=0).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, self.mpo[i].shape[3])
            ff2 = np.concatenate([zero_arr2, other.mpo[i]], axis=0).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3])
            new_element = np.concatenate([ff1, ff2], axis = 3).reshape(other.mpo[i].shape[0] + self.mpo[i].shape[0], 2, 2, other.mpo[i].shape[3] + self.mpo[i].shape[3])
            common_tensor.append(new_element)
        common_tensor.append(np.concatenate([self.mpo[-1], other.mpo[-1]], axis = 0).reshape(self.mpo[-1].shape[0] + other.mpo[-1].shape[0], 2, 2, 1))
        return MatrixProductOperator(common_tensor)
    
    def __mul__(self, number):
        new_mpo = self.mpo.copy()
        new_mpo[0] = new_mpo[0]*number
        return MatrixProductState(new_mpo)
    
    def __truediv__(self, number):
        new_mpo = self.mpo.copy()
        new_mpo[0] = new_mpo[0]/number
        return MatrixProductOperator(new_mpo)
    
#class MPSEulerSolver():
#    def __init__(self)