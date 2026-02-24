import numpy as np
from typing import List
from numba import njit
from tqdm import tqdm
from operator import add
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
import pickle
import MPS_classes


def zero_filter(num):
    if num > 1e-6:
        return True
    else:
        return False    
    
def vec_to_mps(psi_vec: np.ndarray, eps: float = 1e-6) -> List[np.ndarray]:
    if not isinstance(psi_vec, np.ndarray):
        raise TypeError("Аргумент должен быть массивом NumPy (np.array)!")
    if not np.log2(len(psi_vec)).is_integer():
        raise ValueError("Длина входного вектора должна быть степенью двойки")
    tensors = []
    
    n = int(np.log2(len(psi_vec)))
    shape = (2, 2**(n-1))
    psi_matrix = psi_vec.reshape(shape)

    for _ in range(n-1):
        psi_svd = np.linalg.svd(psi_matrix, full_matrices = False)
        bond = (psi_svd.S / psi_svd.S.sum() > eps).sum()

        psi_svd_U = psi_svd.U.reshape(int(psi_svd.U.shape[0]/2), 2, int(psi_svd.U.shape[1]))[:, :, :bond]
        tensors.append(psi_svd_U)

        pmatrix = np.diag(psi_svd.S[:bond]) @ psi_svd.Vh[:bond, :]
        psi_matrix = pmatrix.reshape(int(pmatrix.shape[0]*2), int(pmatrix.shape[1]/2))

    last_tensor = pmatrix.reshape(int(pmatrix.shape[0]), 2, -1)
    tensors.append(last_tensor)
    return tensors

def identity(num1: np.ndarray, num2: np.ndarray) -> int:
    return num1 @ num2 / (np.linalg.norm(num1) * np.linalg.norm(num2))

def mps_sum(psi_one: List[np.ndarray], psi_two: List[np.ndarray]) -> List[np.ndarray]:

    if not isinstance(psi_one, list) or not isinstance(psi_two, list):
        raise TypeError("Аргументы должны быть списками")
    if len(psi_one) != len(psi_two):
        raise ValueError("Длины двух mps должны быть одинаковы")

    sum_mps = []
    sum_mps.append(np.concatenate([psi_one[0], psi_two[0]], axis = 2).reshape(1, 2, -1))
    for i in range(1, len(psi_one)-1):
        zero_arr1 = np.zeros((psi_two[i].shape[0], 2, psi_one[i].shape[2]))
        zero_arr2 = np.zeros((psi_one[i].shape[0], 2, psi_two[i].shape[2]))
        ff1 = np.concatenate([psi_one[i], zero_arr1], axis=0).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_one[i].shape[2])
        ff2 = np.concatenate([zero_arr2, psi_two[i]], axis=0).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_two[i].shape[2])
        new_element = np.concatenate([ff1, ff2], axis = 2).reshape(psi_two[i].shape[0] + psi_one[i].shape[0], 2, psi_two[i].shape[2] + psi_one[i].shape[2])
        sum_mps.append(new_element)
    sum_mps.append(np.concatenate([psi_one[-1], psi_two[-1]], axis = 0).reshape(psi_one[-1].shape[0] + psi_two[-1].shape[0], 2, 1))
    return sum_mps


# Функция переводит mps в левую каноническую форму и сжимает его, переводя в правую каноническую форму
def mps_compression(mps, eps = 1e-6, bond_limit = 1000):

    if not isinstance(mps, list):
        raise TypeError("Аргумент должен быть списком")

    compressed_mps = []

    mps = left_canonical_MPS(mps)
    temp_tensor = [mps[-1]]
    for i in range(len(mps)-1):
        shapi = temp_tensor[0].shape
        element = temp_tensor[-1].reshape(shapi[0], 2 * shapi[2]) # Переводим тензор в матрицу
        element_svd = np.linalg.svd(element, full_matrices=False) # Делаем сингулярное разложение матрицы
        b_tensor = element_svd.Vh.reshape(element_svd.Vh.shape[0], 2, shapi[2]) # Переводим матрицу Vh в тензор
        bond = min((element_svd.S / element_svd.S.sum() > eps).sum(), bond_limit)
        b_tensor = b_tensor[:bond, :, :] # Обрезаем тензор
        compressed_mps.append(b_tensor) # Вносим тензор в лист хранения тензоров
        element_svd_S = element_svd.S[:bond] # Обрезаем матрицу S
        newpmatrx = element_svd.U[:,:bond] @ np.diag(element_svd_S) # Обрезаем матрицу U и умножаем на матрицу S
        pmatrix = np.tensordot(mps[-2-i], newpmatrx,axes = 1) # Перемножаем более левый тензор с перемноженными матрицами U и S
        temp_tensor = [pmatrix]
    compressed_mps.append(pmatrix) # Добавляем последний тензор в лист хранения тензоров
    compressed_mps[::] = compressed_mps[::-1] # Инвертируем порядок хранения листа хранения
    return compressed_mps

def QR_decomposition(psi):

    if not isinstance(psi, np.ndarray):
        raise TypeError("Аргумент должен быть массивом NumPy (np.array)!")
    if not np.log2(len(psi)).is_integer():
        raise ValueError("Длина входного вектора должна быть степенью двойки")

    n = int(np.log2(len(psi)))
    arr_store = []
    shape = (2,2**(n-1))
    psi_matrix = psi.reshape(shape)
    for i in range(0,n-1):
        Q, R = np.linalg.qr(psi_matrix, mode = 'reduced')
        Q_reshaped = Q.reshape(int(Q.shape[0]/2),2,int(Q.shape[1])) # Переводим матрицу Q в тензор
        arr_store.append(Q_reshaped)
        pmatrix = R 
        psi_matrix = pmatrix.reshape(int(pmatrix.shape[0]*2),int(pmatrix.shape[1]/2))    
    Q = pmatrix.reshape(int(pmatrix.shape[0]),2,int(pmatrix.shape[1]/2))
    arr_store.append(Q)
    return arr_store

def QR_rc_decomposition(psi):

    if not isinstance(psi, np.ndarray):
        raise TypeError("Аргумент должен быть массивом NumPy (np.array)!")
    if not np.log2(len(psi)).is_integer():
        raise ValueError("Длина входного вектора должна быть степенью двойки")

    n = int(np.log2(len(psi)))
    arr_store = []
    shape = (2**(n-1), 2)
    psi_matrix = psi.reshape(shape)
    for i in range(0,n-1):
        Q, R = np.linalg.qr(psi_matrix.T, mode = 'reduced')
        Q_reshaped = Q.T.reshape(int(Q.T.shape[0]),2,int(Q.T.shape[1]/2)) # Переводим матрицу Q в тензор
        arr_store.append(Q_reshaped)
        pmatrix = R.T 
        psi_matrix = pmatrix.reshape(int(pmatrix.shape[0]/2),int(pmatrix.shape[1]*2))    
    Q = pmatrix.reshape(int(pmatrix.shape[0]/2),2,int(pmatrix.shape[1]))
    arr_store.append(Q)
    return arr_store[::-1]

def left_canonical_MPS(mps):

    if not isinstance(mps, list):
        raise TypeError("Аргументы должны быть списками")

    mps_copy = mps.copy()
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

def compute_norm(mps):
    mps_for_change = mps.copy()
    mps_for_change = left_canonical_MPS(mps_for_change)
    return calculate_lc_mpsnorm(mps_for_change)

#def convert_mps_to_mixed(mps, first_mps_length, eps = 1e-6):
    singular_values = 0
    tensor = mps.copy()
    final_tensor = []  # Создаём лист для хранения тензоров
    inter_tensor = []
    ten_ten = [tensor[0]]
    for i in range(first_mps_length - 1):
        sh = ten_ten[0].shape
        element = ten_ten[0].reshape(sh[0] * 2, sh[2])  # Переводим тензор в матрицу
        element_svd = np.linalg.svd(element, full_matrices=False)
        bond = (element_svd.S / element_svd.S.sum() > eps).sum()
        svd_U = element_svd.U.reshape(sh[0], 2, -1)[:, :, :bond]  # Переводим матрицу Vh в тензор
        final_tensor.append(svd_U)  # Вносим тензор в лист хранения тензоров
        if i < first_mps_length - 2:
            newpmatrx = np.diag(element_svd.S[:bond]) @ element_svd.Vh[:bond, :]
        if i == first_mps_length - 2:
            newpmatrx = element_svd.Vh[:bond, :]
            singular_values = element_svd.S[:bond]
        pmatrix = np.tensordot(newpmatrx, tensor[i + 1], axes=1)  # Перемножаем более левый тензор с перемноженными матрицами U и S
        ten_ten = [pmatrix]
    tensor[i + 1] = ten_ten[0]
    ten_ten = [tensor[-1]]
    for i in range(len(mps) - first_mps_length):
        sh = ten_ten[0].shape
        element = ten_ten[-1].reshape(sh[0], 2 * sh[2])  # Переводим тензор в матрицу
        element_svd = np.linalg.svd(element, full_matrices=False)  # Делаем сингулярное разложение матрицы
        b_tensor = element_svd.Vh.reshape(element_svd.Vh.shape[0], 2, sh[2])  # Переводим матрицу Vh в тензор
        bond = (element_svd.S / element_svd.S.sum() > eps).sum()
        b_tensor = b_tensor[:bond, :, :]  # Обрезаем тензор
        inter_tensor.append(b_tensor)  # Вносим тензор в лист хранения тензоров
        element_svd_S = element_svd.S[:bond]  # Обрезаем матрицу S
        newpmatrx = element_svd.U[:, :bond] @ np.diag(element_svd_S)  # Обрезаем матрицу U и умножаем на матрицу S
        pmatrix = np.tensordot(absnew_tensor[-2 - i], newpmatrx, axes=1)  # Перемножаем более левый тензор с перемноженными матрицами U и S
        ten_ten = [pmatrix]

    final_tensor.append(pmatrix)  # Добавляем последний тензор в лист хранения тензоров
    return final_tensor, singular_values

def el_by_el_multiplication(mps1, mps2):

    if not isinstance(mps1, list) or not isinstance(mps2, list):
        raise TypeError("Аргументы должны быть списками")
    if len(mps1) != len(mps2):
        raise ValueError("Длины двух mps должны быть одинаковы")

    resulting_tensor = []
    for i in range(len(mps1)):
        sh1 = mps1[i].shape
        sh2 = mps2[i].shape
        resulting_tensor.append(np.einsum('ijk,mjl->imjkl', mps1[i], mps2[i]).reshape(sh1[0]*sh2[0], 2, sh1[2]*sh2[2]))
    return resulting_tensor

def mps_scalar_multiplication(mps_1, mps_2):

    if not isinstance(mps_1, list) or not isinstance(mps_2, list):
        raise TypeError("Аргументы должны быть списками")
    if len(mps_1) != len(mps_2):
        raise ValueError("Длины двух mps должны быть одинаковы")

    mps1 = mps_1
    mps2 = mps_2
    sh1 = mps1[0].shape
    sh2 = mps2[0].shape
    nnn = np.einsum('ijk,mjl->imkl', np.conj(mps1[0]), mps2[0]).reshape(sh1[0] * sh2[0], sh1[2] * sh2[2])
    for i in range(1, len(mps1)):
        sh1 = mps1[i].shape
        sh2 = mps2[i].shape
        nnn = nnn @ np.einsum('ijk,mjl->imkl', np.conj(mps1[i]), mps2[i]).reshape(sh1[0]*sh2[0], sh1[2]*sh2[2])
    nnn = nnn.reshape(-1)[0]
    return nnn

def der_mpo(n, step = 1):
    der_mpo_list = []
    mpo_derivative_first = 0.5 * np.array([1,0,0,0,1,0,0,0,-1,1,0,0]).reshape(1,2,2,3) / step
    mpo_derivative_last = np.array([0,1,-1,0,0,0,1,0,0,1,0,0]).reshape(3,2,2,1)
    mpo_derivative_intermediate = np.array([1,0,0,0,1,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]).reshape(3,2,2,3)
    der_mpo_list.append(mpo_derivative_first)
    for i in range(n-2):
        der_mpo_list.append(mpo_derivative_intermediate)
    der_mpo_list.append(mpo_derivative_last)
    return der_mpo_list

def der2_matrix(n, step = 1, bound = False):
    mpo_derivative2_first = np.array([1,0,0,0,1,0,0,0,1,1,0,0]).reshape(1,2,2,3) / step**2
    mpo_derivative2_last = np.array([-2,1,1,-2,0,0,1,0,0,1,0,0]).reshape(3,2,2,1)
    mpo_derivative2_intermediate = np.array([1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]).reshape(3,2,2,3)
    a_first2 = mpo_derivative2_first
    for i in range(n-2):
        a_first2 = np.tensordot(a_first2, mpo_derivative2_intermediate, axes = 1)
    a_end = np.tensordot(a_first2, mpo_derivative2_last, axes = 1)
    a_end = a_end.reshape(2*np.ones(2*n, dtype = int))
    sh_t = np.arange(0,2*n,2)
    shh_t = np.arange(1,2*n,2)
    sh_t = list(np.concatenate((sh_t, shh_t)))
    a_end = np.transpose(a_end, axes = sh_t)
    a_end = a_end.reshape(2**n,2**n)
    if bound == "periodic":
        a_end[0][-1] = 1/step**2
        a_end[-1][0] = 1/step**2
    return a_end

def der2_mpo(n, step = 1):
    mpo_derivative2_first = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]).reshape(1, 2, 2, 3) / step**2
    mpo_derivative2_last = np.array([-2, 1, 1, -2, 0, 0, 1, 0, 0, 1, 0, 0]).reshape(3, 2, 2, 1)
    mpo_derivative2_intermediate = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0]).reshape(3, 2, 2, 3)
    return [mpo_derivative2_first] + [mpo_derivative2_intermediate] * (n - 2) + [mpo_derivative2_last]

def der_matrix(n, step = 1):
    mpo_derivative_first = 0.5 * np.array([1,0,0,0,1,0,0,0,-1,1,0,0]).reshape(1,2,2,3) / step
    mpo_derivative_last = np.array([0,1,-1,0,0,0,1,0,0,1,0,0]).reshape(3,2,2,1)
    mpo_derivative_intermediate = np.array([1,0,0,0,1,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]).reshape(3,2,2,3)
    a_first = mpo_derivative_first
    for i in range(n-2):
        a_first = np.tensordot(a_first, mpo_derivative_intermediate, axes = 1)
    a_end = np.tensordot(a_first, mpo_derivative_last, axes = 1)
    a_end = a_end.reshape(2*np.ones(2*n, dtype = int))
    sh_t = np.arange(0,2*n,2)
    shh_t = np.arange(1,2*n,2)
    sh_t = list(np.concatenate((sh_t, shh_t)))
    a_end.shape
    a_end = np.transpose(a_end, axes = sh_t)
    a_end = a_end.reshape(2**n,2**n)
    return a_end

def mpo_to_matrix(mpo):
    mul = mpo[0]
    for i in range(1, len(mpo)):
        mul = np.tensordot(mul, mpo[i], axes = 1)
    mul = mul.reshape(2 * np.ones(2 * len(mpo), dtype = int))
    sh_t = np.arange(0, 2 * len(mpo), 2)
    shh_t = np.arange(1, 2 * len(mpo), 2)
    sh_t = list(np.concatenate((sh_t, shh_t)))
    mul = np.transpose(mul, axes = sh_t)
    mul = mul.reshape(2**(len(mpo)), 2**(len(mpo)))
    return mul

def x2_mps(n):
    mps_x2_first = np.array([1,0,0,1,1,1]).reshape(1,2,3)
    mps_x2_last = np.array([0,1,0,4,4,4]).reshape(3,2,1)
    a_first3 = mps_x2_first
    for i in range(1,n-1):
        mps_x2_intermediate = np.array([1,0,0,1,1,1,0,2,0,0,2,4,0,0,4,0,0,4]).reshape(3,2,3)
        a_first3 = np.tensordot(a_first3, mps_x2_intermediate, axes = 1)
    a_end = np.tensordot(a_first3, mps_x2_last, axes = 1)
    a_end = a_end.reshape(-1)
    return a_end

def x_mps(n):
    mps_x_first = np.array([1,0,1,1]).reshape(1,2,2)
    mps_x_last = np.array([0,1,2,2]).reshape(2,2,1)
    mps_x_intermediate = np.array([1,0,1,1,0,2,0,2]).reshape(2,2,2)
    return [mps_x_first] + [mps_x_intermediate] * (n - 2) + [mps_x_last]

def x_vector(n):
    mps_x_first = np.array([1,0,1,1]).reshape(1,2,2)
    mps_x_last = np.array([0,1,2,2]).reshape(2,2,1)
    mps_x_intermediate = np.array([1,0,1,1,0,2,0,2]).reshape(2,2,2)
    a_first4 = mps_x_first
    for i in range(n-2):
        a_first4 = np.tensordot(a_first4, mps_x_intermediate, axes = 1)
    a_end = np.tensordot(a_first4, mps_x_last, axes = 1)
    a_end = a_end.reshape(-1)
    return a_end

def x_mpo(n, x0 = 0, step = 1):
    x_mpo_first = [np.array([1, 0, 0, 0, 0, 0, 1, 1]).reshape(1, 2, 2, 2)]
    x_mpo_intermediate = [np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2]).reshape(2, 2, 2, 2)]
    x_mpo_last = [np.array([x0/step, 0, 0, x0/step + 1, 2, 0, 0, 2]).reshape(2, 2, 2, 1) * step]
    return x_mpo_first + (n - 2) * x_mpo_intermediate + x_mpo_last

def x2_mpo(n, x0 = 0, step = 1):
    return 0

def x_matrix(n, x0 = 0, step = 1):
    x_mpo_first = np.array([1, 0, 0, 0, 0, 0, 1, 1]).reshape(1, 2, 2, 2)
    x_mpo_intermediate = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2]).reshape(2, 2, 2, 2)
    x_mpo_last = np.array([x0/step, 0, 0, x0/step + 1, 2, 0, 0, 2]).reshape(2, 2, 2, 1) * step
    st_first = x_mpo_first
    for i in range(n-2):
        st_first = np.tensordot(st_first, x_mpo_intermediate, axes = 1)
    st_end = np.tensordot(st_first, x_mpo_last, axes = 1)
    st_end = st_end.reshape(2*np.ones(2*n, dtype = int))
    sh_t = np.arange(0,2*n,2)
    shh_t = np.arange(1,2*n,2)
    sh_t = list(np.concatenate((sh_t, shh_t)))
    st_end = np.transpose(st_end, axes = sh_t)
    st_end = st_end.reshape(2**n,2**n)
    return(st_end)

def exp_mps(n, alpha, step):
    exp_mps = []
    for i in range(n):
        exp_mps.append(np.array([1, np.exp(step * alpha * 2**(n-i-1))]).reshape(1,2,1))
    return exp_mps

def exp_mps_smart(n, step, begin, end):
    exp_mps = []
    alpha = (end - begin)/2**n
    for i in range(n):
        exp_mps.append(np.array([1, np.exp(step * alpha * 2**(n + begin - i - 1))]).reshape(1,2,1))
    return exp_mps

def tensors_multiplication(mps):
    if not isinstance(mps, list):
        raise TypeError("Аргумент должен быть списком")

    tensor_second = mps[0]
    for i in range(len(mps)-1):
        tensor_second = np.tensordot(tensor_second, mps[i+1],axes = 1)
    return tensor_second.reshape(-1)

def gauss_mps(n, l, alpha, sigma):
    x_base = np.arange(2**n)
    x_squared = x_base**2
    gauss_state = np.exp(-alpha * x_squared / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    gauss_n = QR_decomposition(gauss_state) # Нахождение MPS для гауссого состояния при некотором начальном n
    for i in range(l):
        gauss_n_one = [np.array([1,0]).reshape(1,2,1)] + gauss_n
        gauss_n_two = gauss_n.copy()
        gauss_n_two[0] = gauss_n_two[0] * np.exp(- alpha / (2 * sigma**2) * 4**(n + i))
        gauss_n_two = el_by_el_multiplication(gauss_n_two, exp_mps(n + i, alpha / (2 * sigma**2), -2**(n + 1 + i)))
        gauss_n_two = [np.array([0,1]).reshape(1,2,1)] + gauss_n_two
        new_gauss_mps = mps_sum(gauss_n_one, gauss_n_two)
        gauss_n = mps_compression(new_gauss_mps)
    return gauss_n

def sym_gauss_mps(x_bond, sigma, n):
    alpha = x_bond**2 / 4**(n-1)
    j = 3
    first_part = gauss_mps(j, n - j - 1, alpha, sigma)
    second_part = []
    for i in first_part:
        second_part.append(i[:,::-1,:])
    first_part = [np.array([0,1]).reshape(1,2,1)] + first_part
    second_part = [np.array([1,0]).reshape(1,2,1)] + second_part
    sym_gauss = mps_sum(first_part, second_part)
    return sym_gauss

def asym_gauss_mps(x_left, x_right, n, sigma, eps=1e-6):
    x_values = np.linspace(-x_left, x_right, 2**n)
    x_squared = x_values**2
    gauss_state = np.exp(-x_squared / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return vec_to_mps(gauss_state, eps=eps)

def vacuum_gauss_mps(n, l, alpha):
    x_base = np.arange(2**n)
    x_squared = x_base**2
    gauss_state = np.exp(-alpha * x_squared / (2)) / (np.sqrt(np.sqrt(np.pi)))
    gauss_n = QR_decomposition(gauss_state) # Нахождение MPS для гауссого состояния при некотором начальном n
    for i in range(l):
        gauss_n_one = [np.array([1,0]).reshape(1,2,1)] + gauss_n
        gauss_n_two = gauss_n.copy()
        gauss_n_two[0] = gauss_n_two[0] * np.exp(- alpha / 2 * 4**(n + i))
        gauss_n_two = el_by_el_multiplication(gauss_n_two, exp_mps(n + i, alpha / 2, -2**(n + 1 + i)))
        gauss_n_two = [np.array([0,1]).reshape(1,2,1)] + gauss_n_two
        new_gauss_mps = mps_sum(gauss_n_one, gauss_n_two)
        gauss_n = mps_compression(new_gauss_mps)
    return gauss_n

def vacuum_state(x_bond, n):
    alpha = x_bond**2 / 4**(n-1)
    j = 3
    first_part = vacuum_gauss_mps(j, n - j - 1, alpha)
    second_part = []
    for i in first_part:
        second_part.append(i[:,::-1,:])
    first_part = [np.array([0,1]).reshape(1,2,1)] + first_part
    second_part = [np.array([1,0]).reshape(1,2,1)] + second_part
    sym_gauss = mps_sum(first_part, second_part)
    return sym_gauss

def mp_solve(mpo, mps_rp, mps_initial, fullshape = True, eps = 1e-6, optimize = True, bond_limit = 1000, renormalize = True):
    n = len(mpo)
    initial_norm = np.sqrt(abs(mps_scalar_multiplication(mps_rp, mps_rp)))
    # Эти 3 блока кода - просто приведение крайних тензоров к нужной размерности - условно, если первый тензор был формы (2, 4), эти функции делают reshape в (1,2,4)
    mps_train = mps_initial.copy()
    mps_train = mps_compression(mps_train, eps, bond_limit)
    if len(mps_train[0].shape) == 3:
        mps_train[-1] = mps_train[-1].reshape(mps_train[-1].shape[0], mps_train[-1].shape[1])
        mps_train[0] = mps_train[0].reshape(mps_train[0].shape[1], mps_train[0].shape[2])

    mpo_train = mpo.copy()
    if len(mpo_train[0].shape) == 4:
        mpo_train[-1] = mpo_train[-1].reshape(mpo_train[-1].shape[0], mpo_train[-1].shape[1], mpo_train[-1].shape[2])
        mpo_train[0] = mpo_train[0].reshape(mpo_train[0].shape[1], mpo_train[0].shape[2], mpo_train[0].shape[3])

    b_mps = mps_rp.copy()
    b_mps = mps_compression(b_mps, eps, bond_limit)
    if len(b_mps[0].shape) == 3:
        b_mps[0] = b_mps[0].reshape(b_mps[0].shape[1], b_mps[0].shape[2])
        b_mps[-1] = b_mps[-1].reshape(b_mps[-1].shape[0], b_mps[-1].shape[1])

    # Вычисление крайних правых слоёв
    right_layers = []
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps_train[-1]), mpo_train[-1], mps_train[-1], optimize=optimize)
    right_layers.append(first_layer)

    right_r_layers = []
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps_train[-1]), optimize=optimize)
    right_r_layers.append(first_r_layer)

    # Первоначальная свёртка справа налево
    for i in range(2, n-1):
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps_train[-i]), optimize=optimize)
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_train[-i], optimize=optimize)
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_train[-i], optimize=optimize)
        right_layers.append(new_layer)

    for i in range(2, n-1):
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[-i], optimize=optimize)
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_train[-i]), optimize=optimize)
        right_r_layers.append(new_r_layer)

    # Начало прохода слева направо
    first_step = np.einsum('abc, ijkb, dei -> ajdcke', right_layers[-1], mpo_train[1], mpo_train[0], optimize=optimize)
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> bjc', right_r_layers[-1], b_mps[1], b_mps[0], optimize=optimize)
    first_step_v = first_step_v.reshape(-1)

    # Вычисление крайнего левого тензора
    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    bond = min((t_1_svd.S / t_1_svd.S.sum() > eps ).sum(), bond_limit)
    t_1_svd_U = t_1_svd.U.reshape(2, -1)
    t_1_svd_U = t_1_svd_U[:, :bond]
    mps_train[0] = t_1_svd_U

    # Обновление крайнего левого слоя
    left_layers = []
    last_layer = np.einsum('ij, ilq, lr -> jqr', np.conj(mps_train[0]), mpo_train[0], mps_train[0], optimize=optimize)
    left_layers.append(last_layer)

    left_r_layers = []
    last_r_layer = np.einsum('ij, ik -> jk', b_mps[0], np.conj(mps_train[0]), optimize=optimize)
    left_r_layers.append(last_r_layer)

    # Проход слева направо
    for i in range(n-3): 
        # Вычисляем матрицу
        t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers[i], right_layers[- 2 - i], mpo_train[i + 1], mpo_train[i + 2], optimize=optimize)
        shape_t = t_inter.shape
        t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
        # Вычисляем правую часть СЛУ
        rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layers[i], right_r_layers[- 2 - i], b_mps[i + 1], b_mps[i + 2], optimize=optimize)
        rp_inter = rp_inter.reshape(-1)
        # Вычисляем решение
        t_new = np.linalg.solve(t_inter, rp_inter)

        # Переводим решение в форму тензора
        t_new = t_new.reshape(mps_train[i].shape[-1] * 2, -1)
        t_new_svd = np.linalg.svd(t_new, full_matrices = False)
        bond = min((t_new_svd.S / t_new_svd.S.sum() > eps ).sum(), bond_limit)
        t_new_svd_U = t_new_svd.U.reshape(mps_train[i].shape[-1], 2, -1)
        t_new_svd_U = t_new_svd_U[:, :, :bond]
        mps_train[i + 1] = t_new_svd_U
     
        # Вычисление следующего слоя с учётом обновлённых тензоров
        new_layer = np.einsum('abc, ajk -> kjbc', left_layers[-1], np.conj(mps_train[i + 1]), optimize=optimize)
        new_layer = np.einsum('abcd, cbij -> ajid', new_layer, mpo_train[i + 1], optimize=optimize)
        new_layer = np.einsum('abcd, dci -> abi', new_layer, mps_train[i + 1], optimize=optimize)
        left_layers.append(new_layer)
    
        new_l_layer = np.einsum('ab, aij -> jib', left_r_layers[-1], b_mps[i + 1], optimize=optimize)
        new_l_layer = np.einsum('abc, cbj -> aj', new_l_layer, np.conj(mps_train[i + 1]), optimize=optimize)
        left_r_layers.append(new_l_layer)

    last_step = np.einsum('abc, bpqr, rij -> apicqj', left_layers[-1], mpo_train[-2], mpo_train[-1], optimize=optimize)
    l_sh = last_step.shape
    last_step = last_step.reshape(l_sh[0] * l_sh[1] * l_sh[2], l_sh[3] * l_sh[4] * l_sh[5])

    last_step_v = np.einsum('ab, aij, jc -> bic', left_r_layers[-1], b_mps[-2], b_mps[-1], optimize=optimize)
    last_step_v = last_step_v.reshape(-1)

    # Вычисление предпоследнего тензора
    t_lasti = np.linalg.solve(last_step, last_step_v)
    t_last = t_lasti
    t_last = t_last.reshape(-1, 2)
    t_last_svd = np.linalg.svd(t_last, full_matrices = False)
    bond = min((t_last_svd.S / t_last_svd.S.sum() > eps ).sum(), bond_limit)
    t_last_svd_U = t_last_svd.U.reshape(mps_train[-3].shape[-1], 2, -1)
    t_last_svd_U = t_last_svd_U[:, :, :bond]
    mps_train[-2] = t_last_svd_U
    t_last_svd_S = t_last_svd.S[:bond]
    t_2_last = np.diag(t_last_svd_S) @ t_last_svd.Vh[:bond,:]
    t_2_last = t_2_last.reshape(mps_train[-2].shape[-1], 2)
    mps_train[-1] = t_2_last
    
    # Начало вычисления прохода справа налево
    t_lasti = t_lasti.reshape(-1, 2)
    t_lasti_svd = np.linalg.svd(t_lasti, full_matrices = False)
    bond = min((t_lasti_svd.S / t_lasti_svd.S.sum() > eps ).sum(), bond_limit)
    zam_tensor = t_lasti_svd.Vh.reshape(-1, 2)
    zam_tensor = zam_tensor[:bond,:]
    mps_train[-1] = zam_tensor.copy()

    right_layers = []
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps_train[-1]), mpo_train[-1], mps_train[-1], optimize=optimize)
    right_layers.append(first_layer)

    right_r_layers = []
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps_train[-1]), optimize=optimize)
    right_r_layers.append(first_r_layer)

    for i in range(n-3):
        # Вычисляем матрицу
        t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers[- 2 - i], right_layers[i], mpo_train[- i - 3], mpo_train[- i - 2], optimize=optimize)
        shape_t = t_inter.shape
        t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
        # Вычисляем правую часть СЛУ
        rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layers[- 2 - i], right_r_layers[i], b_mps[- i - 3], b_mps[- i - 2], optimize=optimize)
        rp_inter = rp_inter.reshape(-1)
        # Вычисляем решение
        t_new = np.linalg.solve(t_inter, rp_inter)
    
        # Переводим решение в форму тензора
        t_new = t_new.reshape(-1, mps_train[- i - 1].shape[0] * 2)
        t_new_svd = np.linalg.svd(t_new, full_matrices = False)
        
        b_tensor = t_new_svd.Vh.reshape(-1, 2, mps_train[- i - 1].shape[0]) # Переводим матрицу Vh в тензор
        #l = len(list(filter(zero_filter, t_new_svd.S))) if len(list(filter(zero_filter, t_new_svd.S))) > 0 else 1 # Находим условие обрезания матриц и тензоров
        bond = min((t_new_svd.S / t_new_svd.S.sum() > eps ).sum(), bond_limit)
        b_tensor = b_tensor[:bond,:,:] # Обрезаем тензор
        mps_train[- i - 2] = b_tensor.copy() # Вносим тензор в лист хранения тензоров
        
        # Вычисление следующего слоя с учётом обновлённых тензоров
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps_train[-i -2]), optimize=optimize)
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_train[-i -2], optimize=optimize)
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_train[-i -2], optimize=optimize)
        right_layers.append(new_layer)
    
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[- i - 2], optimize=optimize)
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_train[- i - 2]), optimize=optimize)
        right_r_layers.append(new_r_layer)

    first_step = np.einsum('abc, ijkb, dei -> djaekc', right_layers[-1], mpo_train[1], mpo_train[0], optimize=optimize)
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> cjb', right_r_layers[-1], b_mps[1], b_mps[0], optimize=optimize)
    first_step_v = first_step_v.reshape(-1)

    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    #l = len(list(filter(zero_filter, t_1_svd.S))) if len(list(filter(zero_filter, t_1_svd.S))) > 0 else 1
    bond = min((t_1_svd.S / t_1_svd.S.sum() > eps ).sum(), bond_limit)
    t_2 = t_1_svd.Vh.reshape(-1, 2, mps_train[2].shape[0])
    t_2 = t_2[:bond,:,:]
    t_1_svd_U = t_1_svd.U
    t_1_svd_U = t_1_svd_U[:, :bond]
    t_1_svd_S = t_1_svd.S[:bond]
    mps_train[0] = t_1_svd_U @ np.diag(t_1_svd_S)
    mps_train[1] = t_2
    if fullshape:
        mps_train[0] = mps_train[0].reshape(1, 2, -1)
        mps_train[-1] = mps_train[-1].reshape(-1, 2, 1)
    if renormalize:
        renormalize_mps(mps_train, initial_norm)
    return mps_train

def mp_solve4(mpo, mps_rp, mps_initial, fullshape = True, eps = 1e-6, optimize = True, bond_limit = 1000, renormalize = True):
    n = len(mpo)
    initial_norm = np.sqrt(abs(mps_scalar_multiplication(mps_rp, mps_rp)))
    # Эти 3 блока кода - просто приведение крайних тензоров к нужной размерности - условно, если первый тензор был формы (2, 4), эти функции делают reshape в (1,2,4)
    mps_train = mps_initial.copy()
    mps_train = mps_compression(mps_train)
    if len(mps_train[0].shape) == 3:
        mps_train[-1] = mps_train[-1].reshape(mps_train[-1].shape[0], mps_train[-1].shape[1])
        mps_train[0] = mps_train[0].reshape(mps_train[0].shape[1], mps_train[0].shape[2])

    mpo_train = mpo.copy()
    if len(mpo_train[0].shape) == 4:
        mpo_train[-1] = mpo_train[-1].reshape(mpo_train[-1].shape[0], mpo_train[-1].shape[1], mpo_train[-1].shape[2])
        mpo_train[0] = mpo_train[0].reshape(mpo_train[0].shape[1], mpo_train[0].shape[2], mpo_train[0].shape[3])

    b_mps = mps_rp.copy()
    b_mps = mps_compression(b_mps)
    if len(b_mps[0].shape) == 3:
        b_mps[0] = b_mps[0].reshape(b_mps[0].shape[1], b_mps[0].shape[2])
        b_mps[-1] = b_mps[-1].reshape(b_mps[-1].shape[0], b_mps[-1].shape[1])

    # Вычисление крайних правых слоёв
    right_layers = []
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps_train[-1]), mpo_train[-1], mps_train[-1], optimize=optimize)
    right_layers.append(first_layer)

    right_r_layers = []
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps_train[-1]), optimize=optimize)
    right_r_layers.append(first_r_layer)

    # Первоначальная свёртка справа налево
    for i in range(2, n-1):
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps_train[-i]), optimize=optimize)
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_train[-i], optimize=optimize)
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_train[-i], optimize=optimize)
        right_layers.append(new_layer)

    for i in range(2, n-1):
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[-i], optimize=optimize)
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_train[-i]), optimize=optimize)
        right_r_layers.append(new_r_layer)

    # Начало прохода слева направо
    first_step = np.einsum('abc, ijkb, dei -> ajdcke', right_layers[-1], mpo_train[1], mpo_train[0], optimize=optimize)
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> bjc', right_r_layers[-1], b_mps[1], b_mps[0], optimize=optimize)
    first_step_v = first_step_v.reshape(-1)

    # Вычисление крайнего левого тензора
    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    bond = min((t_1_svd.S / t_1_svd.S.sum() > eps ).sum(), bond_limit)
    t_1_svd_U = t_1_svd.U.reshape(2, -1)
    t_1_svd_U = t_1_svd_U[:, :bond]
    mps_train[0] = t_1_svd_U

    # Обновление крайнего левого слоя
    left_layers = []
    last_layer = np.einsum('ij, ilq, lr -> jqr', np.conj(mps_train[0]), mpo_train[0], mps_train[0], optimize=optimize)
    left_layers.append(last_layer)

    left_r_layers = []
    last_r_layer = np.einsum('ij, ik -> jk', b_mps[0], np.conj(mps_train[0]), optimize=optimize)
    left_r_layers.append(last_r_layer)

    # Проход слева направо
    for i in range(n-3): 
        # Вычисляем матрицу
        t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers[i], right_layers[- 2 - i], mpo_train[i + 1], mpo_train[i + 2], optimize=optimize)
        shape_t = t_inter.shape
        t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
        # Вычисляем правую часть СЛУ
        rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layers[i], right_r_layers[- 2 - i], b_mps[i + 1], b_mps[i + 2], optimize=optimize)
        rp_inter = rp_inter.reshape(-1)
        # Вычисляем решение
        t_new = np.linalg.solve(t_inter, rp_inter)

        # Переводим решение в форму тензора
        t_new = t_new.reshape(mps_train[i].shape[-1] * 2, -1)
        t_new_svd = np.linalg.svd(t_new, full_matrices = False)
        bond = min((t_new_svd.S / t_new_svd.S.sum() > eps ).sum(), bond_limit)
        t_new_svd_U = t_new_svd.U.reshape(mps_train[i].shape[-1], 2, -1)
        t_new_svd_U = t_new_svd_U[:, :, :bond]
        mps_train[i + 1] = t_new_svd_U
     
        # Вычисление следующего слоя с учётом обновлённых тензоров
        new_layer = np.einsum('abc, ajk -> kjbc', left_layers[-1], np.conj(mps_train[i + 1]), optimize=optimize)
        new_layer = np.einsum('abcd, cbij -> ajid', new_layer, mpo_train[i + 1], optimize=optimize)
        new_layer = np.einsum('abcd, dci -> abi', new_layer, mps_train[i + 1], optimize=optimize)
        left_layers.append(new_layer)
    
        new_l_layer = np.einsum('ab, aij -> jib', left_r_layers[-1], b_mps[i + 1], optimize=optimize)
        new_l_layer = np.einsum('abc, cbj -> aj', new_l_layer, np.conj(mps_train[i + 1]), optimize=optimize)
        left_r_layers.append(new_l_layer)

    last_step = np.einsum('abc, bpqr, rij -> apicqj', left_layers[-1], mpo_train[-2], mpo_train[-1], optimize=optimize)
    l_sh = last_step.shape
    last_step = last_step.reshape(l_sh[0] * l_sh[1] * l_sh[2], l_sh[3] * l_sh[4] * l_sh[5])

    last_step_v = np.einsum('ab, aij, jc -> bic', left_r_layers[-1], b_mps[-2], b_mps[-1], optimize=optimize)
    last_step_v = last_step_v.reshape(-1)

    # Вычисление предпоследнего тензора
    t_lasti = np.linalg.solve(last_step, last_step_v)
    t_last = t_lasti
    t_last = t_last.reshape(-1, 2)
    t_last_svd = np.linalg.svd(t_last, full_matrices = False)
    bond = min((t_last_svd.S / t_last_svd.S.sum() > eps ).sum(), bond_limit)
    t_last_svd_U = t_last_svd.U.reshape(mps_train[-3].shape[-1], 2, -1)
    t_last_svd_U = t_last_svd_U[:, :, :bond]
    mps_train[-2] = t_last_svd_U
    t_last_svd_S = t_last_svd.S[:bond]
    t_2_last = np.diag(t_last_svd_S) @ t_last_svd.Vh[:bond,:]
    t_2_last = t_2_last.reshape(mps_train[-2].shape[-1], 2)
    mps_train[-1] = t_2_last
    
    # Начало вычисления прохода справа налево
    t_lasti = t_lasti.reshape(-1, 2)
    t_lasti_svd = np.linalg.svd(t_lasti, full_matrices = False)
    bond = min((t_lasti_svd.S / t_lasti_svd.S.sum() > eps ).sum(), bond_limit)
    zam_tensor = t_lasti_svd.Vh.reshape(-1, 2)
    zam_tensor = zam_tensor[:bond,:]
    mps_train[-1] = zam_tensor.copy()

    right_layers = []
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps_train[-1]), mpo_train[-1], mps_train[-1], optimize=optimize)
    right_layers.append(first_layer)

    right_r_layers = []
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps_train[-1]), optimize=optimize)
    right_r_layers.append(first_r_layer)

    for i in range(n-3):
        # Вычисляем матрицу
        t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers[- 2 - i], right_layers[i], mpo_train[- i - 3], mpo_train[- i - 2], optimize=optimize)
        shape_t = t_inter.shape
        t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
        # Вычисляем правую часть СЛУ
        rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layers[- 2 - i], right_r_layers[i], b_mps[- i - 3], b_mps[- i - 2], optimize=optimize)
        rp_inter = rp_inter.reshape(-1)
        # Вычисляем решение
        t_new = np.linalg.solve(t_inter, rp_inter)
    
        # Переводим решение в форму тензора
        t_new = t_new.reshape(-1, mps_train[- i - 1].shape[0] * 2)
        t_new_svd = np.linalg.svd(t_new, full_matrices = False)
        
        b_tensor = t_new_svd.Vh.reshape(-1, 2, mps_train[- i - 1].shape[0]) # Переводим матрицу Vh в тензор
        #l = len(list(filter(zero_filter, t_new_svd.S))) if len(list(filter(zero_filter, t_new_svd.S))) > 0 else 1 # Находим условие обрезания матриц и тензоров
        bond = min((t_new_svd.S / t_new_svd.S.sum() > eps ).sum(), bond_limit)
        b_tensor = b_tensor[:bond,:,:] # Обрезаем тензор
        mps_train[- i - 2] = b_tensor.copy() # Вносим тензор в лист хранения тензоров
        
        # Вычисление следующего слоя с учётом обновлённых тензоров
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps_train[-i -2]), optimize=optimize)
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_train[-i -2], optimize=optimize)
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_train[-i -2], optimize=optimize)
        right_layers.append(new_layer)
    
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[- i - 2], optimize=optimize)
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_train[- i - 2]), optimize=optimize)
        right_r_layers.append(new_r_layer)

    first_step = np.einsum('abc, ijkb, dei -> djaekc', right_layers[-1], mpo_train[1], mpo_train[0], optimize=optimize)
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> cjb', right_r_layers[-1], b_mps[1], b_mps[0], optimize=optimize)
    first_step_v = first_step_v.reshape(-1)

    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    #l = len(list(filter(zero_filter, t_1_svd.S))) if len(list(filter(zero_filter, t_1_svd.S))) > 0 else 1
    bond = min((t_1_svd.S / t_1_svd.S.sum() > eps ).sum(), bond_limit)
    t_2 = t_1_svd.Vh.reshape(-1, 2, mps_train[2].shape[0])
    t_2 = t_2[:bond,:,:]
    t_1_svd_U = t_1_svd.U
    t_1_svd_U = t_1_svd_U[:, :bond]
    t_1_svd_S = t_1_svd.S[:bond]
    mps_train[0] = t_1_svd_U @ np.diag(t_1_svd_S)
    mps_train[1] = t_2
    if fullshape:
        mps_train[0] = mps_train[0].reshape(1, 2, -1)
        mps_train[-1] = mps_train[-1].reshape(-1, 2, 1)
    if renormalize:
        renormalize_mps(mps_train, initial_norm)
    return mps_train

def lay3conv(mps, mpo, right_layers):
    n = len(mpo)
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps[-1]), mpo[-1], mps[-1])
    right_layers.append(first_layer)
    for i in range(2, n-1):
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps[-i]))
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo[-i])
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps[-i])
        right_layers.append(new_layer)
    
def lay2conv(mps, b_mps, right_r_layers):
    n = len(mps)
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps[-1]))
    right_r_layers.append(first_r_layer)
    for i in range(2, n-1):
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[-i])
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps[-i]))
        right_r_layers.append(new_r_layer)

def mps_check(mps):
    mps_train = mps.copy()
    if len(mps_train[0].shape) == 3:
        mps_train[-1] = mps_train[-1].reshape(mps_train[-1].shape[0], mps_train[-1].shape[1])
        mps_train[0] = mps_train[0].reshape(mps_train[0].shape[1], mps_train[0].shape[2])
    return mps_train
    
def mpo_check(mpo):
    mpo_train = mpo.copy()
    if len(mpo_train[0].shape) == 4:
        mpo_train[-1] = mpo_train[-1].reshape(mpo_train[-1].shape[0], mpo_train[-1].shape[1], mpo_train[-1].shape[2])
        mpo_train[0] = mpo_train[0].reshape(mpo_train[0].shape[1], mpo_train[0].shape[2], mpo_train[0].shape[3])
    return mpo_train

def t0comp(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps):
    
    first_step = np.einsum('abc, ijkb, dei -> ajdcke', right_layers[-1], mpo[1], mpo[0])
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> bjc', right_r_layers[-1], b_mps[1], b_mps[0])
    first_step_v = first_step_v.reshape(-1)

    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    bond = (t_1_svd.S / t_1_svd.S.sum() > eps ).sum()
    t_1_svd_U = t_1_svd.U.reshape(2, -1)
    t_1_svd_U = t_1_svd_U[:, :bond]
    mps[0] = t_1_svd_U
    
    last_layer = np.einsum('ij, ilq, lr -> jqr', np.conj(mps[0]), mpo[0], mps[0])
    left_layers.append(last_layer)
    
    last_r_layer = np.einsum('ij, ik -> jk', b_mps[0], np.conj(mps[0]))
    left_r_layers.append(last_r_layer)
    
def t0comp2(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps):
    first_step = np.einsum('abc, ijkb, dei -> djaekc', right_layers[-1], mpo[1], mpo[0])
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> cjb', right_r_layers[-1], b_mps[1], b_mps[0])
    first_step_v = first_step_v.reshape(-1)

    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    bond = (t_1_svd.S / t_1_svd.S.sum() > eps ).sum()
    t_2 = t_1_svd.Vh.reshape(-1, 2, mps[2].shape[0])
    t_2 = t_2[:bond,:,:]
    t_1_svd_U = t_1_svd.U
    t_1_svd_U = t_1_svd_U[:, :bond]
    t_1_svd_S = t_1_svd.S[:bond]
    mps[0] = t_1_svd_U @ np.diag(t_1_svd_S)
    mps[1] = t_2

def compute_newlayer3lr(layer, mps_i, mpo_i):
    new_layer = np.einsum('abc, ajk -> kjbc', layer, np.conj(mps_i))
    new_layer = np.einsum('abcd, cbij -> ajid', new_layer, mpo_i)
    new_layer = np.einsum('abcd, dci -> abi', new_layer, mps_i)
    return new_layer  

def compute_newlayer2lr(layer, mps_i, b_mps_i):
    new_l_layer = np.einsum('ab, aij -> jib', layer, b_mps_i)
    new_l_layer = np.einsum('abc, cbj -> aj', new_l_layer, np.conj(mps_i))
    return new_l_layer 

def compute_newlayer3rl(layer, mps_i, mpo_i):
    new_layer = np.einsum('abc, ija -> ijbc', layer, np.conj(mps_i))
    new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_i)
    new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_i)
    return new_layer  

def compute_newlayer2rl(layer, mps_i, b_mps_i):
    new_r_layer = np.einsum('ab, ija -> ijb', layer, b_mps_i)
    new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_i))
    return new_r_layer    

def compute_2tsol(left_layer_i, right_layer_i, mpo_i, mpo_j, left_r_layer_i, right_r_layer_i, b_mps_i, b_mps_j):
    t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layer_i, right_layer_i, mpo_i, mpo_j)
    #print(t_inter.shape)
    shape_t = t_inter.shape
    t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
    rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layer_i, right_r_layer_i, b_mps_i, b_mps_j)
    rp_inter = rp_inter.reshape(-1)
    return np.linalg.solve(t_inter, rp_inter)

def tintercomp(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, i, eps):
    # Вычисление решения, соответсвующего паре тезноров
    t_new = compute_2tsol(left_layers[i], right_layers[- 2 - i], mpo[i + 1], mpo[i + 2], left_r_layers[i], right_r_layers[- 2 - i], b_mps[i + 1], b_mps[i + 2])
    # Переводим решение в форму матрицы
    t_new = t_new.reshape(mps[i].shape[-1] * 2, -1)
    # Сингулярное разложение матрицы
    t_new_svd = np.linalg.svd(t_new, full_matrices = False)
    # Перевод матрицы в форму тензора + обрезание тензора
    bond = (t_new_svd.S / t_new_svd.S.sum() > eps ).sum()
    t_new_svd_U = t_new_svd.U.reshape(mps[i].shape[-1], 2, -1)[:, :, :bond]
    mps[i + 1] = t_new_svd_U
    # Вычисление следующего слоя с учётом обновлённых тензоров и добавление в листы хранения
    left_layers.append(compute_newlayer3lr(left_layers[-1], mps[i + 1], mpo[i + 1]))
    left_r_layers.append(compute_newlayer2lr(left_r_layers[-1], mps[i + 1], b_mps[i + 1])) 
    
def tintercomp2(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, i, eps):
    t_new = compute_2tsol(left_layers[- 2 - i], right_layers[i], mpo[- i - 3], mpo[- i - 2], left_r_layers[- 2 - i], right_r_layers[i], b_mps[- i - 3], b_mps[- i - 2])
    # Переводим решение в форму тензора
    t_new = t_new.reshape(-1, mps[- i - 1].shape[0] * 2)
    t_new_svd = np.linalg.svd(t_new, full_matrices = False)
    #l = len(list(filter(zero_filter, t_new_svd.S))) if len(list(filter(zero_filter, t_new_svd.S))) > 0 else 1 # Находим условие обрезания матриц и тензоров
    bond = (t_new_svd.S / t_new_svd.S.sum() > eps ).sum()
    b_tensor = t_new_svd.Vh.reshape(-1, 2, mps[- i - 1].shape[0])[:bond,:,:] # Переводим матрицу Vh в тензор и обрезаем его
    mps[- i - 2] = b_tensor.copy() # Вносим тензор в лист хранения тензоров
    # Вычисление следующего слоя с учётом обновлённых тензоров
    right_layers.append(compute_newlayer3rl(right_layers[-1], mps[-i - 2], mpo[-i - 2]))
    right_r_layers.append(compute_newlayer2rl(right_r_layers[-1], mps[-i - 2], b_mps[-i - 2]))
    
def tlastcomp(mps, mpo, b_mps, left_layers, left_r_layers, eps):
    last_step = np.einsum('abc, bpqr, rij -> apicqj', left_layers[-1], mpo[-2], mpo[-1])
    l_sh = last_step.shape
    last_step = last_step.reshape(l_sh[0] * l_sh[1] * l_sh[2], l_sh[3] * l_sh[4] * l_sh[5])

    last_step_v = np.einsum('ab, aij, jc -> bic', left_r_layers[-1], b_mps[-2], b_mps[-1])
    last_step_v = last_step_v.reshape(-1)
    
    t_lasti = np.linalg.solve(last_step, last_step_v)
    t_last = t_lasti
    t_last = t_last.reshape(-1, 2)
    t_last_svd = np.linalg.svd(t_last, full_matrices = False)
    bond = (t_last_svd.S / t_last_svd.S.sum() > eps ).sum()
    t_last_svd_U = t_last_svd.U.reshape(mps[-3].shape[-1], 2, -1)
    t_last_svd_U = t_last_svd_U[:, :, :bond]
    mps[-2] = t_last_svd_U
    t_last_svd_S = t_last_svd.S[:bond]
    t_2_last = np.diag(t_last_svd_S) @ t_last_svd.Vh[:bond,:]
    t_2_last = t_2_last.reshape(mps[-2].shape[-1], 2)
    mps[-1] = t_2_last

def tlastcomp2(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, eps):
    
    last_step = np.einsum('abc, bpqr, rij -> apicqj', left_layers[-1], mpo[-2], mpo[-1])
    l_sh = last_step.shape
    last_step = last_step.reshape(l_sh[0] * l_sh[1] * l_sh[2], l_sh[3] * l_sh[4] * l_sh[5])

    last_step_v = np.einsum('ab, aij, jc -> bic', left_r_layers[-1], b_mps[-2], b_mps[-1])
    last_step_v = last_step_v.reshape(-1)
    
    t_lasti = np.linalg.solve(last_step, last_step_v)
    t_lasti = t_lasti.reshape(-1, 2)
    t_lasti_svd = np.linalg.svd(t_lasti, full_matrices = False)
    bond = (t_lasti_svd.S / t_lasti_svd.S.sum() > eps ).sum()
    zam_tensor = t_lasti_svd.Vh.reshape(-1, 2)
    zam_tensor = zam_tensor[:bond,:]
    mps[-1] = zam_tensor.copy()

    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps[-1]), mpo[-1], mps[-1])
    right_layers.append(first_layer)

    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps[-1]))
    right_r_layers.append(first_r_layer)

def left_to_right_compute(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps):
    n = len(mpo)
    t0comp(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps)
    for i in range(n-3):
        tintercomp(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, i, eps)
    tlastcomp(mps, mpo, b_mps, left_layers, left_r_layers, eps)

def right_to_left_compute(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps):
    n = len(mpo)
    tlastcomp2(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, eps)
    for i in range(n-3):
        tintercomp2(mps, mpo, b_mps, left_layers, left_r_layers, right_layers, right_r_layers, i, eps)
    t0comp2(mpo, mps, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps)

def solution_check(a, b):
    return 0

def mpt_prepare(mps, mpo, b_mps):
    mps_train = mps_check(mps)
    b_mps = mps_check(b_mps)  
    mpo_train = mpo_check(mpo)
    return mps_train, mpo_train, b_mps

def calculate_rc_mpsnorm(mps):
    return np.sqrt(np.trace(np.conj(mps[0])[:, 0, :].T @ mps[0][:, 0, :]) + np.trace(np.conj(mps[0])[:, 1, :].T @ mps[0][:, 1, :]))

def calculate_lc_mpsnorm(mps):
    return np.sqrt(np.trace(np.conj(mps[-1])[:, 0, :].T @ mps[-1][:, 0, :]) + np.trace(np.conj(mps[-1])[:, 1, :].T @ mps[-1][:, 1, :]))

def renormalize_mps(mps, initial_norm):
    mps[0] = mps[0] * initial_norm / calculate_rc_mpsnorm(mps)

def solve2(mpo, mps_rp, mps_initial, fullshape = True, eps = 1e-6):
    # Создаём путсые листы, в которых будут храниться вычисляемые слои
    right_layers = []
    right_r_layers = []
    left_layers = []
    left_r_layers = []

    initial_norm = np.sqrt(abs(mps_scalar_multiplication(mps_rp, mps_rp)))

    mps_train, mpo_train, b_mps = mpt_prepare(mps_initial, mpo, mps_rp) # Предподготовка mpo, mps и b_mps
    
    lay3conv(mps_train, mpo_train, right_layers) # Выполняется первоначальное вычисление слоёв сети
    lay2conv(mps_train, b_mps, right_r_layers) # Выполняется первоначальное вычисление слоёв сети
    
    #while not solution_check(left_to_right, right_to_left):
    left_to_right_compute(mpo_train, mps_train, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps) # Вычисление первого приближения решения - проход слева направо
    right_layers = []
    right_r_layers = []
    right_to_left_compute(mpo_train, mps_train, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps) # Вычисление второго приближения решения - проход справа налево
    #№left_layers = []
    #left_r_layers = []
    #left_to_right_compute(mpo_train, mps_train, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps) # Вычисление первого приближения решения - проход слева направо
    #right_layers = []
    #right_r_layers = []
    #right_to_left_compute(mpo_train, mps_train, b_mps, right_layers, right_r_layers, left_layers, left_r_layers, eps) # Вычисление второго приближения решения - проход справа налево

    if fullshape:
        full_reshape(mps_train) # Возврат mps в полной форме
    #renormalize_mps(mps_train, initial_norm)
    #print(calculate_rc_mpsnorm(mps_train))
    return mps_train

def full_reshape(mps):
        mps[0] = mps[0].reshape(1, 2, -1)
        mps[-1] = mps[-1].reshape(-1, 2, 1)

'''              
def solve3(mpo, mps_rp, mps_initial, fullshape = True):
    
    n = len(mpo)
    
    right_layers = []
    right_r_layers = []
    left_layers = []
    left_r_layers = []
    
    mps_train, mpo_train, b_mps = mpt_prepare(mps_initial, mpo, mps_rp) # Предподготовка mpo, mps и b_mps
    
    lay3conv(mps_train, mpo_train, right_layers) # Выполняется первоначальное вычисление слоёв сети
    lay2conv(mps_train, b_mps, right_r_layers) # Выполняется первоначальное вычисление слоёв сети
    
    left_to_right_compute(mpo_train, mps_train, b_mps, right_layers, right_r_layers, left_layers, left_r_layers) # Вычисление первого приближения решения - проход слева направо
    
    last_step = np.einsum('abc, bpqr, rij -> apicqj', left_layers[-1], mpo_train[-2], mpo_train[-1])
    l_sh = last_step.shape
    last_step = last_step.reshape(l_sh[0] * l_sh[1] * l_sh[2], l_sh[3] * l_sh[4] * l_sh[5])

    last_step_v = np.einsum('ab, aij, jc -> bic', left_r_layers[-1], b_mps[-2], b_mps[-1])
    last_step_v = last_step_v.reshape(-1)
    
    t_lasti = np.linalg.solve(last_step, last_step_v)
    # Начало вычисления прохода справа налево
    t_lasti = t_lasti.reshape(-1, 2)
    t_lasti_svd = np.linalg.svd(t_lasti, full_matrices = False)
    l = len(list(filter(zero_filter, t_lasti_svd.S))) if len(list(filter(zero_filter, t_lasti_svd.S))) > 0 else 1
    zam_tensor = t_lasti_svd.Vh.reshape(-1, 2)
    zam_tensor = zam_tensor[:l,:]
    mps_train[-1] = zam_tensor.copy()

    right_layers = []
    first_layer = np.einsum('ij, njr, qr -> inq', np.conj(mps_train[-1]), mpo_train[-1], mps_train[-1])
    right_layers.append(first_layer)

    right_r_layers = []
    first_r_layer = np.einsum('ij, kj -> ik', b_mps[-1], np.conj(mps_train[-1]))
    right_r_layers.append(first_r_layer)

    for i in range(n-3):
        # Вычисляем матрицу
        t_inter = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers[- 2 - i], right_layers[i], mpo_train[- i - 3], mpo_train[- i - 2])
        shape_t = t_inter.shape
        t_inter = t_inter.reshape(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3], shape_t[4] * shape_t[5] * shape_t[6] * shape_t[7])
        # Вычисляем правую часть СЛУ
        rp_inter = np.einsum("ab, jk, apq ,qdj -> bpdk", left_r_layers[- 2 - i], right_r_layers[i], b_mps[- i - 3], b_mps[- i - 2])
        rp_inter = rp_inter.reshape(-1)
        # Вычисляем решение
        t_new = np.linalg.solve(t_inter, rp_inter)
    
        # Переводим решение в форму тензора
        t_new = t_new.reshape(-1, mps_train[- i - 1].shape[0] * 2)
        t_new_svd = np.linalg.svd(t_new, full_matrices = False)
        
        b_tensor = t_new_svd.Vh.reshape(-1, 2, mps_train[- i - 1].shape[0]) # Переводим матрицу Vh в тензор
        l = len(list(filter(zero_filter, t_new_svd.S))) if len(list(filter(zero_filter, t_new_svd.S))) > 0 else 1 # Находим условие обрезания матриц и тензоров
        b_tensor = b_tensor[:l,:,:] # Обрезаем тензор
        mps_train[- i - 2] = b_tensor.copy() # Вносим тензор в лист хранения тензоров
        
        # Вычисление следующего слоя с учётом обновлённых тензоров
        new_layer = np.einsum('abc, ija -> ijbc', right_layers[-1], np.conj(mps_train[-i -2]))
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_train[-i -2])
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_train[-i -2])
        right_layers.append(new_layer)
    
        new_r_layer = np.einsum('ab, ija -> ijb', right_r_layers[-1], b_mps[- i - 2])
        new_r_layer = np.einsum('abc, ibc -> ai', new_r_layer, np.conj(mps_train[- i - 2]))
        right_r_layers.append(new_r_layer)

    first_step = np.einsum('abc, ijkb, dei -> djaekc', right_layers[-1], mpo_train[1], mpo_train[0])
    f_sh = first_step.shape
    first_step = first_step.reshape(f_sh[0] * f_sh[1] * f_sh[2], f_sh[3] * f_sh[4] * f_sh[5])

    first_step_v = np.einsum('ab, ija, ci -> cjb', right_r_layers[-1], b_mps[1], b_mps[0])
    first_step_v = first_step_v.reshape(-1)

    t_1 = np.linalg.solve(first_step, first_step_v)
    t_1 = t_1.reshape(2, -1)
    t_1_svd = np.linalg.svd(t_1, full_matrices = False)
    l = len(list(filter(zero_filter, t_1_svd.S))) if len(list(filter(zero_filter, t_1_svd.S))) > 0 else 1
    t_2 = t_1_svd.Vh.reshape(-1, 2, mps_train[2].shape[0])
    t_2 = t_2[:l,:,:]
    t_1_svd_U = t_1_svd.U
    t_1_svd_U = t_1_svd_U[:, :l]
    t_1_svd_S = t_1_svd.S[:l]
    mps_train[0] = t_1_svd_U @ np.diag(t_1_svd_S)
    mps_train[1] = t_2
    if fullshape:
        mps_train[0] = mps_train[0].reshape(1, 2, -1)
        mps_train[-1] = mps_train[-1].reshape(-1, 2, 1)
    return mps_train     
'''
       
def mpo_multiplication(mpo_one, mpo_two):
    mult_mpo = []
    for i in range(len(mpo_one)):
        sh1 = mpo_one[i].shape
        sh2 = mpo_two[i].shape
        mult_mpo.append(np.einsum('ijkl, mkop -> imjolp', mpo_one[i], mpo_two[i]).reshape(sh1[0]*sh2[0], 2, 2, sh1[3]*sh2[3]))
    return mult_mpo

def mpo_sum(mpo_one, mpo_two):
    common_tensor = []
    common_tensor.append(np.concatenate([mpo_one[0], mpo_two[0]], axis = 3).reshape(1, 2, 2, -1))
    for i in range(1,len(mpo_one)-1):
        zero_arr1 = np.zeros((mpo_two[i].shape[0], 2, 2, mpo_one[i].shape[3]))
        zero_arr2 = np.zeros((mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3]))
        ff1 = np.concatenate([mpo_one[i], zero_arr1], axis=0).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_one[i].shape[3])
        ff2 = np.concatenate([zero_arr2, mpo_two[i]], axis=0).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3])
        new_element = np.concatenate([ff1, ff2], axis = 3).reshape(mpo_two[i].shape[0] + mpo_one[i].shape[0], 2, 2, mpo_two[i].shape[3] + mpo_one[i].shape[3])
        common_tensor.append(new_element)
    common_tensor.append(np.concatenate([mpo_one[-1], mpo_two[-1]], axis = 0).reshape(mpo_one[-1].shape[0] + mpo_two[-1].shape[0], 2, 2, 1))
    return common_tensor

def one_mpo(n):
    massive = [np.eye(2).reshape(1,2,2,1)] * n
    return massive

def chislo(n, k):
    massive = [np.eye(2).reshape(1,2,2,1)] * n
    massive[0] = massive[0] * k
    return massive

def mpo_mul_num(mpo, num):
    mpo_ch = mpo.copy()
    mpo_ch[0] = mpo_ch[0] * num
    return mpo_ch

def mps_mul_num(mps, num):
    mps_in_change = mps.copy()
    mps_in_change[0] = mps_in_change[0] * num
    return mps_in_change

def mpo_x_mps(mpo, mps):
    new_mps = []
    for i in range(len(mps)):
        new_mps.append(np.einsum("ijnl, mno -> imjlo", mpo[i], mps[i]).reshape(mpo[i].shape[0] * mps[i].shape[0], 2, mpo[i].shape[-1] * mps[i].shape[-1]))
    return new_mps

def mps_x_num(mps, num):
    mps[0] = mps[0] * num
    return mps

def b_part_funk(n, b_0, b_last):
    b_mps_1 = []
    b_mps_2 = []
    for i in range(n-1):
        b_mps_1.append(np.array([1, 0]).reshape(1, 2, 1))
        b_mps_2.append(np.array([0, 1]).reshape(1, 2, 1))
    b_mps_1.append(np.array([b_0, 0]).reshape(1, 2, 1))
    b_mps_2.append(np.array([0, b_last]).reshape(1, 2, 1))
    return mps_sum(b_mps_1, b_mps_2)

def ga_uss(i, x_0, sigma, amplitude):
    return amplitude * np.exp((-(i - x_0)**2)/(2 * sigma**2))    

def get_bond(mps):
    dim1 = []
    dim2 = []
    for i in mps:
        dim1.append(i.shape[0])
        dim2.append(i.shape[-1])
    return max(max(dim1), max(dim2)) 

#def mpo_mul_chislo(mpo, chislo):
#    mpo[0] = mpo[0] * chislo
#    return mpo

def find_reduced_density_matrix(mps, position, length):
    new_tensor = []
    mps_test = mps.copy()
    if position == 0:
        real_position = 1
    else:
        real_position = 0

    length2 = len(mps) - length
    frin = real_position * length
    frin2 = position * length2

    # Делаем свёртку тензоров MPS по физическим индексам тензоров, которые хотим исключить из рассмотрения
    for i in range(frin, frin + length2):
        new_tensor.append(np.einsum('abc, ibk -> aick', mps_test[i], np.conj(mps_test[i]), optimize=True))
    first = new_tensor[0]
    # Сворачиваем эти же тензоры, но теперь по бонд-индексам
    for i in range(len(new_tensor) - 1):
        first = np.einsum('abcd, cdkl -> abkl', first, new_tensor[i + 1], optimize=True)
    sh = first.shape
    first = first.reshape(sh[0] * sh[1], sh[2] * sh[3])
        
    # Объединяем пары тензоров нужного MPS в один тензор
    new_tensor = []
    for i in range(frin2, frin2 + length):
        t = np.einsum('inj, kml -> iknmjl', mps_test[i], np.conj(mps_test[i]), optimize=True)
        sh = t.shape
        t = t.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4] * sh[5])
        new_tensor.append(t)
    second = new_tensor[0]
    if position == 1:
        second = np.einsum('ab, bijk -> aijk', first, second, optimize=True)

    # Сворачиваем нужные mps по bond-ногам
    for i in range(length - 1):
        second = np.einsum('abcd, djkl -> abjckl', second, new_tensor[i + 1], optimize=True)
        sh = second.shape
        second = second.reshape(sh[0], sh[1] * sh[2], sh[3] * sh[4], sh[5])
    if position == 0:
        last = np.einsum('abcd, dk -> abck', second, first, optimize=True)
    if position == 1:
        last = second
    return last.reshape(last.shape[1], last.shape[2])

def find_density_matrix(mps):
    new_tensor = []
    mps_test = mps.copy()
    for i in range(len(mps)):
        t = np.einsum('inj, kml -> iknmjl', mps_test[i], np.conj(mps_test[i]))
        sh = t.shape
        print(sh)
        t = t.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4] * sh[5])
        new_tensor.append(t)
    second = new_tensor[0]
    for i in range(len(mps) - 1):
        second = np.einsum('abcd, djkl -> abjckl', second, new_tensor[i + 1])
        sh = second.shape
        second = second.reshape(sh[0], sh[1] * sh[2], sh[3] * sh[4], sh[5])
    return second.reshape(second.shape[1], second.shape[2])

def glue_mps(mps1, mps2):
    f_part = [np.array([1, 0]).reshape(1, 2, 1)] + mps1
    s_part = [np.array([0, 1]).reshape(1, 2, 1)] + mps2
    new_mps = mps_sum(f_part, s_part)
    return mps_compression(new_mps)

def squeezed_vacuum(x, alpha, t):
    return np.exp(alpha * t / 2) * np.exp(- (np.exp(2 * alpha * t) * x**2) / 2) / np.pi**(0.25)

def vacuum_state_density_matrix(x, m, omega, hbar):
    N = len(x)
    rho = np.zeros((N, N), dtype=complex)
    prefactor = (m * omega / (np.pi * hbar)) ** 0.5
    for i in range(N):
        for j in range(N):
            rho[i, j] = prefactor * np.exp(-m * omega * (x[i]**2 + x[j]**2) / (2 * hbar))
    return rho

def wigner_function(rho, x, p, hbar = 1):
    N = len(x)
    W = np.zeros((N, N), dtype=complex)
    for i in tqdm(range(0, N)):
        for j in range(0, N):
            sum_w = 0
            for k in range(- int(N/2), int(N/2)):
                if N > i + k >= 0  and N > i - k >= 0  and N > k + int(N/2) >= 0:
                    sum_w += rho[i + k, i - k] * np.exp(2j * p[j] * x[k + int(N/2)] / hbar)
            W[i, j] = sum_w / (np.pi * hbar)
    return W.real

# p - Реальный диапазон p, в котором находится функция Вигнера
# x - Область определения волновой функции
# На выходе мы получим W(q, p) - функция Вигнера, где диапазон q будет соответствовать диапазону x

def wigner_function3(rho, x, p, hbar):
    N1 = len(x)
    N2 = len(p)
    matrix_one = [[1 * i] * N2 for i in range(N1)]
    matrix_two = [[1 * i for i in range(N2)]] * N1
    @njit
    def wig_help_function(i, j):
        sum_w = 0
        for k in range(-int(N1/2), int(N1/2)):
            if N1 > i + k >= 0  and N1 > i - k >= 0  and N1 > k + int(N1/2) >= 0:
                sum_w += rho[i + k, i - k] * np.exp(2j * p[j] * x[k + int(N1/2)] / hbar)
        return sum_w / (np.pi * hbar)
    vectorized_function = np.vectorize(wig_help_function)
    W = vectorized_function(matrix_one, matrix_two)
    return W.real

def wigner_function4(rho, rho_left_edge, rho_right_edge, q, p, hbar):
    
    delta_x_step = (rho_right_edge-rho_left_edge)/len(rho)
    N_q = len(q)
    N_p = len(p)
    
    rho_x_grid = np.linspace(rho_left_edge, rho_right_edge, len(rho))
    
    matrix_one = [[1 * i] * N_p for i in range(N_q)]
    matrix_two = [[1 * i for i in range(N_p)]] * N_q
    
    @njit
    def wig_help_function(i, j):
        if q[i] > rho_right_edge or q[i] < rho_left_edge:
            return 0
        else:
            sum_w = 0
            x = np.linspace(-len(rho)*delta_x_step/2, len(rho)*delta_x_step/2, len(rho))
            bra_index = False
            ket_index = False
            for k in range(len(x)):
                # Проверяем, что координаты точки находятся внутри области определения матрицы плотности
                if rho_right_edge > q[i]+x[k] >= rho_left_edge and rho_right_edge > q[i]-x[k] >= rho_left_edge:
                    # Проверяем, был ли уже найден индекс элемента k
                    if k == 0 or bra_index == False or ket_index == False:
                        condition_bra = (-delta_x_step < q[i] - x[k] - rho_x_grid) & (q[i] - x[k] - rho_x_grid < delta_x_step)
                        condition_ket = (-delta_x_step < q[i] + x[k] - rho_x_grid) & (q[i] + x[k] - rho_x_grid < delta_x_step)
                        bra_indices = np.where(condition_bra)[0]
                        ket_indices = np.where(condition_ket)[0]
                        if len(bra_indices) > 0 and len(ket_indices) > 0:
                            bra_index = bra_indices[0]   
                            ket_index = ket_indices[0]
                            k_f = k
                            sum_w += rho[bra_index, ket_index] * np.exp(2j * p[j] * x[k] / hbar) 
                        else:
                            continue
                    # После того, как первый индекс был найден, сразу понимаем индексы остальных элементов
                    else:
                        sum_w += rho[bra_index-k+k_f, ket_index+k-k_f] * np.exp(2j * p[j] * x[k] / hbar)             
        return sum_w / (np.pi * hbar)
    vectorized_function = np.vectorize(wig_help_function)
    W = vectorized_function(matrix_one, matrix_two)
    return W.real[:,::-1]

def ph_num_operator(n, x0, step, bound = False):
    return (x_matrix(n, x0, step) @ x_matrix(n, x0, step) - list(np.eye(2**n)) - der2_matrix(n, step, bound)) / 2

def calculate_ph_num(rho, n, x0, step, bound = False):  
    return np.real(np.trace(rho @ ph_num_operator(n, x0, step, bound)))

def mult_mpo_by_mps(mpo, mps):
    new_mps = []
    for i in range(len(mpo)):
        new_mps.append(np.einsum('abcd, ick -> aibdk', mpo[i], mps[i]).reshape(mpo[i].shape[0]*mps[i].shape[0], 2, mpo[i].shape[-1]*mps[i].shape[-1]))
    return new_mps

def bound_part(n, c1, c2):
    mpo = [[[1,0],[0,0]]]*(n-1)

# Функция, вычисляющая все шиги эволюции
# def evolute(state, operator, steps_number, solver, initial_guess = "best", eps = 1e-6):
#     fig, ax = plt.subplots()
#     line1, = ax.plot(np.arange(1,steps_number+1), np.zeros(steps_number))
#     line2, = ax.plot(np.arange(1,steps_number+1), np.zeros(steps_number))
#     #text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center')

#     times_container = []
#     start_time = time.time()
#     if initial_guess == "best":
#         guess = state
#     elif isinstance(initial_guess, list):
#         if initial_guess[0] == "approx":
#             guess = mps_compression(state, initial_guess[1])
#         elif len(initial_guess) == len(state):
#             guess = initial_guess  
#         else:
#             raise ValueError("Invalid initial_guess list format")
#     else:
#         raise ValueError("Invalid initial_guess type")

#     list_for_matrices = []
#     list_for_matrices.append(state)
#     solution = solver(operator, state, guess, eps = eps)
#     list_for_matrices.append(solution)
#     times_container.append(time.time()-start_time)
#     for i in tqdm(range(steps_number-1)):
#         if initial_guess == "best":
#             guess = solution
#         elif isinstance(initial_guess, list):
#             if initial_guess[0] == "approx":
#                 guess = mps_compression(solution, initial_guess[1])
#             elif len(initial_guess) == len(state):
#                 guess = initial_guess 
#             else:
#                 raise ValueError("Invalid initial_guess list format")
#         else:
#             raise ValueError("Invalid initial_guess type")

#         solution = solver(operator, solution, guess, eps = eps)
#         list_for_matrices.append(solution)
#         times_container.append(time.time()-start_time)
#         poly_coeffs = np.polyfit(np.arange(1,i+3), times_container, 10)
#         poly_func = np.poly1d(poly_coeffs)
#         line1.set_ydata(times_container)
#         line2.set_ydata(poly_func)
#         display(fig)
#         clear_output(wait=True)
#     return list_for_matrices

def get_bond(mps):
    shapes = [tensor.shape for tensor in mps]
    shapes = [number for tuple in shapes for number in tuple]
    return max(shapes)


def evolute(state, operator, steps_number, solver, initial_guess="best", eps=1e-6, chart = True, bond_limit = 1000, renormalize = True):
    energy = []
    bonds = []
    times_container = []
    norms = []
    exp_out = False
    epsilon = initial_guess[1]
    error = 0
    if chart == True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
        line1, = ax1.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), 'r.', label='Times')
        line2, = ax1.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label="Time's Fit")
        line3, = ax2.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label='Bond')
        line4, = ax3.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label="State's norm")
        text1 = ax1.text(0.5, 0.75, '', transform=ax1.transAxes, ha='center')
        text2 = ax2.text(0.2, 0.9, '', transform=ax2.transAxes, ha='center')
        text3 = ax3.text(0.2, 0.9, '', transform=ax3.transAxes, ha='center')
        ax1.grid(which = 'major', alpha = 0.6)
        ax1.grid(which = 'minor', alpha = 0.2)
        ax1.minorticks_on()
        ax2.grid(which = 'major', alpha = 0.6)
        ax2.grid(which = 'minor', alpha = 0.2)
        ax2.minorticks_on()
        ax3.grid(which = 'major', alpha = 0.6)
        ax3.grid(which = 'minor', alpha = 0.2)
        ax3.minorticks_on()
        ax1.legend()
        ax2.legend()
        ax3.legend()

    start_time = time.time()
    if initial_guess == "best":
        guess = state
    elif isinstance(initial_guess, list):
        if initial_guess[0] == "approx":
            guess = mps_compression(state, epsilon)
        elif initial_guess[0] == "adaptive":
            n1, n2 , x_bias, x_left, x_bond2, step1_v1, step2 = initial_guess[4][0], initial_guess[4][1], initial_guess[4][2], initial_guess[4][3], initial_guess[4][4], initial_guess[4][5], initial_guess[4][6] 
            exp_out = True
            guess = mps_compression(state, epsilon)
            pump_energy = calculate_photon_number3(state, 0, n1, x_bias-x_left, step1_v1)
            signal_energy = calculate_photon_number3(state, 1, n2, -x_bond2, step2)
            total_energy = 2*pump_energy + signal_energy
            energy.append((pump_energy, signal_energy, total_energy))
        elif len(initial_guess) == len(state):
            guess = initial_guess
        else:
            raise ValueError("Invalid initial_guess list format")
    else:
        raise ValueError("Invalid initial_guess type")

    list_for_matrices = []
    list_for_matrices.append(state)
    solution = solver(operator, state, guess, eps=eps, bond_limit=bond_limit, renormalize = renormalize)
    list_for_matrices.append(solution)
    norm = calculate_rc_mpsnorm(solution)

    norms.append(norm)
    times_container.append(time.time() - start_time)
    bonds.append(get_bond(solution))

    try:
        for i in range(2, steps_number + 1):
            if initial_guess == "best":
                guess = solution
            elif isinstance(initial_guess, list):
                if initial_guess[0] == "approx":
                    guess = mps_compression(solution, epsilon)
                elif initial_guess[0] == "adaptive":
                    number = steps_number // initial_guess[2]
                    guess = mps_compression(solution, epsilon)
                    if i % (number + 1) == 0:
                        pump_energy = calculate_photon_number3(solution, 0, n1, x_bias-x_left, step1_v1)
                        signal_energy = calculate_photon_number3(solution, 1, n2, -x_bond2, step2)
                        total_energy = 2*pump_energy + signal_energy
                        error = abs(energy[0][2] - total_energy)/energy[0][2]
                        if error < initial_guess[3]:
                            energy.append((pump_energy, signal_energy, total_energy))
                        else:
                            i = i - number
                            list_for_matrices = list_for_matrices[:-number]
                            solution = list_for_matrices[-1]
                            epsilon = epsilon/2
                            #print(epsilon)
                            guess = mps_compression(solution, epsilon)
                elif len(initial_guess) == len(state):
                    guess = initial_guess
                else:
                    raise ValueError("Invalid initial_guess list format")
            else:
                raise ValueError("Invalid initial_guess type")

            solution = solver(operator, solution, guess, eps=eps, bond_limit=bond_limit, renormalize = renormalize)
            list_for_matrices.append(solution)
            norm = np.round(calculate_rc_mpsnorm(solution), 2)

            times_container.append(time.time() - start_time)
            norms.append(norm)       
            bonds.append(get_bond(solution)) 
            if chart == True:
                cut_time = 10 + i//10
                if i < cut_time+1:
                    poly_coeffs = np.polyfit(np.arange(1, i + 1), times_container, 2)
                else:
                    poly_coeffs = np.polyfit(np.arange(i-(cut_time-1), i+1), times_container[i-cut_time:i], 2)
                poly_func = np.poly1d(poly_coeffs)
                line1.set_ydata(times_container + [0]*(steps_number-len(times_container)))
                line2.set_ydata(poly_func(np.arange(1, steps_number + 1)))
                line3.set_ydata(bonds + [0]*(steps_number-len(times_container)))
                line4.set_ydata(norms + [0]*(steps_number-len(times_container)))
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                ax3.relim()
                ax3.autoscale_view()
                last_time = np.round(times_container[-1]-times_container[-2])
                text1.set_text(f'Общее время выполнения {np.round(poly_func(steps_number))} секунды, \n Оставшееся время {np.round(poly_func(steps_number)-times_container[i-1])} секунды \n Начальное приб. {epsilon} \n Ошибка общ.эн. {error} \n Кол.вып.ит. {i} \n Время посл. ит. {last_time}')
                text2.set_text(f'Бонд {bonds[-1]}')
                text3.set_text(f'Норма \n состояния \n {norms[-1]}')
                text1.set_fontsize(8) 
                text2.set_fontsize(8) 
                text3.set_fontsize(8) 
                clear_output(wait=True)
                display(fig)
                plt.pause(0.001)  # Пауза для обновления графика
                plt.show()

    except KeyboardInterrupt:
        print("Цикл остановлен пользователем.")
    if exp_out:
        return list_for_matrices, energy
    else:
        return list_for_matrices

# Функция, вычисляющая количество фотонов в состояниях на всех шагах эволюции
def calculate_photon_number(set_of_solutions, n1, n2, x_bond1, x_bond2, step1, step2, x_bias, bound = False, step = 1):
    photon_number_in_mode1 = []
    photon_number_in_mode2 = []
    photon_number_common = []
    for i in tqdm(range(0, len(set_of_solutions), step)):
        reddm1 = find_reduced_density_matrix(set_of_solutions[i], 0, n1)
        reddm2 = find_reduced_density_matrix(set_of_solutions[i], 1, n2) 
        reddm1 = reddm1 / np.trace(reddm1)
        reddm2 = reddm2 / np.trace(reddm2)
        photon_number_in_mode1.append(np.round(calculate_ph_num(reddm1, n1, x_bias - x_bond1, step1, bound), 3))
        photon_number_in_mode2.append(np.round(calculate_ph_num(reddm2, n2, - x_bond2, step2, bound), 3))
        photon_number_common = list(map(add, photon_number_in_mode1, photon_number_in_mode2))
    return photon_number_in_mode1, photon_number_in_mode2, photon_number_common

def plot_photon_number_evolution(photon_num1, photon_num2, time_step, delta_t, title, text1, text2, xlim = False, ylim = False, sum_ph_num = True):    
    plt.figure(figsize = (12,6))
    plt.title(title, fontsize = 16)
    plt.plot(np.arange(len(photon_num1)) * delta_t * time_step, photon_num1, label = text1)
    plt.plot(np.arange(len(photon_num2)) * delta_t * time_step, photon_num2, label = text2)
    if sum_ph_num:
        plt.plot(np.arange(len(photon_num2)) * delta_t * time_step, list(map(add, list(map(add, photon_num1, photon_num1)), photon_num2)), label = "Sum")
    plt.xlabel("time, s", fontsize = 16)
    plt.ylabel("Photon number", fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(which = 'major', alpha = 0.6)
    plt.grid(which = 'minor', alpha = 0.2)
    if xlim != False:
        plt.xlim(xlim)
    if ylim != False:
        plt.ylim(ylim)
    plt.minorticks_on()

def photon_number_operator_mpo(n, x0, step):
    return mpo_mul_num(mpo_sum(mpo_multiplication(x_mpo(n, x0, step), x_mpo(n, x0, step)), mpo_mul_num(der2_mpo(n, step), -1)), 1/2)

def calculate_photon_number2(mps, position, length, x0, step):
    common_trace_list = []
    for i in range(len(mps)):
        common_trace = np.einsum('inj, knl -> ikjl', mps[i], np.conj(mps[i]), optimize=True)
        common_trace = common_trace.reshape(common_trace.shape[0]*common_trace.shape[1], common_trace.shape[2]*common_trace.shape[3])
        common_trace_list.append(common_trace)

    first_blood = common_trace_list[0]
    for i in range(len(mps)-1):
        first_blood = np.einsum('ab, bc -> ac',first_blood, common_trace_list[i+1], optimize=True)

    new_tensor = []
    mps_test = mps.copy()
    if position == 0:
        real_position = 1
    else:
        real_position = 0

    length2 = len(mps) - length
    frin = real_position * length
    frin2 = position * length2

    # Делаем свёртку тензоров MPS, которые хотим исключить из рассмотрения, по физическим индексам
    for i in range(frin, frin + length2):
        new_tensor.append(np.einsum('abc, ibk -> aick', mps_test[i], np.conj(mps_test[i])))
    first = new_tensor[0]
    # Сворачиваем эти же тензоры, но теперь по бонд-индексам
    for i in range(len(new_tensor) - 1):
        first = np.einsum('abcd, cdkl -> abkl', first, new_tensor[i + 1], optimize=True)
    sh = first.shape
    first = first.reshape(sh[0] * sh[1], sh[2] * sh[3])
        
    # Объединяем пары тензоров нужного MPS в один тензор
    new_tensor = []
    for i in range(frin2, frin2 + length):
        t = np.einsum('inj, kml -> iknmjl', mps_test[i], np.conj(mps_test[i]), optimize=True)
        sh = t.shape
        t = t.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4] * sh[5])
        new_tensor.append(t)

    # Выполняем свертку с оператором числа фотонов
    new_storage = []
    photon_number_operator_ = photon_number_operator_mpo(length, x0, step)
    for i in range(len(photon_number_operator_)):
        value = np.einsum('abcd, icjk -> aibjdk', new_tensor[i], photon_number_operator_[i], optimize=True)
        value = value.reshape(value.shape[0]*value.shape[1], value.shape[2], value.shape[3], value.shape[4]*value.shape[5])
        value = np.einsum('ijjl -> il', value, optimize=True)
        new_storage.append(value)
    
    second = new_storage[0]
    if position == 1:
        second = np.einsum('ab, bk -> ak', first, second, optimize=True)

    # Сворачиваем нужные mps по bond-ногам
    for i in range(length - 1):
        second = np.einsum('ab, bc -> ac', second, new_storage[i + 1])
    if position == 0:
        last = np.einsum('cd, dk -> ck', second, first)
    if position == 1:
        last = second
    return last.reshape(-1)/first_blood.reshape(-1)

def calculate_photon_number_in_set_of_solutions(set_of_solutions, n1, n2, x_bond1, x_bond2, step1, step2, x_bias, solver, bound = False, step = 1, chart = True):
    if chart == True:
        real_length = len(range(1, len(set_of_solutions)+1, step))
        fig, ax = plt.subplots()
        line1, = ax.plot(np.arange(1, real_length + 1), np.zeros(real_length), 'r.', label='Times')
        line2, = ax.plot(np.arange(1, real_length + 1), np.zeros(real_length), label='Poly Fit')
        text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center')
        ax.legend()

    times_container = []
    start_time = time.time()
    j = 0

    photon_number_in_mode1 = []
    photon_number_in_mode2 = []
    photon_number_common = []
    for i in tqdm(range(1, len(set_of_solutions)+1, step)):
        j += 1
        photon_number_in_mode1.append(np.round(solver(set_of_solutions[i-1], 0, n1, x_bias-x_bond1, step1), 3))
        photon_number_in_mode2.append(np.round(solver(set_of_solutions[i-1], 1, n2, -x_bond2, step2), 3))
        photon_number_common = list(map(add, photon_number_in_mode1, photon_number_in_mode2))

        times_container.append(time.time() - start_time)
        if chart == True:
            cut_time = 7
            if j < cut_time+1:
                poly_coeffs = np.polyfit(np.arange(1, j + 1), times_container, 2)
            else:
                poly_coeffs = np.polyfit(np.arange(j-(cut_time-1), j+1), times_container[j-cut_time:j], 2)
            
            poly_func = np.poly1d(poly_coeffs)  
            line1.set_ydata(times_container + [0]*(real_length-len(times_container)))
            line2.set_ydata(poly_func(np.arange(1, real_length + 1)))
            ax.relim()
            ax.autoscale_view()
            text.set_text(f'Общее время выполнения {np.round(poly_func(real_length))} секунды, \n Оставшееся время {np.round(poly_func(real_length)-times_container[j-1])} секунды')
            clear_output(wait=True)
            display(fig)
            plt.pause(0.001)  # Пауза для обновления графика
            plt.show()

    return photon_number_in_mode1, photon_number_in_mode2, photon_number_common

def calculate_photon_number3(mps, position, length, x0, step):
    l = len(mps)
    photon_number_op = photon_number_operator_mpo(length, x0, step)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[-1], np.conj(env_part[-1]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], np.conj(env_part[-i-1]), optimize=True)
    else:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[0], np.conj(env_part[0]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], np.conj(env_part[i]), optimize=True)

    # Свертка матрицы плотности состояния с оператором числа фотонов
    if position == 0:
        first_state_step = np.einsum("abc, ibjk, ejf -> aieckf", state_part[0], photon_number_op[0],  np.conj(state_part[0]), optimize=True)
        for i in range(1, len(state_part)):
            first_state_step = np.einsum("abcdef, djk, ejop, fow -> abckpw", first_state_step, state_part[i], photon_number_op[i], np.conj(state_part[i]), optimize=True)
    else:
        first_state_step = np.einsum("abc, ibjk, ejf -> aieckf", state_part[-1], photon_number_op[-1],  np.conj(state_part[-1]), optimize=True)
        for i in range(1, len(state_part)):
            first_state_step = np.einsum("abcdef, gja, hjob, woc -> ghwdef", first_state_step, state_part[-i-1], photon_number_op[-i-1], np.conj(state_part[-i-1]), optimize=True)

    # Вычисление следа матрицы плотности
    if position == 0:
        rho_trace_first_step = np.einsum("abc, ibk -> aick", state_part[0], np.conj(state_part[0]), optimize=True)
        for i in range(1, len(state_part)):
            rho_trace_first_step = np.einsum("abcd, cij, dif -> abjf", rho_trace_first_step, state_part[i], np.conj(state_part[i]), optimize=True)
    else:
        rho_trace_first_step = np.einsum("abc, ibk -> aick", state_part[-1], np.conj(state_part[-1]), optimize=True)
        for i in range(1, len(state_part)):
            rho_trace_first_step = np.einsum("abcd, ika, jkb -> ijcd", rho_trace_first_step, state_part[-i-1], np.conj(state_part[-i-1]), optimize=True)
    
    if position == 1:
        common_trace = np.einsum("abcd, cdij -> abij", first_env_step, rho_trace_first_step, optimize=True).reshape(-1)
        return np.einsum("abcd, cfdklm -> afbklm", first_env_step, first_state_step, optimize=True).reshape(-1)/common_trace
    else:
        common_trace = np.einsum("abcd, cdij -> abij", rho_trace_first_step, first_env_step, optimize=True).reshape(-1)
        return np.einsum("abcd, ijkahb -> ijkchd", first_env_step, first_state_step, optimize=True).reshape(-1)/common_trace
    
def find_reduced_density_matrix2(mps, position, length, n_compression):
    l = len(mps)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]
    state_compressed_part = state_part[:len(state_part)-n_compression]
    compressed_part = state_part[len(state_part)-n_compression:]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[-1], np.conj(env_part[-1]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], np.conj(env_part[-i-1]), optimize=True)
    else:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[0], np.conj(env_part[0]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], np.conj(env_part[i]), optimize=True)

    # Усреднение (угрубление) матрицы плотности состояния 
    for i in range(len(compressed_part)):
        compressed_part[i] = np.einsum("ijk -> ik", compressed_part[i], optimize=True)
        compressed_part[i] = np.einsum("ab, cd -> acbd", compressed_part[i], np.conj(compressed_part[i]), optimize=True)

    if position == 0:
        first_part = first_env_step
        for i in range(len(compressed_part)):
            first_part = np.einsum("abcd, cdjk -> abjk", compressed_part[-1-i], first_part, optimize=True)
        
        first_part = np.einsum("abc, ijk, ckgh ->aibjgh", state_compressed_part[-1], np.conj(state_compressed_part[-1]), first_part, optimize=True)
        for i in range(1, len(state_compressed_part)):
            first_part = np.einsum("abc, ijk, ckghml ->aibgjhml", state_compressed_part[-i-1], np.conj(state_compressed_part[-i-1]), first_part, optimize=True)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])
    
    else:
        first_part = np.einsum("abcd, cjk, dgh -> abjgkh", first_env_step, state_compressed_part[0], np.conj(state_compressed_part[0]), optimize=True)
        for i in range(1, len(state_compressed_part)):
            first_part = np.einsum("abcdef, ejk, fgh -> abcjdgkh", first_part, state_compressed_part[i], np.conj(state_compressed_part[i]), optimize=True)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        for i in range(len(compressed_part)):
            first_part = np.einsum("abcdef, efjk -> abcdjk", first_part, compressed_part[i], optimize=True)
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])
    
def find_reduced_density_matrix_for_3_modes(mps, position, length, length2, length3, n_compression):
    l = len(mps)
    if position == 0:
        state_part = mps[:length]
        env_part = mps[length:]
    elif position == 2:
        state_part = mps[l-length:]
        env_part = mps[:l-length]
    else:
        state_part = mps[length2:l-length3]
        env_part1 = mps[:length2]
        env_part2 = mps[l-length3:]

    state_compressed_part = state_part[:len(state_part)-n_compression]
    compressed_part = state_part[len(state_part)-n_compression:]

    # Свертка состояния окружения
    if position == 0:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[-1], np.conj(env_part[-1]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, ika, jkb -> ijcd", first_env_step, env_part[-i-1], np.conj(env_part[-i-1]), optimize=True)
    elif position == 2:
        first_env_step = np.einsum("abc, ibk -> aick", env_part[0], np.conj(env_part[0]), optimize=True)
        for i in range(1, len(env_part)):
            first_env_step = np.einsum("abcd, cij, dik -> abjk", first_env_step, env_part[i], np.conj(env_part[i]), optimize=True)
    else:
        first_env1_step = np.einsum("abc, ibk -> aick", env_part1[0], np.conj(env_part1[0]), optimize=True)
        for i in range(1, len(env_part1)):
            first_env1_step = np.einsum("abcd, cij, dik -> abjk", first_env1_step, env_part1[i], np.conj(env_part1[i]), optimize=True)

        first_env2_step = np.einsum("abc, ibk -> aick", env_part2[-1], np.conj(env_part2[-1]), optimize=True)
        for i in range(1, len(env_part2)):
            first_env2_step = np.einsum("abcd, ika, jkb -> ijcd", first_env2_step, env_part2[-i-1], np.conj(env_part2[-i-1]), optimize=True)

    # Усреднение (угрубление) матрицы плотности состояния 
    for i in range(len(compressed_part)):
        compressed_part[i] = np.einsum("ijk -> ik", compressed_part[i], optimize=True)
        compressed_part[i] = np.einsum("ab, cd -> acbd", compressed_part[i], np.conj(compressed_part[i]), optimize=True)

    if position == 0:
        first_part = first_env_step
        for i in range(len(compressed_part)):
            first_part = np.einsum("abcd, cdjk -> abjk", compressed_part[-1-i], first_part, optimize=True)
        
        first_part = np.einsum("abc, ijk, ckgh ->aibjgh", state_compressed_part[-1], np.conj(state_compressed_part[-1]), first_part, optimize=True)
        for i in range(1, len(state_compressed_part)):
            first_part = np.einsum("abc, ijk, ckghml ->aibgjhml", state_compressed_part[-i-1], np.conj(state_compressed_part[-i-1]), first_part, optimize=True)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])
    
    elif position == 2:
        first_part = np.einsum("abcd, cjk, dgh -> abjgkh", first_env_step, state_compressed_part[0], np.conj(state_compressed_part[0]), optimize=True)
        for i in range(1, len(state_compressed_part)):
            first_part = np.einsum("abcdef, ejk, fgh -> abcjdgkh", first_part, state_compressed_part[i], np.conj(state_compressed_part[i]), optimize=True)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        for i in range(len(compressed_part)):
            first_part = np.einsum("abcdef, efjk -> abcdjk", first_part, compressed_part[i], optimize=True)
        sh = first_part.shape
        return first_part.reshape(sh[2], sh[3])
    
    else:
        first_part = first_env2_step
        for i in range(len(compressed_part)):
            first_part = np.einsum("abcd, cdjk -> abjk", compressed_part[-1-i], first_part, optimize=True)

        first_part = np.einsum("abc, ijk, ckgh ->aibjgh", state_compressed_part[-1], np.conj(state_compressed_part[-1]), first_part, optimize=True)
        for i in range(1, len(state_compressed_part)):
            first_part = np.einsum("abc, ijk, ckghml ->aibgjhml", state_compressed_part[-i-1], np.conj(state_compressed_part[-i-1]), first_part, optimize=True)
            sh = first_part.shape
            first_part = first_part.reshape(sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6], sh[7])

        final_step = np.einsum("abcd, cdijkl -> abijkl", first_env1_step, first_part, optimize=True)
        return final_step.reshape(final_step.shape[2], final_step.shape[3])

def calculate_Wigner_function_for_set_of_solutions(set_of_solutions, rho_left_edge, rho_right_edge, q, p, number_of_modes, list_with_lengths, position, n_compression, step=1):
    wigner_function_storage = []
    if number_of_modes == 2:
        for i in tqdm(range(0, len(set_of_solutions), step)):
            density_matrix = find_reduced_density_matrix2(set_of_solutions[i], position, list_with_lengths[0], n_compression)
            density_matrix = density_matrix/np.trace(density_matrix)
            wigner_function_storage.append(wigner_function4(density_matrix, rho_left_edge, rho_right_edge, q, p, 1))

    elif number_of_modes == 3:
        for i in tqdm(range(0, len(set_of_solutions), step)):
            density_matrix = find_reduced_density_matrix_for_3_modes(set_of_solutions[i], position, list_with_lengths[0], list_with_lengths[1], list_with_lengths[2], n_compression)
            density_matrix = density_matrix/np.trace(density_matrix)
            wigner_function_storage.append(wigner_function4(density_matrix, rho_left_edge, rho_right_edge, q, p, 1))

    return wigner_function_storage

def annihilate_operator_mpo(n):
    first_tensor = np.array([1,0,0,1,0,0,1,0]).reshape(1,2,2,2)
    inter_tensor = np.array([1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0]).reshape(2,2,2,2)
    last_tensor = np.array([0,1,0,0,0,0,1,0]).reshape(2,2,2,1)
    return [first_tensor] + (n-2)*[inter_tensor] + [last_tensor]

def find_reduced_dm_mpo(mps, position, length):
    mpo = []
    l = len(mps)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]

    #if position == 0:
    #    env_tensor = np.einsum("ij, lj -> il", env_part[-1].reshape(env_part[-1].shape[0], env_part[-1].shape[1]), np.conj(env_part[-1].reshape(env_part[-1].shape[0], env_part[-1].shape[1])))
    #    for i in range(len(env_part)-1):
    #        g = 1

    for tensor in state_part:
        mpo_tensor = np.einsum("ijk, abc -> iajbkc", tensor, np.conj(tensor))
        sh = mpo_tensor.shape
        mpo_tensor = mpo_tensor.reshape(sh[0]*sh[1], sh[2], sh[3], sh[4]*sh[5])
        mpo.append(mpo_tensor)
    return mpo

# Эта функция работает некорректно
def compute_mps1_mpo_mps2(mps1, mpo, mps2):
    n = len(mps1)
    mps_1 = mps1.copy()
    mps_2 = mps2.copy()
    mpo_mpo = mpo.copy() 

    mps_1[-1] = mps_1[-1].reshape(mps_1[-1].shape[0], mps_1[-1].shape[1])
    mps_2[-1] = mps_2[-1].reshape(mps_2[-1].shape[0], mps_2[-1].shape[1])
    mpo_mpo[-1] = mpo_mpo[-1].reshape(mpo_mpo[-1].shape[0], mpo_mpo[-1].shape[1], mpo_mpo[-1].shape[2])
    first_layer = np.einsum('ij, njr, qr -> inq', mps_1[-1], mpo_mpo[-1], mps_2[-1], optimize=True)
    for i in range(2, n):
        new_layer = np.einsum('abc, ija -> ijbc', first_layer, mps_1[-i], optimize=True)
        new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, mpo_mpo[-i], optimize=True)
        new_layer = np.einsum('abcd, icd -> abi', new_layer, mps_2[-i], optimize=True)
        first_layer = new_layer
    return first_layer[0][0][0]

def contract_two_mps(mps1, mps2, output = "tensor"):
    mps_one = mps1.copy()
    mps_two = mps2.copy()

    begin = mps_one[0].shape[0] * mps_two[0].shape[0]
    end = mps_one[-1].shape[-1] * mps_two[-1].shape[-1]

    result = np.array(1).reshape(1,1)
    if begin <= end:
        for i in range(len(mps1)):
            result = np.einsum("ia, ijk, ajc -> kc", result, mps_one[i], mps_two[i], optimize = True)
    else:
        for i in range(len(mps1)):
            result = np.einsum("kc, ijk, ajc -> ia", result, mps_one[-i-1], mps_two[-i-1], optimize = True)
    if output == 'tensor':
        return result
    else:
        return result[0][0]

def contract_mpo_and_two_mps(mps1, mps2, mpo, output = 'tensor'):
    mps_one = mps1.copy()
    mps_two = mps2.copy()
    mpo_use = mpo.copy()

    begin = mps_one[0].shape[0] * mps_two[0].shape[0] * mpo_use[0].shape[0]
    end = mps_one[-1].shape[-1] * mps_two[-1].shape[-1] * mpo_use[-1].shape[-1]

    result = np.array(1).reshape(1,1,1)
    if begin <= end:
        for i in range(len(mps1)):
            result = np.einsum("abc, ajk, bjgh, cgm -> khm", result, mps_one[i], mpo_use[i], mps_two[i], optimize = True)
    else:
        for i in range(len(mps1)):
            result = np.einsum("abc, ija, gjhb, phc -> igp", result, mps_one[-i-1], mpo_use[-i-1], mps_two[-i-1], optimize = True)
    if output == 'tensor':
        return result
    else:
        return result[0][0][0]

def calculate_number_of_elements_in_MPS(mps):
    count = 0
    for tensor in mps:
        count += tensor.shape[0]*tensor.shape[1]*tensor.shape[2]
    return count

def get_bond(mps):
    shapes = [tensor.shape for tensor in mps]
    shapes = [number for tuple in shapes for number in tuple]
    return max(shapes)

# Функции для оптимизированного солвера

def compute_mps_layer_from_left_to_right(layer, tensors, optimize):
    new_layer = np.einsum('ab, aij -> jib', layer, tensors[0], optimize=optimize)
    new_layer = np.einsum('abc, cbj -> aj', new_layer, tensors[1], optimize=optimize)
    return new_layer

def compute_mps_layer_from_right_to_left(layer, tensors, optimize):
    new_layer = np.einsum('ab, ija -> ijb', layer, tensors[0], optimize=optimize)
    new_layer = np.einsum('abc, ibc -> ai', new_layer, tensors[1], optimize=optimize)
    return new_layer

def compute_mpo_layer_from_left_to_right(layer, tensors, optimize):
    new_layer = np.einsum('abc, ajk -> kjbc', layer, tensors[0], optimize=optimize)
    new_layer = np.einsum('abcd, cbij -> ajid', new_layer, tensors[1], optimize=optimize)
    new_layer = np.einsum('abcd, dci -> abi', new_layer, tensors[2], optimize=optimize)
    return new_layer

def compute_mpo_layer_from_right_to_left(layer, tensors, optimize):
    new_layer = np.einsum('abc, ija -> ijbc', layer, tensors[0], optimize=optimize)
    new_layer = np.einsum('abcd, ibjc -> aijd', new_layer, tensors[1], optimize=optimize)
    new_layer = np.einsum('abcd, icd -> abi', new_layer, tensors[2], optimize=optimize)
    return new_layer

def compute_reduced_A(left_layers, right_layers, mpo_1, mpo_2, optimize):
    tensor = np.einsum("abc, ijk, bopw, wefj -> aoeicpfk", left_layers, right_layers, mpo_1, mpo_2, optimize=optimize)
    shape = tensor.shape
    return tensor.reshape(shape[0]*shape[1]*shape[2]*shape[3], shape[4]*shape[5]*shape[6]*shape[7])

def compute_reduced_b(left_layers, right_layers, mps_1, mps_2, optimize):
    tensor = np.einsum("ab, jk, apq ,qdj -> bpdk", left_layers, right_layers, mps_1, mps_2, optimize=optimize)
    return tensor.reshape(-1)

def compute_pair_of_tensors(A, b, shape, direction, eps, bond_limit):
    solution = np.linalg.solve(A, b)

    if direction == "left_to_right":
        solution = solution.reshape(shape*2, -1)
    else:
        solution = solution.reshape(-1, shape*2)

    svd_solution = np.linalg.svd(solution, full_matrices = False)
    bond = min((svd_solution.S / svd_solution.S.sum() > eps ).sum(), bond_limit)

    if direction == "left_to_right":
        tensor_1 = svd_solution.U.reshape(shape, 2, -1)[:, :, :bond]
        tensor_2 = (np.diag(svd_solution.S[:bond]) @ svd_solution.Vh[:bond, :]).reshape(tensor_1.shape[-1], 2, -1)
    else:
        tensor_2 = svd_solution.Vh.reshape(-1, 2, shape)[:bond, :, :]
        tensor_1 = (svd_solution.U[:, :bond] @ np.diag(svd_solution.S[:bond])).reshape(-1, 2, tensor_2.shape[0])[:, :, :bond]
    return [tensor_1, tensor_2]

def sweep(mps_train_, b_mps, mpo_train, right_mps_layers_, right_mpo_layers_, eps, bond_limit, optimize):

    mps_train = mps_train_.copy()
    right_mps_layers = right_mps_layers_.copy()
    right_mpo_layers = right_mpo_layers_.copy()

    n = len(mps_train)

    left_mpo_layers = [np.ones((1,1,1))]
    left_mps_layers = [np.ones((1,1))]

    # Вычисление тензоров слева направо
    for i in range(n-1):
        #print(i)
        #print(left_mps_layers[i].shape, b_mps[i].shape, b_mps[i+1].shape, right_mps_layers[-1-i].shape)
        local_A = compute_reduced_A(left_mpo_layers[i], right_mpo_layers[-1-i], mpo_train[i], mpo_train[i+1], optimize)
        local_b = compute_reduced_b(left_mps_layers[i], right_mps_layers[-1-i], b_mps[i], b_mps[i+1], optimize)
        solution = compute_pair_of_tensors(local_A, local_b, left_mps_layers[i].shape[-1], "left_to_right", eps, bond_limit)
        mps_train[i] = solution[0]
        if i < n-2:
            left_mpo_layers.append(compute_mpo_layer_from_left_to_right(left_mpo_layers[-1], [np.conj(mps_train[i]), mpo_train[i], mps_train[i]], optimize))
            left_mps_layers.append(compute_mps_layer_from_left_to_right(left_mps_layers[-1], [b_mps[i], np.conj(mps_train[i])], optimize))
    mps_train[-1] = solution[1]

    right_mpo_layers = [np.ones((1,1,1))]
    right_mps_layers = [np.ones((1,1))]

    # Вычисление тензоров справа налево
    for i in range(n-1):
        #print(left_mps_layers[-i-1].shape, b_mps[-i-2].shape, b_mps[-i-1].shape, right_mps_layers[i].shape)
        local_A = compute_reduced_A(left_mpo_layers[-i-1], right_mpo_layers[i], mpo_train[-i-2], mpo_train[-i-1], optimize)
        local_b = compute_reduced_b(left_mps_layers[-i-1], right_mps_layers[i], b_mps[-i-2], b_mps[-i-1], optimize)
        solution = compute_pair_of_tensors(local_A, local_b, right_mps_layers[i].shape[-1], "right_to_left", eps, bond_limit)
        mps_train[-i-1] = solution[1]
        if i < n-2:
            right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [np.conj(mps_train[-i-1]), mpo_train[-i-1], mps_train[-i-1]], optimize))
            right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i-1], np.conj(mps_train[-i-1])], optimize))
    mps_train[0] = solution[0]

    return mps_train, right_mps_layers, right_mpo_layers

def initialize_layers(mps_train, b_mps, mpo_train, optimize):
    
    n = len(mps_train)
    
    right_mpo_layers = [np.ones((1,1,1))]
    right_mps_layers = [np.ones((1,1))]
    
    for i in range(1, n-1):
        right_mpo_layers.append(compute_mpo_layer_from_right_to_left(right_mpo_layers[-1], [np.conj(mps_train[-i]), mpo_train[-i], mps_train[-i]], optimize))    
        right_mps_layers.append(compute_mps_layer_from_right_to_left(right_mps_layers[-1], [b_mps[-i], np.conj(mps_train[-i])], optimize))
        
    return right_mpo_layers, right_mps_layers

def initialize_mps_and_mpo(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps, bond_limit):
    mps_train = mps_compression(ex_mps_initial_guess.mps.copy(), eps, bond_limit)
    mpo_train = ex_mpo.mpo.copy()
    b_mps = mps_compression(ex_mps_right_part.mps.copy(), eps, bond_limit)
    
    return mps_train, mpo_train, b_mps

def initialize_mps_and_mpo10(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps, bond_limit):
    mps_train = mps_compression(ex_mps_initial_guess.copy(), eps, bond_limit)
    mpo_train = ex_mpo.copy()
    b_mps = mps_compression(ex_mps_right_part.copy(), eps, bond_limit)
    
    return mps_train, mpo_train, b_mps

# def mp_solve5(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps=1e-6, optimize=True, bond_limit=1000, sweep_number=1, renormalize=True):
    
#     # Вычисление начальной нормы состояния
#     initial_norm = compute_norm(ex_mps_right_part.mps)
#     # Инициализация MPO и MPS из объектов
#     mps_train, mpo_train, b_mps = initialize_mps_and_mpo(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps, bond_limit)
#     # Первоначальная инициализация слоев сети
#     left_mpo_layers, right_mpo_layers, left_mps_layers, right_mps_layers = initialize_layers(mps_train, b_mps, mpo_train, optimize)
    
#     # Выполнение нескольких свипов вычисления решения
#     for i in range(sweep_number):
#         mps_train, right_mps_layers, right_mpo_layers = sweep(mps_train, b_mps, mpo_train, right_mps_layers, right_mpo_layers, eps, bond_limit, optimize)

#     if renormalize:
#         renormalize_mps(mps_train, initial_norm)
        
#     return MatrixProductState(mps_train)

def mp_solve10(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps=1e-9, optimize=True, bond_limit=1000, sweep_number=1, renormalize=True):
    
    # Вычисление начальной нормы состояния
    initial_norm = compute_norm(ex_mps_right_part)
    # Инициализация MPO и MPS из объектов
    mps_train, mpo_train, b_mps = initialize_mps_and_mpo10(ex_mpo, ex_mps_right_part, ex_mps_initial_guess, eps, bond_limit)
    # Первоначальная инициализация слоев сети
    right_mpo_layers, right_mps_layers = initialize_layers(mps_train, b_mps, mpo_train, optimize)
    
    # Выполнение нескольких свипов вычисления решения
    for i in range(sweep_number):
        mps_train, right_mps_layers, right_mpo_layers = sweep(mps_train, b_mps, mpo_train, right_mps_layers, right_mpo_layers, eps, bond_limit, optimize)

    if renormalize:
        renormalize_mps(mps_train, initial_norm)
        
    return mps_train

# Вспомогательные функции для функции evolute

def setup_plot(steps_number):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
    
    # Создание линий
    line1, = ax1.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), 'r.', label='Times')
    line2, = ax1.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label="Time's Fit")
    line3, = ax2.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label='Bond')
    line4, = ax3.plot(np.arange(1, steps_number + 1), np.zeros(steps_number), label="State's norm")
    
    # Создание текстовых элементов
    text1 = ax1.text(0.5, 0.75, '', transform=ax1.transAxes, ha='center')
    text2 = ax2.text(0.2, 0.9, '', transform=ax2.transAxes, ha='center')
    text3 = ax3.text(0.2, 0.9, '', transform=ax3.transAxes, ha='center')
    
    # Настройка сетки для всех осей
    for ax in [ax1, ax2, ax3]:
        ax.grid(which='major', alpha=0.6)
        ax.grid(which='minor', alpha=0.2)
        ax.minorticks_on()
        ax.legend()
    
    return fig, (ax1, ax2, ax3), (line1, line2, line3, line4), (text1, text2, text3)

import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

def update_plot(i, times_container, bonds, norms, steps_number, lines, axes, texts, 
                epsilon, fig, cut_time_base=10):

    # Вычисление cut_time
    cut_time = cut_time_base + i // 10
    
    # Полиномиальная аппроксимация
    if i < cut_time + 1:
        poly_coeffs = np.polyfit(np.arange(1, i + 1), times_container, 2)
    else:
        poly_coeffs = np.polyfit(np.arange(i - (cut_time - 1), i + 1), 
                                times_container[i - cut_time:i], 2)
    
    poly_func = np.poly1d(poly_coeffs)
    
    # Обновление данных на графиках
    lines[0].set_ydata(times_container + [0] * (steps_number - len(times_container)))
    lines[1].set_ydata(poly_func(np.arange(1, steps_number + 1)))
    lines[2].set_ydata(bonds + [0] * (steps_number - len(times_container)))
    lines[3].set_ydata(norms + [0] * (steps_number - len(times_container)))
    
    # Автомасштабирование осей
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    
    # Вычисление времени последней итерации
    if len(times_container) > 1:
        last_time = np.round(times_container[-1] - times_container[-2])
    else:
        last_time = 0
    
    # Обновление текстовой информации
    total_time_estimate = np.round(poly_func(steps_number))
    remaining_time = np.round(total_time_estimate - times_container[i - 1]) if i > 0 else 0
    
    texts[0].set_text(
        f'Общее время выполнения {total_time_estimate} секунды\n'
        f'Оставшееся время {remaining_time} секунды\n'
        f'Начальное приб. {epsilon}\n'
        f'Кол.вып.ит. {i}\n'
        f'Время посл. ит. {last_time}'
    )
    texts[1].set_text(f'Бонд {bonds[-1]}')
    texts[2].set_text(f'Норма\nсостояния\n{norms[-1]}')
    
    # Установка размера шрифта
    for text in texts:
        text.set_fontsize(8)
    
    # Обновление отображения
    clear_output(wait=True)
    display(fig)
    plt.pause(0.001)  # Пауза для обновления графика
    
    return poly_func, total_time_estimate, remaining_time, last_time

def setup_guess(initial_guess, state, epsilon, bond_limit):
    if initial_guess == "best":
        guess = state
        return guess
    elif isinstance(initial_guess, list):
        if initial_guess[0] == "approx":
            guess = mps_compression(state, epsilon, bond_limit)
            return guess
        else:
            raise ValueError("Invalid initial_guess list format")
    else:
        raise ValueError("Invalid initial_guess type")
    
def evolute10(state, operator, steps_number, solver, initial_guess="best", eps=1e-6, chart=True, bond_limit=1000, sweep_number=1, renormalize=True, way=False):
    list_for_matrices = [state]
    times_container = []
    norms = []
    bonds = []

    epsilon = initial_guess[1]

    start_time = time.time()

    if chart == True:
        fig, axes, lines, texts = setup_plot(steps_number)
    
    solution = state
    try:
        for i in range(1, steps_number + 1):
            guess = setup_guess(initial_guess, solution, epsilon, bond_limit)
            solution = solver(operator, solution, guess, eps=eps, bond_limit=bond_limit, sweep_number = sweep_number, renormalize = renormalize)

            list_for_matrices.append(solution)
            times_container.append(time.time() - start_time)
            norms.append(np.round(compute_norm(solution), 2))       
            bonds.append(get_bond(solution)) 

            if chart == True:
                update_plot(i, times_container, bonds, norms, steps_number, lines, axes, texts, epsilon, fig, cut_time_base=10)

    except KeyboardInterrupt:
        print("Цикл остановлен пользователем.")

    if isinstance(way, str):
        with open(way+'solutions.pkl', 'wb') as f:
            pickle.dump(list_for_matrices, f)
    
    return list_for_matrices