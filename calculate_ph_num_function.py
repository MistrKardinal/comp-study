import numpy as np 

def calculate_photon_number3(mps, position, length, x0, step):
    l = len(mps)
    photon_number_op = photon_number_operator_mpo(length, x0, step)
    state_part = mps[position*(l - length):length*(1-position)+l*position]
    env_part = mps[(1-position)*length:(l-length)*position+l*(1-position)]

    # Свертка состояния окружения
    first_env_step = np.einsum("abc, ibk - > aick", env_part[0], np.conj(env_part[0]))
    for i in range(1, len(env_part)):
        first_env_step = np.einsum("abcd, cij, dif -> abjf", first_env_step, env_part[i], np.conj(env_part[i]))

    # Свертка матрицы плотности состояния с оператором числа фотонов
    first_state_step = np.einsum("abc, ibjk, ejf - > aieckf", state_part[0], photon_number_op[0],  np.conj(state_part[0]))
    for i in range(1, len(state_part)):
        first_state_step = np.einsum("abcdef, djk, ejop, fow -> abckpw", first_state_step, state_part[i], photon_number_op[i], np.conj(state_part[i]))

    # Вычисление следа матрицы плотности
    rho_trace_first_step = np.einsum("abc, ibk - > aick", state_part[0], np.conj(state_part[0]))
    for i in range(1, len(state_part)):
        rho_trace_first_step = np.einsum("abcd, cij, dif -> abjf", rho_trace_first_step, state_part[i], np.conj(state_part[i]))
    
    if position == 1:
        common_trace = np.einsum("abcd, cdij -> abij", first_env_step, rho_trace_first_step).reshape(-1)
        return np.einsum("abcd, cfdklm -> afbklm", first_env_step, first_state_step).reshape(-1)/common_trace
    else:
        common_trace = np.einsum("abcd, cdij -> abij", rho_trace_first_step, first_env_step).reshape(-1)
        return np.einsum("abcd, ijkahb -> ijkchd", first_env_step, first_state_step).reshape(-1)/common_trace
