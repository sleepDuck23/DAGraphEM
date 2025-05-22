import numpy as np

from tools.loss import ComputeMaj_D1, Compute_Prior_D1
from tools.prox import prox_L1, prox_ML_D1, prox_stable

def Smoothing_update(zk_kal,Pk_kal,zk_smooth_past,Pk_smooth_past,A,H,R,Q):
    zk_minus = A @ zk_kal
    Pk_minus = A @ Pk_kal @ A.T + Q

    Gk = Pk_kal @ A.T @ np.linalg.pinv(Pk_minus)
    zk_smooth_new = zk_kal + Gk @ (zk_smooth_past - zk_minus)
    Pk_smooth_new = Pk_kal + Gk @ (Pk_smooth_past - Pk_minus) @ Gk.T

    return zk_smooth_new, Pk_smooth_new, Gk

def MLEM_update_truncated(Phi,C,Mask):
    N1, N2 = Mask.shape

    Mask_vec = Mask.flatten()

    Phi_Large = np.kron(Phi.T, np.eye(N1))

    Phi_Large_truncated = Phi_Large[Mask_vec == 1, :][:, Mask_vec == 1]

    c = C.flatten()

    c_truncated = c[Mask_vec == 1]

    d = np.zeros(N1 * N2)

    d[Mask_vec == 1] = np.linalg.inv(Phi_Large_truncated) @ c_truncated

    D = d.reshape(N1, N2)

    return D

def Kalman_update(y_k,xk_mean_past,Pk_past,A,H,R,Q):
    xk_minus = A @ xk_mean_past
    Pk_minus = A @ Pk_past @ A.T + Q

    vk = y_k - H @ xk_minus
    Sk = H @ Pk_minus @ H.T + R
    Kk = Pk_minus @ H.T @ np.linalg.inv(Sk)
    xk_mean_new = xk_minus + Kk @ vk
    Pk_new = Pk_minus - Kk @ Sk @ Kk.T

    return xk_mean_new, Pk_new, vk, Sk

def EM_parameters(x, z_mean_smooth, P_smooth, G_smooth, z_mean_smooth0, P_smooth0, G_smooth0):

    K = x.shape[1]
    Nx = x.shape[0]
    Nz = z_mean_smooth.shape[0]

    Sigma = np.zeros((Nz, Nz))
    Phi = np.zeros((Nz, Nz))
    B = np.zeros((Nx, Nz))
    C = np.zeros((Nz, Nz))
    D = np.zeros((Nx, Nx))

    for k in range(1, K):  
        Sigma += (1/K) * (P_smooth[:, :, k] + np.outer(z_mean_smooth[:, k], z_mean_smooth[:, k]))
        Phi += (1/K) * (P_smooth[:, :, k-1] + np.outer(z_mean_smooth[:, k-1], z_mean_smooth[:, k-1]))
        B += (1/K) * np.outer(x[:, k], z_mean_smooth[:, k])
        C += (1/K) * (P_smooth[:, :, k] @ G_smooth[:, :, k-1].T + np.outer(z_mean_smooth[:, k], z_mean_smooth[:, k-1]))
        D += (1/K) * np.outer(x[:, k], x[:, k])

    
    Sigma += (1/K) * (P_smooth[:, :, 0] + np.outer(z_mean_smooth[:, 0], z_mean_smooth[:, 0]))
    Phi += (1/K) * (P_smooth0 + np.outer(z_mean_smooth0, z_mean_smooth0))
    B += (1/K) * np.outer(x[:, 0], z_mean_smooth[:, 0])
    C += (1/K) * (P_smooth[:, :, 0] @ G_smooth0.T + np.outer(z_mean_smooth[:, 0], z_mean_smooth0))
    D += (1/K) * np.outer(x[:, 0], x[:, 0])

    return Sigma, Phi, B, C, D

#def EM_parameters(x, z_mean_smooth, P_smooth, G_smooth, z_mean_smooth0, P_smooth0, G_smooth0):
#    K = x.shape[1]
#    Nx = x.shape[0]
#    Nz = z_mean_smooth.shape[0]
#
#    Sigma = np.zeros((Nz, Nz))
#    Phi   = np.zeros((Nz, Nz))
#    B     = np.zeros((Nx, Nz))
#    C     = np.zeros((Nz, Nz))
#    D     = np.zeros((Nx, Nx))
#
#    for k in range(1, K):  # 1-based loop in MATLAB => starts at index 1
#        zk   = z_mean_smooth[:, k].reshape(-1, 1)
#        zk_1 = z_mean_smooth[:, k-1].reshape(-1, 1)
#        xk   = x[:, k].reshape(-1, 1)
#
#        Sigma += (1/K) * (P_smooth[:, :, k] + zk @ zk.T)
#        Phi   += (1/K) * (P_smooth[:, :, k-1] + zk_1 @ zk_1.T)
#        B     += (1/K) * (xk @ zk.T)
#        C     += (1/K) * (P_smooth[:, :, k] @ G_smooth[:, :, k-1].T + zk @ zk_1.T)
#        D     += (1/K) * (xk @ xk.T)
#
#    # Handle k=0 (MATLAB's k=1)
#    z0     = z_mean_smooth[:, 0].reshape(-1, 1)
#    z0_0   = z_mean_smooth0.reshape(-1, 1)
#    x0     = x[:, 0].reshape(-1, 1)
#
#    Sigma += (1/K) * (P_smooth[:, :, 0] + z0 @ z0.T)
#    Phi   += (1/K) * (P_smooth0 + z0_0 @ z0_0.T)
#    B     += (1/K) * (x0 @ z0.T)
#    C     += (1/K) * (P_smooth[:, :, 0] @ G_smooth0.T + z0 @ z0_0.T)
#    D     += (1/K) * (x0 @ x0.T)
#
#    return Sigma, Phi, B, C, D


def update_case_0(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    return C @ np.linalg.inv(0.5 * (Phi + Phi.T))

def update_case_2(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    gamma1 = reg['gamma1']
    temp = K / sigma_Q**2
    return temp * C @ np.linalg.inv(Phi * temp + gamma1 * np.eye(Phi.shape[0]))

def update_case_1(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    gamma1 = reg['gamma1']
    ItDR = 1000
    precision = 1e-4
    D1 = D10.copy()
    Y = D10.copy()
    obj = []

    for i in range(ItDR):
        D1 = prox_L1(1, gamma1, Y)
        V = prox_ML_D1(C, Phi, sigma_Q, 1, 2 * D1 - Y, K)
        Y = Y + V - D1

        obj_val = ComputeMaj_D1(sigma_Q, D1, Sigma, Phi, C, K) + Compute_Prior_D1(D1, reg)
        obj.append(obj_val)

        if i > 0 and abs(obj[i] - obj[i - 1]) <= precision and obj[i] < Maj_D1:
            break

    return D1

def update_case_13(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    gamma1 = reg['gamma1']
    ItDR = 1000
    precision = 1e-4
    X = D10.copy()
    V = D10.copy()
    lam = 0.49
    gam = (1 - lam) * 0.99
    obj = []

    for i in range(ItDR):
        Y1 = X - gam * V
        Y2 = V + gam * X
        P1 = prox_L1(gam, gamma1, Y1)
        temp = prox_ML_D1(C, Phi, sigma_Q, 1 / gam, Y2 / gam, K)
        P2 = Y2 - gam * temp
        Q1 = P1 - gam * P2
        Q2 = P2 + gam * P1
        X = X - Y1 + Q1
        V = V - Y2 + Q2

        obj_val = ComputeMaj_D1(sigma_Q, P1, Sigma, Phi, C, K) + Compute_Prior_D1(P1, reg)
        obj.append(obj_val)

        if i > 0 and abs(obj[i] - obj[i - 1]) <= precision and obj[i] < Maj_D1:
            break

    return P1

def update_case_113(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    gamma1 = reg['gamma1']
    eta = 0.99
    ItDR = 1000
    precision = 1e-4
    X = D10.copy()
    V1 = D10.copy()
    V2 = D10.copy()
    lam = 0.99 * (1 / 3)
    gam = (1 - lam) * 0.99
    obj = []

    for i in range(ItDR):
        Y1 = X - gam * (V1 + V2)
        Y21 = V1 + gam * X
        Y22 = V2 + gam * X

        P1 = prox_L1(gam, gamma1, Y1)
        temp = prox_ML_D1(C, Phi, sigma_Q, 1 / gam, Y21 / gam, K)
        P21 = Y21 - gam * temp
        temp = prox_stable(Y22 / gam, eta)
        P22 = Y22 - gam * temp

        Q1 = P1 - gam * (P21 + P22)
        Q21 = P21 + gam * P1
        Q22 = P22 + gam * P1

        X = X - Y1 + Q1
        V1 = V1 - Y21 + Q21
        V2 = V2 - Y22 + Q22

        obj_val = ComputeMaj_D1(sigma_Q, P1, Sigma, Phi, C, K) + Compute_Prior_D1(P1, reg)
        obj.append(obj_val)

        if i > 0 and abs(obj[i] - obj[i - 1]) <= precision and obj[i] < Maj_D1:
            break

    return P1

def update_case_3(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    Mask = reg['Mask']
    return MLEM_update_truncated(Phi, C, Mask)

def update_case_4(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    eta = 0.99
    ItDR = 1000
    precision = 1e-4
    D1 = D10.copy()
    Y = D10.copy()
    obj = []

    for i in range(ItDR):
        D1 = prox_stable(Y, eta)
        V = prox_ML_D1(C, Phi, sigma_Q, 1, 2 * D1 - Y, K)
        Y = Y + V - D1

        obj_val = ComputeMaj_D1(sigma_Q, D1, Sigma, Phi, C, K)
        obj.append(obj_val)

        if i > 0 and abs(obj[i] - obj[i - 1]) <= precision and np.linalg.norm(D1) <= eta:
            break

    return D1

def GRAPHEM_update(Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1):
    reg1 = reg.get('reg1', 0)

    dispatch = {
        0: update_case_0,
        2: update_case_2,
        1: update_case_1,
        13: update_case_13,
        113: update_case_113,
        3: update_case_3,
        4: update_case_4,
    }

    if reg1 in dispatch:
        return dispatch[reg1](Sigma, Phi, C, K, sigma_Q, reg, D10, Maj_D1)
    else:
        raise ValueError(f"Unsupported reg1 value: {reg1}")

