import numpy as np
from numpy.linalg import norm, svd

from tools.matrix import reshape_my_A

def norm_21(W):
    nve = np.sqrt(np.sum(W**2, axis=1))
    F = np.sum(nve)
    return F

def ComputeMaj_D1(sigma_Q, D1, Sigma, Phi, C, K):
    term = (1 / (sigma_Q**2)) * (Sigma - C @ D1.T - D1 @ C.T + D1 @ Phi @ D1.T)
    Maj_D1 = (K / 2) * np.trace(term)
    return Maj_D1

def ComputeMaj(z0, P0, Q, R, z_mean_smooth0, P_smooth0, D1, D2, Sigma, Phi, B, C, D, K):
    det_P0 = np.linalg.det(2 * np.pi * P0)
    det_Q = np.linalg.det(2 * np.pi * Q)
    det_R = np.linalg.det(2 * np.pi * R)
    term3 = Sigma - C @ D1.T - D1 @ C.T + D1 @ Phi @ D1.T
    term4 = D - B @ D2.T - D2 @ B.T + D2 @ Sigma @ D2.T

    Maj1 = 0.5 * np.log(det_P0) + (K / 2) * np.log(det_Q) + (K / 2) * np.log(det_R)
    Maj2 = 0.5 * np.trace(np.linalg.inv(P0) @ (P_smooth0 + np.outer(z_mean_smooth0 - z0, z_mean_smooth0 - z0))) 
    #Maj2 = 0.5 * np.trace(np.linalg.inv(P0) @ (P_smooth0 + (z_mean_smooth0 - z0) @ (z_mean_smooth0 - z0).T)) 
    Maj3 = (K / 2) * np.trace(np.linalg.inv(Q) @ term3)
    Maj4 = (K / 2) * np.trace(np.linalg.inv(R) @ term4)

    Maj = Maj1 + Maj2 + Maj3 + Maj4
    return Maj

def ComputeGrad_ML(K, sigma_Q, C, Phi, D1):
    g = K * (-1 / (sigma_Q**2) * C + (1 / (sigma_Q**2)) * D1 @ Phi)
    return g

def Compute_PhiK(Phi0, Sk_kal, yk_kal):
    K = Sk_kal.shape[2]
    PhiK = Phi0  

    for k in range(K):
      Sk_k = Sk_kal[:, :, k]
      yk_k = yk_kal[:, k]

      PhiK = PhiK + 0.5 * np.log(np.linalg.det(2 * np.pi * Sk_k)) + 0.5 * yk_k.T @ np.linalg.inv(Sk_k) @ yk_k

    return PhiK

def Compute_Prior_D1(D1, reg):
    reg1_type = reg.get('reg1')
    gamma1 = reg.get('gamma1', 0)

    reg_functions = {
        0: lambda d, g: 0,
        1: lambda d, g: g * np.sum(np.abs(d)),
        10: lambda d, g: g * np.sum(np.abs(d)),
        13: lambda d, g: g * np.sum(np.abs(d)),
        100: lambda d, g: g * np.sum(np.abs(d)),
        113: lambda d, g: g * np.sum(np.abs(d)),
        2: lambda d, g: (g / 2) * norm(d, 'fro')**2,
        #11: lambda d, g: g * np.sum(np.abs(d)) + fun_barriers(svd(d), 1e-3, 0, 1),             #fun_barriers is Not defined even on matlab code (investigate this function)
        3: lambda d, g: 0,
        4: lambda d, g: 0,
        21: lambda d, g: g * norm_21(reshape_my_A(d)),
        210: lambda d, g: g * norm_21(reshape_my_A(d)),
        2100: lambda d, g: g * norm_21(reshape_my_A(d)),
        213: lambda d, g: g * norm_21(reshape_my_A(d)),
        5: lambda d, g: g * np.sum(np.abs(d)),
        50: lambda d, g: g * np.sum(np.abs(d)),
        6: lambda d, g: 0,
    }

    Reg1 = reg_functions[reg1_type](D1, gamma1)

    return Reg1