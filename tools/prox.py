import numpy as np
from scipy.linalg import schur, solve
import control 


from tools.matrix import Extend_matA

def Scaling_to_stabilize(A, bnd):
    p = A.shape[2]
    A_ex = Extend_matA(A)  # Extended matrix A
    a = bnd / np.max(np.abs(np.linalg.eigvals(A_ex)))
    A_stable = np.copy(A)  # Initialize A_stable as a copy of A
    for i in range(p):
      A_stable[:, :, i] = (a**(i+1)) * A[:, :, i]  

    A_ex_stable = Extend_matA(A_stable)  # Extended matrix A_stable
    return A_ex_stable, A_stable

def remove_bias(x_in, K, y, seuil):
    index_unb = np.where(np.abs(x_in) >= seuil)[0]
    K_unb = K[:, index_unb]
    x_out = np.copy(x_in)
    x_out[index_unb] = np.linalg.pinv(K_unb.T @ K_unb) @ K_unb.T @ y
    return x_out

def prox_stable(D, eta):
    #U, s, V = np.linalg.svd(D)
    #S = np.diag(np.minimum(s, eta))
    #Dprox = U @ S @ V.T
    U, s, Vh = np.linalg.svd(D, full_matrices=False)
    S_clipped = np.minimum(s, eta)
    Dprox = U @ np.diag(S_clipped) @ Vh
    return Dprox

def prox_ML_D1(C, Phi, sigma_Q, gamma, D1, K):
    temp = (gamma * K) / (sigma_Q**2)
    #D1prox = (temp * C + D1) @ np.linalg.pinv(temp * Phi + np.eye(Phi.shape[0]))
    # Build A and B
    A = temp * Phi + np.eye(Phi.shape[0])
    B = temp * C   + D1
    
    # Solve A X^T = B^T  =>  X = (solve(A, B.T)).T
    D1prox = np.linalg.solve(A, B.T).T
    return D1prox

def prox_L1plus(gamma, gammareg, D):
    temp = gamma * gammareg
    Dprox = np.sign(D) * np.maximum(0, np.abs(D) - temp)
    Dprox = np.maximum(Dprox, 0)
    return Dprox

def prox_L1(gamma, gammareg, D):
    temp = gamma * gammareg
    Dprox = np.sign(D) * np.maximum(0, np.abs(D) - temp)
    return Dprox

def prox_21(W, lambda_reg):
    Nw = W.shape[1]
    Norm_rows = np.sqrt(np.sum(W**2, axis=1))
    Norm_rows_large = np.tile(Norm_rows[:, np.newaxis], (1, Nw))
    X = W * np.maximum(1 - lambda_reg / Norm_rows_large, 0)
    return X

def lyap(A, B, C=None, E=None):
    """
    Solve continuous-time Lyapunov or Sylvester equations.
    """
    if C is None and E is None:
        # A*X + X*A' + B = 0
        X = sylv(A, None, B)
        if np.isrealobj(A) and np.isrealobj(B):
            X = np.real(X)
    elif C is not None and E is None:
        # A*X + X*B + C = 0
        X = sylv(A, B, C)
        if np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(C):
            X = np.real(X)
    else:
        # A*X*E' + E*X*A' + B = 0
        X = bartels_stewart(A, E, None, None, -B)
        if np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(E):
            X = np.real(X)
    return X

def sylv(A, B, C):
    """
    Solve the Sylvester matrix equation:
    A*X + X*B + C = 0
    or Lyapunov matrix equation when B is None.
    """
    m, n = C.shape
    ZA, TA = schur(A, output='complex')

    if B is None or np.allclose(B, A.T):
        ZB = ZA
        TB = TA.T
        solve_direction = 'backward'
    elif np.allclose(B, A.T.conj()):
        ZB = ZA.conj()
        TB = TA.T
        solve_direction = 'backward'
    else:
        ZB, TB = schur(B, output='complex')
        solve_direction = 'forward'

    F = ZA.T @ C @ ZB

    if np.allclose(TA, np.diag(np.diagonal(TA))) and np.allclose(TB, np.diag(np.diagonal(TB))):
        L = -1.0 / (np.diagonal(TA)[:, None] + np.diagonal(TB))
        return ZA @ (L * F) @ ZB.T

    Y = np.zeros_like(F, dtype=complex)
    idx = np.diag_indices(m)
    p = np.diagonal(TA)

    kk = range(n - 1, -1, -1) if solve_direction == 'backward' else range(n)

    for k in kk:
        rhs = F[:, k] + Y @ TB[:, k]
        TA[idx] = p + TB[k, k]
        Y[:, k] = solve(TA, -rhs)

    return ZA @ Y @ ZB.T

def bartels_stewart(A, B, C, D, E):
    """
    Solve generalised Sylvester equation:
    A*X*B^T + C*X*D^T = E
    If C and D are None, solves:
    A*X*B^T + B*X*A^T = -E
    """
    if C is None and D is None:
        C = B
        D = A

    UA, TA = schur(A, output='complex')
    UB, TB = schur(B, output='complex')

    F = UA.T @ E @ UB
    m, n = F.shape
    X = np.zeros((m, n), dtype=complex)

    for i in range(m):
        for j in range(n):
            denom = TA[i, i] * TB[j, j] + C[i, i] * D[j, j]
            X[i, j] = F[i, j] / denom

    return UA @ X @ UB.T