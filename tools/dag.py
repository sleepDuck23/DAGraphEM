import numpy as np
import torch

def numpy_to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

def logdet_dag(A):
    # A∘A = elementwise square, I = identity
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    mat = I - A * A
    # To ensure numerical stability add small jitter
    jitter = 1e-6 * I
    mat = mat + jitter
    # logdet can be computed with torch.slogdet for stability
    sign, logdet = torch.slogdet(mat)
    if (sign <= 0).any():
        # handle non-positive definite case
        return torch.tensor(float('-inf'))
    return logdet

def compute_loss(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5):
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * torch.trace(Q @ inside)

    # L1 norm
    f2 = lambda_reg * torch.norm(A, p=1)

    # logdet penalty
    #h = -alpha * logdet_dag(A)
    h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]

    return f1 + f2 + h 

def compute_new_loss(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * torch.trace(Q @ inside)

    # L1 norm
    f2 = lambda_reg * torch.sum(torch.sqrt(A**2 + delta**2))

    # logdet penalty
    h = -alpha * logdet_dag(A)
    #h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]

    return f1 + f2 + h 