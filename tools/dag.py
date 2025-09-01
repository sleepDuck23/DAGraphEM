import numpy as np
import torch
import scipy.linalg as sla

def numpy_to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

def logdet_dag_torch(A):
    # A∘A = elementwise square, I = identity
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    mat = I - A * A
    # To ensure numerical stability add small jitter
    #jitter = 1e-6 * I
    #mat = mat + jitter
    # logdet can be computed with torch.slogdet for stability
    sign, logdet = torch.linalg.slogdet(mat)
    if (sign <= 0).any():
        # handle non-positive definite case
        return torch.tensor(float('-inf'))
    return logdet

def logdet_dag(A):
    # A∘A = elementwise square, I = identity
    I = np.eye(A.shape[0])
    mat = I - A * A
    # To ensure numerical stability add small jitter
    #jitter = 1e-6 * I
    #mat = mat + jitter
    # logdet can be computed with torch.slogdet for stability
    sign, logdet = np.linalg.slogdet(mat)
    if (sign <= 0).any():
        # handle non-positive definite case
        return float('-inf')
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
    h = -alpha * logdet_dag_torch(A)
    #h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]
    #print(h)

    return f1 + f2 + h 

def compute_loss_zero(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5):
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * torch.trace(Q @ inside)

    # L1 norm
    f2 = lambda_reg * torch.norm(A, p=1)
    
    
    return f1 + f2 

def compute_new_loss(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    print("Computing new loss")
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * torch.trace(Q @ inside)

    # L1 norm
    f2 = lambda_reg * torch.sum(torch.sqrt(A**2 + delta**2))

    # logdet penalty
    h = -alpha * logdet_dag_torch(A)
    #h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]

    return f1 + f2 + h 

def compute_new_loss_np(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * np.trace(Q @ inside)

    # L1 norm
    f2 = lambda_reg * np.sum(np.sqrt(A**2 + delta**2))

    # logdet penalty
    h = -alpha * logdet_dag(A)
    #h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]

    return f1 + f2 + h 

def grad_newloss(A,K,Q,Sigma,C,Phi,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    
    grad_f1 = 0.5 * K * (-(Q @ C) - (Q.T @ C) + (Q.T @ A @ Phi.T) + (Q @ A @ Phi)) 

    
    grad_f2 = lambda_reg * (A/(np.sqrt(A**2 + delta**2)))

    
    grad_h = 2 * alpha * A * (sla.inv(np.eye(A.shape[0]) - A*A)).T
    

    return grad_f1 + grad_f2 + grad_h 


def pipa_f1_h_loss(A,K,Q,Sigma,C,Phi,alpha=0.5):
    # f1: trace(Q^{-1} (Sigma - CA^T - AC^T + A Phi A^T))
    CA_T = C @ A.T
    AC_T = A @ C.T
    APhiA_T = A @ Phi @ A.T
    inside = Sigma - CA_T - AC_T + APhiA_T
    f1 = 0.5 * K * torch.trace(Q @ inside)

    # logdet penalty
    h = -alpha * logdet_dag_torch(A)
    #h = torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0]

    return f1 + h 

def pipa_f1_h_grad(A,K,Q,Sigma,C,Phi,alpha=0.5):
    
    f1_g = 0.5 * K * (-2*Q @ C + (Q @ A @ Phi.T + Q @ A @ Phi))

    M = torch.eye(A[0]) - A*A
    h_grad = 2 * M * torch.linalg.inv(M).T
    

    return f1_g + h_grad

def grad_desc_penalty_torch(A,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    # L1 norm
    f2 = lambda_reg * torch.sum(torch.sqrt(A**2 + delta**2)) 
    #f2 = lambda_reg * torch.sum((A**2)) #l2 

    # L1 norm gradient
    grad_f2 = lambda_reg * A/(torch.sqrt(A**2 + delta**2))
    #grad_f2 = lambda_reg * A * 2 #l2

    # logdet penalty
    #h = -alpha * logdet_dag_torch(A)
    h = alpha * (torch.trace(torch.linalg.matrix_exp(A*A)) - A.shape[0])

    # logdet gradient
    #grad_h = alpha * 2 * A * torch.linalg.inv((torch.eye(A.shape[0])- A*A).T)
    grad_h = alpha * 2 * A * torch.linalg.matrix_exp(A*A).T

    return f2 + h, grad_f2 + grad_h

def grad_desc_penalty(A,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    # L1 norm
    f2 = lambda_reg * np.sum(np.sqrt(A**2 + delta**2))

    # L1 norm gradient
    grad_f2 = lambda_reg * (A/(np.sqrt(A**2 + delta**2)))

    # logdet penalty
    h = -alpha * logdet_dag(A)

    # logdet gradient
    grad_h = alpha * 2 * A * (sla.inv(np.eye(A.shape[0])- A*A)).T 

    return f2 + h, grad_f2 + grad_h
