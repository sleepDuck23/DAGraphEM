import torch
import numpy as np

def commutmatrix_torch(m, n):
    """
    Computes the commutation matrix K of size (m*n) x (m*n) using PyTorch.
    """
    # Create index matrix like Fortran order (column-major)
    I = torch.arange(1, m * n + 1).reshape(n, m).t().flatten() - 1  # 0-based index for PyTorch

    # Identity matrix
    Y = torch.eye(m * n)

    # Reorder rows
    Y = Y[I, :]

    # Transpose to get K
    K = Y.T
    return K

def commutmatrix(m, n):
    """
    Computes the commutation matrix K of size (m*n) x (m*n) using PyTorch.
    """
    # Create index matrix like Fortran order (column-major)
    I = np.arange(1, m * n + 1).reshape(n, m).t().flatten() - 1  # 0-based index for PyTorch

    # Identity matrix
    Y = np.eye(m * n)

    # Reorder rows
    Y = Y[I, :]

    # Transpose to get K
    K = Y.T
    return K
