import numpy as np

def reshape_my_A(A):
    Nx = A.shape[0]

    if A.shape[0] != A.shape[1]:
      raise ValueError("Input matrix A must be square.")
    if Nx % 2 != 0:
      raise ValueError("The number of rows/columns of A must be even.")

    Nx_2 = Nx // 2  # Integer division

    A_11 = A[:Nx_2, :Nx_2]
    A_12 = A[:Nx_2, Nx_2:]
    A_21 = A[Nx_2:, :Nx_2]
    A_22 = A[Nx_2:, Nx_2:]

    A_block = np.zeros((Nx_2**2, 4))

    A_block[:, 0] = A_11.flatten()
    A_block[:, 1] = A_12.flatten()
    A_block[:, 2] = A_21.flatten()
    A_block[:, 3] = A_22.flatten()

    return A_block

def reshape_my_A_block(A_block):
    Ntemp = A_block.shape[0]
    Nx_2_squared = int(Ntemp)  

    Nx_2_float = np.sqrt(Nx_2_squared)
    if Nx_2_float != int(Nx_2_float):
      raise ValueError("The number of rows in A_block must be a perfect square (Nx_2^2).")

    Nx_2 = int(Nx_2_float)
    Nx = 2 * Nx_2

    A = np.zeros((Nx, Nx))

    A[:Nx_2, :Nx_2] = A_block[:, 0].reshape((Nx_2, Nx_2))
    A[:Nx_2, Nx_2:] = A_block[:, 1].reshape((Nx_2, Nx_2))
    A[Nx_2:, :Nx_2] = A_block[:, 2].reshape((Nx_2, Nx_2))
    A[Nx_2:, Nx_2:] = A_block[:, 3].reshape((Nx_2, Nx_2))

    return A

def Recover_short_A(A_long):
    Nx = A_long.shape[0]
    Nx_p = A_long.shape[1]

    if Nx_p % Nx != 0:
      raise ValueError("The number of columns of A_long (Nx_p) must be divisible by the number of rows (Nx).")

    p = Nx_p // Nx  

    A = np.zeros((Nx, Nx, p))

    for l in range(p):
      A[:, :, l] = A_long[:, Nx * l:Nx * (l + 1)]

    return A


def Extend_matA(A):
    p = A.shape[2]
    Nx = A.shape[0]

    A_row = np.concatenate([A[:, :, i] for i in range(p)], axis=1)

    lower_block_left = np.eye(Nx * (p - 1))
    lower_block_right = np.zeros((Nx * (p - 1), Nx))

    lower_block = np.concatenate((lower_block_left, lower_block_right), axis=1)

    A_ex = np.concatenate((A_row, lower_block), axis=0)

    return A_ex


def Create_long_A(A):
    Nx = A.shape[0]
    p = A.shape[2]

    A_long = np.zeros((Nx, Nx * p))

    for l in range(p):
      A_long[:, Nx * l:Nx * (l + 1)] = A[:, :, l]

    return A_long

def calError(trueMat, predictedMat):
    if trueMat.shape != predictedMat.shape:
        raise ValueError("Input matrices must have the same shape.")

    TP = np.sum(trueMat & predictedMat)        # Both True
    TN = np.sum(~trueMat & ~predictedMat)       # Both False
    FP = np.sum(~trueMat & predictedMat)        # True is False, Predicted is True
    FN = np.sum(trueMat & ~predictedMat)        # True is True, Predicted is False

    return TP, FP, TN, FN