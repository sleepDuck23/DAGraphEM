import numpy as np

def commutation_matrix(m, n):
  """
  Computes the commutation matrix K of size (m*n) x (m*n).

  Args:
    m (int): The number of rows in the conceptual matrix.
    n (int): The number of columns in the conceptual matrix.

  Returns:
    numpy.ndarray: The commutation matrix K.
  """
  # Initialize a matrix of indices of size (m, n)
  I = np.arange(1, m * n + 1).reshape(m, n, order='F') # order='F' for Fortran-like (column-major) reshaping

  # Transpose it
  I = I.T

  # Vectorize the required indices (flatten in row-major order by default)
  I = I.flatten()

  # Initialize an identity matrix
  Y = np.eye(m * n)

  # Re-arrange the rows of the identity matrix
  Y = Y[I, :]

  # Transpose Y to get K
  K = Y.T

  return K
