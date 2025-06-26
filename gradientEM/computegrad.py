import numpy as np

from gradientEM import comutmatrix
from tools.EM import Kalman_update
from tools.loss import Compute_PhiK
from tools.dag import grad_desc_penalty

def gradient_A_mu(grad_A_mu_in, grad_A_sig, A, H, Sk, vk, Kk, xk_mean_new):
  Sk_inv = np.linalg.pinv(Sk)
  Nx = A.shape[0]  # Get the number of rows of A

  wk = Sk_inv @ vk
  Jk = A @ Kk
  Lk = A - Jk @ H

  term1 = grad_A_mu_in @ Lk.T

  term2 = grad_A_sig @ np.kron(H.T @ wk, Lk.T)

  term3 = np.kron(xk_mean_new, np.eye(Nx))

  grad_A_mu_new = term1 + term2 + term3

  return grad_A_mu_new


def gradient_A_sig(grad_A_sig_in, A, H, Pk_minus, Kk):
  """
  Computes the updated gradient of A with respect to sigma (covariance).

  Args:
    grad_A_sig_in (numpy.ndarray): The previous gradient of A with respect to sigma.
    A (numpy.ndarray): The state transition matrix.
    H (numpy.ndarray): The observation matrix.
    Pk_minus (numpy.ndarray): The a priori error covariance matrix.
    Kk (numpy.ndarray): The Kalman gain.

  Returns:
    numpy.ndarray: The updated gradient of A with respect to sigma.
  """

  Jk = A @ Kk
  Lk = A - Jk @ H
  Nx = A.shape[0]  # Get the number of rows of A

  Kom = comutmatrix(Nx, Nx)

  Nm = (np.eye(Nx**2) + Kom) / 2

  term1 = grad_A_sig_in @ np.kron(Lk.T, Lk.T)

  term2 = 2 * np.kron(Pk_minus @ Lk.T, np.eye(Nx)) @ Nm

  grad_A_sig_new = term1 + term2

  return grad_A_sig_new

def gradient_A_phik(grad_A_mu, H, vk, Sk, grad_A_sig):
    # Compute pseudo-inverse of Sk
    Sk_inv = np.linalg.pinv(Sk)
    
    # Compute wk and Mk
    wk = Sk_inv @ vk
    Mk = np.outer(wk, wk) - Sk_inv
    
    # Compute temp = H' * Mk * H
    temp = H.T @ Mk @ H
    
    # Compute grad_A_phik
    grad_A_phik = grad_A_mu @ H.T @ wk + 0.5 * grad_A_sig @ temp.flatten()
    
    return grad_A_phik

def compute_loss_gradient(A, Q, x, z0, P0, H, R, Nx, Nz, K,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    Kom = comutmatrix(Nx, Nx)
    Nm = (np.eye(Nx**2) + Kom) / 2

    # Initialize storage
    z_mean_kalman_em = np.zeros((Nz, K))
    P_kalman_em = np.zeros((Nz, Nz, K))
    yk_kalman_em = np.zeros((Nx, K))
    Sk_kalman_em = np.zeros((Nx, Nx, K))



    Pk_minus = np.zeros((Nx, Nx, K))
    Kk = np.zeros((Nx, Nz, K))

    grad_A_mu = np.kron(z0, np.eye(Nx))
    grad_A_sig = 2 * np.kron(P0 @ A.T, np.eye(Nx)) @ Nm
    grad_A_phik = np.zeros((Nx**2, K))

    # First step
    z_mean_kalman_em[:, 0], P_kalman_em[:, :, 0], yk_kalman_em[:, 0], Sk_kalman_em[:, :, 0], Pk_minus[:, :, 0], Kk[:, :, 0] = Kalman_update(
        x[:, 0], z0, P0, A, H, R, Q)

    grad_A_phik[:, 0] = gradient_A_phik(grad_A_mu, H, yk_kalman_em[:, 0], Sk_kalman_em[:, :, 0], grad_A_sig)
    

    grad_A_mu = gradient_A_mu(grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, 0], yk_kalman_em[:, 0], Kk[:, :, 0], z_mean_kalman_em[:, 0])
    grad_A_sig = gradient_A_sig(grad_A_sig, A, H, Pk_minus[:, :, 0], Kk[:, :, 0])

    # Loop over time steps
    for k in range(1, K):
        z_mean_kalman_em[:, k], P_kalman_em[:, :, k], yk_kalman_em[:, k], Sk_kalman_em[:, :, k], Pk_minus[:, :, k], Kk[:, :, k] = Kalman_update(
            x[:, k], z_mean_kalman_em[:, k-1], P_kalman_em[:, :, k-1], A, H, R, Q)

        grad_A_phik[:, k] = gradient_A_phik(grad_A_mu, H, yk_kalman_em[:, k], Sk_kalman_em[:, :, k], grad_A_sig)
        

        grad_A_mu = gradient_A_mu(grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, k], yk_kalman_em[:, k], Kk[:, :, k], z_mean_kalman_em[:, k])
        grad_A_sig = gradient_A_sig(grad_A_sig, A, H, Pk_minus[:, :, k], Kk[:, :, k])

        
    penalty, grad_penalty = grad_desc_penalty(A,lambda_reg,alpha,delta)

    phi = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em) + penalty

    dphiA = -np.reshape(np.sum(grad_A_phik, axis=1), (Nx, Nx)) + grad_penalty
  
    dphi = dphiA.flatten()

    print('+')
    return phi, dphi, dphiA