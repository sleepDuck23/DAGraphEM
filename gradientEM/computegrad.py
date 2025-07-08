import torch

from gradientEM.comutation import commutmatrix
from tools.EM import Kalman_update_torch
from tools.loss import Compute_PhiK
from tools.dag import grad_desc_penalty

def gradient_A_mu(grad_A_mu_in, grad_A_sig, A, H, Sk, vk, Kk, xk_mean_new):
    Sk_inv = torch.linalg.pinv(Sk)
    Nx = A.shape[0]

    wk = Sk_inv @ vk # Nx,1
    Jk = A @ Kk # Nx,Nx
    Lk = A - Jk @ H # Nx,Nx

    term1 = grad_A_mu_in @ Lk.T # Nx**2, Nx

    term2 = grad_A_sig @  torch.kron((H.T @ wk).view(-1, 1), Lk.T.contiguous()) # Nx**2, Nx

    term3 = torch.kron(xk_mean_new.view(-1, 1), torch.eye(Nx, device=A.device))  # Nx**2, Nx

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

  Jk = A @ Kk # Nx,Nx
  Lk = A - Jk @ H # Nx,Nx
  Nx = A.shape[0]  # Get the number of rows of A

  Kom = commutmatrix(Nx, Nx)

  Nm = (torch.eye(Nx**2) + Kom) / 2 # Nx**2,Nx**2

  term1 = grad_A_sig_in @ torch.kron(Lk.T, Lk.T) # Nx**2,Nx**2

  term2 = 2 * torch.kron(Pk_minus @ Lk.T, torch.eye(Nx)) @ Nm # Nx**2,Nx**2

  grad_A_sig_new = term1 + term2

  return grad_A_sig_new # Nx**2,Nx**2

def gradient_A_phik(grad_A_mu, H, vk, Sk, grad_A_sig):
    # Compute pseudo-inverse of Sk
    Sk_inv = torch.linalg.pinv(Sk) # Nx,Nx
    
    # Compute wk and Mk
    wk = Sk_inv @ vk # vk is (Nx,) in your use case # Nx,1
    Mk = torch.outer(wk, wk) - Sk_inv # torch.outer with 1D vector returns 2D matrix #Nx,Nx
    
    # Compute temp = H' * Mk * H
    temp = H.T @ Mk @ H # Nx,Nx
    
    # Compute grad_A_phik
    # 0.5 * grad_A_sig @ temp.flatten() is correct if temp.flatten() produces 1D
    grad_A_phik = grad_A_mu @ H.T @ wk + 0.5 * grad_A_sig @ temp.flatten()
    
    return grad_A_phik # Nx**2,1

def compute_loss_gradient(A, Q, x, z0, P0, H, R, Nx, Nz, K,lambda_reg=0.1,alpha=0.5,delta=1e-4):
    
    dtype = torch.float32
    
    A = A.to(dtype)
    Q = Q.to(dtype)
    x = x.to(dtype)
    z0 = z0.to(dtype)
    P0 = P0.to(dtype)
    H = H.to(dtype)
    R = R.to(dtype)

    Kom = commutmatrix(Nx, Nx)
    Nm = (torch.eye(Nx**2) + Kom) / 2

    print(f"size of Nm: {Nm}")

    # Initialize storage
    z_mean_kalman_em = torch.zeros((Nz, K))
    P_kalman_em = torch.zeros((Nz, Nz, K))
    yk_kalman_em = torch.zeros((Nx, K)) 
    Sk_kalman_em = torch.zeros((Nx, Nx, K))

    Pk_minus = torch.zeros((Nx, Nx, K))
    Kk = torch.zeros((Nx, Nz, K))

    grad_A_mu = torch.kron(z0, torch.eye(Nx)) 
    grad_A_sig = 2 * torch.kron(P0 @ A.T, torch.eye(Nx)) @ Nm
    grad_A_phik = torch.zeros((Nx**2, K))

    # First step
    z_mean_t0, P_t0, yk_t0, Sk_t0, Pk_minus_t0, Kk_t0 = Kalman_update_torch(
        x[:, 0].unsqueeze(1), z0, P0, A, H, R, Q)

    # Assign to storage:
    z_mean_kalman_em[:, 0] = z_mean_t0.squeeze()
    P_kalman_em[:, :, 0] = P_t0
    yk_kalman_em[:, 0] = yk_t0.squeeze() 
    Sk_kalman_em[:, :, 0] = Sk_t0
    Pk_minus[:, :, 0] = Pk_minus_t0
    Kk[:, :, 0] = Kk_t0

    # These gradient functions expect 1D vectors for yk and xk_mean_new (which are yk_kalman_em[:, k] and z_mean_kalman_em[:, k] respectively)

    print(f"shape sk: {Sk_kalman_em[:, :, 0].size()}")
    print(f"shape yk: {yk_kalman_em[:, 0].size()}")
    print(f"shape kk: {Kk[:, :, 0].size()}")
    print(f"shape z: {z_mean_kalman_em[:, 0].size()}")

    grad_A_phik[:, 0] = gradient_A_phik(grad_A_mu, H, yk_kalman_em[:, 0], Sk_kalman_em[:, :, 0], grad_A_sig)
    grad_A_mu = gradient_A_mu(grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, 0], yk_kalman_em[:, 0], Kk[:, :, 0], z_mean_kalman_em[:, 0])
    grad_A_sig = gradient_A_sig(grad_A_sig, A, H, Pk_minus[:, :, 0], Kk[:, :, 0])

    # Loop over time steps
    for k in range(1, K):
        # The z_mean_kalman_em[:, k-1] passed here is already 1D. Kalman_update_torch expects a column vector for state.
        # So, we need to unsqueeze it. This was missing in your original loop for the z_mean_kalman_em[:, k-1] input.
        # This is another crucial point for consistency.
        z_mean_tk, P_tk, yk_tk, Sk_tk, Pk_minus_tk, Kk_tk = Kalman_update_torch(
            x[:, k].unsqueeze(1), z_mean_kalman_em[:, k-1].unsqueeze(1), P_kalman_em[:, :, k-1], A, H, R, Q)

        # Assign back to storage, squeezing the column vectors to fit the 1D slices
        z_mean_kalman_em[:, k] = z_mean_tk.squeeze()
        P_kalman_em[:, :, k] = P_tk
        yk_kalman_em[:, k] = yk_tk.squeeze() # CRUCIAL CHANGE HERE TOO
        Sk_kalman_em[:, :, k] = Sk_tk
        Pk_minus[:, :, k] = Pk_minus_tk
        Kk[:, :, k] = Kk_tk

        grad_A_phik[:, k] = gradient_A_phik(grad_A_mu, H, yk_kalman_em[:, k], Sk_kalman_em[:, :, k], grad_A_sig)
        grad_A_mu = gradient_A_mu(grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, k], yk_kalman_em[:, k], Kk[:, :, k], z_mean_kalman_em[:, k])
        grad_A_sig = gradient_A_sig(grad_A_sig, A, H, Pk_minus[:, :, k], Kk[:, :, k])

        
    penalty, grad_penalty = grad_desc_penalty(A,lambda_reg,alpha,delta)

    phi = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em) + penalty

    dphiA = -torch.reshape(torch.sum(grad_A_phik, axis=1), (Nx, Nx)) + grad_penalty
  
    dphi = dphiA.flatten()

    print('+')
    return phi, dphi, dphiA