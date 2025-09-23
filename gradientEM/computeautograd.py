import torch
import numpy as np
import time

from gradientEM.comutation import commutmatrix, commutmatrix_torch
from tools.EM import Kalman_update_torch, Kalman_update
from tools.loss import Compute_PhiK_torch, Compute_PhiK
from tools.dag import grad_desc_penalty, grad_desc_penalty_torch
from tools.dag import numpy_to_torch


def gradient_A_mu_torch(grad_A_mu_in, grad_A_sig, A, H, Sk, vk, Kk, xk_mean_new):
    Sk_inv = torch.linalg.pinv(Sk)
    Nx = A.shape[0]

    wk = Sk_inv @ vk # Nx,1
    Jk = A @ Kk # Nx,Nx
    Lk = A - Jk @ H # Nx,Nx


    term1 = grad_A_mu_in @ Lk.T # Nx**2, Nx

    term2 = grad_A_sig @  torch.kron((H.T @ wk).reshape(-1, 1), Lk.T.contiguous()) # Nx**2, Nx

    term3 = torch.kron(xk_mean_new.reshape(-1, 1), torch.eye(Nx))  # Nx**2, Nx

    grad_A_mu_new = term1 + term2 + term3

    return grad_A_mu_new


def gradient_A_sig_torch(grad_A_sig_in, A, H, Pk_minus, Kk):
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

  Kom = commutmatrix_torch(Nx, Nx)

  Nm = (torch.eye(Nx**2) + Kom) / 2 # Nx**2,Nx**2

  term1 = grad_A_sig_in @ torch.kron(Lk.T, Lk.T) # Nx**2,Nx**2

  term2 = 2 * torch.kron(Pk_minus @ Lk.T, torch.eye(Nx)) @ Nm # Nx**2,Nx**2

  grad_A_sig_new = term1 + term2

  return grad_A_sig_new # Nx**2,Nx**2



def gradient_A_phik_torch(grad_A_mu, H, vk, Sk, grad_A_sig):

    if not torch.isfinite(Sk).all():
        print("Sk:", Sk)
        raise ValueError("Sk contains NaNs or Infs")

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



def compute_loss_gradient_torch(
    A, Q, x, z0, P0, H, R, Nx, Nz, K,
    lambda_reg=20, alpha=1, delta=1e-4
):
    """
    Compute the loss and its exact gradient w.r.t. matrix A.
    This version uses .reshape() and .contiguous() to avoid
    PyTorch view errors when passing gradients to LBFGS.
    """

    # Symmetric commutation projection
    Kom = commutmatrix_torch(Nx, Nx)
    Nm = (torch.eye(Nx**2) + Kom) / 2

    # Storage initialization
    z_mean_kalman_em = torch.zeros((Nz, K))
    P_kalman_em = torch.zeros((Nz, Nz, K))
    yk_kalman_em = torch.zeros((Nx, K))
    Sk_kalman_em = torch.zeros((Nx, Nx, K))
    Pk_minus = torch.zeros((Nx, Nx, K))
    Kk = torch.zeros((Nx, Nz, K))

    # Initial gradient states
    grad_A_mu = torch.kron(z0, torch.eye(Nx))
    grad_A_sig = 2 * torch.kron(P0 @ A.T, torch.eye(Nx)) @ Nm
    grad_A_phik = torch.zeros((Nx**2, K))

    # --- First Kalman step ---
    z_mean_t0, P_t0, yk_t0, Sk_t0, Pk_minus_t0, Kk_t0 = Kalman_update_torch(
        x[:, 0:1], z0, P0, A, H, R, Q
    )

    # Save results
    z_mean_kalman_em[:, 0] = z_mean_t0.squeeze()
    P_kalman_em[:, :, 0] = P_t0
    yk_kalman_em[:, 0] = yk_t0.squeeze()
    Sk_kalman_em[:, :, 0] = Sk_t0
    Pk_minus[:, :, 0] = Pk_minus_t0
    Kk[:, :, 0] = Kk_t0

    # Gradient updates for first step
    grad_A_phik[:, 0] = gradient_A_phik_torch(
        grad_A_mu, H, yk_kalman_em[:, 0], Sk_kalman_em[:, :, 0], grad_A_sig
    ).reshape(-1).contiguous()

    grad_A_mu = gradient_A_mu_torch(
        grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, 0],
        yk_kalman_em[:, 0], Kk[:, :, 0], z_mean_kalman_em[:, 0]
    )

    grad_A_sig = gradient_A_sig_torch(
        grad_A_sig, A, H, Pk_minus[:, :, 0], Kk[:, :, 0]
    )

    # --- Loop over time steps ---
    for k in range(1, K):
        x_k = x[:, k].reshape(-1, 1)
        z_mean_tk, P_tk, yk_tk, Sk_tk, Pk_minus_tk, Kk_tk = Kalman_update_torch(
            x_k, z_mean_kalman_em[:, k-1:k], P_kalman_em[:, :, k-1], A, H, R, Q
        )

        # Save results
        z_mean_kalman_em[:, k] = z_mean_tk.squeeze()
        P_kalman_em[:, :, k] = P_tk
        yk_kalman_em[:, k] = yk_tk.squeeze()
        Sk_kalman_em[:, :, k] = Sk_tk
        Pk_minus[:, :, k] = Pk_minus_tk
        Kk[:, :, k] = Kk_tk

        # Gradient updates
        grad_A_phik[:, k] = gradient_A_phik_torch(
            grad_A_mu, H, yk_kalman_em[:, k], Sk_kalman_em[:, :, k], grad_A_sig
        ).reshape(-1).contiguous()

        grad_A_mu = gradient_A_mu_torch(
            grad_A_mu, grad_A_sig, A, H, Sk_kalman_em[:, :, k],
            yk_kalman_em[:, k], Kk[:, :, k], z_mean_kalman_em[:, k]
        )

        grad_A_sig = gradient_A_sig_torch(
            grad_A_sig, A, H, Pk_minus[:, :, k], Kk[:, :, k]
        )

    # Regularization penalty and gradient
    penalty, grad_penalty = grad_desc_penalty_torch(A, lambda_reg, alpha, delta)


    # Loss: negative log-likelihood + penalty
    phi = Compute_PhiK_torch(0, Sk_kalman_em, yk_kalman_em) + penalty

    # Matrix gradient (make contiguous)
    dphiA = -torch.sum(grad_A_phik, axis=1).reshape(Nx, Nx).T + grad_penalty
    dphiA = dphiA.contiguous()

    # Flattened gradient for optimizers that expect 1D (optional)
    dphi = dphiA.flatten().contiguous()

    return phi, dphi, dphiA


def run_kalman_full_torch(A_kf, Q, x_data, z0, P0, H, R, Nx, Nz, K):
    """
    Fully differentiable Kalman filter over a time series.
    Returns all relevant KF outputs.
    PyTorch entries and outputs.
    """
    device = A_kf.device
    dtype = A_kf.dtype

    # Allocate tensors
    yk_kalman_em = torch.zeros((Nx, K), dtype=dtype, device=device)
    Sk_kalman_em = torch.zeros((Nx, Nx, K), dtype=dtype, device=device)
    z_mean_kalman_em = torch.zeros((Nz, K), dtype=dtype, device=device)
    Pk_minus = torch.zeros((Nz, Nz, K), dtype=dtype, device=device)
    Kk = torch.zeros((Nz, Nx, K), dtype=dtype, device=device)

    z_prev = z0
    P_prev = P0

    for k in range(K):
        x_k = x_data[:, k:k+1]

        z_new, P_new, vk, Sk, P_pred, K_gain = Kalman_update_torch(
            x_k, z_prev, P_prev, A_kf, H, R, Q
        )

        # Save results (no in-place on leaves)
        z_mean_kalman_em[:, k:k+1] = z_new
        yk_kalman_em[:, k:k+1] = H @ z_prev  # predicted observation
        Sk_kalman_em[:, :, k] = Sk
        Pk_minus[:, :, k] = P_pred
        Kk[:, :, k] = K_gain

        # Prepare for next iteration
        z_prev = z_new
        P_prev = P_new

    return {
        "yk": yk_kalman_em,
        "Sk": Sk_kalman_em,
        "Pk_minus": Pk_minus,
        "Kk": Kk,
        "z_mean": z_mean_kalman_em
    }


def compute_loss_gradient_v2_torch(
    A_current,
    A_kf,
    kf_results,
    z0, P0, H,
    Nx, Nz, K,
    lambda_reg=20, alpha=1, delta=1e-4
):
    
    print("Starting compute_loss_gradient_v2_torch...")

    yk_kalman_em = kf_results["yk"]
    Sk_kalman_em = kf_results["Sk"]
    Pk_minus = kf_results["Pk_minus"]
    Kk = kf_results["Kk"]
    z_mean_kalman_em = kf_results["z_mean"]

    # symmetric commutation projection
    Kom = commutmatrix_torch(Nx, Nx)
    Nm = (torch.eye(Nx**2) + Kom) / 2

    # initial gradients — initialize consistently with A_kf (the one used by KF)
    grad_A_mu = torch.kron(z0, torch.eye(Nx))                           # (Nx^2, Nx)
    grad_A_sig = 2 * torch.kron(P0 @ A_kf.T, torch.eye(Nx)) @ Nm                    # (Nx^2, Nx^2)
    grad_A_phik = torch.zeros((Nx**2, K))

    # recursion over stored time steps (use A_current when propagating gradient states)
    for k in range(K):
        yk = yk_kalman_em[:, k]
        Sk = Sk_kalman_em[:, :, k]
        Pk_m = Pk_minus[:, :, k]
        Kk_t = Kk[:, :, k]
        zmean = z_mean_kalman_em[:, k]

        # gradient contribution for time k
        # gradient_A_phik returns shape (Nx^2, 1) in your earlier code
        tgrad_start = time.perf_counter()
        grad_A_phik[:, k] = gradient_A_phik_torch(grad_A_mu, H, yk, Sk, grad_A_sig).reshape(-1)
        tgrad_end = time.perf_counter() - tgrad_start
        print(f"Gradient gradient_A_phik computation for time step {k} took {tgrad_end} seconds")

        # update recurrent gradient states but using A_current (variable we optimize)
        tgrad_start = time.perf_counter()
        grad_A_mu = gradient_A_mu_torch(grad_A_mu, grad_A_sig, A_current, H, Sk, yk, Kk_t, zmean)
        tgrad_end = time.perf_counter() - tgrad_start
        print(f"Gradient gradient_A_mu computation for time step {k} took {tgrad_end} seconds")


        tgrad_start = time.perf_counter()
        grad_A_sig = gradient_A_sig_torch(grad_A_sig, A_current, H, Pk_m, Kk_t)
        tgrad_end = time.perf_counter() - tgrad_start
        print(f"Gradient grad_A_sig computation for time step {k} took {tgrad_end} seconds")

    # penalty computed at A_current
    penalty, grad_penalty = grad_desc_penalty_torch(A_current, lambda_reg, alpha, delta)

    # likelihood part uses stored Sk, yk (computed with A_kf)
    phi = Compute_PhiK_torch(0, Sk_kalman_em, yk_kalman_em) + penalty

    dphiA = -torch.reshape(torch.sum(grad_A_phik, axis=1), (Nx, Nx)).T + grad_penalty

    print("Finished compute_loss_gradient_v2_torch.")

    return phi, dphiA


def compute_loss_grad_alternative(
    A_current,
    A_kf,
    kf_results,
    z0, P0, H,
    Nx, Nz, K,
    lambda_reg=20, alpha=1, delta=1e-4
):
    

    yk_kalman_em = kf_results["yk"]
    Sk_kalman_em = kf_results["Sk"]
    # recursion over stored time steps (use A_current when propagating gradient states)

    # penalty computed at A_current
    penalty, _ = grad_desc_penalty_torch(A_current, lambda_reg, alpha, delta)

    # likelihood part uses stored Sk, yk (computed with A_kf)
    phi = Compute_PhiK_torch(0, Sk_kalman_em, yk_kalman_em) + penalty
    
    return phi

