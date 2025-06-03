import numpy as np
import scipy.io
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import os
import time
import networkx as nx
import torch
from scipy.linalg import expm

from tools.matrix import calError
from tools.loss import ComputeMaj_D1, ComputeMaj, Compute_PhiK, Compute_Prior_D1
from tools.EM import Smoothing_update, Kalman_update, EM_parameters, GRAPHEM_update
from tools.prox import prox_stable
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG
from tools.dag import numpy_to_torch, logdet_dag, compute_loss

"""
The goal of this file is to understand the behavior of each hyper-parameter in the solving process
At each run we will test a chosen parameter with multiple graphs and increasing the number of nodes
The result of this file should be a series of plots to analyze the behavior of such parameter
"""
hyperparam = [10,20,50,75,100]
nodes_size = [4,6,8,10,15]
random_seed = [40,41,42,43,44]
RMSE_results = []
meanRMSE = [ [ 0 for i in range(len(nodes_size)) ] for j in range(len(hyperparam)) ] 

for param in range(len(hyperparam)):
    print(f"-------------------- hyperparam: {hyperparam[param]} --------------------")
    for nodex in range(len(nodes_size)):
        print(f"------------ Nodes: {nodes_size[nodex]} ------------")
        for seeds in random_seed:
            print(f"---- Seed: {seeds} ----")
            if __name__ == "__main__":
                K = 2000  # length of time series
                flag_plot = 0

                ## Load ground truth matrix D1
                #try:
                #    data = scipy.io.loadmat('dataset/D1_datasetA_icassp.mat')
                #    D1 = data['D1']
                #except FileNotFoundError:
                #    print("Error: datasets/D1_datasetA_icassp.mat not found. Using a dummy D1.")
                #    Nx = 15  # Dummy size
                #    D1 = prox_stable(np.random.rand(Nx, Nx) - 0.5, 1)
                #Nx = D1.shape[0]  # number of nodes
                #Nz = Nx
                #D2 = np.eye(Nz)  # for simplicity and identifiability purposes

                #Lets try new things: let's generate a DAG and use it on yhe following
                D1, Graph = generate_random_DAG(nodes_size[nodex], graph_type='ER', edge_prob=0.2, seed=seeds) # Could also use the prox stable too (test it after)
                Nx = D1.shape[0]  # number of nodes
                Nz = Nx
                D2 = np.eye(Nz)  # for simplicity and identifiability purposes

                p = 1  # markov order
                sigma_Q = 1  # observation noise std
                Q = sigma_Q**2 * np.eye(Nz)
                sigma_R = 1  # state noise std
                R = sigma_R**2 * np.eye(Nx)
                sigma_P = 0.0001  # prior state std
                P0 = sigma_P**2 * np.eye(Nz)
                z0 = np.ones((Nz, 1))

                Q_inv = np.linalg.inv(Q)
                Q_inv_torch = torch.linalg.inv(numpy_to_torch(Q))

                reg1 = 113
                gamma1 = 20
                num_adam_steps = 1000
                #lambda_reg = 50 #Being tested as study parameter
                alpha = 50 
                stepsize = 0.1

                reg = {}
                reg['reg1'] = reg1
                reg['gamma1'] = gamma1

                Mask_true = (D1 != 0)
                reg['Mask'] = Mask_true  # only used to try the OracleEM option ie reg.reg1=3

                Nreal = 1  # Number of independent runs
                tEnd = np.zeros(Nreal)
                RMSE = np.zeros(Nreal)
                accuracy = np.zeros(Nreal)
                precision = np.zeros(Nreal)
                recall = np.zeros(Nreal)
                specificity = np.zeros(Nreal)
                F1score = np.zeros(Nreal)
                saveX = np.zeros((Nx, K, Nreal))


                for real in range(Nreal):
                    # Synthetic data generation
                    y, x = GenerateSynthetic_order_p(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)
                    saveX[:, :, real] = x[real]

                    # Inference (GRAPHEM algorithm)
                    print('-- GRAPHEM + DAG --')

                    Err_D1 = []
                    charac_dag = []
                    Nit_em = 50  # number of iterations maximum for EM loop
                    prec = 1e-2  # precision for EM loop
                    w_threshold = 0.1

                    tStart = time.perf_counter() 
                    # initialization of GRAPHEM
                    D1_em = prox_stable(CreateAdjacencyAR1(Nz, 0.1), 0.99)
                    D1_em_save = np.zeros((Nz, Nz, Nit_em))
                    PhiK = np.zeros(Nit_em)
                    MLsave = np.zeros(Nit_em)
                    Regsave = np.zeros(Nit_em)
                    Maj_before = np.zeros(Nit_em)
                    Maj_after = np.zeros(Nit_em)

                    #x = np.stack(x, axis=1)  

                    for i in range(Nit_em):  # EM iterations
                        # Just for visualization purposes
                        if i % 25 == 0:
                                print(f"EM Step {i}")
                        
                        # 1/ Kalman filter filter
                        z_mean_kalman_em = np.zeros((Nz, K))
                        P_kalman_em = np.zeros((Nz, Nz, K))
                        yk_kalman_em = np.zeros((Nx, K))
                        Sk_kalman_em = np.zeros((Nx, Nx, K))

                        x_k_initial = x[:, 0].reshape(-1, 1)  # Reshape to a column vector
                        z_mean_kalman_em_temp, P_kalman_em[:, :, 0], yk_kalman_em_temp, Sk_kalman_em[:, :, 0] = \
                            Kalman_update(x_k_initial, z0, P0, D1_em, D2, R, Q)
                        z_mean_kalman_em[:, 0] = z_mean_kalman_em_temp.flatten()
                        yk_kalman_em[:, 0] = yk_kalman_em_temp.flatten()

                        for k in range(1, K):
                            x_k = x[:, k].reshape(-1, 1)      # Reshape each observation
                            z_mean_kalman_em_temp, P_kalman_em[:, :, k], yk_kalman_em_temp, Sk_kalman_em[:, :, k] = \
                                Kalman_update(x_k, z_mean_kalman_em[:, k - 1].reshape(-1, 1), P_kalman_em[:, :, k - 1], D1_em, D2, R, Q)
                            z_mean_kalman_em[:, k] = z_mean_kalman_em_temp.flatten()
                            yk_kalman_em[:, k] = yk_kalman_em_temp.flatten()

                        # compute loss function (ML for now, no prior)
                        PhiK[i] = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em)

                        # compute penalty function before update
                        Reg_before = Compute_Prior_D1(D1_em, reg)
                        MLsave[i] = PhiK[i]
                        Regsave[i] = Reg_before
                        PhiK[i] = PhiK[i] + Reg_before  # update loss function

                        # 2/ Kalman smoother
                        z_mean_smooth_em = np.zeros((Nz, K))
                        P_smooth_em = np.zeros((Nz, Nz, K))
                        G_smooth_em = np.zeros((Nz, Nz, K))
                        z_mean_smooth_em[:, K - 1] = z_mean_kalman_em[:, K - 1]
                        P_smooth_em[:, :, K - 1] = P_kalman_em[:, :, K - 1]
                        for k in range(K - 2, -1, -1):
                            z_mean_smooth_em[:, k], P_smooth_em[:, :, k], G_smooth_em[:, :, k] = \
                                Smoothing_update(z_mean_kalman_em[:, k], P_kalman_em[:, :, k],
                                                z_mean_smooth_em[:, k + 1], P_smooth_em[:, :, k + 1], D1_em, D2, R, Q)     
                        z_mean_smooth0_em, P_smooth0_em, G_smooth0_em = \
                            Smoothing_update(z0, P0, z_mean_smooth_em[:, 0].reshape(-1, 1), P_smooth_em[:, :, 0], D1_em, D2, R, Q)


                        # compute EM parameters
                        Sigma, Phi, B, C, D = EM_parameters(x, z_mean_smooth_em, P_smooth_em, G_smooth_em,
                                                            z_mean_smooth0_em, P_smooth0_em, G_smooth0_em)
                        
                        #Implementation of the DAG caractherization function while using Adam solver for a gradient descent
                        A = torch.tensor(D1_em, dtype=torch.float32, requires_grad=True)
                        optimizer = torch.optim.Adam([A], lr=1e-4)

                        for step in range(num_adam_steps):
                            #Sigma_scaled = Sigma / np.linalg.norm(Sigma, 'fro')
                            #C_scaled = C / np.linalg.norm(C, 'fro')
                            #Phi_scaled = Phi / np.linalg.norm(Phi, 'fro')

                            # Convert all to PyTorch tensors
                            Sigma_torch = numpy_to_torch(Sigma)
                            C_torch = numpy_to_torch(C)
                            Phi_torch = numpy_to_torch(Phi)

                            optimizer.zero_grad()
                            loss = compute_loss(A,K,Q_inv_torch,Sigma_torch,C_torch,Phi_torch,hyperparam[param],alpha)
                            if not torch.isfinite(loss):
                                print("Non-finite loss encountered")
                                break
                            loss.backward()
                            #torch.nn.utils.clip_grad_norm_([A], max_norm=10.0)
                            optimizer.step()
                            if step % 1001 == 0 and i % 10 == 0:
                                print(f"Adam Step {step}, Loss: {loss.item():.2f}")
                                grad_norm = A.grad.norm().item()
                                print(f"Grad norm: {grad_norm:.2f}")

                        D1_em = A.detach().cpu().numpy()
                        #D1_em = D1_em_  # D1 estimate updated
                        D1_em_save[:, :, i] = D1_em  # keep track of the sequence

                        Err_D1.append(np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro'))

                        charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)


                        if i > 0:
                            if np.linalg.norm(D1_em_save[:, :, i - 1] - D1_em_save[:, :, i], 'fro') / \
                            np.linalg.norm(D1_em_save[:, :, i - 1], 'fro') < prec and charac_dag[i] < prec:
                                print(f"EM converged after iteration {i + 1}")
                                break


                    tEnd[real] = time.perf_counter() - tStart

                    D1_em[np.abs(D1_em) < w_threshold] = 0 #Eliminate edges that are close to zero0

                    D1_em_save_realization = D1_em_save[:, :, :len(Err_D1)]
                    D1_em_final = D1_em

                    threshold = 1e-10
                    D1_binary = np.abs(D1) >= threshold
                    D1_em_binary = np.abs(D1_em_final) >= threshold
                    TP, FP, TN, FN = calError(D1_binary, D1_em_binary)

                    precision[real] = TP / (TP + FP + 1e-8)
                    recall[real] = TP / (TP + FN + 1e-8)
                    specificity[real] = TN / (TN + FP + 1e-8)
                    accuracy[real] = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    RMSE[real] = Err_D1[-1] if Err_D1 else np.nan
                    F1score[real] = 2 * TP / (2 * TP + FP + FN + 1e-8)
                    RMSE_results.append(RMSE[real])

        meanRMSE[param][nodex] = np.mean(RMSE_results)
        RMSE_results = []
meanRMSE = np.array(meanRMSE)
print(meanRMSE)
for j, lam in enumerate(hyperparam):
    plt.plot(
        nodes_size,           
        meanRMSE[j, :],       
        marker='o',
        label=f'lambda = {lam}'
    )

plt.xlabel('Number of nodes')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE vs Node Count for different lambda')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

    