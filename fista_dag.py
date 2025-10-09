#Implementation of the FISTA strategy into the DAGraphEM structure
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
from tools.prox import prox_stable, prox_f3
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG
from tools.dag import numpy_to_torch, logdet_dag, compute_loss, compute_new_loss, compute_loss_zero, grad_f1_f2, compute_F

if __name__ == "__main__":
    K = 500  # length of time series
    flag_plot = 1
    D1, Graph = generate_random_DAG(5, graph_type='ER', edge_prob=0.2, seed=40)  # Could also use the prox stable too (test it after)
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

    lambda_reg = 20
    alpha = 0
    factor_alpha = 1.1
    upper_alpha = 1e8  # upper bound for alpha
    stepsize = 0.1

    #FISTA parameters
    L_prev = 1
    eta = 1.5
    jmax = 10
    kmax = 10
    tk = 1
    tk_prev = 1

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
        print(f"---- REALIZATION {real + 1} ----")

        # Synthetic data generation
        y, x = GenerateSynthetic_order_p(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)
        saveX[:, :, real] = x[real]

        Err_D1 = []
        charac_dag = []
        loss_dag = []
        A_norm = []
        spectral_norm = []
        alpha_values = []
        Nit_em = 50  # number of iterations maximum for EM loop
        prec = 1e-2  # precision for EM loop
        prec_dag = 1e-9

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
            if i % 10 == 0:
                    print(f"EM Step {i}")
            
            # 1/ Kalman filter filter
            z_mean_kalman_em = np.zeros((Nz, K))
            P_kalman_em = np.zeros((Nz, Nz, K))
            yk_kalman_em = np.zeros((Nx, K))
            Sk_kalman_em = np.zeros((Nx, Nx, K))

            x_k_initial = x[:, 0].reshape(-1, 1)  # Reshape to a column vector
            z_mean_kalman_em_temp, P_kalman_em[:, :, 0], yk_kalman_em_temp, Sk_kalman_em[:, :, 0],_,_ = \
                Kalman_update(x_k_initial, z0, P0, D1_em, D2, R, Q)
            z_mean_kalman_em[:, 0] = z_mean_kalman_em_temp.flatten()
            yk_kalman_em[:, 0] = yk_kalman_em_temp.flatten()

            for k in range(1, K):
                x_k = x[:, k].reshape(-1, 1)      # Reshape each observation
                z_mean_kalman_em_temp, P_kalman_em[:, :, k], yk_kalman_em_temp, Sk_kalman_em[:, :, k],_,_ = \
                    Kalman_update(x_k, z_mean_kalman_em[:, k - 1].reshape(-1, 1), P_kalman_em[:, :, k - 1], D1_em, D2, R, Q)
                z_mean_kalman_em[:, k] = z_mean_kalman_em_temp.flatten()
                yk_kalman_em[:, k] = yk_kalman_em_temp.flatten()


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
            
            
            #M-Step
            Bk = D1_em.copy()
            Ak_prev = D1_em.copy()
            Ak = D1_em.copy()
            Lk = L_prev
            for k in range(kmax):
                tk_prev = tk
                Gk = grad_f1_f2(Bk, K, Q_inv, C, Phi, alpha) #possible error here!
                #Backtracking line search
                for j in range(jmax):
                    Lkj = L_prev * (eta**j)
                    Akj = prox_f3(Bk - (1.0 / Lkj) * Gk, Lkj, Gk, lambda_reg)
                    if compute_F(Akj, K, Q_inv, Sigma, C, Phi, lambda_reg, alpha) <= \
                        compute_F(Bk, K, Q_inv, Sigma, C, Phi, lambda_reg, alpha) + \
                        np.trace(Gk.T@(Akj - Bk)) + (Lkj/2)*np.linalg.norm(Akj - Bk, 'fro')**2:
                        Lk = Lkj
                        Ak = Akj
                        break
                tk = (1 + np.sqrt(1 + 4 * tk_prev**2)) / 2
                Bk = Ak + ((tk_prev - 1) / tk) * (Ak - Ak_prev)
                Ak_prev = Ak.copy()
                L_prev = Lk

            D1_em = Bk.copy()
                    

            D1_em_save[:, :, i] = D1_em  # keep track of the sequence

            Err_D1.append(np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro'))

            charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)
            dagness = numpy_to_torch(-alpha * logdet_dag(D1_em))
            loss_dag.append(dagness)
            alpha_values.append(alpha)
            A_norm.append(np.linalg.norm(D1_em, np.inf))
            spectral_norm.append(np.linalg.norm(D1_em,ord=2))


            if i > 0:
                if np.linalg.norm(D1_em_save[:, :, i - 1] - D1_em_save[:, :, i], 'fro') / \
                   np.linalg.norm(D1_em_save[:, :, i - 1], 'fro') < prec and charac_dag[i] < prec_dag:
                    print(f"EM converged after iteration {i + 1}")
                    break


        tEnd[real] = time.perf_counter() - tStart
        print(f"final alpha: {alpha}")

        D1_em_save_realization = D1_em_save[:, :, :len(Err_D1)]
        D1_em_final = D1_em

        threshold = 1e-10
        D1_binary = np.abs(D1) >= threshold
        D1_em_binary = np.abs(D1_em_final) >= threshold
        TP, FP, TN, FN = calError(D1_binary, D1_em_binary)

        #RMSE support matrix
        Err_support = np.linalg.norm(D1_binary.astype(int)  - D1_em_binary.astype(int) , 'fro') / np.linalg.norm(D1_binary.astype(int) , 'fro')
        print(f"Final error on D1 = {Err_support:.4f}")
        print(f"binary D1: {D1_binary}")
        print(f"binary D1_est: {D1_em_binary}")

        plt.figure(30, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        G_true = nx.DiGraph(D1)
        weights_true = np.array([abs(G_true[u][v]['weight']) for u, v in G_true.edges()])
        if weights_true.size > 0:
            linewidths_true = 5 * weights_true / np.max(weights_true)
        else:
            linewidths_true = 1
        pos = nx.spring_layout(G_true, seed=42)  # You might need to adjust the layout algorithm
        nx.draw(G_true, pos, width=linewidths_true, with_labels=False, node_size=30, arrowsize=10)
        plt.title('True D1 Network')

        plt.subplot(1, 2, 2)
        G_est = nx.DiGraph(D1_em_final)
        weights_est = np.array([abs(G_est[u][v]['weight']) for u, v in G_est.edges()])
        if weights_est.size > 0:
            linewidths_est = 5 * weights_est / np.max(weights_est)
        else:
            linewidths_est = 1
        nx.draw(G_est, pos, width=linewidths_est, with_labels=False, node_size=30, arrowsize=10)
        plt.title('Estimated D1 Network')
        plt.tight_layout()
        plt.show()

        precision[real] = TP / (TP + FP + 1e-8)
        recall[real] = TP / (TP + FN + 1e-8)
        specificity[real] = TN / (TN + FP + 1e-8)
        accuracy[real] = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        RMSE[real] = Err_D1[-1] if Err_D1 else np.nan
        F1score[real] = 2 * TP / (2 * TP + FP + FN + 1e-8)

        TestDAG = nx.from_numpy_array(D1_em_final, create_using=nx.DiGraph)
        print(int(nx.is_directed_acyclic_graph(TestDAG)))

        print(f"Final error on D1 = {RMSE[real]:.4f}")
        print(f"accuracy = {accuracy[real]:.4f}; precision = {precision[real]:.4f}; recall = {recall[real]:.4f}; specificity = {specificity[real]:.4f}")

    print(f"Total time = {np.mean(tEnd):.4f}")

    print(f"average RMSE = {np.nanmean(RMSE):.4f}")
    print(f"average accuracy = {np.nanmean(accuracy):.4f}")
    print(f"average precision = {np.nanmean(precision):.4f}")
    print(f"average recall = {np.nanmean(recall):.4f}")
    print(f"average specificity = {np.nanmean(specificity):.4f}")
    print(f"average F1 score = {np.nanmean(F1score):.4f}")


    if flag_plot == 1:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(D1, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('True D1')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(D1_em_final, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Estimated D1')
        plt.axis('off')
        plt.show()

        plt.figure(3)
        plt.semilogy(Err_D1)
        plt.title('Error on A')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('Frobenius Norm Error')
        plt.grid(True)
        plt.show()

        plt.figure(4)
        plt.plot(PhiK[:len(Err_D1)])
        plt.title('Loss function')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.show()

        plt.figure(5)
        plt.semilogy(charac_dag)
        plt.title('DAG characterization of A')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('Characterization')
        plt.grid(True)
        plt.show()

       