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
    D1, Graph = generate_random_DAG(3, graph_type='ER', edge_prob=0.2, seed=40)  # Could also use the prox stable too (test it after)
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

    lambda_reg = 15
    alpha = 1
    factor_alpha = 2
    upper_alpha = 1e8  # upper bound for alpha
    stepsize = 0.1

    #FISTA parameters
    eta = 1.5
    jmax = 20
    kmax = 10
    tk = 1

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
        Nit_em = 20  # number of iterations maximum for EM loop
        prec = 1e-3  # precision for EM loop
        prec_dag = 1e-9

        tStart = time.perf_counter() 
        # initialization of GRAPHEM
        #D1_em = prox_stable(CreateAdjacencyAR1(Nz, 0.1), 0.99)
        D1_em = np.zeros((Nz, Nz)) 
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

            PhiK[i] = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em)
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
            # I loop initialization
            Bk = D1_em.copy()
            Bk_prev = D1_em.copy()
            Ak_prev = D1_em.copy()
            Ak = D1_em.copy()
            L_prev = 1.0
            Lk = 1.0
            tk_prev = 1.0

            # K loop
            for k in range(kmax):
                Gk = grad_f1_f2(Bk, K, Q_inv, C, Phi, alpha) # Gradient at iteration k
                
                #Backtracking line search
                # J loop
                for j in range(jmax):
                    Lkj = L_prev * (eta**j)

                    Akj = prox_f3(Bk, Lkj, Gk, lambda_reg) # Soft Thresholding Operator

                    F_Akj = compute_F(Akj, K, Q_inv, Sigma, C, Phi, lambda_reg, alpha) # Backtracking condition term
                    F_Bk = compute_F(Bk, K, Q_inv, Sigma, C, Phi, lambda_reg, alpha)   # Backtracking condition term 
                    ker_k = np.trace(Gk.T @ (Akj - Bk))                                # Backtracking condition term
                    lip_norm = (Lkj/2)*(np.linalg.norm(Akj - Bk, 'fro')**2)            # Backtracking condition term

                    #print(f"Iteration I= {i}, k={k}, j={j}")


                    #print(f"backtracking iter L={Lkj}, F(Akj)={F_Akj}, F(Bk)={F_Bk}, ker_k={ker_k}, lip_norm={lip_norm}")
                    #print(f"bactracking condition left: {F_Akj} right: {(F_Bk + ker_k + lip_norm)}")

                    #print(f"matrix Akj:\n{Akj}")
                    #print("---------------------------")
                    #print(f"matrix Bk:\n{Bk}")


                    if  F_Akj <= F_Bk + ker_k + lip_norm: # Backtracking condition
                        print(f"Condition satisfied at loop k={k} and loop j={j} with L={Lkj}")
                        L_prev = Lk # L from previous iteration k
                        Lk = Lkj
                        Ak_prev = Ak.copy()
                        Ak = Akj
                        break

                tk_prev = tk # tk from previous iteration k
                tk = (1 + np.sqrt(1 + 4 * tk_prev**2)) / 2
                Bk_prev = Bk.copy()
                Bk = Ak + ((tk_prev - 1) / tk) * (Ak - Ak_prev)

                # Stopping criterion for the K loop
                #if k > 4 and np.linalg.norm(Bk - Bk_prev, 'fro') / np.linalg.norm(Bk_prev, 'fro') < 1e-4:
                #    print(f"  FISTA converged after iteration {k + 1}")
                #    break
                

            D1_em = Bk.copy()

            err = np.linalg.norm(D1_em - D1, 'fro') / np.linalg.norm(D1, 'fro')

            if alpha < upper_alpha:
                alpha = factor_alpha * alpha
                print(f"update alpha: {alpha}")
                    

            D1_em_save[:, :, i] = D1_em  # keep track of the sequence

            Err_D1.append(err)

            charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)
            dagness = -alpha * logdet_dag(D1_em)
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

        print(f"final matrix D1:\n{D1_em_final}")
        print(f"true matrix D1:\n{D1}")

        #RMSE support matrix
        Err_support = np.linalg.norm(D1_binary.astype(int)  - D1_em_binary.astype(int) , 'fro') / np.linalg.norm(D1_binary.astype(int) , 'fro')
        

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

       