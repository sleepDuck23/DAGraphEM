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
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG, create_fixed_upper_triangular_dag
from tools.dag import numpy_to_torch, logdet_dag, compute_loss, compute_new_loss, compute_loss_zero, grad_newloss
from solvers.adam import adam

if __name__ == "__main__":
    K = 500  # length of time series
    flag_plot = 1

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
    D1, Graph = generate_random_DAG(20, graph_type='ER', edge_prob=0.2, seed=41,weight_range=(0.1, 0.99)) # Could also use the prox stable too (test it after)
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

    #reg1 = 113
    gamma1 = 0
    num_adam_steps = 1000
    lambda_reg = 20
    alpha = 1
    factor_alpha = 10 # factor to increase alpha
    stepsize = 1e-4
    upper_alpha = 1e15  # upper bound for alpha

    w_threshold = 1e-2  # threshold to eliminate small weights
    

    reg = {}
    #reg['reg1'] = reg1
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

    alpha_values = []

    for real in range(Nreal):
        print(f"---- REALIZATION {real + 1} ----")

        # Synthetic data generation
        y, x = GenerateSynthetic_order_p(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)
        saveX[:, :, real] = x[real]

        # Inference (GRAPHEM algorithm)
        print('-- GRAPHEM + DAG --')
        #print(f"Regularization on D1: norm {reg1} with gamma1 = {gamma1}")

        Err_D1 = []
        charac_dag = []
        loss_dag = []
        A_norm = []
        spectral_norm = []
        Nit_em = 50  # number of iterations maximum for EM loop
        prec = 1e-4  # precision for EM loop
        prec_DAG = 1e-12
        grad_norm_threshold = 1e-5  # threshold for gradient norm

        tStart = time.perf_counter() 
        # initialization of GRAPHEM
        #D1_em = prox_stable(CreateAdjacencyAR1(Nz, 0.1), 0.99)
        D1_em = create_fixed_upper_triangular_dag(Nz, weight=0.5)  # Initialize D1_em to a fixed upper triangular DAG
        #D1_em = np.zeros((Nz, Nz))  # Initialize D1_em to a zero matrix
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

            # compute loss function (ML for now, no prior)
            PhiK[i] = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em)

            # compute penalty function before update
            #Reg_before = Compute_Prior_D1(D1_em, reg)
            #MLsave[i] = PhiK[i]
            #Regsave[i] = Reg_before
            #PhiK[i] = PhiK[i] + Reg_before  # update loss function

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
            
           #running adam solver builded in this code:
            grad_loss = lambda D1_em: grad_newloss(D1_em,K,Q_inv,Sigma,C,Phi,lambda_reg,alpha)
            D1_em, grad_norm = adam(grad_loss, D1_em,step_size=stepsize, num_iters=num_adam_steps, callback=None)

            
            D1_em_save[:, :, i] = D1_em  # keep track of the sequence

            Err_D1.append(np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro'))

            charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)
            dagness = numpy_to_torch(-alpha * logdet_dag(D1_em/np.linalg.norm(D1_em, np.inf)))  # compute DAGness
            loss_dag.append(dagness)
            alpha_values.append(alpha)
            A_norm.append(np.linalg.norm(D1_em, np.inf))
            spectral_norm.append(np.linalg.norm(D1_em,ord=2))

            stop_crit = np.linalg.norm(D1_em_save[:, :, i - 1] - D1_em_save[:, :, i], 'fro') /np.linalg.norm(D1_em_save[:, :, i - 1], 'fro')
            print(f"Iteration {i + 1}: stop criterion = {stop_crit:.4f}, loss = {dagness:.4f}, alpha = {alpha:.4f}")
            if alpha < upper_alpha and stop_crit < 1e-2:
                alpha *= factor_alpha # increase alpha

            #if i % 5 == 0:
            #    print(f"Iteration {i + 1}:")
            #    print(f"alpha: {alpha}")
            #    print(f"Matrix A: {D1_em}")
            
            if i > 0:
                if stop_crit < prec and loss_dag[i] < prec_DAG or alpha >= upper_alpha:
                    print(f"EM converged after iteration {i + 1}")
                    break


        tEnd[real] = time.perf_counter() - tStart
        print(f"final alpha: {alpha}")

        D1_em[np.abs(D1_em) < w_threshold] = 0 #Eliminate edges that are close to zero0

        D1_em_save_realization = D1_em_save[:, :, :len(Err_D1)]
        D1_em_final = D1_em
        
        print(f"Final D1_em: {D1_em_final}")

        print(f"True graph: {D1}")

        threshold = 1e-10
        D1_binary = np.abs(D1) >= threshold
        D1_em_binary = np.abs(D1_em_final) >= threshold
        TP, FP, TN, FN = calError(D1_binary, D1_em_binary)

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
        plt.savefig('D1_networks.png', dpi=300)
        plt.show()

        precision[real] = TP / (TP + FP + 1e-8)
        recall[real] = TP / (TP + FN + 1e-8)
        specificity[real] = TN / (TN + FP + 1e-8)
        accuracy[real] = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        RMSE[real] = Err_D1[-1] if Err_D1 else np.nan
        F1score[real] = 2 * TP / (2 * TP + FP + FN + 1e-8)

        TestDAG = nx.from_numpy_array(D1_em_final, create_using=nx.DiGraph)
        print(f"Is it a real DAG: {nx.is_directed_acyclic_graph(TestDAG)}")

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

        plt.figure(6)
        plt.semilogy(loss_dag)
        plt.title('DAG loss of A')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('Characterization')
        plt.grid(True)
        plt.show()

        plt.figure(7)
        plt.semilogy(alpha_values)
        plt.title('Alpha evolution')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('alpha')
        plt.grid(True)
        plt.show()

        plt.figure(8)
        plt.semilogy(A_norm)
        plt.title('A norm evolution')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('A norm')
        plt.grid(True)
        plt.show()

        plt.figure(9)
        plt.semilogy(spectral_norm)
        plt.title('Spectral norm evolution')
        plt.xlabel('GRAPHEM iterations')
        plt.ylabel('Spectral norm')
        plt.grid(True)
        plt.show()