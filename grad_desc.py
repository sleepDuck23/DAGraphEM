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
from tools.EM import Smoothing_update, Kalman_update, EM_parameters, GRAPHEM_update, Kalman_update_torch
from tools.prox import prox_stable
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG
from tools.dag import numpy_to_torch, logdet_dag, compute_loss, compute_new_loss
from gradientEM.computegrad import compute_loss_gradient
from solvers.adam import adam, adam_alpha

if __name__ == "__main__":
    K = 500  # length of time series
    flag_plot = 1
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


    reg1 = 113
    gamma1 = 20
    lambda_reg = 15
    alpha = 1
    factor_alpha = 1.1
    stepsize = 0.1 # This stepsize is not directly used by L-BFGS, but can be for other parts.
    

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
        print(f"---- REALIZATION {real + 1} ----")

        # Synthetic data generation
        y, x = GenerateSynthetic_order_p(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)
        saveX[:, :, real] = x[real]

        Err_D1 = []
        charac_dag = []
        stop_crit = []
        Nit_em = 2  # number of iterations maximum for EM loop
        prec = 1e-4  # precision for EM loop
        precDAG = 1e-3
        w_threshold = 1e-4

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
            
        # 1/ Kalman filter filter
        z_mean_kalman_em = np.zeros((Nz, K))
        P_kalman_em = np.zeros((Nz, Nz, K))
        yk_kalman_em = np.zeros((Nx, K))
        Sk_kalman_em = np.zeros((Nx, Nx, K))

        #grad_fn  = lambda D1_em: compute_loss_gradient(D1_em,Q,x,z0,P0,D2,R,Nx,Nz,K,lambda_reg,alpha)[2]
            
        #D1_em, _ = adam(grad_fn, D1_em,step_size=1e-4, num_iters=1000)

        grad_fn  = lambda D1_em, alpha: compute_loss_gradient(D1_em,Q_inv,x,z0,P0,D2,R,Nx,Nz,K,lambda_reg,alpha)[2]

            
        D1_em, _ = adam_alpha(grad_fn, D1_em, alpha, step_size=1e-4, num_iters=500,clip=100,clip_flag=True)

        tEnd[real] = time.perf_counter() - tStart

        D1_em[D1_em < w_threshold] = 0 #Eliminate edges that are close to zero0

        D1_em_final = D1_em
        print(f"Final D1 estimated:\n{D1_em_final}")

        TestDAG = nx.from_numpy_array(D1_em_final, create_using=nx.DiGraph)

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
        plt.show()

        precision[real] = TP / (TP + FP + 1e-8)
        recall[real] = TP / (TP + FN + 1e-8)
        specificity[real] = TN / (TN + FP + 1e-8)
        accuracy[real] = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        RMSE[real] = Err_D1[-1] if Err_D1 else np.nan
        F1score[real] = 2 * TP / (2 * TP + FP + FN + 1e-8)

        print(f"Final error on D1 = {RMSE[real]:.4f}")
        print(f"accuracy = {accuracy[real]:.4f}; precision = {precision[real]:.4f}; recall = {recall[real]:.4f}; specificity = {specificity[real]:.4f}")

    print(f"Total time = {np.mean(tEnd):.4f}")

    print(f"average RMSE = {np.nanmean(RMSE):.4f}")
    print(f"average accuracy = {np.nanmean(accuracy):.4f}")
    print(f"average precision = {np.nanmean(precision):.4f}")
    print(f"average recall = {np.nanmean(recall):.4f}")
    print(f"average specificity = {np.nanmean(specificity):.4f}")
    print(f"average F1 score = {np.nanmean(F1score):.4f}")
    print(f"Is it a DAG = {nx.is_directed_acyclic_graph(TestDAG)}")

    print(f"Final D1 estimated:\n{D1_em_final}")
    print(f"Final D1 true:\n{D1}")

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
