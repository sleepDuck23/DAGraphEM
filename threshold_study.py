import numpy as np
import scipy.io
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import os
import time
import networkx as nx
import torch
from scipy.linalg import expm
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm

from tools.matrix import calError
from tools.loss import ComputeMaj_D1, ComputeMaj, Compute_PhiK, Compute_Prior_D1
from tools.EM import Smoothing_update, Kalman_update, EM_parameters, GRAPHEM_update
from tools.prox import prox_stable
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG, create_fixed_upper_triangular_dag
from tools.dag import numpy_to_torch, logdet_dag, compute_loss, compute_new_loss, compute_loss_zero, grad_newloss
from solvers.adam import adam

nodes_size = [10, 15]
random_seed = np.linspace(40, 50, num=10, dtype=int)  # Random seeds for reproducibility
roc_results = []


if __name__ == "__main__":

    for nodex in range(len(nodes_size)):
        print(f"------------ Nodes: {nodes_size[nodex]} ------------")
        for seeds in random_seed:
            print(f"---- Seed: {seeds} ----")
            K = 500  # length of time series
        
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
            D1, Graph = generate_random_DAG(nodes_size[nodex], graph_type='ER', edge_prob=0.2, seed=seeds,weight_range=(0.1, 0.99)) # Could also use the prox stable too (test it after)
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
            # ROC-like analysis: sweep threshold from 1e-10 to 1e-2
            threshold_range = np.logspace(-10, -2, num=50)
            
        
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
            
                    D1_em_final = D1_em
            
                    D1_binary = np.abs(D1) >= 1e-10
            
                    for thr in threshold_range:
                        D1_em_binary_thr = np.abs(D1_em_final) >= thr
                        TP, FP, TN, FN = calError(D1_binary, D1_em_binary_thr)
            
                        accuracy_thr = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                        f1_thr = 2 * TP / (2 * TP + FP + FN + 1e-8)
            
                        roc_results.append({
                            "nodes": nodes_size[nodex],
                            "seed": seeds,
                            "w_threshold": thr,
                            "Accuracy": accuracy_thr,
                            "F1": f1_thr
                        })
            
                    
                    
            
roc_df = pd.DataFrame(roc_results)
roc_csv_path = "dagraphem_roc_threshold_sweep.csv"
roc_df.to_csv(roc_csv_path, index=False)
print(f"ROC threshold sweep saved to {roc_csv_path}")            
roc_df["w_threshold"] = pd.to_numeric(roc_df["w_threshold"])
roc_df["nodes"] = pd.to_numeric(roc_df["nodes"])

# Group by both w_threshold and node size, compute mean and std
agg_df = roc_df.groupby(["w_threshold", "nodes"]).agg({
    "Accuracy": ["mean", "std"],
    "F1": ["mean", "std"]
}).reset_index()

# Flatten multi-level columns
agg_df.columns = ["w_threshold", "nodes", "Accuracy_mean", "Accuracy_std", "F1_mean", "F1_std"]

# Plotting
plt.figure(figsize=(10, 6))
for nodes in agg_df["nodes"].unique():
    node_df = agg_df[agg_df["nodes"] == nodes]

    # Plot Accuracy
    plt.plot(node_df["w_threshold"], node_df["Accuracy_mean"], label=f"Accuracy (nodes={nodes})")
    plt.fill_between(node_df["w_threshold"],
                     node_df["Accuracy_mean"] - node_df["Accuracy_std"],
                     node_df["Accuracy_mean"] + node_df["Accuracy_std"],
                     alpha=0.2)

    # Plot F1
    plt.plot(node_df["w_threshold"], node_df["F1_mean"], linestyle='--', label=f"F1 (nodes={nodes})")
    plt.fill_between(node_df["w_threshold"],
                     node_df["F1_mean"] - node_df["F1_std"],
                     node_df["F1_mean"] + node_df["F1_std"],
                     alpha=0.2)

plt.xscale("log")
plt.xlabel("w_threshold")
plt.ylabel("Score")
plt.title("Mean ± Std of Accuracy and F1 vs. w_threshold by Node Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_mean_std_w_threshold.png", dpi=300)
plt.show()