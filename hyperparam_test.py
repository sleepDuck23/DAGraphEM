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

from tools.matrix import calError
from tools.loss import ComputeMaj_D1, ComputeMaj, Compute_PhiK, Compute_Prior_D1
from tools.EM import Smoothing_update, Kalman_update, EM_parameters, GRAPHEM_update
from tools.prox import prox_stable
from simulators.simulators import GenerateSynthetic_order_p, CreateAdjacencyAR1, generate_random_DAG
from tools.dag import numpy_to_torch, logdet_dag, compute_loss



if __name__ == "__main__":
    print("Begining computation:")
    # Experiment settings
    hyperparam = [0.5, 1, 10]
    nodes_size = [5,10,15,20]
    random_seed = [40,41,42,43,44,45,46,47,48,49]

    print(f"Defined Hyperparameter values: {hyperparam}")
    print(f"Defined node sizes: {nodes_size}")
    print(f"Defined random seeds: {random_seed}")
    
    
    # Store RMSEs as: results[alpha_idx][nodes_idx] = list of RMSEs over seeds
    all_RMSE = [[[] for _ in range(len(nodes_size))] for _ in range(len(hyperparam))]
    all_accuracy = [[[] for _ in range(len(nodes_size))] for _ in range(len(hyperparam))]
    all_time = [[[] for _ in range(len(nodes_size))] for _ in range(len(hyperparam))]
    all_f1 = [[[] for _ in range(len(nodes_size))] for _ in range(len(hyperparam))]
    all_DAG = [[[] for _ in range(len(nodes_size))] for _ in range(len(hyperparam))]

    for param in range(len(hyperparam)):
        print(f"-------------------- hyperparam: {hyperparam[param]} --------------------")
        for nodex in range(len(nodes_size)):
            print(f"------------ Nodes: {nodes_size[nodex]} ------------")
            for seeds in random_seed:
                print(f"---- Seed: {seeds} ----")

                K = 2000
                flag_plot = 0

                D1, Graph = generate_random_DAG(nodes_size[nodex], graph_type='ER', edge_prob=0.2, seed=seeds)
                Nx = D1.shape[0]
                Nz = Nx
                D2 = np.eye(Nz)

                p = 1
                sigma_Q = 1
                Q = sigma_Q**2 * np.eye(Nz)
                sigma_R = 1
                R = sigma_R**2 * np.eye(Nx)
                sigma_P = 0.0001
                P0 = sigma_P**2 * np.eye(Nz)
                z0 = np.ones((Nz, 1))

                Q_inv_torch = torch.linalg.inv(numpy_to_torch(Q))

                reg = {'reg1': 113, 'gamma1': 20, 'Mask': (D1 != 0)}

                saveX = np.zeros((Nx, K, 1))

                # Single realization
                y, x = GenerateSynthetic_order_p(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)

                # GRAPHEM Init
                Err_D1 = []
                charac_dag = []
                D1_em = prox_stable(CreateAdjacencyAR1(Nz, 0.1), 0.99)
                Nit_em = 50
                prec = 1e-2
                w_threshold = 0.1
                num_adam_steps = 1000
                lambda_reg = 50
                alpha = 50
                D1_em_save = np.zeros((Nz, Nz, Nit_em))

                tStart = time.perf_counter() 

                for i in range(Nit_em):
                    z_mean_kalman_em = np.zeros((Nz, K))
                    P_kalman_em = np.zeros((Nz, Nz, K))
                    yk_kalman_em = np.zeros((Nx, K))
                    Sk_kalman_em = np.zeros((Nx, Nx, K))

                    x_k_initial = x[:, 0].reshape(-1, 1)
                    z_mean_kalman_em_temp, P_kalman_em[:, :, 0], yk_kalman_em_temp, Sk_kalman_em[:, :, 0] = Kalman_update(
                        x_k_initial, z0, P0, D1_em, D2, R, Q
                    )
                    z_mean_kalman_em[:, 0] = z_mean_kalman_em_temp.flatten()
                    yk_kalman_em[:, 0] = yk_kalman_em_temp.flatten()

                    for k in range(1, K):
                        x_k = x[:, k].reshape(-1, 1)
                        z_mean_kalman_em_temp, P_kalman_em[:, :, k], yk_kalman_em_temp, Sk_kalman_em[:, :, k] = Kalman_update(
                            x_k, z_mean_kalman_em[:, k - 1].reshape(-1, 1), P_kalman_em[:, :, k - 1], D1_em, D2, R, Q
                        )
                        z_mean_kalman_em[:, k] = z_mean_kalman_em_temp.flatten()
                        yk_kalman_em[:, k] = yk_kalman_em_temp.flatten()

                    PhiK = Compute_PhiK(0, Sk_kalman_em, yk_kalman_em)
                    Reg_before = Compute_Prior_D1(D1_em, reg)
                    PhiK += Reg_before

                    z_mean_smooth_em = np.zeros((Nz, K))
                    P_smooth_em = np.zeros((Nz, Nz, K))
                    G_smooth_em = np.zeros((Nz, Nz, K))
                    z_mean_smooth_em[:, K - 1] = z_mean_kalman_em[:, K - 1]
                    P_smooth_em[:, :, K - 1] = P_kalman_em[:, :, K - 1]

                    for k in range(K - 2, -1, -1):
                        z_mean_smooth_em[:, k], P_smooth_em[:, :, k], G_smooth_em[:, :, k] = Smoothing_update(
                            z_mean_kalman_em[:, k], P_kalman_em[:, :, k],
                            z_mean_smooth_em[:, k + 1], P_smooth_em[:, :, k + 1], D1_em, D2, R, Q
                        )
                    z_mean_smooth0_em, P_smooth0_em, G_smooth0_em = Smoothing_update(
                        z0, P0, z_mean_smooth_em[:, 0].reshape(-1, 1), P_smooth_em[:, :, 0], D1_em, D2, R, Q
                    )

                    Sigma, Phi, B, C, D = EM_parameters(x, z_mean_smooth_em, P_smooth_em, G_smooth_em,
                                                        z_mean_smooth0_em, P_smooth0_em, G_smooth0_em)

                    A = torch.tensor(D1_em, dtype=torch.float32, requires_grad=True)
                    optimizer = torch.optim.Adam([A], lr=1e-4)
                    for step in range(num_adam_steps):
                        Sigma_torch = numpy_to_torch(Sigma)
                        C_torch = numpy_to_torch(C)
                        Phi_torch = numpy_to_torch(Phi)

                        optimizer.zero_grad()
                        loss = compute_loss(A, K, Q_inv_torch, Sigma_torch, C_torch, Phi_torch, hyperparam[param], alpha)
                        if not torch.isfinite(loss): break
                        loss.backward()
                        optimizer.step()

                    D1_em = A.detach().cpu().numpy()

                    D1_em_save[:, :, i] = D1_em
                    Err_D1.append(np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro'))
                    charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)

                    if i > 0:
                        if np.linalg.norm(D1_em_save[:, :, i - 1] - D1_em_save[:, :, i], 'fro') / \
                           np.linalg.norm(D1_em_save[:, :, i - 1], 'fro') < prec and charac_dag[i] < prec:
                            print(f"EM converged after iteration {i + 1}")
                            break

                tEnd = time.perf_counter() - tStart

                D1_em[np.abs(D1_em) < w_threshold] = 0
                D1_em_save_realization = D1_em_save[:, :, :len(Err_D1)]
                D1_em_final = D1_em

                threshold = 1e-10
                D1_binary = np.abs(D1) >= threshold
                D1_em_binary = np.abs(D1_em) >= threshold

                TP, FP, TN, FN = calError(D1_binary, D1_em_binary)
                RMSE = np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro')

                all_RMSE[param][nodex].append(RMSE)

                accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                RMSE = Err_D1[-1] if Err_D1 else np.nan
                F1score = 2 * TP / (2 * TP + FP + FN + 1e-8)

                all_accuracy[param][nodex].append(accuracy)
                all_f1[param][nodex].append(F1score)
                all_time[param][nodex].append(tEnd)

                TestDAG = nx.from_numpy_array(D1_em_final, create_using=nx.DiGraph)
                all_DAG[param][nodex].append(int(nx.is_directed_acyclic_graph(TestDAG)))

    # Plotting RMSE using boxplots
    import seaborn as sns
    import matplotlib.pyplot as plt

    for j in range(len(hyperparam)):
        data = all_RMSE[j]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, positions=range(len(nodes_size)), patch_artist=True)

        plt.title(f'Boxplot of RMSE vs Number of Nodes (lambda = {hyperparam[j]})')
        plt.xticks(range(len(nodes_size)), nodes_size)
        plt.xlabel("Number of Nodes")
        plt.ylabel("RMSE")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"rmse_boxplot_lambda_{hyperparam[j]}.png", dpi=300)

        plt.show()


    for j in range(len(hyperparam)):
        data = all_accuracy[j]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, positions=range(len(nodes_size)), patch_artist=True)

        plt.title(f'Boxplot of Accuracy vs Number of Nodes (lambda = {hyperparam[j]})')
        plt.xticks(range(len(nodes_size)), nodes_size)
        plt.xlabel("Number of Nodes")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"acc_boxplot_lambda_{hyperparam[j]}.png", dpi=300)

        plt.show()

    for j in range(len(hyperparam)):
        data = all_f1[j]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, positions=range(len(nodes_size)), patch_artist=True)

        plt.title(f'Boxplot of F1 Score vs Number of Nodes (lambda = {hyperparam[j]})')
        plt.xticks(range(len(nodes_size)), nodes_size)
        plt.xlabel("Number of Nodes")
        plt.ylabel("F1 Score")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"f1_boxplot_lambda_{hyperparam[j]}.png", dpi=300)

        plt.show()

    for j in range(len(hyperparam)):
        data = all_time[j]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, positions=range(len(nodes_size)), patch_artist=True)

        plt.title(f'Boxplot of Comp. Time vs Number of Nodes (lambda = {hyperparam[j]})')
        plt.xticks(range(len(nodes_size)), nodes_size)
        plt.xlabel("Number of Nodes")
        plt.ylabel("Comp. Time")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"time_boxplot_lambda_{hyperparam[j]}.png", dpi=300)

        plt.show()

    for j in range(len(hyperparam)):
        data = all_DAG[j]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, positions=range(len(nodes_size)), patch_artist=True)

        plt.title(f'Boxplot of is DAG or not vs Number of Nodes (lambda = {hyperparam[j]})')
        plt.xticks(range(len(nodes_size)), nodes_size)
        plt.xlabel("Number of Nodes")
        plt.ylabel("DAG")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"dag_boxplot_lambda_{hyperparam[j]}.png", dpi=300)

        plt.show()

    

    # Prepare a list of dictionaries for DataFrame rows
    results_list = []

    for i, lambda_val in enumerate(hyperparam):
        for j, n_nodes in enumerate(nodes_size):
            for k, seed_idx in enumerate(random_seed):
                result_dict = {
                    "lambda": lambda_val,
                    "nodes_size": n_nodes,
                    "seed": seed_idx,
                    "RMSE": all_RMSE[i][j][k] if k < len(all_RMSE[i][j]) else None,
                    "Accuracy": all_accuracy[i][j][k] if k < len(all_accuracy[i][j]) else None,
                    "F1": all_f1[i][j][k] if k < len(all_f1[i][j]) else None,
                    "Time": all_time[i][j][k] if k < len(all_time[i][j]) else None,
                    "Is_DAG": all_DAG[i][j][k] if k < len(all_DAG[i][j]) else None
                }
                results_list.append(result_dict)

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    csv_path = "GRAPHEM_experiment_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")
