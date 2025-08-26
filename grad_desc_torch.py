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
from tools.loss import ComputeMaj_D1, ComputeMaj, Compute_PhiK, Compute_Prior_D1, Compute_PhiK_torch
from tools.EM import Smoothing_update, Kalman_update, EM_parameters, GRAPHEM_update, Kalman_update_torch
from tools.prox import prox_stable
from simulators.simulators import GenerateSynthetic_order_p_torch, CreateAdjacencyAR1, generate_random_DAG
from tools.dag import numpy_to_torch, logdet_dag, compute_loss, compute_new_loss, grad_desc_penalty_torch
from gradientEM.computeautograd import compute_loss_gradient_torch, run_kalman_full_torch, compute_loss_gradient_v2_torch, compute_loss_grad_alternative
from solvers.adam import adam, adam_alpha


if __name__ == "__main__":
    K = 500  # length of time series
    flag_plot = 1
    #Lets try new things: let's generate a DAG and use it on yhe following
    D1, Graph = generate_random_DAG(4, graph_type='ER', edge_prob=0.2, seed=41,weight_range=(0.1, 0.99)) # Could also use the prox stable too (test it after)
    Nx = D1.shape[0]  # number of nodes
    Nz = Nx
    D2 = torch.eye(Nz)  # for simplicity and identifiability purposes

    D1 = numpy_to_torch(D1)  # Convert to torch tensor

    p = 1  # markov order
    sigma_Q = 1  # observation noise std
    Q = sigma_Q**2 * torch.eye(Nz)
    sigma_R = 1  # state noise std
    R = sigma_R**2 * torch.eye(Nx)
    sigma_P = 0.0001  # prior state std
    P0 = sigma_P**2 * torch.eye(Nz)
    z0 = torch.ones((Nz, 1))


    reg1 = 1
    gamma1 = 20
    lambda_reg = 10
    alpha = 1
    factor_alpha = 10
    upper_bound_alpha = 1e15
    delta = 1e-4
    stepsize = 0.1 # This stepsize is not directly used by L-BFGS, but can be for other parts.
    max_outer_iters = 10
    num_lbfgs_steps = 10  # Number of steps for L-BFGS optimizer
    prec_dag = 1e-15
    prec = 1e-2
    
    kf_change_log = {
        "yk": [],
        "Sk": [],
        "Pk_minus": [],
        "Kk": [],
        "z_mean": []
    }

    kf_results_prev = None

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
        y, x = GenerateSynthetic_order_p_torch(K, D1, D2, p, z0, sigma_P, sigma_Q, sigma_R)
       
        saveX[:, :, real] = x[real]

        Err_D1 = []
        charac_dag = []
        stop_crit = []
        w_threshold = 1e-4

        tStart = time.perf_counter() 
        
        #D1_em = prox_stable(CreateAdjacencyAR1(Nz, 0.1), 0.99)
        D1_em = np.zeros((Nz, Nz))
        D1_em_save = np.zeros((Nz, Nz, max_outer_iters))
        PhiK = torch.zeros(max_outer_iters)
        MLsave = torch.zeros(max_outer_iters)
        Regsave = torch.zeros(max_outer_iters)
        Maj_before = torch.zeros(max_outer_iters)
        Maj_after = torch.zeros(max_outer_iters)
            
        # 1/ Kalman filter filter
        z_mean_kalman_em = torch.zeros((Nz, K))
        P_kalman_em = torch.zeros((Nz, Nz, K))
        yk_kalman_em = torch.zeros((Nx, K))
        Sk_kalman_em = torch.zeros((Nx, Nx, K))

        
        A = torch.tensor(D1_em, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([A], lr=1e-2, max_iter=5, history_size=50)

                
        for iter_outer in range(max_outer_iters):        
            
            def closure():
                optimizer.zero_grad()

                # Compute exact loss and gradients
                phi, _, dphiA = compute_loss_gradient_torch(
                    A, Q, x, z0, P0, D2, R, Nx, Nz, K,
                    lambda_reg=lambda_reg, alpha=alpha, delta=delta
                )

                # Assign gradient manually
                A.grad = dphiA.clone().contiguous()

                print("Loss:", phi.item())
                return phi
        
            
            optimizer.step(closure)

            D1_em = A.detach().cpu().numpy()
            D1_em_save[:, :, iter_outer] = D1_em
            charac_dag.append(np.trace(expm(D1_em*D1_em))-D1_em[0].shape)
            Err_D1.append(np.linalg.norm(D1 - D1_em, 'fro') / np.linalg.norm(D1, 'fro'))

            print(f"End of outer iteration {iter_outer + 1}, current D1 estimate:\n{A.detach().cpu().numpy()}")
            print(f"Current alpha: {alpha}")
            print("--------------------------------------------------")

            stop_crit.append(np.linalg.norm(D1_em_save[:, :, iter_outer] - D1_em_save[:, :, iter_outer - 1], 'fro') / \
                             np.linalg.norm(D1_em_save[:, :, iter_outer - 1], 'fro'))
            
            if alpha <= upper_bound_alpha: # and stop_crit[iter_outer] <= 1e-1:
                alpha = alpha * factor_alpha

            
            if iter_outer > 0:
                if stop_crit[iter_outer] < prec and charac_dag[iter_outer] < prec_dag:
                    print(f"EM converged after iteration {iter_outer + 1}")
                    print(f"Final alpha: {alpha}")
                    print(f"final charac dag = {charac_dag[iter_outer]}")
                    break

        tEnd[real] = time.perf_counter() - tStart

        D1_em[D1_em < w_threshold] = 0 #Eliminate edges that are close to zero0

        D1_em_final = D1_em

        print(f"Final D1 estimated:\n{D1_em_final}")
        print(f"Final D1 true:\n{D1}")

        TestDAG = nx.from_numpy_array(D1_em_final, create_using=nx.DiGraph)

        threshold = 1e-10
        D1_binary = np.abs(D1.detach().cpu().numpy()) >= threshold
        D1_em_binary = np.abs(D1_em_final) >= threshold
        TP, FP, TN, FN = calError(D1_binary, D1_em_binary)

        plt.figure(30, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        G_true = nx.from_numpy_array(D1.detach().cpu().numpy(), create_using=nx.DiGraph)
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

        plt.figure()
        plt.semilogy(charac_dag)
        plt.title('DAG characterization of A')
        plt.xlabel('DAGrad iterations')
        plt.ylabel('Characterization')
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.semilogy(Err_D1)
        plt.title('Error on A')
        plt.xlabel('DAGrad iterations')
        plt.ylabel('Frobenius Norm Error')
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.semilogy(stop_crit)
        plt.title('Stopping criterion')
        plt.xlabel('DAGrad iterations')
        plt.ylabel('Frobenius Norm change')
        plt.grid(True)
        plt.show()


    