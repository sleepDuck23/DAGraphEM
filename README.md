# DAGraphEM
DAGraphEM algorithm 
------------------------------------------------------------------------------------
Code for comparing DAGraphEM with other methods that provide the inference of graphs from LG-SSM time-series.
------------------------------------------------------------------------------------
DESCRIPTION:
This toolbox allows to estimate the transition matrix in linear-Gaussian state space model. 
We implement a method called GraphEM based on the expectation-maximization (EM) 
methodology for inferring the transition matrix jointly with the smoothing/filtering of the  
observed data. We implement a gradient descent optimization solver  for solving the 
M-step. This approach enables an efficient and versatile processing of various sophisticated 
priors on the graph structure, such as parsimony constraints, while benefiting from  
convergence guarantees towards the DAG space.  
This toolbox consists of 5 subfolders:
1) dataset  : contains the synthetic dataset from [Elvira & Chouzenoux, 2022]. 
2) gradientEM: gradients and important fucntions for the models to compute the solution of the loss fucntions
3) solvers: ADAM and L-BFGS solvers in NumPy
4) simulators: contains functions to generate DAG adjacency matrices, time series and initialization
5) tools: Fucntions for the correct work of all models
    * EM: contains functions for building EM algorithm updates
    * loss: contains functions for evaluating the likelihood and prior loss
    * matrix: contains functions for block sparsity models and for score computations
    * prox: contains functions for evaluating useful proximity operators
    * dag: DAG characterization and sparsity contraints for the loss functions
    * dagma: DAGMA implementation
    * l_bfgs: L-BFGS implementation
------------------------------------------------------------------------------------
SPECIFICATIONS for using the implemented models:
* main.py : GraphEM model with DAG generation
* main_expm.py : DAGraphEM model with ADAM optimizer (Torch)
* main_dagma.py : DAGraphEM using original DAGMA implementation (loss function and solver) 
* lbfgs_dagma.py : DAGraphEM model with L-BFGS optimizer
* test_numpy.py : DAGraphEM model with ADAM optimizer (NumPy)
* hyperparam_graphem.py : MLEM and GraphEM study script
* hyperparam_test.py : DAGraphEM study script
* threshold_study.py : Threshold study script
* grad_desc.py : DAGrad implemented with ADAM (NumPy) - First proposition
* grad_desc_2.py : DAGrad implemented with ADAM (NumPy) - Second proposition
* grad_desc_torch.py : DAGrad implemented with L-BFGS optimizer - First proposition

------------------------------------------------------------------------------------
RELATED PUBLICATIONS:
 * V. Elvira and E. Chouzenoux. Graphical Inference in Linear-Gaussian State-Space Models. IEEE Transactions on Signal Processing, vol. 70, pp. 4757 - 4771, Sep. 2022
 * E. Chouzenoux and V. Elvira.  GraphEM: EM Algorithm for Blind Kalman Filtering under Graphical Sparsity Constraints. In Proceedings of the 45th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2020), Virtual Conference, May 4-8 2020.
---------
