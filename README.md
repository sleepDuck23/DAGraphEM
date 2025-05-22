# GraphEM
GraphEM algorithm converted into python language

------------------------------------------------------------------------------------
DESCRIPTION:
This toolbox allows to estimate the transition matrix in linear-Gaussian state space model. 
We implement a method called GraphEM based on the expectation-maximization (EM) 
methodology for inferring the transition matrix jointly with the smoothing/filtering of the  
observed data. We implement a proximal-based convex optimization solver  for solving the 
M-step. This approach enables an efficient and versatile processing of various sophisticated 
priors on the graph structure, such as parsimony constraints, while benefiting from  
convergence guarantees.  
This toolbox consists of 6 subfolders:
1) dataset  : contains the synthetic dataset from [Elvira & Chouzenoux, 2022]. 
2) EMtools: contains functions for building EM algorithm updates
3) losstools: contains functions for evaluating the likelihood and prior loss
4) matrixtools: contains functions for block sparsity models and for score computations
5) proxtools: contains functions for evaluating useful proximity operators
6) simulators: contains functions to generate time series and initialization
------------------------------------------------------------------------------------
SPECIFICATIONS for using GraphEM:
A demo file is provided :
* main.py 


------------------------------------------------------------------------------------
RELATED PUBLICATIONS:
 * V. Elvira and E. Chouzenoux. Graphical Inference in Linear-Gaussian State-Space Models. IEEE Transactions on Signal Processing, vol. 70, pp. 4757 - 4771, Sep. 2022
 * E. Chouzenoux and V. Elvira.  GraphEM: EM Algorithm for Blind Kalman Filtering under Graphical Sparsity Constraints. In Proceedings of the 45th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2020), Virtual Conference, May 4-8 2020.
---------
