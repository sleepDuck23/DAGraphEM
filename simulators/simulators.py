import numpy as np
import networkx as nx

def CreateAdjacencyAR1(N,rho):
    A = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            A[i, j] = rho ** abs((i-1)-(j-1))

    return A

def GenerateSynthetic_order_p(K,A,H,p,x0,sigma_P, sigma_Q, sigma_R):
    Ny, Nx = H.shape

    x = np.zeros((Nx,K))
    y = np.zeros((Ny,K))

    for pp in range(p):
        x[:, pp] = (x0.flatten() + sigma_P * np.random.randn(Nx)).flatten()

    for k in range(p, K):
        deterministic_state = np.zeros(Nx)
        for pp in range(p):
            deterministic_state += A @ x[:, k - pp - 1]
        x[:, k] = deterministic_state + sigma_Q * np.random.randn(Nx)
        y[:, k] = H @ x[:, k] + sigma_R * np.random.randn(Ny)

    return y, x

def generate_random_DAG(N, 
                         graph_type='ER', 
                         edge_prob=0.3, 
                         weight_range=(0.5, 2.0), 
                         seed=None,
                         enforce_ar1=True):
    """
    Generate a random DAG with specified graph type.

    Args:
        N (int): Number of nodes (variables).
        graph_type (str): Type of random graph: 'ER', 'SF', or 'BP'.
        edge_prob (float): For ER graphs, probability of edge between any two nodes.
        weight_range (tuple): Range (min, max) for random edge weights.
        seed (int or None): Random seed for reproducibility.

    Returns:
        A (np.ndarray): [N x N] weighted adjacency matrix representing a DAG.
        G (networkx.DiGraph): Directed graph object.
    """
    if seed is not None:
        np.random.seed(seed)

    nodes = np.arange(N)
    np.random.shuffle(nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    if graph_type == 'ER':
        # Erdős-Rényi random DAG
        for i in range(N):
            for j in range(i + 1, N):
                if np.random.rand() < edge_prob:
                    weight = np.random.uniform(*weight_range)
                    G.add_edge(nodes[i], nodes[j], weight=round(weight, 2))

    elif graph_type == 'SF':
        # Scale-Free DAG using Barabási-Albert model
        m = max(1, int(edge_prob * N))  # number of edges to attach from a new node to existing nodes
        G_temp = nx.barabasi_albert_graph(N, m, seed=seed)
        G_temp = nx.DiGraph(G_temp)

        # Orient edges according to node ordering to avoid cycles
        for u, v in G_temp.edges():
            if nodes.tolist().index(u) < nodes.tolist().index(v):
                source, target = u, v
            else:
                source, target = v, u
            weight = np.random.uniform(*weight_range)
            G.add_edge(nodes[source], nodes[target], weight=round(weight, 2))

    elif graph_type == 'BP':
        # Bipartite DAG
        top_size = int(0.2 * N)
        bottom_size = N - top_size
        top_nodes = nodes[:top_size]
        bottom_nodes = nodes[top_size:]

        possible_edges = [(u, v) for u in top_nodes for v in bottom_nodes]
        selected_edges = np.random.choice(len(possible_edges), size=int(edge_prob * len(possible_edges)), replace=False)
        
        for idx in selected_edges:
            u, v = possible_edges[idx]
            weight = np.random.uniform(*weight_range)
            G.add_edge(u, v, weight=round(weight, 2))
    
    else:
        raise ValueError(f"Unknown graph type '{graph_type}'. Choose from 'ER', 'SF', or 'BP'.")

    # Build adjacency matrix
    A = np.zeros((N, N))
    for u, v, attr in G.edges(data=True):
        A[u, v] = attr['weight']

    if enforce_ar1:
        for j in range(N):
            col_norm = np.linalg.norm(A[:, j], ord=2)
            if col_norm >= 1.0:
                A[:, j] = A[:, j] / (col_norm + 1e-6) * 0.99  # enforce strict < 1

    return A, G