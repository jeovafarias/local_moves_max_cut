import sklearn.metrics.pairwise as skl
import numpy as np
import sdp_solvers


# CLUSTERING ASSESSMENT TOOLS ==========================================================================================
def stats_clustering(C, lb, gt):
    """
    Compute the purity, the clustering energy and the energy percentage of a partition (clustering)

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :return: (float) - purity,
             (float) - clustering energy,
             (float) - energy percentage
    """
    return purity(lb, gt), energy_clustering(C, lb), percentage_energy_clustering(C, lb)


def energy_clustering(C, lb):
    """
    Compute clustering energy, i.e. the sum of edges cut from the graph

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - clustering energy
    """
    assert C.shape[0] == C.shape[1]
    ene = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            if lb[i] != lb[j]:
                ene += C[i, j]
    return ene


def percentage_energy_clustering(C, lb):
    """
    Compute energy percentage

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - energy percentage
    """
    return energy_clustering(C, lb) / np.sum(C)


def purity(lb, gt):
    """
    Compute clustering purity

    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :return: (float) - clustering purity
    """
    lb = np.array(lb)
    gt = np.array(gt)
    K = len(set(gt))

    population_vec = [float(np.sum([lb == i])) for i in range(K)]
    h = np.zeros(K)
    for i in range(K):
        if population_vec[i] > 0:
            h[i] = np.max(np.bincount(gt[lb == i])) / population_vec[i]
        else:
            h[i] = 0
    return np.average(h, weights=population_vec)


# CLUSTERING METHODS USING SDP SOLVERS =================================================================================
def iterate_sdp(C, K, solver='admm', num_max_it=100):
    """
    Iterate SDP clustering

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param K: (integer) - Number of clusters (partitions)
    :param solver: (string) - SDP solver to be used ("admm" for ADMM and "IPM" for Interior Point Method)
    :param num_max_it: (integer) - Maximum number of SDP iterations
    :return: (2d array[float], NxN) - Final pairwise partitioning ,
             (float) - Total elapsed time after iterations,
             (integer) - Number of executed SDP iterations
    """
    X = np.zeros_like(C)
    V = np.linalg.cholesky(X + 1e-9 * np.trace(X) * np.eye(X.shape[0]))
    err, num_it_SDP, total_elapsed_time = np.inf, 0, 0
    while num_it_SDP < num_max_it and err > 1e-8:
        X_int = np.copy(X)

        if solver == 'admm':
            C = skl.pairwise_distances(V, metric='sqeuclidean')
            X, _, elapsed_time, _ = sdp_solvers.maxkcut_admm_solver(C, K)
        elif solver == 'ipm':
            X, _, elapsed_time = sdp_solvers.maxkhypercut_ipm_solver(V, K, 2)

        V = np.linalg.cholesky(X + 1e-9 * np.trace(X) * np.eye(X.shape[0]))

        total_elapsed_time += elapsed_time
        err = np.linalg.norm(X - X_int)
        num_it_SDP += 1

    if num_it_SDP < num_max_it:
        print("Converged at iteration: %d" % num_it_SDP)

    return X, total_elapsed_time, num_it_SDP


def sdp_clustering(P, K, l=2, solver='admm', delta=0, do_iterate=False):
    """
    Cluster a dataset in K partitions in subspaces of dimestion (l-2) using a SDP formulation

    :param P: (2d array[float], Nxd) - Dataset of N points in d dimensions
    :param K: (integer) - Number of clusters
    :param l: (integer) - Subspace dimension
    :param solver: (string) - SDP solver to be used ("admm" for ADMM and "IPM" for Interior Point Method)
    :param delta: (float) - parameter used in the hypergraph reduction sampling
    :param do_iterate: (boolean) - Iterate clustering
    :return: (1d array[integer]) - Final labeling (clustering)
    """
    C = skl.pairwise_distances(P, metric='sqeuclidean')
    if do_iterate:
        X, _, _ = iterate_sdp(C, K, solver=solver)
    else:
        if solver == 'admm':
            C = skl.pairwise_distances(P, metric='sqeuclidean')
            X, _, _, _ = sdp_solvers.maxkcut_admm_solver(C, K)
        elif solver == 'ipm':
            X, _, _ = sdp_solvers.maxkhypercut_ipm_solver(P, K, l, delta=delta)

    V = np.linalg.cholesky(X + 1e-9 * np.trace(X) * np.eye(X.shape[0]))

    best_lb_nn = np.zeros(X[0], dtype=int)
    num_rounding_trials = max(1000, np.floor_divide(C.shape[0], 2))
    min_ene = np.inf
    for i in range(num_rounding_trials):

        lb_nn = nearest_neighbours(V, K)

        energy = energy_clustering(C, lb_nn)
        if energy < min_ene:
            best_lb_nn = lb_nn
            min_ene = energy

    return best_lb_nn


# OTHER CLUSTERING METHODS =============================================================================================
def local_search(C, K, lb_init, num_max_it=100):
    """
    Simple local search algorithm to clustering and Max-K-Cut solving

    :param C: (2d array[float], NxN) - Weight matrix from Max-K-Cut problem
    :param K: (integer) - Number of clusters
    :param lb_init: (1d array[integer]) - Initial labeling
    :param num_max_it: (integer) - Maximum number of iterations on the local search
    :return: (1d array[integer]) - Final labeling
    """
    def min_cost(C, K, cl, i):
        c = C[:, i]
        return np.argmin([np.sum(c[np.nonzero(cl == k)[0]]) for k in range(K)])

    err, it = np.inf, 0
    lb = lb_init
    while err > 1e-10 and it < num_max_it:
        lb_prev = np.copy(lb)
        it += 1
        for i in range(len(lb_init)):
            lb[i] = min_cost(C, K, lb, i)
        err = np.linalg.norm(lb - lb_prev)

    return lb


# OTHER FUNCTIONS ======================================================================================================
def k_simplex(K):
    """
    Generate a K-uniform simplex in K dimensions
    """
    if K != 2:
        M = 1. / (float(K) - 1) * np.ones((K, K))
        np.fill_diagonal(M, 1.0)
        V = np.linalg.cholesky(M)
        new_V = V
        new_V = new_V - np.mean(new_V, axis=0)
        new_V = new_V / np.linalg.norm(new_V, axis=1)[:, np.newaxis]
    else:
        new_V = np.array([[1, -1], [-1, 1]])

    return new_V


def nearest_neighbours(V, K, post_processing=True, max_tol=100):
    """
    Compute the K nearest neighbours rounding procedure to Max-K-Cut SDP problem

    :param V: (2d array[float], NxN) - Embedding created by the SDP solver
    :param K: (integer) - Number of partitions to be found
    :param post_processing: (boolean) - Do post processing in order to find neigbours to all K vectors
                            representing the partitions
    :param max_tol: (integer) - Maximum number of nearest neighbours retrials in the post-processing step
    :return: (1d array[integer]) -  Final partitioning
    """
    N = V.shape[0]
    P = np.random.randn(N, K)
    P /= np.linalg.norm(P, axis=0)

    lb = np.argmin(skl.pairwise_distances(P.T, V), axis=0)
    if post_processing:
        num_attempts = 0
        prev_len_ind = 0
        while len(np.unique(lb)) != K and num_attempts < max_tol:
            ind_non_used = list(set(range(K)) - set(lb))
            new_P = np.random.randn(N, len(ind_non_used))
            new_P /= np.linalg.norm(new_P, axis=0)
            P[:, ind_non_used] = new_P
            lb = np.argmin(skl.pairwise_distances(P.T, V), axis=0)
            if prev_len_ind <= len(ind_non_used):
                num_attempts += 1
                prev_len_ind = len(ind_non_used)
            else:
                num_attempts = 0
                prev_len_ind = 0

    return lb


def compute_volume(P, ind, squared_dist=False):
    """
    Compute volume given by the points P
    """
    M = P[ind[0:-1], :] - P[ind[-1], :]
    if squared_dist:
        return np.abs(np.linalg.det(np.dot(M, M.T)))
    else:
        return np.abs(np.sqrt(np.linalg.det(np.dot(M, M.T))))



