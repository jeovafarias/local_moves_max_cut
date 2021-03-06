import sklearn.metrics as skl
import numpy as np
import sdp_solvers as solvers
import itertools


# CLUSTERING ASSESSMENT TOOLS ==========================================================================================
def stats_clustering_pairwise(C, lb, gt, P=None, use_other_measures=False):
    """
    Compute the purity, the min purity, the clustering energy and the energy percentage of a partition (clustering)
    whose weights are pairwise

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :return: (float) - purity,
             (float) - clustering energy,
             (float) - CH index,
             (float) - Silhouette,
             (float) - DB index,
             (float) - DU index
    """
    if use_other_measures:
        return purity(lb, gt, type="ave"), purity(lb, gt, type="min"), \
                energy_clustering_pairwise(C, lb), percentage_energy_clustering_pairwise(C, lb), \
                CH(P, lb), SI(C, lb), DB(P, lb), DU(P, lb)
    else:
        return purity(lb, gt, type="ave"), purity(lb, gt, type="min"), \
               energy_clustering_pairwise(C, lb), percentage_energy_clustering_pairwise(C, lb)

def stats_clustering_triples(C, lb, gt):
    """
    Compute the purity, the min purity, the clustering energy and the energy percentage of a partition (clustering)

    :param C: (3d array[float], NxNxN) - Triple weights
    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :return: Purity, Min purity, Energy, Percent of total energy
    """
    return purity(lb, gt, type="ave"), purity(lb, gt, type="min"), \
           energy_clustering_triples(C, lb), percentage_energy_clustering_triples(C, lb)



def stats_clustering_high_order(E, w, lb, gt):
    """
    Compute the purity, the min purity, the clustering energy and the energy percentage of a partition (clustering)

    :param E: (List of list of Integers) - Hypergraph vertices
    :param w: (1d array[float], |E|) - Hypergraph weights
    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :return: Purity, Min purity, Energy, Percent of total energy
    """
    return purity(lb, gt, type="ave"), purity(lb, gt, type="min"), \
           energy_clustering_high_order(E, w, lb), percentage_energy_clustering_high_order(E, w, lb)


def energy_clustering_pairwise(C, lb):
    """
    Compute clustering energy for pairwise clustering, i.e. the sum of edges cut from the graph

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

def energy_clustering_triples(C, lb):
    """
    Compute clustering energy in triples clustering, i.e. the sum of edges cut from the hypergraph

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - clustering energy
    """
    ene = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            for k in range(C.shape[0]):
                if (lb[i] != lb[j]) or (lb[i] != lb[k]):
                    ene += C[i, j, k]
    return ene


def energy_clustering_high_order(E, w, lb):
    """
    Compute clustering energy in hypergraph clustering, i.e. the sum of edges cut from the hypergraph

    :param E: (List of list of Integers) - Hypergraph vertices
    :param w: (1d array[float], |E|) - Hypergraph weights
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - clustering energy
    """
    num_edges = len(w)
    ene = 0
    for i in range(num_edges):
        labels = np.array([lb[j] for j in E[i]])
        if not np.all(labels == labels[0]):
            ene += w[i]
    return ene


def percentage_energy_clustering_pairwise(C, lb):
    """
    Compute energy percentage for pairwise clustering

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - energy percentage
    """
    return energy_clustering_pairwise(C, lb) / np.sum(C)

def percentage_energy_clustering_triples(C, lb):
    """
    Compute energy percentage for triples clustering

    :param C: (3d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - energy percentage
    """
    return energy_clustering_triples(C, lb) / np.sum(C)


def percentage_energy_clustering_high_order(E, w, lb):
    """
    Compute energy percentage for hypergraph clustering

    :param E: (List of list of Integers) - Hypergraph vertices
    :param w: (1d array[float], |E|) - Hypergraph weights
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - energy percentage
    """
    return energy_clustering_high_order(E, w, lb) / np.sum(w)


def purity(lb, gt, type="ave"):
    """
    Compute clustering purity

    :param lb: (1d array[integer]) - Labeling to be assessed
    :param gt: (1d array[integer]) - Ground Truth labeling
    :param type: string - Purity type: "min" for min-purity; "ave" for average
    :return: (float) - clustering purity
    """
    assert type in ["min", "ave"]

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

    if type == "ave":
        return np.average(h, weights=population_vec)
    else:
        return np.min(h)


def SI(C, lb):
    """
    Compute silhouette score of a labeling

    :param C: (2d array[float], NxN) - Weight matrix from the data (N points)
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - score
    """
    return skl.silhouette_score(C, lb, metric='precomputed')


def CH(P, lb):
    """
    Compute Calinski-Harabaz Index of a labeling

    :param P: (2d array[float], Nxdim) - Clustered points
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - CH index
    """
    return skl.calinski_harabaz_score(P, lb)


def DB(P, lb):
    """
    Compute the Davies-Bouldin Index of a labeling

    :param P: (2d array[float], Nxdim) - Clustered points
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - DB index
    """

    labels = np.unique(lb)
    K = len(labels)

    c = np.matrix([np.mean(P[np.nonzero(lb == i)[0], :], axis=0) for i in range(K)])
    mean_dist_to_centers = [np.mean(skl.pairwise.pairwise_distances(P[np.nonzero(lb == i)[0], :], c[i]))
                            for i in range(K)]
    dist_center_to_center = skl.pairwise.pairwise_distances(c)

    score = np.zeros([K, K])
    for i in range(K):
        for j in range(K):
            if i != j:
                score[i, j] = (mean_dist_to_centers[i] + mean_dist_to_centers[j]) / dist_center_to_center[i, j]

    return np.mean(np.max(score, axis=0))


def DU(P, lb):
    """
    Compute the Dunn Index of a labeling

    :param P: (2d array[float], Nxdim) - Clustered points
    :param lb: (1d array[integer]) - Labeling to be assessed
    :return: (float) - DU index
    """

    labels = np.unique(lb)
    K = len(labels)

    numerator = np.inf * np.ones([K, K])
    for i in range(K):
        for j in range(K):
            if i != j:
                numerator[i, j] = np.min(skl.pairwise.pairwise_distances(P[np.nonzero(lb == i)[0], :],
                                                                         P[np.nonzero(lb == j)[0], :]))

    denominator = np.max([np.max(skl.pairwise.pairwise_distances(P[np.nonzero(lb == j)[0], :])) for j in range(K)])

    return np.min(numerator) / denominator


# CLUSTERING METHODS USING SDP SOLVERS =================================================================================
def iterate_sdp(C, K, num_max_it=50, alpha=0):
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
    X = np.ones_like(C)
    err, num_it_SDP, total_elapsed_time = np.inf, 0, 0
    while num_it_SDP < num_max_it and err > 1e-6:
        X_int = np.copy(X)

        X, _, elapsed_time, _ = solvers.maxkcut_admm_solver(C, K, num_max_it=num_max_it, alpha=alpha)
        C = 1 - X

        total_elapsed_time += elapsed_time
        err = np.linalg.norm(X - X_int)
        num_it_SDP += 1

    # if num_it_SDP < num_max_it:
    #     print("Converged at iteration: %d" % num_it_SDP)

    return X, total_elapsed_time, num_it_SDP, err


def cluster_integer_sol(X, K):
    N = X.shape[0]
    lb = -1 * np.ones(N)
    k_idx = 0
    for i in range(N):
        vec_aux = X[i, :]
        for j in range(i):
            if vec_aux[j] >= 0.4:
                lb[i] = lb[j]

        if lb[i] == -1:
            lb[i] = k_idx
            k_idx += 1

    if k_idx == K:
        return lb
    else:
        raise Exception('Error on rounding')

# # Old code ===========================================================================================================
# def sdp_clustering(P, K, l=2, solver='admm', delta=0, do_iterate=False):
#     """
#     Cluster a dataset in K partitions in subspaces of dimestion (l-2) using a SDP formulation
#
#     :param P: (2d array[float], Nxd) - Dataset of N points in d dimensions
#     :param K: (integer) - Number of clusters
#     :param l: (integer) - Subspace dimension
#     :param solver: (string) - SDP solver to be used ("admm" for ADMM and "IPM" for Interior Point Method)
#     :param delta: (float) - parameter used in the hypergraph reduction sampling
#     :param do_iterate: (boolean) - Iterate clustering
#     :return: (1d array[integer]) - Final labeling (clustering)
#     """
#     C = skl.pairwise.pairwise_distances(P, metric='sqeuclidean')
#     N = P.shape[0]
#     if do_iterate:
#         X, _, _ = iterate_sdp(C, K, solver=solver)
#     else:
#         if solver == 'admm':
#             C = skl.pairwise.pairwise_distances(P, metric='sqeuclidean')
#             X, _, _, _ = solvers.maxkcut_admm_solver(C, K)
#         elif solver == 'ipm':
#             E = np.array(list(itertools.combinations(range(N), l)))
#             w = np.array([compute_volume(P, s) for s in C])
#
#             if delta != 0:
#                 E, w = solvers.graph_sampling(E, w, delta)
#
#             X, _, _ = solvers.maxkhypercut_ipm_solver(E, w, K, N, l)
#
#     V = np.linalg.cholesky(X + 1e-9 * np.trace(X) * np.eye(X.shape[0]))
#
#     best_lb_nn = np.zeros(X[0], dtype=int)
#     num_rounding_trials = max(1000, np.floor_divide(C.shape[0], 2))
#     min_ene = np.inf
#     for i in range(num_rounding_trials):
#
#         lb_nn = nearest_neighbours(V, K)
#
#         energy = energy_clustering_pairwise(C, lb_nn)
#         if energy < min_ene:
#             best_lb_nn = lb_nn
#             min_ene = energy
#
#     return best_lb_nn


# OTHER CLUSTERING METHODS =============================================================================================
def local_search(C, K, lb_init, num_max_it=1000):
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

    lb = np.copy(lb_init)
    it, max_ene, err = 1, 0, np.inf
    while err > 1e-5 and it < num_max_it:
        prev_ene = max_ene
        for i in range(len(lb_init)):
            lb[i] = min_cost(C, K, lb, i)
        max_ene = energy_clustering_pairwise(C,lb)
        err = max_ene-prev_ene
        it += 1

    return lb, it, err


def local_search_triples(C,K,lb_init,num_max_it=1000):
    """
    Simple local search algorithm which iteratively finds the best cluster
    for each point but adapated for triple hyperedges
    
    Args:
        C (3d array[float], NxN): weight of triple hyperedges
        k (int): number of clusters
        lb_init (array of int): clustering labels for each v in V
        num_max_it (int): maximum number of iterations run
    Returns:
        lb (1d array[integer]) - Final labeling
    """
    
    def min_cost(C, K, cl, i):
        costs = [0 for _ in range(K)]
        for k in range(K):
            k_ind = np.nonzero(cl == k)[0]
            for x in range(len(k_ind)):
                for y in range(x,len(k_ind)):
                    costs[k] += C[i,k_ind[x],k_ind[y]]
        return np.argmin(costs)

    lb = np.copy(lb_init)
    it, max_ene, err = 1, 0, np.inf
    while err > 1e-5 and it < num_max_it:
        prev_ene = max_ene
        for i in range(len(lb_init)):
            lb[i] = min_cost(C, K, lb, i)
        max_ene = energy_clustering_triples(C,lb)
        err = max_ene - prev_ene
        it += 1
    return lb, it, err

##def local_search_high_order(E,w,k,origLabels,maxIt=1000):
##    """
##    Simple local search algorithm which iteratively finds the best cluster
##    for each point but adapated for hyperedges
##    
##    Args:
##        E (list of list of int): list of hyperedges
##        w (list of float): corresponding weights of edges in E
##        k (int): number of clusters
##        origLabels (array of int): clustering labels for each v in V
##        maxIt (int): maximum number of iterations run
##    Returns:
##        labels (array of int): updated clustering labels for each v in V
##    """
##    
##    labels = np.copy(origLabels)
##    it, maxEne = 0, energy_clustering_high_order(E,w,labels)
##    updated = True
##    while updated and it <= maxIt:
##        #print(it)
##        updated = False
##        for v in range(len(labels)):
##            #print("Updating (%d), Current Class.: %s" % (v, labels))
##            for i in range(k):
##                label0 = labels[v]
##                labels[v] = i
##                ene = energy_clustering_high_order(E,w,labels)
##                if ene > maxEne:
##                    updated = True
##                    maxEne = ene
##                else:
##                    labels[v] = label0
##        it += 1
##
##
##    return labels, it


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


def compute_volume(P, ind, squared_dist=False):
    """
    Compute volume given by the points P
    """
    M = P[ind[0:-1], :] - P[ind[-1], :]
    if squared_dist:
        return np.abs(np.linalg.det(np.dot(M, M.T)))
    else:
        return np.sqrt(np.abs(np.linalg.det(np.dot(M, M.T))))

def sdp_reduction(n,E,w):
    C = np.zeros(shape=(n,n))
    for i in range(len(E)):
        e, vol = E[i], w[i]
        if len(e) == 3:
            i1, i2, i3 = e[0], e[1], e[2]
            C[i1,i2] += vol
            C[i2,i1] += vol
            C[i2,i3] += vol
            C[i3,i2] += vol
            C[i1,i3] += vol
            C[i3,i1] += vol
        elif len(e) == 2:
            i1, i2 = e[0], e[1]
            C[i1, i2] += vol
            C[i2, i1] += vol
    return C

def index_volumes(n,E):
    indices = [[] for _ in range(n)]
    m = len(E)
    for j in range(m):
        for i in e:
            indices[i].append(m)
    return indices
        
