import numpy as np
import scipy as sp
import picos as pic
import itertools
import time
import clustering_utils as cu
import itertools


# BRUTE FORCE METHODS (TODO) ===========================================================================================
def maxcut_brute_force_solver(C):
    N = C.shape[0]
    lst = list(itertools.product([0, 1], repeat=N))
    all_partitions = list(itertools.product([-1, 1], repeat=N))
    max_ene = 0
    best = all_partitions[0]
    for lb in all_partitions:
        ene = cu.energy_clustering(C, lb)
        if ene > max_ene:
            best = lb
            max_ene = ene

    return best

# INTERIOR POINT METHODS & UTILS =======================================================================================
def maxcut_ipm_solver(C):
    """
    Solve the Max-Cut SDP problem with Interior Point Method

    :param C: (2d array[float], NxN) - Weight Matrix representd the graph to be partitioned
    :return: (2d array[float], NxN) - SDP solution,
             (1d array[integer]) - Final objective value,
             (float) - Final elapsed time

    """
    N = C.shape[0]

    # SDP Creation =====================================================================================================
    max_cut = pic.Problem()
    C = pic.new_param('C', C)
    X = max_cut.add_variable('X', (N, N), 'symmetric')
    max_cut.add_constraint(pic.tools.diag_vect(X) == 1)
    max_cut.add_constraint(X >> 0)
    max_cut.set_objective('max', C | (1 - X))

    # Solve SDP ========================================================================================================
    start_time = time.time()
    max_cut.solve(verbose=0)  # Solve SDP
    elapsed_time = time.time() - start_time  # Calculate execution time

    return np.array(X.value), 0.5 * max_cut.obj_value(), elapsed_time


def maxkhypercut_ipm_solver(P, K, l, delta=0, squared_dist=False, use_clique_expansion=False):
    """
    Solve Max-K-Hypercut SDP problem with Interior Point Method

    :param P: (2d array[float], Nxd) - Dataset of N points in d dimensions
    :param K: (integer) - Number of clusters
    :param l: (integer) - Subspace dimension
    :param delta: (float) - parameter used in the hypergraph reduction sampling
    :param squared_dist: (boolean) - use squared distances to calculate volumes
    :param use_clique_expansion: (boolean) - use the clique expansion technique
    :return: (1d array[integer]) - Partition,
             (1d array[integer]) - Final objective value,
             (float) - Final elapsed time
    """
    N = P.shape[0]

    # Calculate Volumes ================================================================================================
    C = np.array(list(itertools.combinations(range(N), l)))  # All possible combinations of size set_size for N p
    w = np.array([cu.compute_volume(P, s, squared_dist) for s in C])  # Vector of weights given by each subset

    # Clique Expansion =================================================================================================
    if use_clique_expansion:
        C, w = clique_expansion(C, w, N)
        l = 2

    # Sampling Procedure ===============================================================================================
    if delta != 0:
        C, w = graph_sampling(C, w, N)

    # SDP Creation =====================================================================================================
    cluster_problem = pic.Problem()

    X = cluster_problem.add_variable('X', (N, N), 'symmetric')  # Matrix of inner products as a parameter
    z = cluster_problem.add_variable('z', len(w))  # Factor indicators as a parameter
    w = pic.new_param('w', w)  # Weights as a parameter

    C1 = pic.new_param('1/(|S_j| - 1) * (K-1)/K',
                       1 / (float(l) - 1) * (float(K) - 1) / float(K))  # Constant for the z's constraints
    C2 = pic.new_param('-1/(K - 1)', -1 / (float(K) - 1))  # Constant for the X_ij constraints

    # Constraint on z
    cluster_problem.add_list_of_constraints(
        [C1 * pic.tools.sum([1 - X[s[0], s[1]] for s in list(itertools.combinations(C[j], 2))]) > z[j]
         for j in range(len(C))],
        ['i', 'k'],
        '|S_j|, i < k, for all j'
    )

    # Constraint on X_ii
    cluster_problem.add_constraint(pic.tools.diag_vect(X) == 1)

    # Constraints on X_ij, i != j
    cluster_problem.add_list_of_constraints(
        [X[i, j] > C2 for i, j in itertools.product(range(N), range(N)) if i > j],
        ['i', 'j'],
        'non-diagonal entries of X'
    )

    # Constraints on the semipositiveness of X and the positiveness of z
    cluster_problem.add_constraint(X >> 0)
    cluster_problem.add_constraint(z > 0)
    cluster_problem.add_constraint(z <= 1)

    cluster_problem.set_objective('max', w | z)  # Set objective

    # Solve SDP ========================================================================================================
    start_time = time.time()
    cluster_problem.solve(verbose=0)  # Solve SDP
    elapsed_time = time.time() - start_time  # Calculate execution time

    return np.array(X.value), cluster_problem.obj_value(), elapsed_time


def clique_expansion(C, w, N):
    mu = 0.5
    pairs = np.array(list(itertools.combinations(range(N), 2)))
    new_w = [np.sum([w[index]/mu for c, index in zip(C, range(len(C))) if set(p).issubset(c)]) for p in pairs]

    return pairs, np.array(new_w)


def graph_sampling(C, w, delta):
    r = int((delta ** (-2)) * 30)
    chosen_idx = np.random.choice(len(w), r, p=w / np.sum(w), replace=True)
    hist = np.histogram(chosen_idx, bins=range(len(C)))[0]
    C = C[np.squeeze(np.argwhere(hist != 0))]
    w = hist[np.squeeze(np.argwhere(hist != 0))]

    return C, w


# ADMM METHODS & UTILS =================================================================================================
def maxkcut_admm_solver(C, K, num_max_it=5000, epsilon=1e-8, alpha=0):
    """
    Solve Max-K-Hypercut SDP problem with Alternate Direction Multipliers Method

    :param C: (2d array[float], NxN) - Weight Matrix represents the graph to be partitioned
    :param K: (integer) - Number of partitions
    :param num_max_it: (integer) - Maximum Number of ADMM iterations
    :param epsilon: (float) - Desired final error
    :param alpha: (float) - Dimensionality of the low-rank eigenvalue problem (alpha=0 for full rank)
    :return: (2d array[float], NxN) - SDP solution,
             (float) - Final error,
             (float) - Final elapsed time
             (integer) - Total number of iterations
    """

    def finish_iteration(X_f, X_i, y, nu, C, b, d, it, params):
        """
        Function used to monitor the error in each ADMM iteration
        """
        # Primal Infeasibility -----------------------------------------------------------------------------------------
        pinf = (np.linalg.norm(np.diag(X_f) - b) + np.linalg.norm(np.minimum(X_f - d, 0))) / (1 + np.linalg.norm(b))

        # Dual Infeasibility -------------------------------------------------------------------------------------------
        dinf = np.linalg.norm(params['mu'] * (X_f - X_i)) / (1 + np.linalg.norm(C, ord=1))

        # Gap ----------------------------------------------------------------------------------------------------------
        if np.remainder(it, params['check_finish_rate']) == 0:
            CX = np.trace(C.dot(X_f))
            y_nu = np.vdot(b, y) + np.vdot(d, nu)
            gap = np.abs(CX - y_nu) / (1 + CX + y_nu)
            params['prev_gap'] = gap
        else:
            gap = params['prev_gap']

        # print("> pinf: %.4e" % pinf)
        # print("> dinf: %.4e" % dinf)
        # print("> gap: %.4e" % gap)
        # print("> mu: %.4e" % params['mu'])

        # Total error assessment ---------------------------------------------------------------------------------------
        error = np.max(np.abs([pinf, dinf, gap]))
        stop = error < params['epsilon']

        # Update mu  ---------------------------------------------------------------------------------------------------
        if pinf / dinf <= 1:
            params['it_pinf'] += 1
            params['it_dinf'] = 0
            if params['it_pinf'] >= params['h']:
                params['mu'] = max(params['gamma'] * params['mu'], params['mu_min'])
                params['it_pinf'] = 0
        else:
            params['it_dinf'] += 1
            params['it_pinf'] = 0
            if params['it_dinf'] >= params['h']:
                params['mu'] = min((1. / params['gamma']) * params['mu'], params['mu_max'])
                params['it_dinf'] = 0

        return stop, params, error

    params = {'mu': 5,
              'pho': 1.6,
              'mu_max': 1e4,
              'mu_min': 1e-4,
              'gamma': .5,
              'epsilon': epsilon,  # Desired error
              'it_pinf': 0,
              'it_dinf': 0,
              'h': 50,
              'num_max_it': num_max_it,
              'check_finish_rate': 5,
              'prev_gap': 0,
              'r': alpha * K}  # Dimension of the low rank apporximation of X

    # SDP Variables ----------------------------------------------------------------------------------------------------
    #  Objective function's term
    N = C.shape[0]
    X = np.zeros_like(C)
    S = np.zeros_like(C)
    b = np.ones(N)  # RHS of equality constraints
    d = (-1.0 / (K - 1)) * np.ones((N, N))  # RHS of inequality constraints
    np.fill_diagonal(d, 0)

    # ADMM iterations (According to [1]) ===============================================================================
    stop = False
    elap_time_step = np.zeros(params['num_max_it'])
    err = np.zeros(params['num_max_it'])
    t_start = time.time()
    it = 0
    error = np.inf
    while not stop and it < params['num_max_it']:
        t_start_it = time.time()

        # Update Y -----------------------------------------------------------------------------------------------------
        y = -(params['mu'] * (np.diag(X) - b) + np.diag(S)).T

        # Update nu ----------------------------------------------------------------------------------------------------
        nu = np.maximum(-(params['mu'] * (X - d) + (S - C)), 0)
        np.fill_diagonal(nu, 0)

        # Update Y (Following the ideas on [2]) ------------------------------------------------------------------------
        W = C - np.diag(y) - nu
        if alpha == 0:
            sigma, U = np.linalg.eigh(X - W / params['mu'])
        else:
            sigma, U = sp.sparse.linalg.eigsh(X - W / params['mu'], params['r'], which='LM')

        X_f = np.linalg.multi_dot([U, np.diag(np.maximum(sigma, 0)), U.T])
        S = W + params['mu'] * (X_f - X)

        # Assess iteration ---------------------------------------------------------------------------------------------
        stop, params, error = finish_iteration(X_f, X, y, nu, C, b, d, it, params)
        elap_time_step[it] = time.time() - t_start_it
        err[it] = error
        # print("> Error: %.6e " % error)
        # print("> Iteration elapsed time: %.3f\n" % elap_time_step[it])
        # print("> ADMM Iteration: %d" % (it + 1))

        X = params['pho'] * X_f + (1 - params['pho']) * X
        it += 1

    elapsed_time = time.time() - t_start
    elapsed_time_step_mean = np.mean(elap_time_step[0:it])

    return X, error, elapsed_time, it


# ROUNDING UTILS =======================================================================================================
def hyper_plane_rounding(V):
    """
    Hyper plane rounding algorithm used in Max-Cut SDP problems

    :param V: (2d array[float], NxN) - Embedded vectors (rows) in a R^{N-1} unit sphere
    :return: (1d array[integer]) - Rounded partition
    """
    N = V.shape[0]
    w = np.random.randn(N)
    v = V.dot(w)
    return (v > 0).astype(int) - (v < 0).astype(int)