import numpy as np
import sdp_solvers as solvers
import clustering_utils as cu
import data_visualization_tools as dv


def large_move_maxcut(C, K, lb_init, move_type="ab", ab_sequence=None, num_max_it=100, use_IPM=False):
    """
    Approximately solve the Max-K-Cut problem represented by C using a large move local search

    :param C: (2d array[float], NxN) - Weight matrix calculated from N data-points that represents the graph to be partitioned
    :param K: (integer) - Number of desired partitions
    :param lb_init: (1d array[integer]) - Initial labeling (partition)
    :param move_type: (string) - Which large move to be used ("ab" for alpha-beta swap, "ae" for alpha-expansion and 
           "ae_bs" for alpha-expansion beta-shrink)
    :param ab_sequence: (2d array[integer], 2xK) - Sequence of alpha labels and beta labels during the iterations
    :param num_max_it: (integer) - Maximum number of iterations
    :param use_IPM: (boolean) - Use Interior Point Method to solve the SDP problem
    :return: (1d array[integer]) - Approximate final labeling (partitioning)
    """
    assert move_type in ("ab", "ae", "ae_bs")
    if ab_sequence is None:
        alpha_sequence = range(K)
        beta_sequence = range(K)
    else:
        if move_type == "ae":
            alpha_sequence = ab_sequence[0]
            beta_sequence = [-1]
        else:
            alpha_sequence = ab_sequence[0, :]
            beta_sequence = ab_sequence[1, :]

    lb = np.copy(lb_init)

    # Iterate moves ----------------------------------------------------------------------------------------------------
    it, max_ene, err = 1, 0, np.inf
    while err > 1e-10 and it < num_max_it:
        lb_prev = np.copy(lb)

        for alpha in alpha_sequence:
            for beta in beta_sequence:
                if alpha != beta:
                    if move_type == "ab":
                        # print("Swapping (%d, %d), Current Class.: %s" % (alpha, beta, lb))
                        new_lb = abswap_sdp(C, np.copy(lb), alpha, beta, use_IPM=use_IPM)
                    elif move_type == "ae":
                        # print("Expanding (%d), Current Class.: %s" % (alpha, lb))
                        new_lb = aexp_sdp(C, np.copy(lb), alpha, use_IPM=use_IPM)
                    elif move_type == "ae_bs":
                        # print("Expanding/shrinking (%d, %d), Current Class.: %s" % (alpha, beta, lb))
                        new_lb = aexp_bshrk_sdp(C, np.copy(lb), alpha, beta, use_IPM=use_IPM)

                    ene = cu.energy_clustering(C, new_lb)
                    if ene >= max_ene:
                        max_ene = ene
                        lb = new_lb

                    # If it is an alpha expansion, there is no need to iterate over beta_sequence
                    if move_type == "ae":
                        break
        it += 1
        err = np.linalg.norm(lb - lb_prev)

    return lb, it


def abswap_sdp(C_initial, lb, alpha, beta, use_IPM=False):
    """
    Executes an alpha-beta swap step

    :param C_initial: (2d array[float], NxN) - initial weight matrix computed from all initial data
    :param lb: (1d array[integer]) - Current labeling
    :param alpha & beta: (integers) Labels to be swapped
    :param use_IPM: (boolean) - Use Interior Point Method to solve the SDP problem
    :return: (1d array[integer]) - New labeling with alpha and bet swapped
    """
    # Select the points whose labels are alpha or beta -----------------------------------------------------------------
    ab_indices = np.nonzero((lb == alpha) | (lb == beta))[0]
    if ab_indices.size == 0:
        return lb

    # Adjacency matrix -------------------------------------------------------------------------------------------------
    C = C_initial[np.ix_(ab_indices, ab_indices)]
    
    # Solve & round ----------------------------------------------------------------------------------------------------
    try:
        int_sol = solve_round_sdp(C, use_IPM=use_IPM)
    except Exception:
        return lb
    
    # Update labels ----------------------------------------------------------------------------------------------------
    lb[ab_indices] = alpha * (int_sol > 0).astype(int) + beta * (int_sol < 0).astype(int)
    return lb


def aexp_sdp(C_initial, lb, alpha, use_IPM=False):
    """
    Executes an alpha-expansion step

    :param C_initial: (2d array[float], NxN) - initial weight matrix computed from all initial data
    :param lb: (1d array[integer]) - Current labeling
    :param alpha: (integer) Label to be expanded
    :param use_IPM: (boolean) - Use Interior Point Method to solve the SDP problem
    :return: (1d array[integer]) - New labeling with alpha expanded
    """
    # Fill adjacency matrix --------------------------------------------------------------------------------------------
    C = create_expansion_gadget(C_initial, lb, alpha)
    if C is None:
        return lb

    # Solve & round ----------------------------------------------------------------------------------------------------
    int_sol = solve_round_sdp(C, use_IPM=use_IPM)

    # Update labels ----------------------------------------------------------------------------------------------------
    non_alpha_indices = np.nonzero(lb != alpha)[0]
    lb[non_alpha_indices[int_sol[0:-1] == int_sol[-1]]] = alpha
    return lb


def aexp_bshrk_sdp(C_initial, lb, alpha, beta, use_IPM=False):
    """
    Executes an alpha-expansion beta-shrink step

    :param C_initial: (2d array[float], NxN) - initial weight matrix computed from all initial data
    :param lb: (1d array[integer]) - Current labeling
    :param alpha: (integer) Label to be expanded
    :param beta: (integer) Label to be shrunk
    :param use_IPM: (boolean) - Use Interior Point Method to solve the SDP problem
    :return: (1d array[integer]) - New labeling with alpha expanded and bet shrunk
    """
    # Turn the current alpha labels into beta --------------------------------------------------------------------------
    lb[np.nonzero(lb == alpha)[0]] = beta

    # Do an alpha expansion on the new labeling-------------------------------------------------------------------------
    new_lb = aexp_sdp(C_initial, np.copy(lb), alpha, use_IPM=use_IPM)
    return new_lb


def create_expansion_gadget(C_initial, lb, alpha):
    """
    Create gadget that will be used on the alpha expansion step

    :param C_initial: (2d array[float], NxN) - initial weight matrix computed from all initial data
    :param lb: (1d array[integer]) - Current labeling
    :param alpha: (integer) Label to be expanded
    :return: (2d array[float]) - Weight matrix that represents the gadget's adjacency matrix
    """
    # Select the points whose labels are alpha and the ones whose labels are not alpha ---------------------------------
    alpha_indices = np.nonzero(lb == alpha)[0]
    non_alpha_indices = np.nonzero(lb != alpha)[0]

    if (non_alpha_indices.size == 0):
        return None

    cl_n_alpha = lb[non_alpha_indices]

    # Initialize the adjacency matrix with the weights (shrink all vertices that are not labeled alpha in one supernode)
    N = len(non_alpha_indices) + 1
    C = np.zeros((N, N))
    C[0:-1, 0:-1] = C_initial[np.ix_(non_alpha_indices, non_alpha_indices)]

    if alpha_indices.shape[0] == 0:
        dists_to_alpha = np.zeros(N - 1)
    else:
        dists_to_alpha = np.sum(C_initial[np.ix_(non_alpha_indices, alpha_indices)], axis=1)

    for i in range(N - 1):
        aux = dists_to_alpha[i]
        C[i, -1] = aux
        C[-1, i] = aux
        for j in range(N - 1):
            if cl_n_alpha[i] != cl_n_alpha[j]:
                C[i, j] /= 2.

                C[i, -1] += C[i, j]
                C[-1, i] += C[i, j]

    return C


def solve_round_sdp(C, use_IPM=False):
    """
    Solve the Max-Cut SDP problem represented by C

    :param C: (2d array[float], NxN) - Max-Cut weight matrix
    :param use_IPM: (boolean) - Use Interior Point Method to solve the SDP problem
    :return: (1d array[integer]) - Approximate optimal partition of the graph represented by C
    """
    # Solve SDP using the Interior Point Method (Picos) or ADMM --------------------------------------------------------
    if use_IPM:
        X, _, _ = solvers.maxcut_ipm_solver(C)
    else:
        X, _, _, _ = solvers.maxkcut_admm_solver(C, 2)

    # dv.plot_matrix(X)

    # Embedding --------------------------------------------------------------------------------------------------------
    V = np.linalg.cholesky(X + 1e-9 * np.trace(X) * np.eye(X.shape[0])) 

    # Select the best cutting plane (the one that maximizes the objective) ---------------------------------------------
    max_ene = 0
    num_rounding_trials = max(100, np.floor_divide(C.shape[0], 2))
    best = solvers.hyper_plane_rounding(V)
    for i in range(num_rounding_trials):

        v = solvers.hyper_plane_rounding(V)

        ene = cu.energy_clustering(C, v)
        if ene > max_ene:
            best = v
            max_ene = ene

    return best