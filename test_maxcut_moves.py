import numpy as np
import sklearn.metrics.pairwise as skl
import data_generation_tools as dg
import data_visualization_tools as dv
import clustering_utils as cu
import moves
import os
import sys
import sdp_solvers

np.set_printoptions(linewidth=1000, precision=5, threshold=np.nan, suppress=True)

if __name__ == "__main__":
    use_experiment_id = False
    if use_experiment_id:
        # Use data from experiments ------------------------------------------------------------------------------------
        experiment_dir = 'experiments_rand_int'
        experiment_id = 10
        experiment_path = experiment_dir + '/' + str(experiment_id)
        assert os.path.exists(experiment_path)

        P = np.loadtxt(experiment_path + "/P.dat")
        ground_truth = np.loadtxt(experiment_path + "/gt.dat", dtype='int')
        K = len(np.unique(ground_truth))
        dv.plot_data(P, K, ground_truth, 2)

    else:
        # Data generation parameters -----------------------------------------------------------------------------------
        n = 1
        K = 3
        sigma_2 = .0
        use_prev_p = 0
        shuffle_data = 0
        params_data = {'sigma_1': 1, 'sigma_2': sigma_2, 'K': K, 'dim_space': 2, 'pop_interv': [n, n],
                       'use_prev_p': use_prev_p, 'shuffle': shuffle_data}

        # Data generation and visualization ----------------------------------------------------------------------------
        P, ground_truth = dg.generate_data_random(params_data)

    N = P.shape[0]
    a = 5
    C = np.array([[0, a, a], [a, 0, 1], [a, 1, 0]])

    # C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, K, ground_truth, 2)
    lb = sdp_solvers.maxcut_brute_force_solver(C)

    # Other parameters -------------------------------------------------------------------------------------------------
    random_init = 0
    use_IPM = 0
    num_max_it = 20

    # Iterate ----------------------------------------------------------------------------------------------------------
    if random_init:
        lb_init = np.random.randint(0, K, N)
        ab_sequence = None
    else:
        lb_init = np.zeros(N, dtype=int)
        ab_sequence = None
        # ab_sequence = np.array([np.random.choice(K, K, replace=False), np.random.choice(K, K, replace=False)])

    # lb_ab = moves.large_move_maxcut(C, K, lb_init, move_type="ab", ab_sequence=ab_sequence,
    #                                 num_max_it=num_max_it, use_IPM=use_IPM)
    # pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ab, ground_truth)
    # print("\n> Alpha-Beta Swap")
    # print("  - Purity: %.4f" % pur)
    # print("  - Energy Clustering: %.4f" % ene_cl)
    # print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    lb_ae = moves.large_move_maxcut(C, K, lb_init, move_type="ae", ab_sequence=ab_sequence,
                                    num_max_it=num_max_it, use_IPM=use_IPM)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ae, ground_truth)
    print("\n> Alpha Expansion")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    # lb_abbs = moves.large_move_maxcut(C, K, lb_init, move_type="ae_bs", ab_sequence=ab_sequence,
    #                                   num_max_it=num_max_it, use_IPM=use_IPM)
    # pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_abbs, ground_truth)
    # print("\n> Alpha Expansion-Beta Shrink")
    # print("  - Purity: %.4f" % pur)
    # print("  - Energy Clustering: %.4f" % ene_cl)
    # print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    lb_ls = cu.local_search(C, K, lb_init)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ls, ground_truth)
    print("\n> Local Search")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)
