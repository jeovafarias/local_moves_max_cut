import numpy as np
import sklearn.metrics.pairwise as skl
import data_generation_tools as dg
import data_visualization_tools as dv
import clustering_utils as cu
import moves

np.set_printoptions(linewidth=1000, precision=5, threshold=np.nan, suppress=True)

if __name__ == "__main__":
    # Generate synthetic points ========================================================================================
    n = 5
    K = 4
    N = n * K

    # Data generation parameters ---------------------------------------------------------------------------------------
    sigma_2 = .2
    use_prev_p = 1
    shuffle_data = 0
    params_data = {'sigma_1': 1, 'sigma_2': sigma_2, 'K': K, 'dim_space': 2, 'pop_interv': [n, n],
                   'use_prev_p': use_prev_p, 'shuffle': shuffle_data}

    # Data generation and visualization --------------------------------------------------------------------------------
    P, ground_truth = dg.generate_data_random(params_data)
    C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, K, ground_truth, 2)

    # Other parameters -------------------------------------------------------------------------------------------------
    random_init = 1
    use_IPM = 1
    num_max_it = 20

    # Iterate ----------------------------------------------------------------------------------------------------------
    if random_init:
        lb_init = np.random.randint(0, K, N)
    else:
        lb_init = np.zeros(N, dtype=int)

    lb_ab = moves.large_move_maxcut(C, K, lb_init, move_type="ab", num_max_it=num_max_it, use_IPM=use_IPM)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ab, ground_truth)
    print("\n> Alpha-Beta Swap")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    # ab_sequence = np.random.choice(K, K)
    lb_ae = moves.large_move_maxcut(C, K, lb_init, move_type="ae", num_max_it=num_max_it, use_IPM=use_IPM)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ae, ground_truth)
    print("\n> Alpha Expansion")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    lb_abbs = moves.large_move_maxcut(C, K, lb_init, move_type="ae_bs", num_max_it=num_max_it, use_IPM=use_IPM)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_abbs, ground_truth)
    print("\n> Alpha Expansion-Beta Shrink")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)

    lb_ls = cu.local_search(C, K, lb_init)
    pur, ene_cl, per_ene_cl = cu.stats_clustering(C, lb_ls, ground_truth)
    print("\n> Local Search")
    print("  - Purity: %.4f" % pur)
    print("  - Energy Clustering: %.4f" % ene_cl)
    print("  - Percentage Energy: %.4f %%" % per_ene_cl)
