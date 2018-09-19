import os
import sys
import time
from numpy.core.multiarray import ndarray
import numpy as np
import sklearn.metrics.pairwise as skl

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)
import matplotlib.pyplot as plt


def plot_line(title, vec, Ks, sigmas, filename):
    num_K = len(vec)

    fig, ax = plt.subplots()
    lines = ax.plot(range(num_K), vec, lw=1)
    plt.yticks(np.linspace(0, np.max(vec), 11))
    plt.xticks(range(num_K), Ks, fontsize=14)
    leg = ax.legend(lines, sigmas, loc='lower left', ncol=3, title="Sigma")
    plt.savefig(filename + '.png', bbox_inches="tight")
    plt.title(title)
    plt.xlabel("K")

    plt.show()


# MAIN CODE ============================================================================================================
# noinspection PyStringFormat
def run_test(n, k, sigma, num_trials, use_D31=False):
    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
        N = P.shape[0]
    else:
        N = n * k
        params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': 0, 'K': k, 'dim_space': 2, 'l': 2,
                  'n': n, 'use_prev_p': False, 'shuffle': False}
        P, ground_truth = dg.generate_data_random(params)

    C = skl.pairwise_distances(P, metric='sqeuclidean')

    # Other parameters -------------------------------------------------------------------------------------------------
    ls_pur, ls_min_pur, ls_ene, ls_per, ls_tim, ls_ch, ls_si, ls_db, ls_du, ls_it = \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        lb_init = np.random.randint(0, k, N)

        start_t = time.time()
        lb_ls, ls_it[t] = cu.local_search(C, k, lb_init, num_max_it=20)
        ls_pur[t], ls_min_pur[t], ls_ene[t], ls_per[t], ls_ch[t], ls_si[t], ls_db[t], ls_du[t]\
            = cu.stats_clustering_pairwise(P, C, lb_ls, ground_truth)
        ls_tim[t] = time.time() - start_t

    props = {"purities": ls_pur, "min_purities": ls_min_pur, "energies": ls_ene, "percentages_energy": ls_per,
              "CH": ls_ch, "SI": ls_si,  "DB": ls_db, "DU": ls_du, "times": ls_tim, "iterations": ls_it}

    return props


if __name__ == "__main__":
    random_init = True

    # dir_name = 'find_failure'
    dir_name = 'test_local_search/plots'
    num_trials = 30

    Ks = range(10, 20, 5)
    num_K = len(Ks)
    n = 5
    sigmas = [0.001, 0.002, 0.005]
    time_start = time.time()
    pur, min_pur, ene, CH, SI, DB, DU, per_ene = np.zeros((num_K, len(sigmas))), np.zeros((num_K, len(sigmas))), \
                                                 np.zeros((num_K, len(sigmas))), np.zeros((num_K, len(sigmas))), \
                                                 np.zeros((num_K, len(sigmas))), np.zeros((num_K, len(sigmas))), \
                                                 np.zeros((num_K, len(sigmas))), np.zeros((num_K, len(sigmas)))
    for ns in range(len(sigmas)):
        for nk in range(num_K):
            print('EXP. (%d, %d) ============================================================' % (nk, ns))
            props = run_test(n, Ks[nk], sigmas[ns], num_trials)
            print Ks[nk]
            pur[nk, ns] = np.mean(props['purities'])
            min_pur[nk, ns] = np.mean(props['min_purities'])
            ene[nk, ns] = np.mean(props['energies'])
            CH[nk, ns] = np.mean(props['CH'])
            SI[nk, ns] = np.mean(props['SI'])
            DB[nk, ns] = np.mean(props['DB'])
            DU[nk, ns] = np.mean(props['DU'])
            per_ene[nk, ns] = np.mean(props['percentages_energy'])

    plot_line("Purity", pur, Ks, sigmas, dir_name + "pur")
    plot_line("Min Purity", min_pur, Ks, sigmas, dir_name + "min_pur")
    plot_line("SI", SI, Ks, sigmas, dir_name + "SI")
    plot_line("CH", CH, Ks, sigmas, dir_name + "CH")
    plot_line("DB", DB, Ks, sigmas, dir_name + "DB")
    plot_line("DU", DU, Ks, sigmas, dir_name + "DU")

    print("Total time: %.4f s\n" % (time.time() - time_start))

    print("SET FINISHED ========================================================================== \n")


