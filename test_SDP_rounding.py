from __future__ import print_function
import os
import sys
import time
import numpy as np
import scipy
import sklearn.metrics.pairwise as skl
from sklearn.cluster import KMeans as kmeans

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv
import moves
import sdp_solvers as solvers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)
import matplotlib.pyplot as plt


def int_sol(K, n):
    M = -1./(float(K)-1) * np.ones((n * K, n * K))
    for i in range(K):
        M[n * i: n * (i + 1), n * i: n * (i + 1)] = np.ones((n, n))
    return M


def run_test(K, n, post_processing=False):
    X = int_sol(K, n)
    gt = np.zeros(n, dtype=int)
    for k in range(K - 1):
        gt = np.concatenate((gt, (k + 1) * np.ones(n, dtype=int)))

    V = np.linalg.cholesky(X + 1e-10 * np.trace(X) * np.eye(n * K))
    labeling, pur = solvers.max_k_cut_rounding(V,
                        {'is_a_hypergraph_problem': False, 'C': 1 - X, 'K': K,
                         'calculate_purities': True, 'gt': gt,
                         'post_processing': post_processing})
    return cu.purity(labeling, gt), np.mean(pur), np.std(pur)


if __name__ == "__main__":
    dir_name = 'test_SDP_rounding'
    Ks = range(5, 55, 5)
    n = 10

    time_start = time.time()
    purities = np.zeros(len(Ks))
    mean_purities = np.zeros_like(purities)
    std_purities = np.zeros_like(purities)
    for k in range(len(Ks)):
        print(k)
        purities[k], mean_purities[k], std_purities[k] = run_test(Ks[k], n, True)

    print("Total time: %.4f s\n" % (time.time() - time_start))

    plt.figure(figsize=(7, 3))
    plt.plot(range(len(Ks)), purities, 'b', lw=1)
    plt.xticks(range(len(Ks)), Ks, fontsize=14)
    plt.ylim(0, 1)
    # plt.savefig(filename + '.png', bbox_inches="tight")
    title = 'SDP Rounding - max_pur (n = %d)' % n
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Purities")
    plt.savefig(title + '.png', bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 3))
    plt.plot(range(len(Ks)), mean_purities, 'b', lw=1)
    plt.fill_between(range(len(Ks)),
                     mean_purities - 2 * std_purities,  mean_purities + 2 * std_purities, color='b', alpha=0.2)
    plt.xticks(range(len(Ks)), Ks, fontsize=14)
    plt.ylim(0, 1)
    # plt.savefig(filename + '.png', bbox_inches="tight")
    title = 'SDP Rounding - mean_pur (n = %d)' % n
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Purities")
    plt.savefig(dir_name + '/' + title + '.png', bbox_inches="tight")
    plt.show()
