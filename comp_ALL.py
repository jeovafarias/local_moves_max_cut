from __future__ import print_function
import os
import sys
import time
import numpy as np
import sklearn.metrics.pairwise as skl
from sklearn.cluster import KMeans as kmeans

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv
import moves
import sdp_solvers as solvers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


def entry_exists(id, method, num_starts=None):
    endname = '' if num_starts is None else '_' + str(num_starts) + 'starts'
    filename = dir_name + '/' + method.__name__ + endname + ".csv"
    if not os.path.exists(filename):
        create_header(filename, num_starts)
        return False
    else:
        with open(filename, "rb") as f:
            M = np.loadtxt(f, delimiter=",", skiprows=1)
        return id in M[:, 0] if M.ndim > 1 else id == M[0]


def create_header(filename, num_starts=None):
    if num_starts is None:
        with open(filename, 'a') as f:
            f.write('id, n, k, gt_energy, energy, purity, time \n')
    else:
        with open(filename, 'a') as f:
            f.write('id, n, k, gt_energy, mean_energy, max_energy, mean_purity, pur_at_max_ene, mean_time \n')


def save_line(id, k, n, gt_energy, method, stats, num_starts=None):
    endname = '' if num_starts is None else '_' + str(num_starts) + 'starts'
    filename = dir_name + '/' + method.__name__ + endname + ".csv"

    if not os.path.exists(filename):
        create_header(filename, num_starts)

    with open(filename, 'a') as f:
        f.write('%d' % id + ',' + '%d' % n + ',' + '%d' % k + ',' + '%.4f' % gt_energy + ',')
        for s in range(len(stats) - 1):
            f.write('%.4f' % stats[s] + ',')
        f.write('%.4f' % stats[-1] + '\n')


def it_sdp(P, C, k):
    itsdp_X, _, _, err = cu.iterate_sdp(C, k, alpha=0)
    itsdp_labeling = cu.cluster_integer_sol(itsdp_X, k)
    return itsdp_labeling, err


def sdp_std_rounding(P, C, k):
    sdp_X, err, _, _ = solvers.maxkcut_admm_solver(C, k)
    V = np.linalg.cholesky(sdp_X + 1e-9 * np.trace(sdp_X) * np.eye(sdp_X.shape[0]))
    sdp_labeling = solvers.max_k_cut_rounding(V, {'is_a_hypergraph_problem': False, 'C': C, 'K': k,
                                                  'post_processing': False})
    return sdp_labeling, err


def sdp_new_rounding(P, C, k):
    sdp_X, err, _, _ = solvers.maxkcut_admm_solver(C, k)
    V = np.linalg.cholesky(sdp_X + 1e-9 * np.trace(sdp_X) * np.eye(sdp_X.shape[0]))
    sdp_labeling = solvers.max_k_cut_rounding(V, {'is_a_hypergraph_problem': False, 'C': C, 'K': k,
                                                  'post_processing': True})
    return sdp_labeling, err


def km(P, C, k):
    km_labeling = kmeans(n_clusters=k, init='random', n_init=1000).fit(P).labels_
    return km_labeling, -1


def kmpp(P, C, k):
    kmpp_labeling = kmeans(n_clusters=k, init='k-means++', n_init=1000).fit(P).labels_
    return kmpp_labeling, -1


def ls(P, C, k, lb_init):
    ls_labeling, _, _ = cu.local_search(C, k, lb_init)
    return ls_labeling


def ab(P, C, k, lb_init):
    ab_labeling, _, err = moves.large_move_maxcut(C, k, lb_init, move_type="ab")
    return ab_labeling


def sample_clusters(P, gt, K, new_k):
    sampled_Ks = np.random.choice(K, new_k, replace=False)
    idx = np.array([i for i in range(len(gt)) if gt[i] in sampled_Ks])
    new_P = P[idx]

    gt = gt[idx]
    k_idx = 0
    aux = np.zeros_like(gt)
    new_gt = np.zeros_like(gt)
    for i in range(len(gt)):
        if aux[i] == 0:
            aux[gt == gt[i]] = 1
            new_gt[gt == gt[i]] = k_idx
            k_idx += 1

    return new_P, new_gt, new_k


# MAIN CODE ============================================================================================================
# noinspection PyStringFormat
def run_experiments(dir_instances, non_it_methods, num_starts, it_methods, dir_name):

    time_start = time.time()

    instances = np.loadtxt(dir_instances + '/index.txt', delimiter=',')
    ids, ns, Ks = instances[:, 0].astype(int), instances[:, 1].astype(int), instances[:, 2].astype(int)

    for id_idx in range(len(ids)):

        print("Instance %d of %d" % (ids[id_idx], len(ids)))

        P = np.loadtxt(dir_instances + '/' + str(ids[id_idx]) + '/P.dat', dtype=float)
        ground_truth = np.loadtxt(dir_instances + '/' + str(ids[id_idx]) + '/gt.dat').astype(int)
        C = skl.pairwise_distances(P, metric='sqeuclidean')
        gt_energy = cu.energy_clustering_pairwise(C, ground_truth)

        dv.plot_data(P, Ks[id_idx], ground_truth, 2)

        print('(', end='')
        for method_idx in range(len(non_it_methods)):
            if entry_exists(ids[id_idx], non_it_methods[method_idx]):
               continue

            start_t = time.time()
            labeling, err = non_it_methods[method_idx](P, C, Ks[id_idx])
            tim = time.time() - start_t

            purity, _, energy, _ = cu.stats_clustering_pairwise(C, labeling, ground_truth)

            stats = [energy, purity, tim]
            save_line(ids[id_idx], Ks[id_idx], ns[id_idx], gt_energy, non_it_methods[method_idx], stats)

            print('%s: %.3f, err: %.1e | ' % (non_it_methods[method_idx].__name__, purity, err), end='')
        print(') ', end='')

        print('(', end='')
        for method_idx in range(len(it_methods)):
            if entry_exists(ids[id_idx], it_methods[method_idx], num_starts=num_starts):
               continue

            energies, purities, times = np.zeros(num_starts), np.zeros(num_starts), np.zeros(num_starts)
            for t in range(num_starts):
                lb_init = np.random.randint(0, Ks[id_idx], P.shape[0])

                start_t = time.time()
                labeling = it_methods[method_idx](P, C, Ks[id_idx], lb_init)
                times[t] = time.time() - start_t

                purities[t], _, energies[t], _ = cu.stats_clustering_pairwise(C, labeling, ground_truth)

            stats = [np.mean(energies), np.max(energies), np.mean(purities), purities[np.argmax(energies)], np.mean(times)]
            save_line(ids[id_idx], Ks[id_idx], ns[id_idx], gt_energy, it_methods[method_idx], stats, num_starts=num_starts)

            print('%s: %.3f | ' % (it_methods[method_idx].__name__, np.mean(purities)), end='')

        print(')')

    print("\nTotal time: %.4f s" % (time.time() - time_start))
    print("SET FINISHED ========================================================================== \n")


if __name__ == "__main__":

    dir_name = 'comp_all_results'
    dir_instances = 'clustering_instances'

    assert os.path.exists(dir_instances)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_starts = 10
    non_it_methods = [sdp_std_rounding, sdp_new_rounding, it_sdp, km, kmpp]
    it_methods = [ls]

    run_experiments(dir_instances, non_it_methods, num_starts, it_methods, dir_name)