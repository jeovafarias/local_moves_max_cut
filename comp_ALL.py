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
import matplotlib.pyplot as plt

num_stats = 4
name_stats = ['pur', 'min_pur', 'ene_per', 'tim']


def get_num_stats():
    return num_stats


def create_stats_vector(tim, labeling, ground_truth, P, C):
    pur, min_pur, ene, ene_per \
        = cu.stats_clustering_pairwise(C, labeling, ground_truth, P, use_other_measures=False)
    stats = np.array([pur, min_pur, ene_per, tim])
    return stats


def it_sdp(P, C, k, ground_truth):
    itsdp_X, tim, _, _ = cu.iterate_sdp(C, k)
    dv.plot_matrix(itsdp_X)
    itsdp_labeling = cu.cluster_integer_sol(itsdp_X, k)
    return create_stats_vector(tim, itsdp_labeling, ground_truth, P, C)


def sdp(P, C, k, ground_truth):
    sdp_X, _, tim, _ = solvers.maxkcut_admm_solver(C, k)
    V = np.linalg.cholesky(sdp_X + 1e-9 * np.trace(sdp_X) * np.eye(sdp_X.shape[0]))
    sdp_labeling = solvers.max_k_cut_rounding(V, {'is_a_hypergraph_problem': False, 'C': C, 'K': k})
    return create_stats_vector(tim, sdp_labeling, ground_truth, P, C)


def ls(P, C, k, ground_truth, lb_init):
    start_t = time.time()
    ls_labeling, _ = cu.local_search(C, k, lb_init)
    tim = time.time() - start_t
    return create_stats_vector(tim, ls_labeling, ground_truth, P, C)


def ab(P, C, k, ground_truth, lb_init):
    start_t = time.time()
    ab_labeling, _ = moves.large_move_maxcut(C, k, lb_init, move_type="ab")
    tim = time.time() - start_t
    return create_stats_vector(tim, ab_labeling, ground_truth, P, C)


def km(P, C, k, ground_truth):
    start_t = time.time()
    km_labeling = kmeans(n_clusters=k, init='random', n_init=1).fit(P).labels_
    tim = time.time() - start_t
    return create_stats_vector(tim, km_labeling, ground_truth, P, C)


def kmpp(P, C, k, ground_truth):
    start_t = time.time()
    kmpp_labeling = kmeans(n_clusters=k, init='k-means++').fit(P).labels_
    tim = time.time() - start_t
    return create_stats_vector(tim, kmpp_labeling, ground_truth, P, C)


def plot_matrix_lines(M, vec, Ks, title, dir_name):
    plt.figure()
    labels = []
    for ns in range(len(vec)):
        plt.plot(M[:, ns])
        if callable(vec[ns]):  # Only works in Python 2.x or Python 3.2+
            labels.append(r'%s' % vec[ns].__name__)
        else:
            labels.append(r'%s' % vec[ns])

    plt.xticks(range(len(Ks)), Ks, fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel('$K$')
    plt.legend(labels, ncol=len(vec), mode="expand")

    plt.savefig(dir_name + '/' + title + '.png')
    plt.show()


def plot_line(title, vec, Ks, sigmas, filename):
    num_K = len(vec)

    fig, ax = plt.subplots()
    lines = ax.plot(range(num_K), vec, lw=1)
    plt.yticks(np.linspace(0, np.max(vec), 11))
    plt.xticks(range(num_K), Ks, fontsize=14)
    leg = ax.legend(lines, sigmas, ncol=3, title="Sigma")
    plt.savefig(filename + '.png', bbox_inches="tight")
    plt.title(title)
    plt.xlabel("K")

    plt.show()


def save_params(dirpath, n, Ks, sigmas, num_starts, use_simplex):
    f = open(str(dirpath) + '/params_data.dat', 'a')
    f.write('param|value\n')
    f.write('n|%d\n' % n)
    f.write('num_starts|%d\n' % num_starts)
    f.write('use_simplex|%d\n' % use_simplex)
    f.close()

    np.savetxt(str(dirpath) + '/Ks.txt', Ks, fmt='%df')
    np.savetxt(str(dirpath) + '/sigmas.txt', sigmas, fmt='%.3f')


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
def run_tests(n, k, sigma, non_it_methods, it_methods, num_starts, use_simplex=False, use_D31=False):
    assert k >= 2

    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
    else:
        params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': 0, 'simplex': use_simplex, 'K': k, 'dim_space': 2, 'l': 2,
                  'n': n, 'use_prev_p': False, 'shuffle': True}
        P, ground_truth = dg.generate_data_random(params)

        per = 0.5
        new_k = int(np.floor(per * k))
        P, ground_truth, k = sample_clusters(P, ground_truth, k, new_k)
        dv.plot_data(P, k, ground_truth, 2)

    assert sigma < 1.0

    N = P.shape[0]
    C = skl.pairwise_distances(P, metric='sqeuclidean')
    num_stats = get_num_stats()   # [pur, min_pur, tim]
    returns = np.zeros((0, num_stats))
    print('(', end='')
    for i in range(len(non_it_methods)):
        stats = non_it_methods[i](P, C, k, ground_truth)
        returns = np.concatenate((returns, np.expand_dims(stats, axis=0)), axis=0)
        print('%s: %.3f, ' % (non_it_methods[i].__name__, stats[0]), end='')
    print('\b\b) ', end='')

    stats = np.zeros((len(it_methods), num_stats, num_starts))
    print(' ', end='')
    for t in range(num_starts):
        lb_init = np.random.randint(0, k, N)
        print('(', end='')
        for i in range(len(it_methods)):
            stats[i, :, t] = it_methods[i](P, C, k, ground_truth, lb_init)
            print('%s: %.3f, ' % (it_methods[i].__name__, stats[0, i, t]), end='')
        print('\b\b) ', end='')

    returns = np.concatenate((returns, np.mean(stats, axis=2)), axis=0)
    print('')
    return returns


def run_experiments(n, Ks, sigmas, non_it_methods, it_methods, num_starts, num_datasets, dir_name, use_simplex=False,
                    title=''):
    time_start = time.time()
    num_stats = get_num_stats()   # [pur, min_pur, tim]
    methods = np.concatenate((non_it_methods, it_methods))

    stats_matrix = np.zeros((len(Ks), len(sigmas), len(methods), num_stats))
    for ns in range(len(sigmas)):
        for nk in range(len(Ks)):
            stats = np.zeros((len(methods), num_stats, num_datasets))
            print('EXP - %s. (K = %d, s = %.2f) ======================================================'
                  % (title, Ks[nk], sigmas[ns]))
            for nd in range(num_datasets):
                print('Dataset %d of %d --------------------------------------------------------------'
                      % (nd, num_datasets))
                returns = run_tests(n, Ks[nk], sigmas[ns], non_it_methods, it_methods, num_starts)
                stats[:, :, nd] = returns
            print('')

            stats_matrix[nk, ns, :, :] = np.mean(stats, axis=2)

    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    save_params(dirpath, n, Ks, sigmas, num_starts, use_simplex)
    np.save(str(dirpath) + '/stats_matrix', stats_matrix)

    for method in range(len(methods)):
        for stat in range(num_stats):
            plot_matrix_lines(stats_matrix[:, :, method, stat], sigmas, Ks,
                              '%s ($n = %d$, n data sets$ = %d$, \n n. starts per data set$ = %d$, method = $%s$)'
                              % (np.char.upper(name_stats[stat]), n, num_starts, num_sample_datasets, np.char.upper(methods[method].__name__)), dirpath)

    for sigma in range(len(sigmas)):
        for stat in range(num_stats):
            plot_matrix_lines(stats_matrix[:, sigma, :, stat], methods, Ks,
                              '%s ($n = %d$, n data sets$ = %d$, \n n. starts per data set$ = %d$, sigma = $%.3f$)'
                              % (np.char.upper(name_stats[stat]), n, num_starts, num_sample_datasets, sigmas[sigma]), dirpath)

    print("Total time: %.4f s\n" % (time.time() - time_start))
    print("SET FINISHED ========================================================================== \n")
    print(stats_matrix)


if __name__ == "__main__":

    dir_name = 'comp_LS_AB_KM_results'

    num_starts = 1
    num_sample_datasets = 1
    Ks = [9, 16, 25, 36]
    sigmas = [0.3, 0.4]
    non_it_methods = [it_sdp]  # [sdp, it_sdp, km, kmpp]
    it_methods = []  # [ab, ls]


    n = 5
    run_experiments(n, Ks, sigmas, non_it_methods, it_methods, num_starts, num_sample_datasets, dir_name, use_simplex=False, title='2')
    #
    # n = 20
    # run_experiments(n, Ks, sigmas, num_starts, num_sample_datasets, dir_name, use_simplex=False, title='3')
