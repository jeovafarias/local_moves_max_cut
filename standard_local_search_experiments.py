from __future__ import print_function
import os
import sys
import time
import numpy as np
import sklearn.metrics.pairwise as skl

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv
import moves


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)
import matplotlib.pyplot as plt

def plot_matrix_lines(M, sigmas, Ks, title, dir_name):
    plt.figure()
    labels = []
    for ns in range(len(sigmas)):
        plt.plot(M[:, ns])
        labels.append(r'$\sigma = %.2f$' % sigmas[ns])

    plt.xticks(range(len(Ks)), Ks, fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel('$K$')
    plt.legend(labels, ncol=len(sigmas), mode="expand", loc=1)

    plt.savefig(dir_name + '/' + title + '.png')
    plt.show()


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


def save_params(dirpath, n, Ks, sigmas, num_starts, num_max_trials, use_simplex):
    f = open(str(dirpath) + '/params_data.dat', 'a')
    f.write('param|value\n')
    f.write('n|%d\n' % n)
    f.write('num_starts|%d\n' % num_starts)
    f.write('num_max_trials|%d\n' % num_max_trials)
    f.write('use_simplex|%d\n' % use_simplex)
    f.close()

    np.savetxt(str(dirpath) + '/Ks.txt', Ks, fmt='%df')
    np.savetxt(str(dirpath) + '/sigmas.txt', sigmas, fmt='%.3f')

# MAIN CODE ============================================================================================================
# noinspection PyStringFormat
def run_tests(n, k, sigma, num_starts, num_max_trials, use_simplex=True, use_D31=False):
    assert k >= 2

    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
        N = P.shape[0]
    else:
        N = n * k
        params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': 0, 'simplex': use_simplex, 'K': k, 'dim_space': 2, 'l': 2,
                  'n': n, 'use_prev_p': False, 'shuffle': False}
        P, ground_truth = dg.generate_data_random(params)

    assert sigma < 1.0

    C = skl.pairwise_distances(P, metric='sqeuclidean')

    num_it_to_max_ls, num_it_to_max_ab = np.zeros(num_starts), np.zeros(num_starts)
    # Iterate ----------------------------------------------------------------------------------------------------------
    print(' ', end='')
    for t in range(num_starts):
        it_ls, pur = 0, 0.0
        while pur < 1.0 and it_ls < num_max_trials:
            lb_init = np.random.randint(0, k, N)
            lb, _ = cu.local_search(C, k, lb_init, num_max_it=100)

            pur = cu.purity(lb, ground_truth, type="ave")
            it_ls += 1
            
        it_ab, pur = 0, 0.0
        while pur < 1.0 and it_ab < num_max_trials:
            lb_init = np.random.randint(0, k, N)
            lb, _ = moves.large_move_maxcut(C, k, lb_init, move_type="ab", num_max_it=100)

            pur = cu.purity(lb, ground_truth, type="ave")
            it_ab += 1

        print('(ls: %d, ab: %d), ' % (it_ls, it_ab),  end='')
        num_it_to_max_ls[t] = it_ls
        num_it_to_max_ab[t] = it_ab
    print('')
    return np.mean(num_it_to_max_ls), np.mean(num_it_to_max_ab)


def run_experiments(Ks, sigmas, n, num_starts, num_sample_datasets, num_max_trials,dir_name, use_simplex=True, title=''):
    time_start = time.time()
    n_it_to_max_ls, n_it_to_max_ab = np.zeros((len(Ks), len(sigmas))), np.zeros((len(Ks), len(sigmas)))
    for ns in range(len(sigmas)):
        for nk in range(len(Ks)):
            nums_it_ls, nums_it_ab = np.zeros(num_sample_datasets), np.zeros(num_sample_datasets)

            print('EXP - %s. (K = %d, s = %.2f) ======================================================'
                  % (title, Ks[nk], sigmas[ns]))
            for nd in range(num_sample_datasets):
                print('Dataset %d of %d --------------------------------------------------------------'
                      % (nd, num_sample_datasets))
                nums_it_ls[nd], nums_it_ab[nd] = run_tests(n, Ks[nk], sigmas[ns], num_starts, num_max_trials)
            print('')
            n_it_to_max_ls[nk, ns] = np.mean(nums_it_ls)
            n_it_to_max_ab[nk, ns] = np.mean(nums_it_ab)

    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    save_params(dirpath, n, Ks, sigmas, num_starts, num_max_trials, use_simplex)

    np.savetxt(str(dirpath) + '/n_it_to_max_ls.txt', n_it_to_max_ls, fmt='%.3f')
    np.savetxt(str(dirpath) + '/n_it_to_max_ab.txt', n_it_to_max_ab, fmt='%.3f')

    plot_matrix_lines(n_it_to_max_ls, sigmas, Ks, 'Num. Iterations (Local Search, $n = %d$, num. experiments $ = %d$)'
                      % (n, num_starts), dirpath)
    plot_matrix_lines(n_it_to_max_ab, sigmas, Ks, 'Num. Iterations (AB Swaps, $n = %d$, num. experiments $ = %d$)'
                      % (n, num_starts), dirpath)

    print("Total time: %.4f s\n" % (time.time() - time_start))
    print("SET FINISHED ========================================================================== \n")


if __name__ == "__main__":
    random_init = True

    dir_name = 'test_local_search_fail_grid'

    num_starts = 10
    num_sample_datasets = 20
    num_max_trials = 50
    Ks = [9, 16, 25, 36, 49, 64]
    sigmas = [0.15, 0.2, 0.3]

    n = 5
    run_experiments(Ks, sigmas, n, num_starts, num_sample_datasets,
                    num_max_trials, dir_name, use_simplex=False, title='1')

    n = 10
    run_experiments(Ks, sigmas, n, num_starts, num_sample_datasets,
                    num_max_trials, dir_name, use_simplex=False, title='2')

    n = 20
    run_experiments(Ks, sigmas, n, num_starts, num_sample_datasets,
                    num_max_trials, dir_name, use_simplex=False, title='3')
