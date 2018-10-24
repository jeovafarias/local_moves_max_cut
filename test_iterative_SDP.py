from __future__ import print_function
import os
import sys
import time
import numpy as np
import sklearn.metrics.pairwise as skl
import data_generation_tools as dg
import data_visualization_tools as dv
import clustering_utils as cu
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)

def plot_matrix_lines(M, sigmas, Ks, title, dir_name):
    plt.figure()
    labels = []
    for ns in range(len(sigmas)):
        plt.plot(M[:, ns])
        labels.append(r'$\sigma = %.2f$' % sigmas[ns])

    plt.xticks(range(len(Ks)), Ks, fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel('$K$')
    plt.legend(labels, ncol=len(sigmas), mode="expand")

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


def save_params(dirpath, n, Ks, sigmas, use_simplex):
    f = open(str(dirpath) + '/params_data.dat', 'a')
    f.write('param|value\n')
    f.write('n|%d\n' % n)
    f.write('use_simplex|%d\n' % use_simplex)
    f.close()

    np.savetxt(str(dirpath) + '/Ks.txt', Ks, fmt='%df')
    np.savetxt(str(dirpath) + '/sigmas.txt', sigmas, fmt='%.3f')


# MAIN CODE ============================================================================================================
# noinspection PyStringFormat
def run_tests(n, k, sigma, use_simplex=False, use_D31=False):
    assert k >= 2

    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
    else:
        params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': 0, 'simplex': use_simplex, 'K': k, 'dim_space': 2, 'l': 2,
                  'n': n, 'use_prev_p': False, 'shuffle': True}
        P, ground_truth = dg.generate_data_random(params)

    assert sigma < 1.0

    C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, k, ground_truth, 2)
    itsdp_X, itsdp_tim, num_it_SDP, err = cu.iterate_sdp(C, k)
    itsdp_labeling = cu.cluster_integer_sol(itsdp_X, k)
    itsdp_pur, itsdp_min_pur, _, _ \
        = cu.stats_clustering_pairwise(C, itsdp_labeling, ground_truth, P, use_other_measures=False)

    print('(itsdp: %.3f, itsdp_min_pur:  %.3f, err: %.6f, %d it)' % (itsdp_pur, itsdp_min_pur, err, num_it_SDP))
    returns = {'itsdp_pur': np.mean(itsdp_pur), 'itsdp_min_pur': np.mean(itsdp_min_pur), 'itsdp_tim': np.mean(itsdp_tim)}
    return returns


def run_experiments(Ks, sigmas, n, num_datasets, dir_name, use_simplex=False, title=''):

    time_start = time.time()
    itsdp_pur_matrix, itsdp_min_pur_matrix, itsdp_tim_matrix = np.zeros((len(Ks), len(sigmas))), \
                                                      np.zeros((len(Ks), len(sigmas))), \
                                                      np.zeros((len(Ks), len(sigmas)))

    for ns in range(len(sigmas)):
        for nk in range(len(Ks)):
            itsdp_pur, itsdp_min_pur, itsdp_tim = np.zeros(num_datasets), np.zeros(num_datasets), np.zeros(num_datasets)

            print('EXP - %s. (K = %d, s = %.2f) ======================================================'
                  % (title, Ks[nk], sigmas[ns]))
            nd = 0
            while nd < num_datasets:
                print('Dataset %d of %d --------------------------------------------------------------'
                      % (nd, num_datasets))
                try:
                    returns = run_tests(n, Ks[nk], sigmas[ns])
                except:
                    print('ERROR HAPPENED')
                    continue

                itsdp_pur[nd] = returns['itsdp_pur']
                itsdp_min_pur[nd] = returns['itsdp_min_pur']
                itsdp_tim[nd] = returns['itsdp_tim']
                nd += 1

            print('')
            itsdp_pur_matrix[nk, ns] = np.mean(itsdp_pur)
            itsdp_min_pur_matrix[nk, ns] = np.mean(itsdp_min_pur)
            itsdp_tim_matrix[nk, ns] = np.mean(itsdp_tim)

    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    save_params(dirpath, n, Ks, sigmas, use_simplex)

    np.savetxt(str(dirpath) + '/itsdp_pur_matrix.txt', itsdp_pur_matrix, fmt='%.3f')
    np.savetxt(str(dirpath) + '/itsdp_min_pur_matrix.txt', itsdp_min_pur_matrix, fmt='%.3f')
    np.savetxt(str(dirpath) + '/itsdp_tim_matrix.txt', itsdp_tim_matrix, fmt='%.3f')

    plot_matrix_lines(itsdp_pur_matrix, sigmas, Ks, 'Purity (IT_SDP, $n = %d$, num. datasets$ = %d$)'
                      % (n, num_sample_datasets), dirpath)
    plot_matrix_lines(itsdp_min_pur_matrix, sigmas, Ks, 'Min Purity (IT_SDP, $n = %d$, num. datasets$ = %d$)'
                      % (n, num_sample_datasets), dirpath)
    plot_matrix_lines(itsdp_tim_matrix, sigmas, Ks, 'Time (IT_SDP, $n = %d$, num. datasets$ = %d$)'
                      % (n, num_sample_datasets), dirpath)

    print("Total time: %.4f s\n" % (time.time() - time_start))
    print("SET FINISHED ========================================================================== \n")


if __name__ == "__main__":
    dir_name = 'test_it_SDP'
    num_sample_datasets = 20
    Ks = [49]
    sigmas = [0.45]

    n = 10
    run_experiments(Ks, sigmas, n, num_sample_datasets, dir_name, use_simplex=False, title='1')

    # n = 10
    # run_experiments(Ks, sigmas, n, num_sample_datasets, dir_name, use_simplex=False, title='2')
    #
    # n = 15
    # run_experiments(Ks, sigmas, n, num_sample_datasets, dir_name, use_simplex=False, title='2')

    # num_points = 3
    # K = 2
    # sigma = 0.001
    # all_vectors = True
    #
    # P = cu.k_simplex(num_points)
    # if all_vectors:
    #     P += sigma * np.random.rand(num_points, num_points)
    # else:
    #     P[:, 0] += sigma * np.random.rand(num_points)
    #
    # P /= np.linalg.norm(P, axis=1)[:, np.newaxis]



