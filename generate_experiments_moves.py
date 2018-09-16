import os
import sys
import time
from numpy.core.multiarray import ndarray
import numpy as np
import sklearn.metrics.pairwise as skl

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv
import moves

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


# noinspection PyStringFormat
def save_data(purities, min_purities, energies, percentages_energy, times, iterations, dirpath, filename="test"):

    f = open(str(dirpath) + '/' + filename + '.dat', 'a')
    f.write('prop|value\n')
    f.write('exp|%d\n' % num_trials)

    f.write('tim|%.4f\n' % np.mean(times))
    f.write('tim_std|%.4f\n' % np.std(times))

    f.write('pur|%.4f\n' % np.mean(purities))
    f.write('pur_std|%.4f\n' % np.std(purities))
    f.write('pur_max|%.4f\n' % np.max(purities))

    f.write('pur|%.4f\n' % np.mean(min_purities))
    f.write('pur_std|%.4f\n' % np.std(min_purities))
    f.write('pur_max|%.4f\n' % np.max(min_purities))

    f.write('ene|%.4f\n' % np.mean(energies))
    f.write('ene_std|%.4f\n' % np.std(energies))
    f.write('ene_max|%.4f\n' % np.max(energies))

    f.write('per|%.2f\n' % (100 * np.mean(percentages_energy)))
    f.write('per_std|%.4f\n' % np.std(percentages_energy))
    f.write('per_max|%.2f\n' % (100 * np.max(percentages_energy)))

    f.write('it|%.4f\n' % np.mean(iterations))

    f.close()


def save_params(params, dirpath):
    f = open(str(dirpath) + '/params_data.dat', 'a')
    f.write('param|value\n')
    f.write('n|%d\n' % params['n'])
    f.write('k|%d\n' % params['K'])
    f.write('dim_space|%d\n' % params['dim_space'])
    f.write('l|%d\n' % params['l'])
    f.write('sigma_1|%.4f\n' % params['sigma_1'])
    f.write('sigma_2|%.4f\n' % params['sigma_2'])
    f.write('min_dist|%.4f\n' % params['min_dist'])
    f.close()


# MAIN CODE ============================================================================================================
def run_test(n, k, sigma, min_dist, num_trials, random_init, dir_name='test'):
    time_start = time.time()

    # Data generation parameters ---------------------------------------------------------------------------------------
    N = n * k
    use_prev_p = 0
    shuffle_data = 1
    params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': min_dist, 'K': k, 'dim_space': 2, 'l': 2,
                   'n': n, 'use_prev_p': use_prev_p, 'shuffle': shuffle_data}

    # Data generation and visualization --------------------------------------------------------------------------------
    P, ground_truth = dg.generate_data_random(params)
    C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, k, ground_truth, 2, show_data=True)

    # Other parameters -------------------------------------------------------------------------------------------------
    use_IPM = 0
    num_max_it = 20

    ab_pur_t, ab_min_pur_t, ab_ene_t, ab_per_t, ab_tim_t, ab_it = np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials)
    ae_pur_t, ae_min_pur_t, ae_ene_t, ae_per_t, ae_tim_t, ae_it = np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials)
    aebs_pur_t, aebs_min_pur_t, aebs_ene_t, aebs_per_t, aebs_tim_t, aebs_it = np.zeros(num_trials), \
                                                                              np.zeros(num_trials), \
                                                                              np.zeros(num_trials), \
                                                                              np.zeros(num_trials), \
                                                                              np.zeros(num_trials), \
                                                                              np.zeros(num_trials)
    ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t, ls_it = np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials), \
                                                                  np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        print ("Running experiment %d of %d" % (t + 1, num_trials))
        if random_init:
            lb_init = np.random.randint(0, k, N)
            ab_sequence = None
        else:
            lb_init = np.zeros(N, dtype=int)
            ab_sequence = np.array([np.random.choice(k, k, replace=False), np.random.choice(k, k, replace=False)])

        start_t = time.time()
        lb_ab, ab_it[t] = moves.large_move_maxcut(C, k, lb_init,
                                        move_type="ab", ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ab_pur_t[t], ab_min_pur_t[t], ab_ene_t[t], ab_per_t[t] = cu.stats_clustering(C, lb_ab, ground_truth)
        ab_tim_t[t] = time.time() - start_t
        # print time.time() - start_t
        # print ab_it[t]

        start_t = time.time()
        lb_ae, ae_it[t] = moves.large_move_maxcut(C, k, lb_init,
                                        move_type="ae", ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ae_pur_t[t], ae_min_pur_t[t], ae_ene_t[t], ae_per_t[t] = cu.stats_clustering(C, lb_ae, ground_truth)
        ae_tim_t[t] = time.time() - start_t
        # print time.time() - start_t
        # print ae_it[t]

        # start_t = time.time()
        # lb_aebs, aebs_it[t] = moves.large_move_maxcut(C, k, lb_init,
        #                                   move_type="ae_bs", ab_sequence=ab_sequence,
        #                                   num_max_it=num_max_it, use_IPM=use_IPM)
        # aebs_pur_t[t], aebs_min_pur_t[t], aebs_ene_t[t], aebs_per_t[t] = cu.stats_clustering(C, lb_aebs, ground_truth)
        # aebs_tim_t[t] = time.time() - start_t
        # print time.time() - start_t
        # print aebs_it[t]

        start_t = time.time()
        lb_ls, ls_it[t] = cu.local_search(C, k, lb_init, num_max_it=num_max_it)
        ls_pur_t[t], ls_min_pur_t[t], ls_ene_t[t], ls_per_t[t] = cu.stats_clustering(C, lb_ls, ground_truth)
        ls_tim_t[t] = time.time() - start_t

        # print cu.purity(lb_ls, ground_truth)
        # print cu.CH(P, lb_ls)
        # print cu.DB(P, lb_ls)
        # print cu.SI(C, lb_ls)
        # print cu.DU(P, lb_ls)
        #
        # print ""
        #
        # print cu.CH(P, ground_truth)
        # print cu.DB(P, ground_truth)
        # print cu.SI(C, ground_truth)
        # print cu.DU(P, ground_truth)

        pass

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save data --------------------------------------------------------------------------------------------------------
    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    dv.plot_data(P, k, ground_truth, 2, show_legend=False, show_data=False, save_to_file=True,
                 file_name=dirpath + '/id_' + str(experiment_id) + '_GT',
                 title='Ground Truth (N = %d, K = %d)' % (n*k, k))
    np.savetxt(str(dirpath) + '/P.dat', P)
    np.savetxt(str(dirpath) + '/gt.dat', ground_truth, fmt='%d')
    np.savetxt(str(dirpath) + '/lb_init.dat', lb_init)
    save_params(params, dirpath)
    if not random_init:
        np.savetxt(str(dirpath) + '/ab_sequence.dat', ab_sequence)

    print("ALPHA-BETA SWAP (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ab_tim_t), np.std(ab_tim_t)))
    print("> Mean min pur:  %.4f + %.4f" % (np.mean(ab_min_pur_t), np.std(ab_min_pur_t)))
    print("> Mean iterations:  %.4f" % np.mean(ab_it))
    save_data(ab_pur_t, ab_min_pur_t, ab_ene_t, ab_per_t, ab_tim_t, ab_it, dirpath, filename="ab_results")

    print("ALPHA EXPANSION (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ae_tim_t), np.std(ae_tim_t)))
    print("> Mean min pur:  %.4f + %.4f" % (np.mean(ae_min_pur_t), np.std(ae_min_pur_t)))
    print("> Mean iterations:  %.4f" % np.mean(ae_it))
    save_data(ae_pur_t, ae_min_pur_t, ae_ene_t, ae_per_t, ae_tim_t, ae_it, dirpath, filename="ae_results")

    # print("ALPHA EXPANSION-BETA SHRINK (id %d):" % experiment_id)
    # print("> Mean time: %.4f + %.4f" % (np.mean(aebs_tim_t), np.std(aebs_tim_t)))
    # print("> Mean min pur:  %.4f + %.4f" % (np.mean(aebs_min_pur_t), np.std(aebs_min_pur_t)))
    # print("> Mean iterations:  %.4f" % np.mean(aebs_it))
    # save_data(aebs_pur_t, aebs_min_pur_t, aebs_ene_t, aebs_per_t, aebs_tim_t, aebs_it, dirpath, filename="aebs_results")

    print("LOCAL SEARCH (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ls_tim_t), np.std(ls_tim_t)))
    print("> Mean min pur:  %.4f + %.4f" % (np.mean(ls_min_pur_t), np.std(ls_min_pur_t)))
    print("> Mean iterations:  %.4f" % np.mean(ls_it))
    save_data(ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t, ls_it, dirpath, filename="ls_results")

    print("Total time (id %d): %.4f s\n" % (experiment_id, time.time() - time_start))


if __name__ == "__main__":
    random_init = True
    if random_init:
        dir_name = 'experiments_rand_int'
    else:
        dir_name = 'experiments_non_rand_int'

    dir_name = 'find_failure'
    num_trials = 10

    K = [20, 50]
    points_per_cluster = [3, 3, 3]
    sigmas = [0.02]
    min_dists = [0.2, 0.2]

    exp_number = 1
    for min_dist in min_dists:
        for sigma in sigmas:
            for k in K:
                for n in points_per_cluster:
                    print('EXP. %d - 1 ============================================================' % (exp_number + 1))
                    print('K: %d, n: %d, min_dist: %.4f, sigma %.4f' % (k, n, min_dist, sigma))
                    exp_number += 1
                    run_test(n, k, sigma, min_dist, num_trials, random_init, dir_name=dir_name)
    print("SET FINISHED ========================================================================== \n")

