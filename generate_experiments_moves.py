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
def save_data(purities, energies, percentages_energy, times, dirpath, filename="test"):

    f = open(str(dirpath) + '/' + filename + '.dat', 'a')
    f.write('prop|value\n')
    f.write('exp|%d\n' % num_trials)

    f.write('tim|%.4f\n' % np.mean(times))
    f.write('tim_std|%.4f\n' % np.std(times))

    f.write('pur|%.4f\n' % np.mean(purities))
    f.write('pur_std|%.4f\n' % np.std(purities))
    f.write('pur_max|%.4f\n' % np.max(purities))

    f.write('ene|%.4f\n' % np.mean(energies))
    f.write('ene_std|%.4f\n' % np.std(energies))
    f.write('ene_max|%.4f\n' % np.max(energies))

    f.write('per|%.2f\n' % (100 * np.mean(percentages_energy)))
    f.write('per_std|%.4f\n' % np.std(percentages_energy))
    f.write('per_max|%.2f\n' % (100 * np.max(percentages_energy)))
    f.close()


# MAIN CODE ============================================================================================================
def run_test(n, K, sigma_2, num_trials, random_init, dir_name='test'):
    time_start = time.time()

    # Data generation parameters ---------------------------------------------------------------------------------------
    N = n * K
    use_prev_p = 0
    shuffle_data = 0
    params_data = {'sigma_1': 1, 'sigma_2': sigma_2, 'K': K, 'dim_space': 2, 'pop_interv': [n, n],
                   'use_prev_p': use_prev_p, 'shuffle': shuffle_data}

    # Data generation and visualization --------------------------------------------------------------------------------
    P, ground_truth = dg.generate_data_random(params_data)
    C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, K, ground_truth, 2)

    # Other parameters -------------------------------------------------------------------------------------------------
    use_IPM = 1
    num_max_it = 20

    ab_pur_t, ab_ene_t, ab_per_t, ab_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials)
    ae_pur_t, ae_ene_t, ae_per_t, ae_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials)
    aebs_pur_t, aebs_ene_t, aebs_per_t, aebs_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                                     np.zeros(num_trials), np.zeros(num_trials)
    ls_pur_t, ls_ene_t, ls_per_t, ls_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        print t
        if random_init:
            lb_init = np.random.randint(0, K, N)
            ab_sequence = None
        else:
            lb_init = np.zeros(N, dtype=int)
            ab_sequence = np.array([np.random.choice(K, K, replace=False), np.random.choice(K, K, replace=False)])

        start_t = time.time()
        lb_ab = moves.large_move_maxcut(C, K, lb_init,
                                        move_type="ab", ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ab_pur_t[t], ab_ene_t[t], ab_per_t[t] = cu.stats_clustering(C, lb_ab, ground_truth)
        ab_tim_t[t] = time.time() - start_t

        start_t = time.time()
        lb_ae = moves.large_move_maxcut(C, K, lb_init,
                                        move_type="ae", ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ae_pur_t[t], ae_ene_t[t], ae_per_t[t] = cu.stats_clustering(C, lb_ae, ground_truth)
        ae_tim_t[t] = time.time() - start_t

        start_t = time.time()
        lb_aebs = moves.large_move_maxcut(C, K, lb_init,
                                          move_type="ae_bs", ab_sequence=ab_sequence,
                                          num_max_it=num_max_it, use_IPM=use_IPM)
        aebs_pur_t[t], aebs_ene_t[t], aebs_per_t[t] = cu.stats_clustering(C, lb_aebs, ground_truth)
        aebs_tim_t[t] = time.time() - start_t

        start_t = time.time()
        lb_ls = cu.local_search(C, K, lb_init, num_max_it=num_max_it)
        ls_pur_t[t], ls_ene_t[t], ls_per_t[t] = cu.stats_clustering(C, lb_ls, ground_truth)
        ls_tim_t[t] = time.time() - start_t

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save data --------------------------------------------------------------------------------------------------------
    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    dv.plot_data(P, K, ground_truth, 2, show_legend=False, show_data=False, save_to_file=True,
                 file_name=dirpath + '/id_' + str(experiment_id) + '_GT',
                 title='Ground Truth (N = %d, K = %d)' % (n*K, K))
    np.savetxt(str(dirpath) + '/P.dat', P)
    np.savetxt(str(dirpath) + '/gt.dat', ground_truth, fmt='%d')
    np.savetxt(str(dirpath) + '/lb_init.dat', lb_init)
    np.savetxt(str(dirpath) + '/ab_sequence.dat', ab_sequence)

    print("ALPHA-BETA SWAP (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ab_tim_t), np.std(ab_tim_t)))
    print("> Mean pur:  %.4f + %.4f" % (np.mean(ab_pur_t), np.std(ab_pur_t)))
    save_data(ab_pur_t, ab_ene_t, ab_per_t, ab_tim_t, dirpath, filename="ab_results")

    print("ALPHA EXPANSION (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ae_tim_t), np.std(ae_tim_t)))
    print("> Mean pur:  %.4f + %.4f" % (np.mean(ae_pur_t), np.std(ae_pur_t)))
    save_data(ae_pur_t, ae_ene_t, ae_per_t, ae_tim_t, dirpath, filename="ae_results")

    print("ALPHA EXPANSION-BETA SHRINK (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(aebs_tim_t), np.std(aebs_tim_t)))
    print("> Mean pur:  %.4f + %.4f" % (np.mean(aebs_pur_t), np.std(aebs_pur_t)))
    save_data(aebs_pur_t, aebs_ene_t, aebs_per_t, aebs_tim_t, dirpath, filename="aebs_results")

    print("LOCAL SEARCH (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ls_tim_t), np.std(ls_tim_t)))
    print("> Mean pur:  %.4f + %.4f" % (np.mean(ls_pur_t), np.std(ls_pur_t)))
    save_data(ls_pur_t, ls_ene_t, ls_per_t, ls_tim_t, dirpath, filename="ls_results")

    print("Total time (id %d): %.4f s\n" % (experiment_id, time.time() - time_start))


if __name__ == "__main__":
    dir_name = 'experiments_rand_int'
    # dir_name = 'experiments_non_rand_int'
    num_trials = 10
    random_init = False

    # points_per_cluster = 10
    # K = 3
    # sigmas = [0.15, 0.15, 0.15]
    # for e in range(len(sigmas)):
    #     print('EXP. %d - 1 =================================================================================' % (e + 1))
    #     sigma_1 = 1
    #     sigma_2 = sigmas[e]
    #     run_test(points_per_cluster, K, sigma_2, num_trials, random_init, dir_name=dir_name)
    # print("SET FINISHED ========================================================================== \n")

    points_per_cluster = 8
    K = 4
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.2]
    for e in range(len(sigmas)):
        print('EXP. %d - 2 =================================================================================' % (e + 1))
        sigma_1 = 1
        sigma_2 = sigmas[e]
        run_test(points_per_cluster, K, sigma_2, num_trials, random_init, dir_name=dir_name)
    print("SET FINISHED ========================================================================== \n")

    points_per_cluster = 2
    K = 4
    sigmas = [0.15, 0.15]
    for e in range(len(sigmas)):
        print('EXP. %d - 3 =================================================================================' % (e + 1))
        sigma_1 = 1
        sigma_2 = sigmas[e]
        run_test(points_per_cluster, K, sigma_2, num_trials, random_init, dir_name=dir_name)
    print("SET FINISHED ========================================================================== \n")

    points_per_cluster = 10
    K = 10
    sigmas = [0.1, 0.1, 0.1, 0.15, 0.15]
    for e in range(len(sigmas)):
        print('EXP. %d - 4 =================================================================================' % (e + 1))
        sigma_1 = 1
        sigma_2 = sigmas[e]
        run_test(points_per_cluster, K, sigma_2, num_trials, random_init, dir_name=dir_name)
    print("SET FINISHED ========================================================================== \n")