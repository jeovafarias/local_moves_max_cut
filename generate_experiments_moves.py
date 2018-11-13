import os
import sys
import time
import numpy as np
import sklearn.metrics.pairwise as skl
import moves

import clustering_utils as cu
import data_generation_tools as dg
import data_visualization_tools as dv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


# noinspection PyStringFormat
def save_data(props, num_trials, dirpath, filename="test"):
    num_trials = float(num_trials)

    header = ['property',
              'num_exp',
              'tim', 'tim_std',
              'pur', 'pur_std', 'pur_max', 'pur_max_n',
              'pur_min', 'pur_min_std', 'pur_min_max', 'pur_min_max_n',
              'ene', 'ene_std', 'ene_max', 'ene_max_n',
              'per', 'per_std', 'per_max', 'per_max_n',
              'CH', 'CH_std', 'CH_max',
              'SI', 'SI_std', 'SI_max',
              'DB', 'DB_std', 'DB_min',
              'DU', 'DU_std', 'DU_max']

    props =  ['value',
              '%d' % num_trials,
              '%.3f' % np.mean(props['times']), '%.2f' % np.std(props['times']),
              '%.3f' % np.mean(props['purities']), '%.2f' % np.std(props['purities']), '%.3f' % np.max(props['purities']), '%.2f' % (np.count_nonzero(props['purities'] == np.max(props['purities']))/num_trials),
              '%.3f' % np.mean(props['min_purities']), '%.2f' % np.std(props['min_purities']), '%.3f' % np.max(props['min_purities']), '%.2f' % (np.count_nonzero(props['min_purities'] == np.max(props['min_purities']))/num_trials),
              '%.3f' % np.mean(props['energies']), '%.2f' % np.std(props['energies']), '%.3f' % np.max(props['energies']), '%.2f' % (np.count_nonzero(props['energies'] == np.max(props['energies']))/num_trials),
              '%.3f' % np.mean(props['percentages_energy']), '%.2f' % np.std(props['percentages_energy']), '%.3f' % np.max(props['percentages_energy']), '%.2f' % (np.count_nonzero(props['percentages_energy'] == np.max(props['percentages_energy']))/num_trials),
              '%.3f' % np.mean(props['CH']), '%.2f' % np.std(props['CH']), '%.3f' % np.max(props['CH']),
              '%.3f' % np.mean(props['SI']), '%.2f' % np.std(props['SI']), '%.3f' % np.max(props['SI']),
              '%.3f' % np.mean(props['DB']), '%.2f' % np.std(props['DB']), '%.3f' % np.max(props['DB']),
              '%.3f' % np.mean(props['DU']), '%.2f' % np.std(props['DU']), '%.3f' % np.max(props['DU'])]

    f = open(str(dirpath) + '/' + filename + '.csv', 'a')
    for i in range(len(header)):
        f.write(header[i] + ',')
    f.write('\n')
    for i in range(len(props)):
        f.write(props[i] + ',')
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
# noinspection PyStringFormat
def run_test(n, k, sigma, min_dist, num_trials, random_init, use_previous_dataset=-1, use_D31=False, dir_name='test'):
    time_start = time.time()

    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
        N = P.shape[0]
    else:
        if use_previous_dataset == -1:
            N = n * k
            params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': min_dist, 'simplex': False, 'K': k, 'dim_space': 2,
                      'l': 2, 'n': n, 'use_prev_p': False, 'shuffle': False}

            P, ground_truth = dg.generate_data_random(params)
        else:
            P = np.loadtxt(dir_name + "/" + str(use_previous_dataset) + "/P.dat")
            ground_truth = np.loadtxt(dir_name + "/" + str(use_previous_dataset) + "/gt.dat", dtype=np.int16)
            k = len(np.unique(ground_truth))
            N = P.shape[0]

    C = skl.pairwise_distances(P, metric='sqeuclidean')
    dv.plot_data(P, k, ground_truth, 2, show_data=True)

    # Other parameters -------------------------------------------------------------------------------------------------
    use_IPM = 0
    num_max_it = 20

    ab_pur, ab_min_pur, ab_ene, ab_per, ab_tim, ab_ch, ab_si, ab_db, ab_du, ab_it = \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials)
    # ae_pur, ae_min_pur, ae_ene, ae_per, ae_tim, ae_ch, ae_si, ae_db, ae_du, ae_it = \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials)
    # aebs_pur, aebs_min_pur, aebs_ene, aebs_per, aebs_tim, aebs_ch, aebs_si, aebs_db, aebs_du, aebs_it\
    #     = \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials), \
    #     np.zeros(num_trials), np.zeros(num_trials)
    ls_pur, ls_min_pur, ls_ene, ls_per, ls_tim, ls_ch, ls_si, ls_db, ls_du, ls_it = \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials), \
        np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        print ("Running experiment %d of %d" % (t + 1, num_trials))
        if random_init:
            lb_init = np.random.randint(0, k, N)
            while len(np.unique(lb_init)) < k:
                lb_init = np.random.randint(0, k, N)
            ab_sequence = None
        else:
            lb_init = np.zeros(N, dtype=int)
            ab_sequence = np.array([np.random.choice(k, k, replace=False), np.random.choice(k, k, replace=False)])

        start_t = time.time()
        ab_labeling, ab_it[t] = moves.large_move_maxcut(C, k, lb_init,
                                        move_type="ab", ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ab_pur[t], ab_min_pur[t], ab_ene[t], ab_per[t], ab_ch[t], ab_si[t], ab_db[t], ab_du[t] \
            = cu.stats_clustering_pairwise(C, ab_labeling, ground_truth, P, use_other_measures=True)
        ab_tim[t] = time.time() - start_t
        print time.time() - start_t
        print ab_it[t]

        # start_t = time.time()
        # ae_labeling, ae_it[t] = moves.large_move_maxcut(C, k, lb_init,
        #                                 move_type="ae", ab_sequence=ab_sequence,
        #                                 num_max_it=num_max_it, use_IPM=use_IPM)
        # ae_pur[t], ae_min_pur[t], ae_ene[t], ae_per[t], ae_ch[t], ae_si[t], ae_db[t], ae_du[t]\
        #     = cu.stats_clustering(P, C, ae_labeling, ground_truth)
        # ae_tim[t] = time.time() - start_t
        # print time.time() - start_t
        #
        # start_t = time.time()
        # aebs_labeling, aebs_it[t] = moves.large_move_maxcut(C, k, lb_init,
        #                                   move_type="ae_bs", ab_sequence=ab_sequence,
        #                                   num_max_it=num_max_it, use_IPM=use_IPM)
        # aebs_pur[t], aebs_min_pur[t], aebs_ene[t], aebs_per[t], aebs_ch[t], aebs_si[t], aebs_db[t], aebs_du[t] \
        #     = cu.stats_clustering(P, C, aebs_labeling, ground_truth)
        # aebs_tim[t] = time.time() - start_t
        # print time.time() - start_t

        start_t = time.time()
        ls_labeling, ls_it[t] = cu.local_search(C, k, lb_init)
        ls_pur[t], ls_min_pur[t], ls_ene[t], ls_per[t], ls_ch[t], ls_si[t], ls_db[t], ls_du[t] \
            = cu.stats_clustering_pairwise(C, ls_labeling, ground_truth, P, use_other_measures=True)
        ls_tim[t] = time.time() - start_t
        # print time.time() - start_t

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save data --------------------------------------------------------------------------------------------------------
    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)

    dv.plot_data(P, k, ground_truth, 2, show_legend=False, show_data=False, save_to_file=True,
                 file_name=dirpath + '/GT',
                 title='Ground Truth (N = %d, K = %d)' % (n*k, k))
    np.savetxt(str(dirpath) + '/P.dat', P)
    np.savetxt(str(dirpath) + '/gt.dat', ground_truth, fmt='%d')
    if not use_D31 and use_previous_dataset == -1:
        save_params(params, dirpath)
    if not random_init:
        np.savetxt(str(dirpath) + '/ab_sequence.dat', ab_sequence)

    print("ALPHA-BETA SWAP (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ab_tim), np.std(ab_tim)))
    print("> Mean min pur:  %.4f + %.4f" % (np.mean(ab_min_pur), np.std(ab_min_pur)))
    print("> Mean SI:  %.4f" % np.mean(ab_si))
    props = {"purities": ab_pur, "min_purities": ab_min_pur, "energies": ab_ene, "percentages_energy": ab_per,
              "CH": ab_ch, "SI": ab_si, "DB": ab_db, "DU": ab_du, "times": ab_tim, "iterations": ab_it}
    save_data(props, num_trials, dirpath, filename="ab_results")

    # print("ALPHA EXPANSION (id %d):" % experiment_id)
    # print("> Mean time: %.4f + %.4f" % (np.mean(ae_tim), np.std(ae_tim)))
    # print("> Mean min pur:  %.4f + %.4f" % (np.mean(ae_min_pur), np.std(ae_min_pur)))
    # print("> Mean SI:  %.4f" % np.mean(ae_si))
    # props = {"purities": ae_pur, "min_purities": ae_min_pur, "energies": ae_ene, "percentages_energy": ae_per,
    #           "CH": ae_ch, "SI": ae_si, "DB": ae_db, "DU": ae_du, "times": ae_tim, "iterations": ae_it}
    # save_data(props, num_trials, filename="ae_results")
    #
    # print("ALPHA EXPANSION-BETA SHRINK (id %d):" % experiment_id)
    # print("> Mean time: %.4f + %.4f" % (np.mean(aebs_tim), np.std(aebs_tim)))
    # print("> Mean min pur:  %.4f + %.4f" % (np.mean(aebs_min_pur), np.std(aebs_min_pur)))
    # print("> Mean SI:  %.4f" % np.mean(aebs_si))
    # props = {"purities": aebs_pur, "min_purities": aebs_min_pur, "energies": aebs_ene, "percentages_energy": aebs_per,
    #           "CH": aebs_ch, "SI": aebs_si,  "DB": aebs_db, "DU": aebs_du, "times": aebs_tim, "iterations": aebs_it}
    # save_data(props, num_trials, dirpath, filename="aebs_results")

    print("LOCAL SEARCH (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ls_tim), np.std(ls_tim)))
    print("> Mean min pur:  %.4f + %.4f" % (np.mean(ls_min_pur), np.std(ls_min_pur)))
    print("> Mean SI:  %.4f" % np.mean(ls_si))
    props = {"purities": ls_pur, "min_purities": ls_min_pur, "energies": ls_ene, "percentages_energy": ls_per,
              "CH": ls_ch, "SI": ls_si,  "DB": ls_db, "DU": ls_du, "times": ls_tim, "iterations": ls_it}
    save_data(props, num_trials, dirpath, filename="ls_results")

    # gt_pur, gt_min_pur, gt_ene, gt_per, gt_ch, gt_si, gt_db, gt_du \
    #     = cu.stats_clustering_pairwise(C, ground_truth, ground_truth, P)
    # props = {"purities": gt_pur, "min_purities": gt_min_pur, "energies": gt_ene, "percentages_energy": gt_per,
    #           "CH": gt_ch, "SI": gt_si,  "DB": gt_db, "DU": gt_du, "times": 0, "iterations": 0}
    # save_data(props, num_trials, dirpath, filename="gt_results")

    print("Total time (id %d): %.4f s\n" % (experiment_id, time.time() - time_start))


if __name__ == "__main__":
    random_init = True

    dir_name = 'test_ls_ab'
    num_trials = 3

    K = [4]

    points_per_cluster = [4]
    sigmas = [0.3, 0.5, 0.6]
    min_dists = [0.01]

    exp_number = 1
    for min_dist in min_dists:
        for k in K:
            for sigma in sigmas:
                for n in points_per_cluster:
                    print('EXP. %d - 1 ============================================================' % (exp_number + 1))
                    print('K: %d, n: %d, min_dist: %.4f, sigma %.4f' % (k, n, min_dist, sigma))
                    exp_number += 1
                    run_test(n, k, sigma, min_dist, num_trials, random_init, dir_name=dir_name)
    print("SET FINISHED ========================================================================== \n")

    # print('EXP - D31 ============================================================')
    # run_test(0, 0, 0, 0, num_trials, random_init, use_D31=True, dir_name=dir_name)
    # print("SET FINISHED ========================================================================== \n")

