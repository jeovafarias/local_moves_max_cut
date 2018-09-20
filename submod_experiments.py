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
import submod_alg as sa
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


# noinspection PyStringFormat
def save_data(purities, min_purities, energies, percentages_energy, times, dirpath, filename="test"):

    f = open(str(dirpath) + '/' + filename + '.dat', 'a')
    f.write('prop|value\n')
    f.write('exp|%d\n' % num_trials)

    f.write('tim|%.4f\n' % np.mean(times))
    f.write('tim_std|%.4f\n' % np.std(times))

    f.write('pur|%.4f\n' % np.mean(purities))
    f.write('pur_std|%.4f\n' % np.std(purities))
    f.write('pur_max|%.4f\n' % np.max(purities))

    f.write('pur_min|%.3f\n' % np.mean(min_purities))
    f.write('pur_min_std|%.3f\n' % np.std(min_purities))
    f.write('pur_min_max|%.3f\n' % np.max(min_purities))

    f.write('ene|%.4f\n' % np.mean(energies))
    f.write('ene_std|%.4f\n' % np.std(energies))
    f.write('ene_max|%.4f\n' % np.max(energies))

    f.write('per|%.2f\n' % (100 * np.mean(percentages_energy)))
    f.write('per_std|%.4f\n' % np.std(percentages_energy))
    f.write('per_max|%.2f\n' % (100 * np.max(percentages_energy)))
    f.close()


# MAIN CODE ============================================================================================================
def run_test(P, K, ground_truth, num_trials, random_init, dir_name='test'):
    time_start = time.time()

    # Data generation --------------------------------------------------------------------------------
    N = len(P)
    V = [i for i in range(N)]
    E = [[c[0],c[1],c[2]] for c in itertools.combinations(V,3)]
    w = [cu.compute_volume(P,e) for e in E]
    
    #C = np.zeros(shape=(N,N))
    #for i in range(len(E)):
    #    e = E[i]
    #    i1, i2, i3 = e[0], e[1], e[2]
    #    vol = w[i]
    #    C[i1,i2] += vol
    #    C[i2,i1] += vol
    #    C[i1,i3] += vol
    #    C[i3,i1] += vol
    #    C[i2,i3] += vol
    #    C[i3,i2] += vol
        
    
    # Save plot
    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1
    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)
    dv.plot_data(P, K, ground_truth, 2, show_legend=False, show_data=False, save_to_file=True,
                 file_name=dirpath + '/id_' + str(experiment_id) + '_GT',
                 title='Ground Truth (N = %d, K = %d)' % (n*K, K))

    # Other parameters -------------------------------------------------------------------------------------------------
    use_IPM = False
    num_max_it = 20

    ab_pur_t, ab_min_pur_t, ab_ene_t, ab_per_t, ab_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)
    #ae_pur_t, ae_min_pur_t, ae_ene_t, ae_per_t, ae_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
    #                                         np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)
    #ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
    #                                         np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        print t
        if random_init:
            lb_init = np.random.randint(0, K, N)
            ab_sequence = None
        else:
            lb_init = np.zeros(N, dtype=int)
            ab_sequence = np.array([np.random.choice(K, K, replace=False), np.random.choice(K, K, replace=False)])

##        lb_dict = dict()
##        for i in range(N):
##            lb_dict[i] = lb_init[i]

        print(lb_init)
        start_t = time.time()
        lb_ab,_ = moves.large_move_maxcut_high_order(E, w, K, lb_init,
                                        ab_sequence=ab_sequence,
                                        num_max_it=num_max_it, use_IPM=use_IPM)
        ab_pur_t[t], ab_min_pur_t[t], ab_ene_t[t], ab_per_t[t] = cu.stats_clustering_high_order(E, w, lb_ab, ground_truth)
        ab_tim_t[t] = time.time() - start_t
        print(lb_ab)

        #start_t = time.time()
        #lb_ae,_ = moves.large_move_maxcut(C, K, lb_init,
        #                                move_type="ae", ab_sequence=ab_sequence,
        #                                num_max_it=num_max_it, use_IPM=use_IPM)
        #ae_pur_t[t], ae_min_pur_t[t], ae_ene_t[t], ae_per_t[t] = cu.stats_clustering_high_order(E, w, lb_ae, ground_truth)
        #ae_tim_t[t] = time.time() - start_t

        #start_t = time.time()
        #lb_ls,_ = cu.local_search(C, K, lb_init, num_max_it)
        #ls_pur_t[t], ls_min_pur_t[t], ls_ene_t[t], ls_per_t[t] = cu.stats_clustering_high_order(E, w, lb_ls, ground_truth)
        #ls_tim_t[t] = time.time() - start_t


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save data --------------------------------------------------------------------------------------------------------
    np.savetxt(str(dirpath) + '/P.dat', P)
    np.savetxt(str(dirpath) + '/gt.dat', ground_truth, fmt='%d')
    np.savetxt(str(dirpath) + '/lb_init.dat', lb_init)
    #np.savetxt(str(dirpath) + '/ab_sequence.dat', ab_sequence)

    print("ALPHA-BETA SWAP (id %d):" % experiment_id)
    print("> Mean time: %.4f + %.4f" % (np.mean(ab_tim_t), np.std(ab_tim_t)))
    print("> Mean pur:  %.4f + %.4f" % (np.mean(ab_pur_t), np.std(ab_pur_t)))
    save_data(ab_pur_t, ab_min_pur_t, ab_ene_t, ab_per_t, ab_tim_t, dirpath, filename="ab_results")

    #print("ALPHA EXPANSION (id %d):" % experiment_id)
    #print("> Mean time: %.4f + %.4f" % (np.mean(ae_tim_t), np.std(ae_tim_t)))
    #print("> Mean pur:  %.4f + %.4f" % (np.mean(ae_pur_t), np.std(ae_pur_t)))
    #save_data(ae_pur_t, ae_min_pur_t, ae_ene_t, ae_per_t, ae_tim_t, dirpath, filename="ae_results")

    #print("LOCAL SEARCH (id %d):" % experiment_id)
    #print("> Mean time: %.4f + %.4f" % (np.mean(ls_tim_t), np.std(ls_tim_t)))
    #print("> Mean pur:  %.4f + %.4f" % (np.mean(ls_pur_t), np.std(ls_pur_t)))
    #save_data(ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t, dirpath, filename="ls_results")

    print("Total time (id %d): %.4f s\n" % (experiment_id, time.time() - time_start))


if __name__ == "__main__":
    dir_name = 'alice_line_experiments'
    num_trials = 10
    random_init = True
    use_prev_p = 0
    shuffle_data = 0

    n = 10
    K = 2
    l = 3
    sigma_2 = 2
    sigmas = [0.1, 0.5, 0.75, 1.]
    for e in range(len(sigmas)):
        for i in range(5):
            sigma_1 = sigmas[e]
            params_data = {'sigma_1': sigma_1, 'sigma_2': sigma_2, 'K': K, 'dim_space': 2, 'n': n,
                   'use_prev_p': use_prev_p, 'shuffle': shuffle_data, 'l': l}
            P, ground_truth = dg.generate_data_random(params_data)
            run_test(P, K, ground_truth, num_trials, random_init, dir_name=dir_name)



