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


# MAIN CODE ============================================================================================================
def run_test(P, K, ground_truth, num_trials, random_init, dir_name='test'):
    time_start = time.time()

    # Data generation --------------------------------------------------------------------------------
    N = len(P)
    C = np.zeros(shape=(N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                vol = cu.compute_volume(P,[i,j,k])
                C[i,j,k] += vol

    # Other parameters -------------------------------------------------------------------------------------------------
    use_IPM = False
    use_reduction = True
    num_max_it = 100

    ab_pur_t, ab_min_pur_t, ab_ene_t, ab_per_t, ab_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)
    ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
                                             np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)

    # Iterate ----------------------------------------------------------------------------------------------------------
    for t in range(num_trials):
        print(t)
        if random_init:
            lb_init = np.random.randint(0, K, N)
            ab_sequence = None
        else:
            lb_init = np.zeros(N, dtype=int)
            ab_sequence = np.array([np.random.choice(K, K, replace=False), np.random.choice(K, K, replace=False)])

        total_ene = np.sum(C)
        gt_ene = cu.energy_clustering_triples(C,ground_truth)
        
        start_t = time.time()
        lb_ab,_,_ = moves.large_move_maxcut_triples(C, K, lb_init, ab_sequence, num_max_it, True)
        ab_pur_t[t], ab_min_pur_t[t], ab_ene_t[t], ab_per_t[t] = cu.stats_clustering_triples(C, lb_ab, ground_truth)
        ab_tim_t[t] = time.time() - start_t

##        start_t = time.time()
##        lb_ls,_,_ = cu.local_search_triples(C, K, lb_init, num_max_it)
##        ls_pur_t[t], ls_min_pur_t[t], ls_ene_t[t], ls_per_t[t] = cu.stats_clustering_triples(C, lb_ls, ground_truth)
##        ls_tim_t[t] = time.time() - start_t

    # Create result string
    result = ""
    result += str(N)+","+str(K)+","+str(total_ene)+","+str(gt_ene)+","
    result += str(np.mean(ab_ene_t))+","+str(np.max(ab_ene_t))+","
    result += str(np.mean(ab_pur_t))+","+str(ab_pur_t[np.argmax(ab_ene_t)])+","
    result += str(np.mean(ab_min_pur_t))+","+str(ab_min_pur_t[np.argmax(ab_ene_t)])+","
    result += str(np.mean(ab_tim_t))
    
    # Create result string
##    result = ""
##    result += str(N)+","+str(K)+","+str(total_ene)+","+str(gt_ene)+","
##    result += str(np.mean(ls_ene_t))+","+str(np.max(ls_ene_t))+","
##    result += str(np.mean(ls_pur_t))+","+str(ls_pur_t[np.argmax(ls_ene_t)])+","
##    result += str(np.mean(ls_min_pur_t))+","+str(ls_min_pur_t[np.argmax(ls_ene_t)])+","
##    result += str(np.mean(ls_tim_t))

    return result

if __name__ == "__main__":
    dir_name = 'line_instances'
    result_file = 'line_ab_10iters.csv'
    num_trials = 10
    random_init = True
    use_prev_p = False
    shuffle_data = 0

##    n = 8
##    K = [4,5,6]
##    l = 3
##    sigma_2 = 2
##    sigmas = [0.25, 0.5, 0.75, 1]
##    ind = 0
##    for k in K:
##        for e in range(len(sigmas)):
##            for i in range(10):
##                N = n*k
##                sigma_1 = sigmas[e]
##                params_data = {'sigma_1': sigma_1, 'sigma_2': sigma_2, 'K': k, 'dim_space': 2, 'n': n,
##                       'use_prev_p': use_prev_p, 'shuffle': shuffle_data, 'l': l}
##                P, ground_truth = dg.generate_data_random(params_data)
##                np.savetxt('line_instances/P_'+str(ind)+'.dat', P)
##                np.savetxt('line_instances/gt_'+str(ind)+'.dat', ground_truth, fmt='%d')
##                dv.plot_data(P, k, ground_truth, 2, show_legend=False, show_data=False, save_to_file=True,
##                     file_name='line_instances/line_test_' + str(ind),
##                     title='Ground Truth (N = %d, K = %d)' % (N, k))
##                ind += 1


    f = open(result_file,"w")
    f.write('instance, n, k, total energy, gt_energy, mean_energy, max_energy, mean_purity, purity_at_max, mean_min_purity, min_purity_at_max, mean_time \n')
    for i in range(4,40):
        print(i)
        P = np.loadtxt('line_instances/P_'+str(i)+'.dat')
        gt = np.loadtxt('line_instances/gt_'+str(i)+'.dat', dtype=np.int16)
        K = gt[-1]+1
        result_str = run_test(P,K,gt,num_trials,random_init,dir_name)
        f.write('line_instances/'+str(i)+","+result_str+"\n")
    f.close()

