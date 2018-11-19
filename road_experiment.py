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
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


def load_road(x_min, x_max, y_min, y_max):
    # load file
    road_points = dict()
    road_list = []
    current_road = ''
    f = open('roads.txt')
    for line in f.readlines():
        words = line.split(',')
        road_id = words[0]
        x,y,z = float(words[1]), float(words[2]), float(words[3])
        if current_road == road_id:
            road_points[road_id].append([x,y,z])
        else:
            road_points[road_id] = [[x,y,z]]
            current_road = road_id
            road_list.append(road_id)
    num_roads = len(road_list)

    gt = []
    P = []
    K = 0
    n = 0
    for i in range(num_roads):
        num_pts = len(road_points[road_list[i]])
        total_in = 0
        for j in range(num_pts):
            point = road_points[road_list[i]][j]
            x,y,z = point[0],point[1],point[2]
            if (x>=x_min) and (x<=x_max) and (y>=y_min) and (y<=y_max):
                total_in += 1
        if (total_in >= 8):
            for j in range(num_pts):
                point = road_points[road_list[i]][j]
                x,y,z = point[0],point[1],point[2]
                if (x>=x_min) and (x<=x_max) and (y>=y_min) and (y<=y_max):
                    P.append(point)
                    gt.append(K)
                    n += 1
                    
            K += 1

    P = np.array(P)
    gt = np.array(gt)
    return P, gt, K


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
    #ls_pur_t, ls_min_pur_t, ls_ene_t, ls_per_t, ls_tim_t = np.zeros(num_trials), np.zeros(num_trials), \
    #                                         np.zeros(num_trials), np.zeros(num_trials), np.zeros(num_trials)

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

    return result


if __name__ == "__main__":
    dir_name = 'road_instances'
    result_file = 'road_ab_10iters.csv'
    num_trials = 10
    random_init = True
    use_prev_p = False
    shuffle_data = 0

##    x_min = 8.1461259
##    x_max = 11.1993265
##    y_min = 56.5824856
##    y_max = 57.750511
##
##    xdiff = (x_max-x_min)/60.
##    ydiff = (y_max-y_min)/25.
##    ind = 0
##    for i in range(25):
##        for j in range(50):
##            print(i+j)
##            x1 = x_min+j*xdiff
##            x2 = x_min+(j+1)*xdiff
##            y1 = y_min+i*ydiff
##            y2 = y_min+(i+1)*ydiff
##            P, gt, K = load_road(x1,x2,y1,y2)
##            if (P.shape[0] > 50) and (P.shape[0] < 150):
##                P[:,2] = P[:,2]/111111.  #scale z
##                P_2d = P[:,0:2]
##                np.savetxt('Test2/P_'+str(ind)+'.dat', P)
##                np.savetxt('Test2/gt_'+str(ind)+'.dat', gt, fmt='%d')
##                dv.plot_data(P_2d,K,gt,2,show_data=False,save_to_file=True,file_name="Test2/road_test_"+str(ind))
##                ind += 1
                #run_test(P,K,gt,num_trials,random_init,dir_name)
    f = open(result_file,"w")
    f.write('instance, n, k, total energy, gt_energy, mean_energy, max_energy, mean_purity, purity_at_max, mean_min_purity, min_purity_at_max, mean_time \n')
    for i in range(65):
        print(i)
        P = np.loadtxt('road_instances/P_'+str(i)+'.dat')
        gt = np.loadtxt('road_instances/gt_'+str(i)+'.dat', dtype=np.int16)
        K = gt[-1]+1
        if (len(P) < 100):
            result_str = run_test(P,K,gt,num_trials,random_init,dir_name)
            f.write('road_instances/'+str(i)+","+result_str+"\n")
    f.close()
        

    # Exp 1: 9.6727262, 9.75, 57.18, 57.213219316



