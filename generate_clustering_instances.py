from __future__ import print_function
import os
import numpy as np
import matplotlib

import data_generation_tools as dg
import data_visualization_tools as dv


def save_vector_txt_file(vec, filename):
    with open(filename, 'a') as f:
        for i in range(len(vec) - 1):
            f.write(str(vec[i]) + ',')
        f.write(str(vec[-1]) + '\n')


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


def generate_instances(n, k, sigma, dir_name, use_simplex=False, use_D31=False):
    assert k >= 2

    # Data generation and visualization --------------------------------------------------------------------------------
    if use_D31:
        P, ground_truth = dg.get_D31_data()
        k = len(np.unique(ground_truth))
    else:
        params = {'sigma_1': 1, 'sigma_2': sigma, 'min_dist': 0, 'simplex': use_simplex, 'K': k, 'dim_space': 2, 'l': 2,
                  'n': n, 'use_prev_p': False, 'shuffle': False}
        P, ground_truth = dg.generate_data_random(params)

        per = 0.5
        new_k = int(np.floor(per * k))
        P, ground_truth, k = sample_clusters(P, ground_truth, k, new_k)
    assert sigma < 1.0

    experiment_id = 0
    while os.path.exists(dir_name + '/' + str(experiment_id)):
        experiment_id += 1

    props = [experiment_id, n, k, sigma]
    print(props)
    save_vector_txt_file(props, dir_name + '/index.txt')

    dirpath = dir_name + '/' + str(experiment_id)
    os.makedirs(dirpath)
    np.savetxt(str(dirpath) + '/gt.dat', ground_truth)
    np.savetxt(str(dirpath) + '/P.dat', P)
    dv.plot_data(P, k, ground_truth, 2,
                 save_to_file=True, file_name=str(dirpath) + '/image', title='', show_data=False)


if __name__ == "__main__":
    dir_name = 'clustering_instances'

    num_sample_datasets = 10
    ns = [5, 10, 20]
    Ks = [9, 16, 25, 36, 49, 64, 81, 100, 121]
    sigmas = [0.3, 0.4, 0.45]

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for n in ns:
        for k in Ks:
            for sigma in sigmas:
                for _ in range(num_sample_datasets):
                    generate_instances(n, k, sigma, dir_name, use_simplex=False, use_D31=False)
                    matplotlib.pyplot.close("all")