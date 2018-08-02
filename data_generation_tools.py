import numpy as np


def generate_data_random(params):
    K = params['K']
    sigma_1 = params['sigma_1']
    sigma_2 = params['sigma_2']
    pop_interval = params['pop_interv']
    dim_ambient_space = params['dim_space']
    use_prev_p = params['use_prev_p']

    if use_prev_p:
        P = np.loadtxt("dataset_previous.dat")
        ground_truth = np.loadtxt("ground_truth_previous.dat", dtype='int')

    else:
        population_vec = np.random.randint(pop_interval[0], pop_interval[1] + 1, size=K)
        P = bump_generator(K, population_vec, sigma_1, sigma_2, dim_ambient_space)

        gt = []
        for i in range(K):
            gt.extend((i * np.ones(population_vec[i])).astype(int))
        ground_truth = np.array(gt)
        np.savetxt('dataset_previous.dat', P)
        np.savetxt('ground_truth_previous.dat', gt, fmt='%d')

    return P, ground_truth


def bump_generator(K, population_vec, sigma_1, sigma_2, dim_ambient_space):
    for i in range(0, K):
        if dim_ambient_space == 2:
            point = np.zeros(2)
            point[0] = 1.5 * np.random.uniform(-sigma_1, sigma_1, 1)
            point[1] = np.random.uniform(-sigma_1, sigma_1, 1)
        else:
            point = np.random.uniform(-sigma_1, sigma_1, dim_ambient_space)

        population = population_vec[i]

        new_points = sigma_2 * np.random.randn(population, dim_ambient_space) + point
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    return P
