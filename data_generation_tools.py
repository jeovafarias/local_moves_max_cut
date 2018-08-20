import numpy as np


def generate_data_random(params):
    """
    Generate a synthetic data-set

    :param params: (Dictionary) - {'sigma_1' (float): First noise parameter, 'sigma_2' (float): Second noise parameter,
                                   'K' (integer): number rof clusters, 'dim_space' (integer): points dimensionality,
                                   'pop_interv' (1d array[integer]): interval from which the population (num. of points)
                                   of each cluster will be drawn, 'set_size' (integer): subspace dimensionality + 2,
                                   'use_prev_p' (boolean): use previous, 'shuffle' (boolean): shuffle data}
    :return: (2d array[float], Nxd) - Points in R^d
             (1d array[float]) - Ground Truth labelling
    """
    pop_interval = params['pop_interv']
    assert (params['set_size'] >= 2), "Wrong set size!"

    if params['use_prev_p']:
        P = np.loadtxt("X_nice.dat")
        population_vec = np.loadtxt("pop_vec_nice.dat", dtype=np.int16)
    else:
        population_vec = np.random.randint(pop_interval[0], pop_interval[1] + 1, size=params['K'])
        if params['set_size'] == 2:
            P = bump_generator(params['K'], population_vec, params['sigma_1'], params['sigma_2'], params['dim_space'])
        else:
            P = subspace_generator(params['K'], population_vec, params['sigma_1'], params['sigma_2'],
                                   params['dim_space'], params['set_size'] - 2)

        np.savetxt('X_nice.dat', P)
        np.savetxt('pop_vec_nice.dat', population_vec)

    gt = []
    for i in range(params['K']):
        gt.extend((i * np.ones(population_vec[i])).astype(int))
    ground_truth = np.array(gt)

    if params['shuffle']:
        sequence = np.array(range(len(ground_truth)))
        np.random.shuffle(sequence)
        P = P[sequence, :]
        ground_truth = ground_truth[sequence]

    return P, ground_truth


def bump_generator(K, population_vec, sigma_1, sigma_2, dim_space):
    """
    Generate points in bumps, i.e. Gaussian distributed with variace sigma_2

    :param K: (integer) - Number of clusters (bumps)
    :param population_vec: interval from which the population of each cluster will be drawn
    :param sigma_1: (float) - Dispersion within the clusters (bumps) centers
    :param sigma_2: (float) - Dispersion within the points in each cluster
    :param dim_space: (integer) - points dimensionality
    :return: (2d array[float], Nxd) - Points in R^d
    """
    for i in range(0, K):
        point = np.random.normal(0, sigma_1, dim_space)
        population = population_vec[i]

        new_points = sigma_2 * np.random.randn(population, dim_space) + point
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    return P


def subspace_generator(K, population_vec, sigma_1, sigma_2, dim_space, set_size):
    """
     Generate points in subspaces of dimensionality lower than the space

    :param K: (integer) - Number of clusters (bumps)
    :param population_vec: interval from which the population of each cluster will be drawn
    :param sigma_1: (float) - Dispersion within the subspace
    :param sigma_2: (float) - Parameter that measures how much the subspaces will cross each other (the greater,
                    more the subspaces will cross each other)
    :param dim_space: (integer) - points dimensionality
    :param set_size: (integer) - subspace dimensionality + 2
    :return: (2d array[float], Nxd) - Points in R^d
    """
    for i in range(0, K):
        c = np.random.randn(dim_space, 1)

        basis = np.linalg.qr(np.random.randn(dim_space, set_size))[0]
        new_points = basis.dot(sigma_2*np.random.randn(set_size, population_vec[i])) + c
        new_points = new_points.T
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    P += sigma_1*np.random.rand(P.shape[0], P.shape[1])

    return P


def get_D31_data():
    """
    Load D31 dataset

    :return: (2d array[float], Nxd) - Points in R^d
             (1d array[float]) - Ground Truth labelling
    """
    filename = 'D31.txt'
    with open(filename) as f:
        data = f.readlines()
    N = len(data)

    P = np.zeros((N, 2))
    gt = np.zeros(N)

    for i in range(N):
        data_i = data[i].split("\t")
        P[i, :] = np.array(data_i[0:2], dtype=float)
        gt[i] = int(data_i[-1]) - 1

    return P, gt.astype(int)