import numpy as np
from sklearn.preprocessing import normalize
import sklearn.metrics.pairwise as skl


def generate_data_random(params):
    """
    Generate a synthetic data-set

    :param params: (Dictionary) - {'sigma_1' (float): First noise parameter, 'sigma_2' (float): Second noise parameter,
                                   'K' (integer): number rof clusters, 'dim_space' (integer): points dimensionality,
                                   'n' (integer): num. of points per cluster, 'l' (integer): subspace
                                   dimensionality + 2, 'use_prev_p' (boolean): use previous, 'shuffle' (boolean): shuffle data}
    :return: (2d array[float], Nxd) - Points in R^d
             (1d array[float]) - Ground Truth labelling
    """

    if params['use_prev_p']:
        P = np.loadtxt("P.dat")
        ground_truth = np.loadtxt("gt.dat", dtype=np.int16)
    else:
        if params['l'] == 2:
            if params['min_dist'] > 0:
                P, ground_truth = generate_gaussian_mindist(params['K'], params['n'], params['min_dist'],
                                                            params['sigma_2'], params['dim_space'])
            else:
                P, ground_truth = bump_generator(params['K'], params['n'], params['sigma_1'],
                                                 params['sigma_2'], params['dim_space'])
        else:
            P, ground_truth = subspace_generator(params['K'], params['n'], params['sigma_1'], params['sigma_2'],
                                                 params['dim_space'], params['l'])

        np.savetxt('P.dat', P)
        np.savetxt('gt.dat', ground_truth)

    if params['shuffle']:
        sequence = np.array(range(len(ground_truth)))
        np.random.shuffle(sequence)
        P = P[sequence, :]
        ground_truth = ground_truth[sequence]

    return P, ground_truth


def bump_generator(K, n, sigma_1, sigma_2, dim_space):
    """
    Generate points in bumps, i.e. Gaussian distributed with variace sigma_2

    :param K: (integer) - Number of clusters (bumps)
    :param n: number of points per cluster
    :param sigma_1: (float) - Dispersion within the clusters (bumps) centers
    :param sigma_2: (float) - Dispersion within the points in each cluster
    :param dim_space: (integer) - points dimensionality
    :return: (2d array[float], Nxd) - Points in R^d
    """
    for i in range(0, K):
        point = np.random.normal(0, sigma_1, dim_space)
        new_points = sigma_2 * np.random.randn(n, dim_space) + point
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    gt = []
    for i in range(K):
        gt.extend([i for _ in range(n)])
    ground_truth = np.array(gt)

    return P, ground_truth


def subspace_generator(K, n, sigma_1, sigma_2, dim_space, l):
    """
     Generate points in subspaces of dimensionality lower than the space

    :param K: (integer) - Number of clusters (bumps)
    :param n: number of points per cluster
    :param sigma_1: (float) - Dispersion within the subspace
    :param sigma_2: (float) - Parameter that measures how much the subspaces will cross each other (the greater,
                    more the subspaces will cross each other)
    :param dim_space: (integer) - points dimensionality
    :param l: (integer) - subspace dimensionality
    :return: (2d array[float], Nxd) - Points in R^d
    """

    assert (l >= 2), "Wrong set size!"
    for i in range(0, K):
        c = np.random.randn(dim_space, 1)

        basis = np.linalg.qr(np.random.randn(dim_space, l - 2))[0]
        new_points = basis.dot(sigma_2*np.random.randn(l - 2, n)) + c
        new_points = new_points.T
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    P += sigma_1*np.random.rand(P.shape[0], P.shape[1])

    gt = []
    for i in range(K):
        gt.extend([i for _ in range(n)])
    ground_truth = np.array(gt)

    return P, ground_truth

# Generate K clusters where each cluster contains n points uniformly distributed on a sphere of radius
# r+N(0,sigma^2) in R^dim_ambient_space
def generate_sphere_surface(K, n, r, sigma, dim_ambient_space):
    # generate uniform centers with range = [-1,1]
    centers= np.zeros(shape=(K,dim_ambient_space))
    for i in range(K):
        centers[i,:] = np.random.uniform(-1, 1, dim_ambient_space)

    # generate points for each cluster around center
    for i in range(K):
        new_points = np.random.randn(n,dim_ambient_space)
        radius = r+sigma*np.random.randn(1,dim_ambient_space)
        new_points = radius*normalize(new_points,axis=1)+centers[i,:]
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    # generate ground truth array
    gt = []
    for i in range(K):
        gt.extend([i for _ in range(n)])
    ground_truth = np.array(gt)
    
    return P, ground_truth

# Generate K clusters where each cluster contains n points uniformly distributed in a sphere of radius
# r+N(0,sigma^2) in R^dim_ambient_space
def generate_unif_ball(K, n, r, sigma, dim_ambient_space):
    # generate uniform centers with range = [-1,1]
    centers= np.zeros(shape=(K, dim_ambient_space))
    for i in range(K):
        centers[i,:] = np.random.uniform(-1, 1, dim_ambient_space)

    # generate points for each cluster around center
    for i in range(K):
        new_points = np.random.randn(n,2)
        u = np.random.uniform(0,1,n)
        radius = r+sigma*np.random.randn(1,dim_ambient_space)
        new_points = radius*normalize(new_points,axis=1)
        for j in range(n):
            new_points[j,:] = (u[j]**(1./dim_ambient_space))*new_points[j,:]+centers[i,:]
            
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))
            
    # create ground truth vector
    gt = []
    for i in range(K):
        gt.extend([i for _ in range(n)])
    ground_truth = np.array(gt)
    
    return P, ground_truth

# Generate K clusters where each cluster contains n points normally distributed in dim_ambient_space
# and the centers are grid points are min_dist apart
def generate_gaussian_mindist(K, n, min_dist, sigma, dim_ambient_space):
    # generate uniform centers with range = [-1,1] but maintain min distance
    centers = np.zeros(shape=(K, dim_ambient_space))
    for i in range(K):
        centers[i, :] = np.random.uniform(-1, 1, dim_ambient_space)
        if i != 0:
            while min(skl.pairwise_distances(centers[0:i, :], centers[i, :].reshape(1, -1), metric='euclidean')) < min_dist:
                centers[i, :] = np.random.uniform(-1, 1, dim_ambient_space)

    # generate points for each cluster around center
    for i in range(K):
        new_points = sigma*np.random.randn(n, dim_ambient_space) + centers[i, :]
        if 'P' not in locals():
            P = new_points
        else:
            P = np.concatenate((P, new_points))

    # generate ground truth array
    gt = []
    for i in range(K):
        gt.extend([i for _ in range(n)])
    ground_truth = np.array(gt)
    
    return P, ground_truth
    


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
