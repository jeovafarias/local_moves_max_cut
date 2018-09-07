import numpy as np
from sklearn.preprocessing import normalize
import sklearn.metrics.pairwise as skl


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
            point[0] = 1.5*np.random.uniform(-sigma_1, sigma_1, 1)
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
    centers= np.zeros(shape=(K,dim_ambient_space))
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
    centers= np.zeros(shape=(K,dim_ambient_space))
    for i in range(K):
        centers[i,:] = np.random.uniform(-1, 1, dim_ambient_space)
        if i != 0:
            while (min(skl.pairwise_distances(centers[0:i,:],centers[i,:].reshape(1,-1),metric='euclidean')) < min_dist):
                centers[i,:] = np.random.uniform(-1, 1, dim_ambient_space)

    # generate points for each cluster around center
    for i in range(K):
        new_points = sigma*np.random.randn(n,dim_ambient_space)+centers[i,:]
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
    
    
