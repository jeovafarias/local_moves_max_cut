from __future__ import print_function
import os
import sys
import time
import numpy as np

import moves
import sdp_solvers as solvers
import clustering_utils as cu

import sklearn.metrics.pairwise as skl
from sklearn.cluster import KMeans as kmeans

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.set_printoptions(linewidth=1000, precision=4, threshold=np.nan, suppress=True)


class ClusteringInstance:
    """
    Clustering Instance information

    Attributes:
        id, k, n    Instance ID, num. of clusters, num. of points per cluster, respectively.
        P, gt, C    Points in R^{d}, ground_truth vector, matrix of pairwise distances, respectively.
    """

    id, k, n = None, None, None
    P, gt, C = None, None, None

    def __init__(self, id, n, k, P, gt):
        # type: (int, int, int, numpy.ndarray, numpy.ndarray) -> ClusteringInstance

        self.k = k
        self.P = P
        self.n = n
        self.id = id
        self.gt = gt
        self.C = skl.pairwise_distances(self.P, metric='sqeuclidean')

    def get_gt_energy(self):
        """
        Compute instance's ground truth energy

        :return: (float) - ground truth energy
        """
        return cu.energy_clustering_pairwise(self.C, self.gt)


class MethodHandler:
    """
    Parent class for the method handlers

    Attributes:
        method      Method that will be handled by this object.
        name        Method's name.
        dir_name    Directory name where the spreadsheets with the clustering results will be saved.
    """

    dir_name = None
    method = None
    name = None

    def __init__(self, method, dir_name):
        # type: (function, str) -> MethodHandler
        self.method = method
        self.name = method.__name__
        self.dir_name = dir_name


    def get_filename(self):
        """
        Get the name of the spreadsheet that is saving the method's results.

        :return: (string) - Filename
        """
        endname = '' if type(self) is NonIterativeMethodHandler else '_' + str(self.num_starts) + 'starts'
        return self.dir_name + '/' + self.name + endname + ".csv"

    @staticmethod
    def entry_exists(id, filename):
        """
        Check if the Clustering Instance that is identified by id already exists in the the spreadsheet named
        'filename'.

        :param id: (integer) - instance id
        :param filename: (string) - spreadsheet file name.
        :return: (boolean) - whether the entry is in the given spreadsheet or not.
        """
        with open(filename, "rb") as f:
            M = np.loadtxt(f, delimiter=",", skiprows=1)
            if M.size == 0:
                return False
            else:
                return id in M[:, 0] if M.ndim > 1 else id == M[0]

    @staticmethod
    def create_header(filename, method_handler):
        """
        Create the header of the spreadsheet named by 'filename'.

        :param filename: (string) - spreadsheet name.
        :param method_handler: (MethodHandler) - the handler of the method whose spreadsheet is named as 'filename'.
        """
        if not os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write(method_handler.header)

    @staticmethod
    def save_line(filename, clustering_instance, stats):
        """
        Add a line to the spreadsheet named as 'filename'

        :param filename: (string) - spreadsheet name.
        :param clustering_instance:  (ClusteringInstance) - clustering instance that is being processed.
        :param stats: (array) - statistics of the labeling given to the clustering instance.
        """
        with open(filename, 'a') as f:
            f.write('%d' % clustering_instance.id + ',' + '%d' % clustering_instance.n + ',' +
                    '%d' % clustering_instance.k + ',' + '%.4f' % clustering_instance.get_gt_energy() + ',')
            for s in range(len(stats) - 1):
                f.write('%.4f' % stats[s] + ',')
            f.write('%.4f' % stats[-1] + '\n')


class NonIterativeMethodHandler(MethodHandler, object):
    """
    Handler for the Non Iterative methods.

    Attributes:
        header      Spreadsheet header for the method that is being handled.
    """
    header = 'id, n, k, gt_energy, mean_energy, max_energy, mean_purity, pur_at_max_ene, mean_time \n'

    def __init__(self, method, dir_name):
        # type: (function, str) -> NonIterativeMethodHandler
        super(NonIterativeMethodHandler, self).__init__(method, dir_name)
        MethodHandler.create_header(self.get_filename(), self)

    def get_stats(self, clustering_instance):
        """
        Apply the method that is bing handled to 'clustering_instance'.

        :param clustering_instance: (ClusteringInstance) - clustering instance that is being processed.
        :return: (array) - statistics of the labeling given to the clustering instance.
        """

        # Compute the labeling
        start_t = time.time()
        labeling, err = self.method(clustering_instance)

        # Gather the labeling statistics
        tim = time.time() - start_t
        purity, _, energy, _ = \
            cu.stats_clustering_pairwise(clustering_instance.C, labeling, clustering_instance.gt)

        return [energy, purity, tim]


class IterativeMethodHandler(MethodHandler, object):
    """
    Handler for the Iterative methods.

    Attributes:
        header      Spreadsheet header for the method that is being handled.
        num_starts  Num. of different initializations on the method being handled.
    """
    header = 'id, n, k, gt_energy, energy, purity, time \n'
    num_starts = 0

    def __init__(self, method, num_starts, dir_name):
        # type: (function, int, str) -> IterativeMethodHandler
        self.num_starts = num_starts
        super(IterativeMethodHandler, self).__init__(method, dir_name)
        MethodHandler.create_header(self.get_filename(), self)

    def get_stats(self, clustering_instance):
        """
        Apply the method that is bing handled to 'clustering_instance'.

        :param clustering_instance: (ClusteringInstance) - clustering instance that is being processed.
        :return: (array) - statistics of the labeling given to the clustering instance.
        """

        energies, purities, times = np.zeros(num_starts), np.zeros(num_starts), np.zeros(num_starts)
        # Iterate over the different initializations
        for t in range(self.num_starts):
            # Sample initial labeling
            lb_init = np.random.randint(0, clustering_instance.k, clustering_instance.k * clustering_instance.n)

            # Compute the labeling
            start_t = time.time()
            labeling = self.method(clustering_instance, lb_init)

            # Gather the labeling statistics
            times[t] = time.time() - start_t
            purities[t], _, energies[t], _ = \
                cu.stats_clustering_pairwise(clustering_instance.C, labeling, clustering_instance.gt)

        return [np.mean(energies), np.max(energies), np.mean(purities), purities[np.argmax(energies)], np.mean(times)]


# METHODS ==============================================================================================================
def it_sdp(clustering_instance):
    itsdp_X, _, _, err = cu.iterate_sdp(clustering_instance.C, clustering_instance.k, alpha=0)
    itsdp_labeling = cu.cluster_integer_sol(itsdp_X, clustering_instance.k)
    return itsdp_labeling, err


def sdp_std_rounding(clustering_instance):
    sdp_X, err, _, _ = solvers.maxkcut_admm_solver(clustering_instance.C, clustering_instance.k)
    V = np.linalg.cholesky(sdp_X + 1e-9 * np.trace(sdp_X) * np.eye(sdp_X.shape[0]))
    sdp_labeling = solvers.max_k_cut_rounding(V, {'is_a_hypergraph_problem': False, 'C': clustering_instance.C,
                                                  'K': clustering_instance.k, 'post_processing': False})
    return sdp_labeling, err


def sdp_new_rounding(clustering_instance):
    sdp_X, err, _, _ = solvers.maxkcut_admm_solver(clustering_instance.C, clustering_instance.k)
    V = np.linalg.cholesky(sdp_X + 1e-9 * np.trace(sdp_X) * np.eye(sdp_X.shape[0]))
    sdp_labeling = solvers.max_k_cut_rounding(V, {'is_a_hypergraph_problem': False, 'C': clustering_instance.C,
                                                  'K': clustering_instance.k, 'post_processing': True})
    return sdp_labeling, err


def km(clustering_instance):
    km_labeling =\
        kmeans(n_clusters=clustering_instance.k, init='random', n_init=1000).fit(clustering_instance.P).labels_
    return km_labeling, -1


def kmpp(clustering_instance):
    kmpp_labeling = \
        kmeans(n_clusters=clustering_instance.k, init='k-means++', n_init=1000).fit(clustering_instance.P).labels_
    return kmpp_labeling, -1


def ls(clustering_instance, lb_init):
    ls_labeling, _, _ = cu.local_search(clustering_instance.C, clustering_instance.k, lb_init)
    return ls_labeling


def ab(clustering_instance, lb_init):
    ab_labeling, _, err = \
        moves.large_move_maxcut(clustering_instance.C, clustering_instance.k, lb_init, move_type="ab")
    return ab_labeling


# MAIN CODE ============================================================================================================
def run_experiments(dir_instances, dir_spreadsheets, non_it_methods, it_methods, num_starts):
    """
    Run the experiments on the clustering instances in the folder 'dir_instances'

    :param dir_instances: (string) - folder where the clustering instances are located.
    :param dir_spreadsheets: (string) - folder where the results' spreadsheets will be saved.
    :param non_it_methods: (array of functions) - non iterative methods to be used.
    :param it_methods: (array of functions) - iterative methods to be used.
    :param num_starts: (integer) - num. of starts for the iterative methods.
    """
    time_start = time.time()

    # Read the clustering instance's index
    instances = np.loadtxt(dir_instances + '/index.txt', delimiter=',')
    ids, ns, Ks = instances[:, 0].astype(int), instances[:, 1].astype(int), instances[:, 2].astype(int)

    # Iterate over the instances
    for id_idx in range(len(ids)):

        # Read instance data and instantiate the clustering object
        P = np.loadtxt(dir_instances + '/' + str(ids[id_idx]) + '/P.dat', dtype=float)
        ground_truth = np.loadtxt(dir_instances + '/' + str(ids[id_idx]) + '/gt.dat').astype(int)
        clu_inst = ClusteringInstance(ids[id_idx], ns[id_idx], Ks[id_idx], P, ground_truth)

        print("Instance %d of %d (n = %d, k = %d)" % (id_idx, len(ids), clu_inst.n, clu_inst.k))

        # Solve the clustering instance by the non iterative methods
        print('(', end='')
        for method_idx in range(len(non_it_methods)):
            # Instantiate the method handler
            mh = NonIterativeMethodHandler(non_it_methods[method_idx], dir_spreadsheets)

            # Check if the clustering instance had already being processed.
            if MethodHandler.entry_exists(clu_inst.id, mh.get_filename()):
               continue

            # Compute the clustering labeling and its statistics and save them
            stats = mh.get_stats(clu_inst)
            MethodHandler.save_line(mh.get_filename(), clu_inst, stats)

            print('%s: %.3f | ' % (mh.name, stats[1]), end='')
        print(') (', end='')

        # Solve the clustering instance by the iterative methods
        for method_idx in range(len(it_methods)):
            # Instantiate the method handler
            mh = IterativeMethodHandler(it_methods[method_idx], num_starts, dir_spreadsheets)

            # Check if the clustering instance had already being processed.
            if MethodHandler.entry_exists(clu_inst.id, mh.get_filename()):
                continue

            # Compute the clustering labeling and its statistics and save them
            stats = mh.get_stats(clu_inst)
            MethodHandler.save_line(mh.get_filename(), clu_inst, stats)

            print('%s: %.3f | ' % (mh.name, stats[2]), end='')
        print(')')

    print("\nTotal time: %.4f s" % (time.time() - time_start))
    print("SET FINISHED ========================================================================== \n")


if __name__ == "__main__":

    dir_spreadsheets = 'comp_all_results2'
    dir_instances = 'clustering_instances'

    assert os.path.exists(dir_instances)
    if not os.path.exists(dir_spreadsheets):
        os.makedirs(dir_spreadsheets)

    num_starts = 10
    non_it_methods = [sdp_std_rounding, sdp_new_rounding, it_sdp, km, kmpp]
    it_methods = [ls]

    run_experiments(dir_instances, dir_spreadsheets, non_it_methods, it_methods, num_starts)
