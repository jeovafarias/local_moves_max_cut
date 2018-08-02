import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as opt
from sklearn.decomposition import PCA


def plot_data(P, K, cl, set_size,
              show_data=True, save_to_file=False, file_name='foo', title='',
              show_legend=True, normalize=False, gt=''):
    N, dim_space = P.shape
    assert (dim_space in (2, 3))

    if normalize:
        assert (len(gt) == len(cl))
        cl = normalize_classification(gt, cl)

    if save_to_file or show_data:
        colors = cm.rainbow(np.linspace(0, 1, K))
        fig = plt.figure()
        if dim_space == 2:
            for i in range(K):
                P_aux = P[np.nonzero(cl == i)[0], :]
                plt.scatter(P_aux[:, 0], P_aux[:, 1], color=colors[i], label=str(i + 1))
                ax = plt.gca()
                ax.set_aspect('equal')

        else:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(K):
                P_aux = P[np.nonzero(cl == i)[0], :]
                s = ax.scatter(P_aux[:, 0], P_aux[:, 1], P_aux[:, 2], c=colors[i], label=str(i))
                s.set_edgecolors = s.set_facecolors = lambda *args: None
                ax = plt.gca()
                ax.set_aspect('equal')

        if show_legend:
            plt.legend(range(K), loc='lower center', ncol=K, mode="expand", shadow=True)

        if save_to_file:
            if title == '':
                plt.suptitle('Total number of points: %d | Set Size: %d '
                             % (P.shape[0], set_size), fontsize=14, fontweight='bold')
            else:
                plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.savefig(file_name + '.png', bbox_inches="tight")

        if show_data:
            plt.show()


def normalize_classification(gt, cl):
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    if not isinstance(cl, np.ndarray):
        gt = np.array(cl)

    K = len(np.unique(gt))
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            gt_i = list(np.nonzero(gt == i)[0])
            cl_j = list(np.nonzero(cl == j)[0])
            cost_matrix[i, j] = np.sum([1 for k in range(len(cl_j)) if cl_j[k] not in gt_i])

    true_pat = opt.linear_sum_assignment(cost_matrix)[1]

    norm_cl = np.copy(cl)
    for i in range(K):
        norm_cl[cl == true_pat[i]] = i * np.ones(np.sum(cl == true_pat[i]))

    return norm_cl


def plot_matrix(M, show_data=True, save_to_file=False, file_name='foo', title=''):
    if save_to_file or show_data:
        fig, ax = plt.subplots()
        cax = ax.imshow(M, cmap='jet')
        fig.colorbar(cax)

        if save_to_file:
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.savefig(file_name + '.png')

        if show_data:
            plt.show()
        else:
            plt.close()


def visualize_binaries(P, X, K, ground_truth, set_size, dim_space, elapsed_time, num_sample_pts=3,
                       show_data=True, save_to_file=False, file_name='foo'):

    assert (dim_space in (2, 3))

    N = P.shape[0]
    sample_points = np.random.choice(P.shape[0], num_sample_pts, replace=False)
    fig = plt.figure()

    colors = cm.rainbow(np.linspace(0, 1, K))
    center = (num_sample_pts / 2, num_sample_pts / 2 + 1) if num_sample_pts % 2 == 0 else int(num_sample_pts / 2) + 1
    if dim_space == 2:
        ax = fig.add_subplot(2, num_sample_pts, center)
        for i in range(K):
            P_aux = P[np.nonzero(ground_truth == i)[0], :]
            ax.scatter(P_aux[:, 0], P_aux[:, 1], color=colors[i])
            ax.set_aspect('equal')
            ax.set_title('Ground Truth')
    else:
        ax = fig.add_subplot(2, num_sample_pts, center, projection='3d')
        for i in range(K):
            P_aux = P[np.nonzero(ground_truth == i)[0], :]
            s = ax.scatter(P_aux[:, 0], P_aux[:, 1], P_aux[:, 2], c=colors[i])
            s.set_edgecolors = s.set_facecolors = lambda *args: None
            ax.set_aspect('equal')
            ax.set_title('Ground Truth')

    axis_id = 0
    axis = []

    for k in sample_points:
        axis_id += 1

        if dim_space == 2:
            ax = fig.add_subplot(2, num_sample_pts, num_sample_pts + axis_id)
            for i in range(0, N):
                if np.array_equal(P[i, :], P[k, :]):
                    im = ax.scatter(P[i, 0], P[i, 1], marker='*', c="r", s=200)
                else:
                    im = ax.scatter(P[i, 0], P[i, 1], vmin=-1, vmax=1, c=np.floor(100 *X[i, k]))
            ax.set_title('Point %d' % k)
            ax.set_aspect('equal')
            axis.append(ax)
        else:
            ax = fig.add_subplot(2, num_sample_pts, num_sample_pts + axis_id, projection='3d')
            for i in range(0, N):
                if np.array_equal(P[i, :], P[k, :]):
                    im = ax.scatter(P[i, 0], P[i, 1], P[i, 2], marker='*', c="r", s=200)
                else:
                    im = ax.scatter(P[i, 0], P[i, 1], P[i, 2], vmin=-1, vmax=1, c=np.floor(100 * X[i, k]))
            ax.set_title('Point %d' % k)
            ax.set_aspect('equal')
            axis.append(ax)
    fig.colorbar(im, ax=axis, ticks=[-1, 0, 1], orientation='horizontal')
    plt.suptitle('Total number of points: %d | Set Size: %d \n Calculation Time: %.3f s'
                 % (P.shape[0], set_size, elapsed_time),
                 fontsize=14, fontweight='bold')

    if save_to_file:
        plt.savefig(file_name + '.png')

    if show_data:
        plt.show()


def visualize_embedding(V, ground_truth, show_data=True, save_to_file=False, file_name='foo'):
    pca = PCA(n_components=3)
    pca.fit(V.T)
    V = pca.components_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(set(ground_truth))))

    for i in range(len(set(ground_truth))):
        V_aux = V[:, np.nonzero(ground_truth == i)[0]]
        s = ax.scatter(V_aux[0, :], V_aux[1, :], V_aux[2, :],  color=colors[i])
        s.set_edgecolors = s.set_facecolors = lambda *args: None
        ax = plt.gca()
        ax.set_aspect('equal')

    if save_to_file:
        plt.savefig(file_name + '.png')

    if show_data:
        plt.show()
