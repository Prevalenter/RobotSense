import numpy as np
import matplotlib.pyplot as plt

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def solve(points, ids):
    #(3, 35) (35,)
    xy = np.array([ids%5, ids//5]).T
    xy1 = np.concatenate([xy, np.ones(xy.shape[0])[:, None]], axis=1)

    xy1_invert = np.linalg.pinv(xy1)
    rst = np.dot(xy1_invert, points.T)
    p0 = rst[2]
    vec_a, vec_b = rst[0], rst[1]
    vec_c = np.cross(vec_a, vec_b)

    vec = np.array([
            (vec_a/np.linalg.norm(vec_a)),
            (vec_b/np.linalg.norm(vec_b)),
            (vec_c/np.linalg.norm(vec_c))
        ])

    return vec, p0

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = Axes3D(fig)
    points = np.load('points_selected.npy')
    ids = np.load('ids.npy')[:, 0]
    ax.scatter(*points, c='k', s=2)
    print(points.shape, ids.shape)


    selected_idx = [0,5,9,10,15,16,20,26,29,34]
    vec, p0 = solve(points[:, selected_idx], ids[selected_idx])
    vec *= 0.04
    ax.scatter(*p0, c='b', s=2)
    ax.plot(*np.array([p0, p0+vec[0]]).T)
    ax.plot(*np.array([p0, p0+vec[1]]).T)
    ax.plot(*np.array([p0, p0+vec[2]]).T)
    for idx, i in enumerate(points.T):
        # print(idx, i)
        ax.text(i[0], i[1]+0.001, i[2], ids[idx])



    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    axisEqual3D(ax)
    plt.show()