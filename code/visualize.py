'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import submission
import helper
import matplotlib.pyplot as plt
import os
import findM2
from mpl_toolkits.mplot3d import Axes3D


def threeD_Visualization():
    M1, C1, M2, C2, F = findM2.findM2()
    points = np.load("../data/templeCoords.npz")
    # print(points.files)
    x1 = points['x1']
    y1 = points['y1']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    # print(x1.shape)
    N, _ = x1.shape
    points2 = []

    for i in range(N):
        x = x1[i, 0]
        y = y1[i, 0]
        x2, y2 = submission.epipolarCorrespondence(im1, im2, F, x, y)
        points2.append([x2, y2])

    points2 = np.stack(points2, axis=0)
    points1 = np.hstack((x1, y1))
    P, error = submission.triangulate(C1, points1, C2, points2)

    print(error)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(3.4, 4.2)
    ax.set_xlim3d(-0.8, 0.6)

    plt.show()
    if not os.path.isfile('../results/q4_2.npz'):
        np.savez('../results/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)


if __name__ == '__main__':
    threeD_Visualization()
    # q42()
