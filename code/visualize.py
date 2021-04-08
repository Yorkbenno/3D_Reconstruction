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


def bundle_test():
    f1 = np.load('../data/intrinsics.npz')
    K1 = f1['K1']
    K2 = f1['K2']
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    points = np.load("../data/some_corresp_noisy.npz")
    pts1 = points['pts1']
    pts2 = points['pts2']
    imheight, imwidth, _ = im1.shape
    M = max(imheight, imwidth)

    F, inliers = submission.ransacF(pts1, pts2, M)
    inliers = inliers.flatten()
    E = submission.essentialMatrix(F, K1, K2)
    M2s = helper.camera2(E)
    M1 = np.hstack((np.diag([1., 1., 1.]), np.zeros((3, 1))))
    C1 = K1 @ M1

    for i in range(M2s.shape[-1]):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        P, err = submission.triangulate(C1, pts1[inliers], C2, pts2[inliers])
        if np.all(P[:, 2] > 0):
            break

    # P, err = submission.triangulate(C1, pts1[inliers], C2, pts2[inliers])
    # print(err)
    M2, P = submission.bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], P)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(P[:, 0], P[:, 1], P[:, 2])
    plt.title("3D reconstruction")
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_zlabel('Z', fontweight='bold')

    plt.show()


if __name__ == '__main__':
    # threeD_Visualization()
    bundle_test()
