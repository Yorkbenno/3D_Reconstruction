"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import helper
import os
import cv2
import matplotlib.pyplot as plt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, [0]]
    x2 = pts2[:, [0]]
    y1 = pts1[:, [1]]
    y2 = pts2[:, [1]]
    N = pts1.shape[0]
    # Starting to construct the matrix
    col1 = x2 * x1
    col2 = x2 * y1
    col3 = x2
    col4 = y2 * x1
    col5 = y2 * y1
    col6 = y2
    col7 = x1
    col8 = y1
    col9 = np.ones((N, 1))
    A = np.hstack((col1, col2, col3, col4, col5, col6, col7, col8, col9))
    # Finish constructing the matrix A
    u, s, vt = np.linalg.svd(A)
    f = vt[-1, :]
    F = f.reshape(3, 3)
    # Now enforce the rank 2
    F = helper._singularize(F)
    # Now refine according to instruction
    F = helper.refineF(F, pts1, pts2)
    # What is the origin transform matrix?
    T = np.vstack(([1. / M, 0, 0], [0, 1. / M, 0], [0, 0, 1]))
    # Denormalize F
    F = T.T @ F @ T

    if not os.path.isdir('../results'):
        os.mkdir('../results')
    # if not os.path.isfile('../results/q2_1.npz'):
    np.savez('../results/q2_1.npz', F=F, M=M)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, [0]]
    x2 = pts2[:, [0]]
    y1 = pts1[:, [1]]
    y2 = pts2[:, [1]]
    N = pts1.shape[0]
    # Starting to construct the matrix
    col1 = x1 * x2
    col2 = x1 * y2
    col3 = x1
    col4 = y1 * x2
    col5 = y1 * y2
    col6 = y1
    col7 = x2
    col8 = y2
    col9 = np.ones((N, 1))
    A = np.hstack((col1, col2, col3, col4, col5, col6, col7, col8, col9))
    # Finish constructing the matrix A
    u, s, vt = np.linalg.svd(A)
    # We get the last two col as the construction dimension
    f1 = vt[-1, :]
    f2 = vt[-2, :]
    # Reshape to get the matrix dimension(elements to construct the real F)
    F1 = f1.reshape(3, 3)
    F2 = f2.reshape(3, 3)
    # Refine the matrix according to the instruction
    F1 = helper.refineF(F1, pts1, pts2)
    F2 = helper.refineF(F2, pts1, pts2)

    # credit to CMU Chendi Lin
    func = lambda x: np.linalg.det(x * F1 + (1 - x) * F2)
    a0 = func(0)
    a1 = (func(1) - func(-1)) / 3 - (func(2) - func(-2)) / 12
    a2 = 0.5 * func(1) + 0.5 * func(-1) - func(0)
    a3 = (func(1) - func(-1)) / 6 + (func(2) - func(-2)) / 12

    coefficient = [a3, a2, a1, a0]
    solution = np.roots(coefficient)
    solution = solution[np.isreal(solution)]
    Farray = []
    T = np.vstack(([1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]))
    for i in range(len(solution)):
        F = T.T @ (solution[i] * F1 + (1 - solution[i]) * F2) @ T
        Farray.append(F)

    if not os.path.isfile('../results/q2_2.npz'):
        np.savez('../results/q2_2.npz', F=Farray, M=M, pts1=pts1, pts2=pts2)
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    essential = K2.T @ F @ K1
    return essential


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''

def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n, _ = pts1.shape
    Ps = []
    # Loop to compute for each point
    for i in range(n):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        # construct A
        row1 = x1 * C1[2] - C1[0]
        row2 = y1 * C1[2] - C1[1]
        row3 = x2 * C2[2] - C2[0]
        row4 = y2 * C2[2] - C2[1]
        A = np.vstack((row1, row2, row3, row4))
        # svd to solve the question
        u, s, v = np.linalg.svd(A)
        # p is a homogeneous coordinate in 3D
        p = v[-1, :]
        p = p / p[3]
        Ps.append(p)

    # compute error
    Ps = np.stack(Ps, axis=0)
    proj1 = C1 @ Ps.T
    proj2 = C2 @ Ps.T
    lam1 = proj1[-1, :]
    proj1 = proj1 / lam1
    lam2 = proj2[-1, :]
    proj2 = proj2 / lam2
    err1 = np.linalg.norm(proj1[[0, 1]].T - pts1, axis=1)
    err1 = np.square(err1).sum()
    err2 = np.linalg.norm(proj2[[0, 1]].T - pts2, axis=1)
    err2 = np.square(err2).sum()
    err = err1 + err2

    return Ps[:, :-1], err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation
    pass


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation
    pass


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


if __name__ == '__main__':
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    points = np.load("../data/some_corresp.npz")
    pts1 = points['pts1']
    pts2 = points['pts2']
    imheight, imwidth, _ = im1.shape
    M = max(imheight, imwidth)
    F = eightpoint(pts1, pts2, M)
    print(F)
    # N, _ = pts1.shape
    # index = np.random.randint(N, size=7)
    # seven_pts1 = pts1[index]
    # seven_pts2 = pts2[index]
    # F = sevenpoint(seven_pts1, seven_pts2, M)
    # helper.displayEpipolarF(im1, im2, F[-1])

    # f1 = np.load('../data/intrinsics.npz')
    # K1 = f1['K1']
    # K2 = f1['K2']
    # f2 = np.load('../results/q2_1.npz')
    # F = f2['F']
    # E = essentialMatrix(F, K1, K2)
    # print(E)
