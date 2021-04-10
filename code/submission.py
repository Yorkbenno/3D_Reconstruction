"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import helper
import os
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.optimize as optimize

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
    if not os.path.isfile('../results/q2_1.npz'):
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
    # print(err)

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
    point1 = np.array([x1, y1, 1]).T
    line = F @ point1
    # pt1 = np.array([[x1], [y1], [1]])
    # line = F.dot(pt1)
    window_size = 4
    a, b, c = line
    imheight, imwidth, _ = im1.shape
    lower_bound = y1 - 25
    upper_bound = y1 + 25

    errors = []
    candidate = []
    # im1_g = ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    # im2_g = ndimage.gaussian_filter(im2, sigma=1, output=np.float64)
    box1 = im1[y1 - window_size:y1 + window_size + 1, x1 - window_size: x1 + window_size + 1]
    box1 = ndimage.gaussian_filter(box1, sigma=1, output=np.float64)

    for y2 in range(lower_bound, upper_bound + 1):
        x2 = int((-(b * y2 + c) / a))
        if y2 < window_size or y2 > imheight - window_size - 1 or x2 < window_size or x2 > imwidth - window_size - 1:
            continue
        elif y1 < window_size or y1 > imheight - window_size - 1 or x1 < window_size or x1 > imwidth - window_size - 1:
            return x1, int((- a * x1 - c) / b)
        box2 = im2[y2 - window_size: y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]
        box2 = ndimage.gaussian_filter(box2, sigma=1, output=np.float64)
        diff = box1 - box2
        diff = np.square(diff).sum()
        errors.append(diff)
        candidate.append([x2, y2])

    index = np.argmin(errors)
    # print(index)
    coor = candidate[index]
    x2, y2 = coor
    points = np.load("../data/some_corresp.npz")
    np.savez('../results/q4_1.npz', F=F, pts1=points['pts1'], pts2=points['pts2'])

    return x2, y2


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
    iter = 200
    N, _ = pts1.shape
    inliners = None
    num_inliner = 0
    p1_3D = np.hstack((pts1, np.ones((N, 1))))
    p2_3D = np.hstack((pts2, np.ones((N, 1))))
    threshold = 0.001

    for i in range(iter):
        index = np.random.randint(N, size=7)
        points1 = pts1[index]
        points2 = pts2[index]
        Farray = sevenpoint(points1, points2, M)

        for j in range(len(Farray)):
            F = Farray[j]
            inliner_candidate = []

            for z in range(N):
                p1 = p1_3D[z]
                p2 = p2_3D[z]
                error = p2 @ F @ p1.T
                error = abs(error)
                if error < threshold:
                    inliner_candidate.append(z)

            if len(inliner_candidate) > num_inliner:
                num_inliner = len(inliner_candidate)
                inliners = inliner_candidate

    inliners1 = pts1[inliners]
    inliners2 = pts2[inliners]

    F = eightpoint(inliners1, inliners2, M)
    result = np.zeros((N, 1)).astype(bool)
    result[inliners] = True
    inliers = result

    return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    norm = np.linalg.norm(r)
    if norm == 0:
        return np.diag([1., 1., 1.])
    K = r / norm
    kx = K[0]
    ky = K[1]
    kz = K[2]
    kc = np.vstack(([0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]))
    R = np.diag([1., 1., 1.]) * np.cos(norm) + (1 - np.cos(norm)) * K @ K.T + np.sin(norm) * kc

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).T
    s = np.linalg.norm(rho)
    c = (np.diag(R).sum() - 1) / 2
    if s == 0 and c == 1:
        return np.zeros((3, 1))
    elif s == 0 and c == -1:
        matrix = R + np.eye(3)
        i = 0
        while np.linalg.norm(matrix[:, i]) == 0:
            i += 1
        v = matrix[:, i]
        u = v / np.linalg.norm(v)
        r = u * np.pi
        r1 = r[0]
        r2 = r[1]
        r3 = r[2]
        if (r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0):
            r = -1 * r
        return r
    else:
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta
        return r


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
    C1 = K1 @ M1
    # Don't have M2, so find it
    N, _ = p1.shape
    Points_3D = x[:3 * N]
    Points_3D = Points_3D.reshape((N, 3))
    homo_points = np.hstack((Points_3D, np.ones((N, 1))))
    # We need the rotation and translation matrix
    r2 = x[3 * N: 3 * N + 3].reshape((3, 1))
    R2 = rodrigues(r2)
    t2 = x[3 * N + 3: 3 * N + 6].reshape((3, 1))
    M2 = np.hstack((R2, t2))
    C2 = K2 @ M2
    # Find the projection error
    proj1 = C1 @ homo_points.T
    proj2 = C2 @ homo_points.T
    proj1 = proj1 / proj1[2]
    proj2 = proj2 / proj2[2]
    # According to the naming strategy given by the pdf instructions
    p1_hat = proj1[:-1].T
    p2_hat = proj2[:-1].T
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])]).reshape((4 * N, 1))

    return residuals


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
    def fun(x):
        return rodriguesResidual(K1, M1, p1, K2, p2, x).flatten().astype(float)

    N, _ = p1.shape
    points_3D = P_init.flatten()
    R2_init = M2_init[:, :3]
    r2_init = invRodrigues(R2_init).flatten()
    t2_init = M2_init[:, 3].flatten()

    x_init = np.concatenate((points_3D, r2_init, t2_init))
    x, temp = optimize.leastsq(fun, x_init)
    print(np.sum(fun(x) ** 2))
    P2 = x[:3 * N].reshape((N, 3))
    r2 = x[3 * N:3 * N + 3].reshape((3, 1))
    t2 = x[3 * N + 3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2


if __name__ == '__main__':
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    points = np.load("../data/some_corresp_noisy.npz")
    pts1 = points['pts1']
    pts2 = points['pts2']
    imheight, imwidth, _ = im1.shape
    M = max(imheight, imwidth)
    # F = eightpoint(pts1, pts2, M)
    # N, _ = pts1.shape
    F, inliers = ransacF(pts1, pts2, M)
    # index = np.random.randint(N, size=7)
    # seven_pts1 = pts1[index]
    # seven_pts2 = pts2[index]
    # F = sevenpoint(seven_pts1, seven_pts2, M)
    print(F)
    helper.displayEpipolarF(im1, im2, F)

    # f1 = np.load('../data/intrinsics.npz')
    # K1 = f1['K1']
    # K2 = f1['K2']
    # f2 = np.load('../results/q2_1.npz')
    # F = f2['F']
    # E = essentialMatrix(F, K1, K2)
    # print(E)
    # helper.epipolarMatchGUI(im1, im2, F)
