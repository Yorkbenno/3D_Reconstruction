'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import helper
import submission
import numpy as np
import matplotlib.pyplot as plt
import os


def findM2():
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    points = np.load("../data/some_corresp.npz")
    pts1 = points['pts1']
    pts2 = points['pts2']
    imheight, imwidth, _ = im1.shape
    M = max(imheight, imwidth)
    F = submission.eightpoint(pts1, pts2, M)
    # print(F)
    f1 = np.load('../data/intrinsics.npz')
    K1 = f1['K1']
    K2 = f1['K2']
    E = submission.essentialMatrix(F, K1, K2)
    print(E)
    M2s = helper.camera2(E)
    M1 = np.diag([1, 1, 1])
    M1 = np.hstack((M1, np.zeros((3, 1))))
    C1 = K1 @ M1
    # error = []

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        P, err = submission.triangulate(C1, pts1, C2, pts2)
        if np.all(P[:, 2] > 0):
            break
        # error.append(err)

    # index = np.argmin(np.array(error))
    # M2 = M2s[:, :, index]
    # C2 = K2 @ M2
    # P, err = submission.triangulate(C1, pts1, C2, pts2)

    if not os.path.isfile('../results/q3_3.npz'):
        np.savez('../results/q3_3.npz', M2=M2, C2=C2, P=P)

    return M1, C1, M2, C2, F


def findM22():
    data = np.load('../data/some_corresp.npz')
    # data = np.load('../data/some_corresp_noisy.npz')

    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    F = submission.eightpoint(data['pts1'], data['pts2'], M)
    E = submission.essentialMatrix(F, K1, K2)
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = helper.camera2(E)
    row, col, num = np.shape(M2s)
    # print(M1)
    C1 = np.matmul(K1, M1)
    # minerr = np.inf
    # res = 0
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = submission.triangulate(C1, pts1, C2, pts2)
        if np.all(P[:, 2] > 0):
            break
    #     if (err<minerr):
    #         minerr = err
    #         res = i
    # M2 = M2s[:,:,res]
    # C2 = np.matmul(K2, M2)
    # if(os.path.isfile('q3_3.npz')==False):
    #     np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)

    return M1, C1, M2, C2, F


if __name__ == '__main__':
    M1, C1, M2, C2, F = findM2()
    print("in main")
    print(C2)
    print(M2)
