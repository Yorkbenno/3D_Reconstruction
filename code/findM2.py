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

if __name__ == '__main__':
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    points = np.load("../data/some_corresp.npz")
    pts1 = points['pts1']
    pts2 = points['pts2']
    imheight, imwidth, _ = im1.shape
    M = max(imheight, imwidth)
    F = submission.eightpoint(pts1, pts2, M)
    f1 = np.load('../data/intrinsics.npz')
    K1 = f1['K1']
    K2 = f1['K2']
    E = submission.essentialMatrix(F, K1, K2)
    M2s = helper.camera2(E)
    M1 = np.diag([1, 1, 1])
    M1 = np.hstack((M1, np.zeros((3, 1))))
    C1 = K1 @ M1
    error = []

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        p, err = submission.triangulate(C1, pts1, C2, pts2)
        error.append(err)

    index = np.argmin(np.array(error))
    M2 = M2s[:, :, index]
    C2 = K2 @ M2
    P, err = submission.triangulate(C1, pts1, C2, pts2)

    # if not os.path.isfile('../results/q3_3.npz'):
    np.savez('../results/q3_3.npz', M2=M2, C2=C2, P=P)

    print(M2)