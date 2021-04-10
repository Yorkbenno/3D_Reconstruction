import numpy as np

if __name__ == '__main__':
    # result = [False] * 7
    # result = np.array(result)
    # result[[2, 4, 5]] = True
    # a = np.concatenate((np.zeros(3), np.ones(4)))
    a = np.ones((3,5))
    a[2] += 1
    a = a/a[2]
    print(a)
