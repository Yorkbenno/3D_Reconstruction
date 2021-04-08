import numpy as np

if __name__ == '__main__':
    result = [False] * 7
    result = np.array(result)
    result[[2, 4, 5]] = True
    a = np.concatenate((np.zeros(3), np.ones(4)))
    print(a[result])
