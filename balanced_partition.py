import numpy as np
from sklearn.datasets import make_regression


def check_error(Q, i):
    if i == 10 or 20 or 30 or 50:
        return 20
    else:
        return 0


def bicrteria(A):
    return 10


def calc_coreset(Q):
    return np.mean(Q)


def balanced_partition(points, index, epsilon, optimum):
    np.sort(points, axis=index)
    Q = []
    D = []

    for i, point in enumerate(points):
        Q.append(point)
        start = i
        end = i+1
        if check_error(Q, i) > epsilon * optimum:
            Q.pop()
            D.append([Q, start, end])

            Q.clear
            Q.append(point)

    coreset = [Q]

    return coreset


def main():
    X, y, coef = make_regression(n_samples=100, n_features=1, noise=0.1, coef=True)
    A = np.append(X, y[:, np.newaxis], axis=1)

    opt = bicrteria(A)

    coreset = balanced_partition(points=A, index=0, epsilon=0.1, optimum=opt)

    # tree...


if __name__ == '__main__':
    main()
