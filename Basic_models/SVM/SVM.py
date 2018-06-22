"""
本程序使用SMO算法，采用高斯核函数，对非线性可分的训练集进行划分
作者: 朱海
地址: https://github.com/Demohai/my_machine_learning/Basic_models
"""


import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as scio


# deifine kernel function
def kernel(x, z, sigma=10, kernelOption='Gauss'):
    m = x.shape[0]
    k = np.zeros((m, 1))
    if kernelOption == 'Linear':
        for i in range(m):
            k[i] = np.dot(x[i], z.T)
        return k
    elif kernelOption == 'Gauss':
        for i in range(m):
            A = x[i] - z
            k[i] = [np.exp(-np.dot(A, A.T)/(2*sigma**2))]
        return k
    else:
        raise NameError("Not support kernel type! You can use 'Linear' or 'Gause'")


# calculate error_i
def calError(x, y, i, alphas, b):
    # ---------------------------------------------------------------
    # Arguments:  x:  x is an array of whole examples, shape: (m, d)
    #             xi: xi is the i'th example, shape: (1, d)
    #             y:  y is an array including labels of whole examples, shape: (m, 1)
    #             yi: yi is the label of the i'th example, shape: (1, 1)
    # ---------------------------------------------------------------
    f_xi = float(np.dot(alphas.T, y*kernel(x, x[i, :]))) + b
    error_i = f_xi - y[i]
    return error_i


# update error cache, 1 stands for corresponding error updated, 0 stands for corresponding unupdated
def updateError(x, y, i, alphas, b, error):
    error_i = calError(x, y, i, alphas, b)
    error[i] = [1, error_i]


def findNonBound(alphas, C):
    nonbound = []
    for i in range(len(alphas)):
        if 0 < alphas[i] < C:
            nonbound.append(i)
    return nonbound


# produce a random j of range 0-m with max step
def select_j(x, y, i, error, error_i, alphas, b):
    m = np.shape(x)[0]
    # related to updated error
    valid_error = np.nonzero(error[:, 0])[0]
    if len(valid_error) > 1:
        j = 0
        maxDelta = 0
        error_j = 0
        for k in valid_error:
            if k == i:
                continue
            error_k = calError(x, y, k, alphas, b)
            if abs(error_i - error_k) > maxDelta:
                j = k
                maxDelta = abs(error_i - error_k)
                error_j = error_k
    else:
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        error_j = calError(x, y, j, alphas, b)
    return j, error_j


# limit alphas[j]
def clipAlpha(alpha_j, H, L):
    if alpha_j > H:
        alpha_j = H

    if alpha_j < L:
        alpha_j = L
    return alpha_j


# inner loop
def inner_loop(x, y, toler, C, alphas, error, i, b):
    # calculate error_i
    error_i = calError(x, y, i, alphas, b)
    # check and pick up the alpha who violates the KKT condition
    # ------------------------------------------------------------
    # satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    # ------------------------------------------------------------
    # violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    # ------------------------------------------------------------------------------
    if (y[i] * error_i < -toler and alphas[i] < C) or (y[i] * error_i > toler and alphas[i] > 0):
        # update error[i]
        error[i] = [1, error_i]
        j, error_j = select_j(x, y, i, error, error_i, alphas, b)

        # copy the old alpha to calculate new alpha below
        alpha_i_old = alphas[i]
        alpha_j_old = alphas[j]

        # calculate L and H
        if y[i] != y[j]:
            L = max(0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])
        else:
            L = max(0, alphas[j] + alphas[i] - C)
            H = min(C, alphas[j] + alphas[i])

        if L == H:
            return alphas, b, 0

        xi = x[i, :].reshape(1, 2)
        xj = x[j, :].reshape(1, 2)
        k11 = kernel(xi, xj)
        k22 = kernel(xj, xj)
        k12 = kernel(xi, xj)

        elta = float(2.0 * k12 - k11 - k22)
        if elta >= 0:
            return alphas, b, 0

        # update alphas[j]
        alphas[j] -= y[j] * (error_i - error_j) / elta
        # limit alphas[j]
        alphas[j] = clipAlpha(alphas[j], H, L)
        # update error[j]
        updateError(x, y, j, alphas, b, error)

        if abs(alphas[j] - alpha_j_old) < 0.00001:
            return alphas, b, 0

        # update alphas[i]
        alphas[i] = alpha_i_old + y[j] * y[i] * (alpha_j_old - alphas[j])

        # update b
        bi = b - error_i - y[i] * (alphas[i] - alpha_i_old) * k11 - y[j] * (alphas[j] - alpha_j_old) * k12
        bj = b - error_j - y[j] * (alphas[i] - alpha_i_old) * k12 - y[j] * (alphas[j] - alpha_j_old) * k22

        if 0 < alphas[i] < C:
            b = bi
        elif 0 < alphas[j] < C:
            b = bj
        else:
            b = (bi + bj) / 2
        return alphas, b, 1
    else:
        return alphas, b, 0


# main SMO
def SMO(x, y, C, toler, maxIter):
    b = 0
    # m is the number of examples, d is the dimension of one example
    m, d = np.shape(x)
    # alphas is an array including m alpha, shape:(m, 1)
    alphas = np.zeros((m, 1))
    # error is a cache: mark the updated error as 1, mark the unupdated error as 0
    error = np.zeros((m, 2))
    # initialize iteration parameter
    iter = 0
    # the flag of going through all examples
    iterEntire = True
    # start training
    while (iter < maxIter):
        iter += 1
        # go through all examples
        if iterEntire:
            alpha_pair_changed = 0
            for i in range(m):
                alphas, b, one_or_zero = inner_loop(x, y, toler, C, alphas, error, i, b)
                alpha_pair_changed += one_or_zero

            if alpha_pair_changed == 0:
                break
            else:
                iterEntire = False

        # go through no bound examples
        else:
            alpha_pair_changed = 0
            nonbound = findNonBound(alphas, C)
            for i in nonbound:
                alphas, b, one_or_zero = inner_loop(x, y, toler, C, alphas, error, i, b)
                alpha_pair_changed += one_or_zero

            if alpha_pair_changed == 0:
                iterEntire = True
    print("Training finished!")
    return alphas, b


# build training examples and testing examples
def loadDataSet(filename):
    data = scio.loadmat(filename)
    dataset = data['nonlinear']
    # the shape of train_x is (200, 2)
    train_x = dataset[0:2, 1000:1200].T
    # the shape of train_y is (200, 1)
    train_y = dataset[2, 1000:1200].reshape(200, 1)
    test_x = dataset[0:2, 1200:1300].T
    test_y = dataset[2, 1200:1300].T.reshape(100, 1)
    return train_x, train_y, test_x, test_y


# plot trained svm model
def show(train_x, train_y, alphas, b):
    # horizontal and vertical axises
    plt.xlabel('X1')
    plt.ylabel('X2')
    m = np.shape(train_x)[0]
    # draw all examples
    for i in range(m):
        if train_y[i] == -1:
            plt.plot(train_x[i, 0], train_x[i, 1], '.', color='r')
        elif train_y[i] == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], '.', color='b')

    # mark support vectors
    supportVectorsIndices = np.nonzero(alphas)[0]
    supportVector = train_x[supportVectorsIndices]
    plt.scatter(supportVector[:, 0], supportVector[:, 1], s=100, c='y', alpha=0.5, marker='o')

    # draw classify line
    X1 = np.arange(-50.0, 50.0, 0.1)
    X2 = np.arange(-50.0, 50.0, 0.1)
    x1, x2 = np.meshgrid(X1, X2)
    w = b
    for i in supportVectorsIndices:
        z1 = x1 - train_x[i, 0]
        z2 = x2 - train_x[i, 1]
        z3 = np.exp(-0.5 * (z1 ** 2 + z2 ** 2) / 10 ** 2)
        w += alphas[i]*train_y[i]*z3

    plt.contour(x1, x2, w, 0, colors='g')
    plt.show()


# test your svm model given test set
def test(test_x, test_y, train_x, train_y, alphas, b):
    # the number of test examples
    num_test = np.shape(test_x)[0]
    # find the indices of support vector
    supportVectorsIndices = np.nonzero(alphas)[0]
    print("the indices of the support vectors is :", supportVectorsIndices)
    supportVectors = train_x[supportVectorsIndices]
    supportVectorsLabels = train_y[supportVectorsIndices]
    supportVectorsAlphas = alphas[supportVectorsIndices]
    matchCount = 0
    for i in range(test_x.shape[0]):
        predict = float(np.dot(supportVectorsAlphas.T, supportVectorsLabels*kernel(supportVectors, test_x[i, :]))) + b

        if np.sign(predict) == np.sign(test_y[i]):
            matchCount += 1
    # calculate accuracy
    accuracy = float(matchCount) / num_test
    return accuracy

print("step 1: load data...")
train_x, train_y, test_x, test_y = loadDataSet('nonlinearData.mat')

print("step 2: training...")
C = 0.8
toler = 0.001
maxIter = 40
alphas, b = SMO(train_x, train_y, C, toler, maxIter)

print("step 3: testing...")
accuracy = test(test_x, test_y, train_x, train_y, alphas, b)
print("Accuracy:", accuracy)

print("step 4: show the trained model...")
show(train_x, train_y, alphas, b)

