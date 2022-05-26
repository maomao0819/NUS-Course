import numpy as np
import math


rows, cols = 12, 2

A = [[0 for j in range(rows)] for i in range(rows)]
A = np.array(A)
A[0,:]  = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]  # A
A[1,:]  = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  # B
A[2,:]  = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # C
A[3,:]  = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # D
A[4,:]  = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # E
A[5,:]  = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]  # F
A[6,:]  = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  # G
A[7,:]  = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # H
A[8,:]  = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # I
A[9,:]  = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]  # J
A[10,:] = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  # K
A[11,:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # L

Aorig = A


if (0):   # Q1

    print('------ A -----------')
    for i in range(rows):
        print(A[i])

    D = [[0 for j in range(rows)] for i in range(rows)]

    for i in range(rows):
        D[i][i] = sum(A[i])

    print('------ D -----------')
    for i in range(rows):
        print(D[i])

    D = np.array(D)

    L = D - A

    print('------ L -----------')
    for i in range(rows):
        print(L[i])

    from numpy import linalg as LA

    eigValues, eigVectors = LA.eig(L)

    #idx = eigValues.argsort()[::-1]  # descending order
    idx = eigValues.argsort()

    eigValues = eigValues[idx]
    eigVectors = eigVectors[:,idx]

    print('----- eigenvalues of L -----')
    print(eigValues)

    print('----- eigenvectors of L -----')
    print(eigVectors)

    print('----- eigenvectors corresponding to zero eigenvalues ----')
    for i in range(rows):
        if (eigValues[i] < 1e-10):
            print(i)
            print(eigVectors[:,i])




if (0):   # Q2

    A = Aorig
    A[5,6] = 0
    A[6,5] = 0

    print('------ A -----------')
    for i in range(rows):
        print(A[i])

    D = [[0 for j in range(rows)] for i in range(rows)]

    for i in range(rows):
        D[i][i] = sum(A[i])

    print('------ D -----------')
    for i in range(rows):
        print(D[i])

    D = np.array(D)

    L = D - A

    print('------ L -----------')
    for i in range(rows):
        print(L[i])

    from numpy import linalg as LA

    eigValues, eigVectors = LA.eig(L)

    #idx = eigValues.argsort()[::-1]  # descending order
    idx = eigValues.argsort()

    eigValues = eigValues[idx]
    eigVectors = eigVectors[:,idx]

    print('----- eigenvalues of L -----')
    print(eigValues)

    print('----- eigenvectors of L -----')
    print(eigVectors)

    print('----- eigenvectors corresponding to zero eigenvalues ----')
    for i in range(rows):
        if (eigValues[i] < 1e-10):
            print(i)
            print(eigVectors[:,i])




if (1):   # Q3

    A = Aorig
    A[5, 6] = 0
    A[6, 5] = 0
    A[9,10] = 0
    A[10,9] = 0

    print('------ A -----------')
    for i in range(rows):
        print(A[i])

    D = [[0 for j in range(rows)] for i in range(rows)]

    for i in range(rows):
        D[i][i] = sum(A[i])

    print('------ D -----------')
    for i in range(rows):
        print(D[i])

    D = np.array(D)

    L = D - A

    print('------ L -----------')
    for i in range(rows):
        print(L[i])

    from numpy import linalg as LA

    eigValues, eigVectors = LA.eig(L)

    #idx = eigValues.argsort()[::-1]  # descending order
    idx = eigValues.argsort()

    eigValues = eigValues[idx]
    eigVectors = eigVectors[:,idx]

    print('----- eigenvalues of L -----')
    print(eigValues)

    print('----- eigenvectors of L -----')
    print(eigVectors)

    print('----- eigenvectors corresponding to zero eigenvalues ----')
    for i in range(rows):
        if (eigValues[i] < 1e-10):
            print(i)
            print(eigVectors[:,i])



