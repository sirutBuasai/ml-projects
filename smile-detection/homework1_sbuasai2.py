import numpy as np
import scipy as sp

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return A.dot(B) - C

def problem3 (A, B, C):
    return (A * B) + C.T

def problem4 (x, S, y):
    return x.T.dot(S.dot(y))

def problem5 (A):
    # this produces a vector with the same amount of rows
    # if A has 2 rows, the result will be [[1],
    #                                      [1]]
    return np.ones((A.shape[1],1))

def problem6 (A):
    # are we assuming that A is a squared matrix?
    # if not, are we assumping that the diagonal 0s wrap around
    B = np.copy(A)
    np.fill_diagonal(B, 0)
    return B

def problem7 (A, alpha):
    I = np.eye(A.shape[0])
    return A + (alpha * I)

def problem8 (A, i, j):
    return A[j,i]

def problem9 (A, i):
    return np.sum(A[i,:]) 

def problem10 (A, c, d):
    S = [num for row in A for num in row if num >= c and num <= d]
    return np.mean(S)

def problem11 (A, k):
    return sp.linalg.eigh(A,subset_by_index=[A.shape[0]-k, A.shape[0]-1])[1]

def problem12 (A, x):
    return np.linalg.solve(A, x)  

def problem13 (x, k):
    col_x = x[:, np.newaxis]
    return np.repeat(col_x, k, axis=1)

def problem14 (A):
    B = np.copy(A)
    for i in range(A.shape[0]):
      B[i,:] = np.random.permutation(A[i,:])
    return B

