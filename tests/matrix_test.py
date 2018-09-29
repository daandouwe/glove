import time
import sys
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix, coo_matrix, lil_matrix


def construct_test(k, n, m, matrix='dok'):
    t0 = time.time()
    for _ in range(k):
        if matrix == 'dok':
            X = construct_dok(n, m)
        if matrix == 'coo':
            X = construct_coo(n, m)
        if matrix == 'csr':
            X = construct_csr(n, m)
        if matrix == 'csc':
            X = construct_csc(n, m)
        if matrix == 'lil':
            X = construct_lil(n, m)
    avg = (time.time() - t0) / k
    print('{}: {} seconds'.format(matrix, avg))
    return avg


def construct_dok(n, m):
    X = dok_matrix((n, n), dtype=np.float32)
    ix = np.random.randint(0, high=n, size=(m,))
    jx = np.random.randint(0, high=n, size=(m,))
    for i,j in zip(ix, jx):
        X[i,j] = i + j
    return X


def construct_lil(n, m):
    X = lil_matrix((n, n), dtype=np.float32)
    ix = np.random.randint(0, high=n, size=(m,))
    jx = np.random.randint(0, high=n, size=(m,))
    for i,j in zip(ix, jx):
        X[i,j] = i + j
    return X


def construct_coo(n, m):
    row = np.random.randint(0, high=n, size=(m,))
    col = np.random.randint(0, high=n, size=(m,))
    data = row + col
    return coo_matrix((data, (row,col)), shape=(n,n))


def construct_csr(n, m):
    row = np.random.randint(0, high=n, size=(m,))
    col = np.random.randint(0, high=n, size=(m,))
    data = row + col
    return csr_matrix((data, (row,col)), shape=(n,n))


def construct_csc(n, m):
    row = np.random.randint(0, high=n, size=(m,))
    col = np.random.randint(0, high=n, size=(m,))
    data = row + col
    return csc_matrix((data, (row,col)), shape=(n,n))


def index_test(matrix_type, k, n):
    m = int(0.05*n*n)
    if matrix_type == 'dok':
        X = construct_dok(n, m=m)
    elif matrix_type == 'lil':
        X = construct_lil(n, m=m)
    elif matrix_type == 'csr':
        X = construct_csr(n, m=m)
    elif matrix_type == 'csc':
        X = construct_csr(n, m=m)
    else:
        X = construct_csr(n, m=m)

    t0 = time.time()
    for _ in range(k):
        ix = np.random.randint(0, high=n, size=(128,))
        submat = np.ix_(ix,ix) # to select the submatrix_type
        x = X[submat].todense()
    avg = (time.time() - t0) / k
    print('{}: {} seconds'.format(matrix_type, avg))

    return avg

def index_test_array(k, n):
    X = np.zeros((n,n), dtype=np.float32)
    t0 = time.time()
    for _ in range(k):
        ix = np.random.randint(0, high=n, size=(128,))
        submat = np.ix_(ix,ix) # to select the submatrix
        x = X[submat]
    avg = (time.time() - t0) / k
    print('array: {} seconds'.format(avg))
    return avg


if __name__ == '__main__':
    n = 1000

    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    print('Testing construction...')
    avg = construct_test(10, n, m=n, matrix='dok')
    avg = construct_test(10, n, m=n, matrix='lil')
    avg = construct_test(10, n, m=n, matrix='coo')
    avg = construct_test(10, n, m=n, matrix='csc')
    avg = construct_test(10, n, m=n, matrix='csr')
    print()

    print('Testing indexing')
    avg = index_test_array(100, n)
    # avg = index_test('dok', 100)
    avg = index_test('csr', 100, n)
    avg = index_test('csc', 100, n)
