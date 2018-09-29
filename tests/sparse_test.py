import numpy as np
from scipy.sparse import dok_matrix


X = dok_matrix((10, 10), dtype=np.float32)

Y = 1 + X

print(Y[1,2])
print(type(Y))


ix = [1,4,6]
print(ix)
submat = np.ix_(ix,ix)
print(submat)

Z = Y[submat]
print(type(Z))

W = Z.todense()
print(W)
