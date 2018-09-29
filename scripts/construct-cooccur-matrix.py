import argparse

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix

from util import load_vocabulary, save_matrix


def weigh(x, x_max=100, alpha=0.75):
	"""The weighing function f applied to X_ij."""
	y = (x / x_max)**alpha
	y[x > x_max] = 1.
	return y


def weigh_(x, x_max=100, alpha=0.75):
	"""The weighing function f applied to X_ij."""
	if x > x_max:
		return 1.0
	else:
		return (x / x_max)**alpha


def construct_sparse_cooccurrence_matrix(path, w2i):
	"""Constructs a sparse co-occurrence matrix.

	Constructs in CSR format from the counts saved in the test file at `path`.
	"""
	n = len(w2i)
	print('Constructing sparse co-occurrence matrix (CSR) of shape [{},{}]'.format(n,n))

	with open(path, 'r') as f:
		npairs = int(f.readline())

		row = np.zeros((2*npairs,), dtype=np.float32)
		col = np.zeros((2*npairs,), dtype=np.float32)
		data = np.zeros((2*npairs,), dtype=np.float32)

		k = 0
		for l, line in enumerate(f, 1):
			w, v, count = line.split()
			i, j = w2i[w], w2i[v]

			count = float(count)
			row[k] = col[k+1] = i
			col[k] = row[k+1] = j
			data[k] = data[k+1] = count

			k += 2

			if l%100000 == 0:
				print('Reading counts to matrix: {:.0f}% ({}/{}).'.format(
					100*l/npairs, l, npairs), end='\r')

		print('Reading counts to matrix: {:.0f}% ({}/{}).'.format(100*l/npairs, l, npairs))

		X = csr_matrix((data, (row, col)), shape=(n,n))
		print('Constructed csr X.')

		logX = csr_matrix((np.log(1 + data), (row, col)), shape=(n,n))
		print('Constructed csr logX.')

		fX = csr_matrix((weigh(data), (row, col)), shape=(n,n))
		print('Constructed csr fX.')

		return X, logX, fX


# def load_cooccurrence_matrix(path, w2i):
# 	n = len(w2i)
# 	print("Making dense co-occurrence matrix of shape [{},{}]".format(n,n))
#
# 	X = np.zeros((n, n), dtype=np.float32)
#
# 	with open(path, 'r') as f:
# 		npairs = int(f.readline())
# 		for l, line in enumerate(f):
# 			w, v, count = line.split()
# 			i, j = w2i[w], w2i[v]
# 			X[i,j] = X[j,i] = float(count)
#
# 			if l%10000 == 0:
# 				print('Processed {} pairs ({:.0f}%).'.format(l, 100*l/npairs), end='\r')
#
# 		print('Processed {} pairs ({:.0f}%).'.format(l, 100*l/npairs))
# 		print('Constructed dense X.')
#
# 		logX = np.log(1 + X)
# 		print('Constructed dense logX.')
#
# 		fX = weigh(X)
# 		print('Constructed dense fX.')
#
# 	return X, logX, fX


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_path', type=str,
						help='input path for vocabulary')
	parser.add_argument('cooccur_path', type=str,
						help='input path for cooccurence counts')
	parser.add_argument('matrix_path', type=str,
						help='output path for cooccurence matrix')
	args = parser.parse_args()

	w2i, i2w = load_vocabulary(args.vocab_path)
	X, logX, fX = construct_sparse_cooccurrence_matrix(args.cooccur_path, w2i)

	print('Saving co-occurrence matrices at '{}'.'.format(args.matrix_path))
	save_matrix(args.matrix_path + 'x', X)
	print('Saved X')
	save_matrix(args.matrix_path + 'logx', logX)
	print('Saved logX')
	save_matrix(args.matrix_path + 'fx', fX)
	print('Saved fX')
