import argparse

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
from tqdm import tqdm


def load_vocabulary(vocab_path):
	w2i, i2w = dict(), dict()
	with open(vocab_path, 'r') as f:
		for i, line in enumerate(f):
			word, count = line.split()
			w2i[word], i2w[i] = i, word
	return w2i, i2w


def save_matrix(f, mat):
	np.savez_compressed(
		f, data=mat.data, indices=mat.indices, indptr=mat.indptr, shape=mat.shape)


def weigh(x, x_max=100, alpha=0.75):
	"""The weighing function f applied to X_ij."""
	y = (x / x_max)**alpha
	y[x > x_max] = 1.
	return y


def construct_sparse_cooccurrence_matrix(path, w2i):
	"""Constructs a sparse co-occurrence matrix.

	Constructs in CSR format from the counts saved in the
	test-file at `path`. Experiments in `tests/test_sparse.py`
	showed that this is the fasted method to construct the matrix.
	"""
	n = len(w2i)
	print('Constructing sparse co-occurrence matrix (CSR) of shape [{:,}, {:,}].'.format(n,n))

	with open(path, 'r') as f:
		npairs = int(f.readline())

		row = np.zeros((2*npairs,), dtype=np.float32)
		col = np.zeros((2*npairs,), dtype=np.float32)
		data = np.zeros((2*npairs,), dtype=np.float32)

		k = 0
		for line in tqdm(f, total=npairs):
			w, v, count = line.split()
			i, j = w2i[w], w2i[v]

			count = float(count)
			row[k] = col[k+1] = i
			col[k] = row[k+1] = j
			data[k] = data[k+1] = count

			k += 2

		X = csr_matrix((data, (row, col)), shape=(n,n))
		print('Constructed csr X.')

		logX = csr_matrix((np.log(1 + data), (row, col)), shape=(n,n))
		print('Constructed csr logX.')

		fX = csr_matrix((weigh(data), (row, col)), shape=(n,n))
		print('Constructed csr fX.')

		return X, logX, fX


def construct_dense_cooccurrence_matrix_slow(path, w2i):
	"""Constructs a dense co-occurrence matrix."""
	n = len(w2i)
	X = np.zeros((n, n), dtype=np.float32)
	print("Making dense co-occurrence matrix of shape [{},{}]".format(n,n))

	with open(path, 'r') as f:
		npairs = int(f.readline())
		for line in tqdm(f, total=npairs):
			w, v, count = line.split()
			i, j = w2i[w], w2i[v]
			X[i,j] = X[j,i] = float(count)

		print('Constructed dense X.')

		logX = np.log(1 + X)
		print('Constructed dense logX.')

		fX = weigh(X)
		print('Constructed dense fX.')

	return X, logX, fX


def main(args):
	w2i, _ = load_vocabulary(args.vocab_path)
	X, logX, fX = construct_sparse_cooccurrence_matrix(args.cooccur_path, w2i)

	print('Saving co-occurrence matrices...')
	save_matrix(args.matrix_path + '.x', X)
	print('Saved X at `{}`.'.format(args.matrix_path + '.x'))

	save_matrix(args.matrix_path + '.logx', logX)
	print('Saved logX at `{}`.'.format(args.matrix_path + '.logx'))

	save_matrix(args.matrix_path + '.fx', fX)
	print('Saved fX at `{}`.'.format(args.matrix_path + '.fx'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_path', type=str,
						help='input path for vocabulary')
	parser.add_argument('cooccur_path', type=str,
						help='input path for cooccurence counts')
	parser.add_argument('matrix_path', type=str,
						help='output path for cooccurence matrix')
	args = parser.parse_args()

	main(args)
