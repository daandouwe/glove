import argparse

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix

from util import save_matrix


def load_vocabulary(vocab_path):
	vocab = dict()
	counts = dict()
	with open(vocab_path, 'r') as f:
		vocab_size, num_tokens = f.readline().split() # First line holds total token and vocabulary count counts
		for i, line in enumerate(f):
			word, count = line.split()
			counts[word] = int(count)
			vocab[word] = i
	return counts, vocab


def construct_sparse_ppmi_matrix(path, word_counts, w2i):
	"""
	Constructs a co-occurrence sparse matrix in CSR format
	from the counts saved in the test file at `path`.
	"""

	n = len(w2i)
	print(f'Constructing sparse co-occurrence matrix (CSR) of shape [{n},{n}]'

	with open(path, 'r') as f:
		npairs = int(f.readline())

		row = np.zeros((2*npairs,), dtype=np.float32)
		col = np.zeros((2*npairs,), dtype=np.float32)
		data = np.zeros((2*npairs,), dtype=np.float32)

		k = 0
		for l, line in enumerate(f, 1):
			w, v, pair_count = line.split()
			i, j = w2i[w], w2i[v]

			p_wv = float(pair_count) /
			row[k] = col[k+1] = i
			col[k] = row[k+1] = j
			data[k] = data[k+1] = ppmi()

			k += 2

			if l%100000 == 0:
				print('Reading counts to matrix: {}/{} ({:.0f}%).'.format(l, npairs, 100*l/npairs), end='\r')

		print('Reading counts to matrix: {}/{} ({:.0f}%).'.format(l, npairs, 100*l/npairs), end='\r')

		X = csr_matrix((data, (row, col)), shape=(n,n))
		print('Constructed csr X.')

		logX = csr_matrix((np.log(1 + data), (row, col)), shape=(n,n))
		print('Constructed csr logX.')

		fX = csr_matrix((weigh(data), (row, col)), shape=(n,n))
		print('Constructed csr fX.')

		return X, logX, fX


def main():
    counts, vocab = load_vocabulary(args.vocab_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

	parser.add_argument('pairs_path', type=str, help='Input path for vocabulary.')
	parser.add_argument('vocab_path', type=str, help='Input path for vocabulary.')

	parser.add_argument('-v', '--verbose', action='store_true', help='Printing results.')

	parser.add_argument('--min_count', type=int, default=5, help='Minimum word count.')
	parser.add_argument('--window_size', type=int, default=15, help='Context window size.')
	parser.add_argument('--context', type=str, default='default', choices=['default', 'dynamic', 'harmonic'],
									help='Type of context: defualt, dynamic (SGNS), harmonic (GloVe).')

	args = parser.parse_args()

    main()
