import time

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix


def load_vocabulary(vocab_path):
	w2i, i2w = dict(), dict()
	with open(vocab_path, 'r') as f:
		f.readline()
		for i, line in enumerate(f):
			word, count = line.split()
			w2i[word], i2w[i] = i, word
	return w2i, i2w


def save_matrix(f, mat):
	np.savez_compressed(
		f, data=mat.data, indices=mat.indices, indptr=mat.indptr, shape=mat.shape)


def load_matrix(path):
	path = path + '.npz' if not path.endswith('.npz') else path
	loader = np.load(path)
	mat = csr_matrix(
		(loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
	return mat


def write_vectors(embeddings, path, dim, i2w, gensim=True):
	with open(path, 'w') as f:
		if gensim:
			# header needed for gensim reading in of vectors
			print(f'{len(i2w)} {dim}', file=f)
		for i, word in i2w.items():
			vec = embeddings[i, :]
			print(f'{word} {" ".join([str(val) for val in vec])}', file=f)


def load_vectors(path, gensim=True):
	with open(path) as f:
		if gensim:
			num_words, dim = f.readline().strip().split()
			# f = tqdm(f, total=num_words)
		w2i, i2w = dict(), dict()
		vecs = []
		for i, line in enumerate(f):
			line = line.strip().split()
			word, vec = line[0], [float(val) for val in line[1:]]
			w2i[word], i2w[i] = i, word
			vec = np.array([vec])
			vecs.append(vec)
	vecs = np.vstack(vecs)
	return vecs, w2i, i2w
