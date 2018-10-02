import os
import sys
import argparse
from collections import Counter

from tqdm import tqdm


def harmonic(window_size):
	fun = lambda x : (window_size - x) / window_size
	return fun


def dynamic():
	fun = lambda x : 1./x
	return fun


def default():
	fun = lambda x : 1
	return fun


def get_context_function(context, window_size):
	if context == 'harmonic':
		return harmonic(window_size)
	elif context == 'dynamic':
		return dynamic()
	else:
		return default()


def load_vocabulary(vocab_path):
	vocab = dict()
	with open(vocab_path, 'r') as f:
		for line in f:
			word, count = line.split()
			vocab[word] = int(count)
	return vocab


def write_pairs(counts, path):
	# Sort the pairs.
	with open(path, 'w') as f:
		print(len(counts), file=f)
		for (word, context), count in counts.most_common():
			print(word, context, round(count, 1), file=f)


def main(args):
	vocab = load_vocabulary(args.vocab_path)
	context_function = get_context_function(args.context, args.window_size)

	print('Counting cooccurences...')
	print('vocabulary size: {}'.format(len(vocab)))
	print('window size: {}'.format(args.window_size))
	print('context: {}'.format(args.context))

	counts = Counter()
	with open(args.corpus_path, 'r') as f:
		step = 0
		words = [[w if w in vocab else None for w in line.strip().split()] for line in f]
	words = words[0] if len(words) == 1 else words
	for i, w in enumerate(tqdm(words)):
		step += 1
		if w is not None:
			left_context = [v for v in words[max(0, i-args.window_size) : i]]
			right_context = [v for v in words[i+1 : i+args.window_size+1]]

			for dist, v in enumerate(reversed(left_context), 1):
				if v is not None:
					word_pair = tuple(sorted([w, v]))
					counts[word_pair] += context_function(dist)

			for dist, v in enumerate(right_context, 1):
				if v is not None:
					word_pair = tuple(sorted([w, v]))
					counts[word_pair] += context_function(dist)

		write_pairs(counts, args.out)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('corpus_path', type=str,
						help='input path for corpus')
	parser.add_argument('vocab_path', type=str,
						help='input path for vocabulary')

	parser.add_argument('--out', type=str, default='pairs.txt',
						help='where to print the pairs')
	parser.add_argument('--window-size', type=int, default=15,
						help='context window size')
	parser.add_argument('--context', default='default', choices=['default', 'dynamic', 'harmonic'],
						help='type of context: defualt, dynamic (sgns), harmonic (glove)')

	args = parser.parse_args()

	assert os.path.exists(args.corpus_path), 'invalid corpus path: {}.'.format(args.corpus_path)
	assert os.path.exists(args.vocab_path), 'invalid vocab path: {}.'.format(args.vocab_path)

	main(args)
