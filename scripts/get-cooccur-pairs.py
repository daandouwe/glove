import os
import sys
import argparse
from collections import Counter


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


def load_vocabulary(args):
	vocab = dict()
	with open(vocab_path, 'r') as f:
  		# First line holds total token and vocabulary count counts.
		vocab_size, num_tokens = f.readline().split()
		for line in f:
			word, count = line.split()
			vocab[word] = int(count)
	return vocab, vocab_size, int(num_tokens)


def main(args):
	vocab, vocab_size, num_tokens = load_vocabulary(args.vocab_path)
	context_function = get_context_function(args.context_function, args.window_size)

	if args.verbose:
		print('Counting cooccurences...', file=sys.stderr)
		print('vocabulary size: {}'.format(vocab_size), file=sys.stderr)
		print('window size: {}'.format(args.window_size), file=sys.stderr)
		print('context: {}'.format(args.context), file=sys.stderr)

	counts = Counter()
	with open(args.corpus_path, 'r') as f:
		step = 0
		for line in f:
			words = [w if w in vocab else None for w in line.strip().split()]
			for i, w in enumerate(words):
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

				if args.verbose and step%10000==0:
					print('Processed {}/{} tokens ({:.0f}%).'.format(step, num_tokens, 100*step/num_tokens), file=sys.stderr, end='\r')

		print('Processed {}/{} tokens ({:.0f}%).'.format(step, num_tokens, 100*step/num_tokens), file=sys.stderr)
		write_pairs(counts)


def write_pairs(counts):
	pairs = counts.most_common()
	print(len(pairs))
	for i, pair in enumerate(pairs, 1):
		(word, context), count = pair
		print('{} {} {:.1f}'.format(word, context, count))

		if args.verbose and i%10000==0:
			print('Written {}/{} pairs ({:.0f}%).'.format(i, len(pairs), 100*i/len(pairs)), file=sys.stderr, end='\r')
	print('Written {}/{} pairs ({:.0f}%).'.format(i, len(pairs), 100*i/len(pairs)), file=sys.stderr)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('corpus_path', type=str,
						help='input path for corpus')
	parser.add_argument('vocab_path', type=str,
						help='input path for vocabulary')

	parser.add_argument('-v', '--verbose', action='store_true',
						help='printing results')

	parser.add_argument('--min_count', type=int, default=5,
						help='minimum word count')
	parser.add_argument('--window_size', type=int, default=15,
						help='context window size')
	parser.add_argument('--context', default='default', choices=['default', 'dynamic', 'harmonic'],
						help='type of context: defualt, dynamic (sgns), harmonic (glove)')

	args = parser.parse_args()

	assert os.path.exists(args.corpus_path), 'invalid corpus path: {}.'.format(args.corpus_path)
	assert os.path.exists(args.vocab_path), 'invalid vocab path: {}.'.format(args.vocab_path)

	main(args)