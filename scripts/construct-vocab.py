import os
import argparse
from collections import Counter


def main(args):
	print('Building vocabulary...')

	word_counts = Counter()
	with open(args.inpath, 'r') as f:
		for line in f:
			words = line.strip().split()
			word_counts.update(words)

	if args.max_size > -1:
		print('Buidling vocab from top {} most-common words.'.format(args.max_size))
		vocabulary = dict([(word, count)
			for word, count in word_counts.most_common(args.max_size)])
	else:
		print('Building vocab from words with more than {} occurences.'.format(args.min_count))
		vocabulary = dict([(word, count)
			for word, count in word_counts.most_common() if count >= args.min_count])

	with open(args.out, 'w') as f:
		for word, count in vocabulary.items():
			print(word, count, file=f)

	print('Processed {} tokens.'.format(sum(word_counts.values())))
	print('Counted {} unique words.'.format(len(word_counts)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('inpath', type=str,
						help='input path for corpus')
	parser.add_argument('--out', type=str, default='vocab.txt',
						help='where to print the pairs')
	parser.add_argument('--min-count', type=int, default=5,
						help='minimum word count')
	parser.add_argument('--max-size', type=int, default=-1,
						help='maximum vocabulary size (keep only k most common words)')
	args = parser.parse_args()

	assert os.path.exists(args.inpath), 'invalid corpus path `{}`.'.format(args.inpath)

	main(args)
