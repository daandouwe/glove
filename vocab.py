import os
import sys
import argparse
from collections import Counter


def main(args):
    if args.verbose:
        print('Building vocabulary...', file=sys.stderr)

    word_counts = Counter()
    with open(args.corpus_path, 'r') as f:
        for line in f:
            words = line.strip().split()
            word_counts.update(words)

    if args.max_size > -1:
        vocabulary = dict([(word, count) for word, count in word_counts.most_common(args.max_size)])
    else:
        vocabulary = dict([(word, count) for word, count in word_counts.most_common() if count >= args.min_count])

    print(len(vocabulary), sum(vocabulary.values()))
    for word, count in vocabulary.items():
        print(word, count)

    if args.verbose:
        print('Processed {} tokens.'.format(sum(word_counts.values())), file=sys.stderr)
        print('Counted {} unique words.'.format(len(word_counts)), file=sys.stderr)
        print('Truncating vocabulary at min count {}.'.format(args.min_count), file=sys.stderr)
        print('Using vocabulary of size {}.\n'.format(len(vocabulary)), file=sys.stderr)


def write_vocabulary(vocabulary):
    with open(args.vocab_path, 'w') as f:
        for word, count in vocabulary.items():
            print(word, count, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', type=str,
                        help="Input path for corpus.'")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Printing results.')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum word count.')
    parser.add_argument('--max_size', type=int, default=-1,
                        help='Maximum vocabulary size (keep only k most common words).')
    args = parser.parse_args()

    assert os.path.exists(args.corpus_path), 'Invalid corpus path: {}'.format(args.corpus_path)

    main(args)
