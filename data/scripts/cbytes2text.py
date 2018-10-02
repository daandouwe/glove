#!/usr/bin/env python
import argparse
import os
import struct
from tqdm import tqdm

NBYTES = 16


def main(args):
    w2i, i2w = dict(), dict()
    with open(args.vocab_path, 'r') as f:
        for i, line in enumerate(f, 1):
            word, count = line.split()
            w2i[word], i2w[i] = i, word

    with open(args.bin_path, mode='rb') as file:
        file_content = file.read()

    npairs = len(file_content) // NBYTES

    print(f'Converting {len(file_content):,} C bytes to Python ({npairs:,} pairs)...')
    pairs = []
    for i in tqdm(range(npairs)):
        # Unpack (int, int, double) from the NBYTES chunk of bytes.
        content = struct.unpack("iid", file_content[i*NBYTES:(i+1)*NBYTES])
        pairs.append(content)

    print(f'Writing out counts to `{args.out}`...')
    with open(args.out, 'w') as f:
        print(npairs, file=f)
        for (i, j, count) in tqdm(pairs):
            print(i2w[i], i2w[j], round(count, 1), file=f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_path', type=str,
						help='vocabulary path')
	parser.add_argument('bin_path', type=str,
						help='counts produced by `cooccur`')
	parser.add_argument('--out', type=str, default='pairs.txt',
						help='where to print the pairs')
	args = parser.parse_args()

	main(args)
