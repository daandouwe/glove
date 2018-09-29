#!/usr/bin/env python
import os
import argparse

import train
import plot
import hogwild


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=['train', 'plot', 'eval', 'hogwild'])

    parser.add_argument('-v', '--verbose', action='store_true')

    # IO arguments.
    parser.add_argument('--name', type=str, default='text8.10k',
                        help='name for model')
    parser.add_argument('--vocab-dir', type=str, default='vocab',
                        help='input path for vocabulary')
    parser.add_argument('--matrix-dir', type=str, default='cooccur',
                        help='input path for cooccurence matrix')
    parser.add_argument('--out-dir', type=str, default='vec',
                        help='ouput directory to write vectors')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='directory to save model')
    parser.add_argument('--vec-path', type=str, default='',
                        help='directory to save model')
    parser.add_argument('--gensim-format', action='store_true',
                        help='save vectors in gensim format')

    # Model arguments.
    parser.add_argument('--emb-dim', type=int, default=50,
                        help='dimension of vectors')

    # Train arguments.
    parser.add_argument('--num-updates', type=int, default=10000,
                        help='number of parameter updates')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Size of minibatches.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--use-schedule', action='store_true',
                        help='using scheduler for optimizer')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='how often to save the model parameters')
    parser.add_argument('--print-every', type=int, default=100,
                        help='how often to print loss to screen')

    args = parser.parse_args()

    if args.mode == 'train':
        train.main(args)
    if args.mode == 'plot':
        plot.main(args)
    if args.mode == 'hogwild':
        hogwild.main(args)


if __name__ == '__main__':
    main()
