import argparse
import os
import csv
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from util import load_vocabulary, save_matrix, load_matrix, write_vectors
from model import GloVe


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    # Set seed for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab_path = os.path.join(args.vocab_dir, f'{args.name}.vocab')
    matrix_path = os.path.join(args.matrix_dir, f'{args.name}')
    model_path = os.path.join(args.model_dir, f'{args.name}.model.dict')
    out_path = os.path.join(args.out_dir, args.name)

    w2i, i2w = load_vocabulary(vocab_path)
    vocab_size = len(i2w)
    sparse = bool(vocab_size > 20000)
    print(f'Loaded vocabulary of size {vocab_size}.')

    # Construct model and optimizer.
    model = GloVe(vocab_size=len(w2i), emb_dim=args.emb_dim, sparse=True).to(device)
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    def load_data(rank):
        # Loading co-occurrence data.
        if rank == 0: print('Loading co-occurrence matrices...')
        logX = load_matrix(matrix_path + '.logx.npz')
        if rank == 0: print('Loaded logX.')
        fX = load_matrix(matrix_path + '.fx.npz')
        if rank == 0: print('Loaded fX.')

        type = 'sparse' if sparse else 'dense'
        if rank == 0: print(f'Using {type} cooccurence matrices during training.')
        if not sparse:
            logX = logX.todense()
            fX = fX.todense()
        return logX, fX

    def train(rank, args, model):
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

        logX, fX = load_data(rank)

        losses = []
        epoch = 0
        prev_loss = np.inf
        lr = args.lr
        t0 = time.time()
        for step in range(1, args.num_updates+1):

            # Sample a random batch.
            idx = np.random.randint(0, high=vocab_size, size=(args.batch_size,))
            indices = torch.LongTensor(idx).to(device)

            submat = np.ix_(idx, idx)  # used to select the submatrix
            if sparse:
                logx = logX[submat].todense()
                weights = fX[submat].todense()
            else:
                logx = logX[submat]
                weights = fX[submat]

            logx = torch.FloatTensor(logx).to(device)
            weights = torch.FloatTensor(weights).to(device)

            # Forward pass
            loss = model(indices, logx, weights)
            del indices, logx, weights  # free some memory

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Bookkeeping
            losses.append(loss.item())

            if step % args.print_every == 0 and rank == 0:
                ls = losses[-args.print_every:]
                avg_loss = sum(ls) / args.print_every
                print('| epoch {:4d} | step {:6d} | loss {:.4f} | pairs/sec {:.1f} | lr {:.1e}'.format(
                    epoch, step, avg_loss, args.print_every * args.batch_size / (time.time() - t0), lr))
                t0 = time.time()
                if args.use_schedule:
                    if avg_loss >= prev_loss:
                        lr /= 4.0
                        optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
                    prev_loss = avg_loss
            k, _ = divmod(step * args.batch_size, vocab_size)
            if k > epoch:
                epoch = k

    nprocs = mp.cpu_count()
    print(f'Hogwild training with {nprocs} processes...')
    processes = []
    for rank in range(nprocs):
        p = mp.Process(target=train, args=(rank, args, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Get the learned embeddings from the model.
    embeddings = model.embedding.weight.data.cpu().numpy()
    if args.gensim_format:
        out_path += f'.{args.emb_dim}d.gensim.txt'
    else:
        out_path += f'.{args.emb_dim}d.txt'
    print(f'Writing vectors to `{out_path}`...')
    write_vectors(embeddings, out_path, args.emb_dim, i2w, gensim=True)
