import os
from collections import Counter

import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, save, output_file
from bokeh.palettes import d3

import matplotlib.pyplot as plt

from util import load_vectors, load_vocabulary, load_matrix


def main(args):
    if args.tsne:
        tsne(args)
    elif args.matrices:
        plot_decomposition(args)
    else:
        exit('Specify plotting.')


def tsne(args):
    """Plots t-SNE."""
    num_words = 1000

    print(f'Reading vectors from `{args.vec_path}`...')
    embeddings, w2i, i2w = load_vectors(args.vec_path, gensim=args.gensim_format)
    vocab_path = os.path.join(args.vocab_dir, f'{args.name}.vocab')

    embeddings = embeddings[:num_words, :]
    most_common_words = [i2w[i] for i in range(num_words)]

    print(f'Loaded {embeddings.shape[0]} vectors.')
    print(f'Plotting t-SNE for {num_words} vectors.')

    # Make bokeh plot.
    emb_scatter(embeddings, list(most_common_words), model_name=args.name)


def emb_scatter(data, names, model_name, perplexity=30.0, k=20):
    """t-SNE plot of embeddings and coloring with K-means clustering.

    Uses t-SNE with given perplexity to reduce the dimension of the
    vectors in data to 2, plots these in a bokeh 2d scatter plot,
    and colors them with k colors using K-means clustering of the
    originial vectors. The colored dots are tagged with labels from
    the list names.

    Args:
        data (np.Array): the word embeddings shape [num_vectors, embedding_dim]
        names (list): num_vectors words same order as data
        perplexity (float): perplexity for t-SNE
        N (int): number of clusters to find by K-means
    """
    assert data.shape[0] == len(names)
    # Find clusters with kmeans.
    print('Finding clusters...')
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    klabels = kmeans.labels_

    # Get a tsne fit.
    print('Fitting t-SNE...')
    tsne = TSNE(n_components=2, perplexity=perplexity)
    emb_tsne = tsne.fit_transform(data)

    # Plot the t-SNE of the embeddings with bokeh.
    # source: https://github.com/oxford-cs-deepnlp-2017/practical-1
    fig = figure(tools='pan,wheel_zoom,reset,save',
               toolbar_location='above',
               title='T-SNE for most common words')

    # Set colormap as a list.
    colormap = d3['Category20'][k]
    colors = [colormap[i] for i in klabels]

    source = ColumnDataSource(
        data=dict(
            x1=emb_tsne[:,0],
            x2=emb_tsne[:,1],
            names=names,
            colors=colors))

    fig.scatter(x='x1', y='x2', size=8, source=source, color='colors')

    labels = LabelSet(x='x1', y='x2', text='names', y_offset=6,
                      text_font_size='8pt', text_color='#555555',
                      source=source, text_align='center')
    fig.add_layout(labels)

    output_path = os.path.join('plots', f'{model_name}.tsne.html')
    output_file(output_path)
    print(f'Saved plot in `{output_path}`.')
    save(fig)


def plot_decomposition(args):
    print(f'Reading vectors from `{args.vec_path}`...')
    embeddings, w2i, i2w = load_vectors(args.vec_path, gensim=args.gensim_format)

    matrix_path = os.path.join(args.matrix_dir, f'{args.name}')
    logX = load_matrix(matrix_path + '.logx.npz')
    fX = load_matrix(matrix_path + '.fx.npz')
    logX, fX = logX.todense(), fX.todense()

    plt.imshow(embeddings)
    plt.savefig(os.path.join('plots', 'emb.pdf'))
    plt.clf()

    plt.imshow(embeddings.T)
    plt.savefig(os.path.join('plots', 'emb.t.pdf'))
    plt.clf()

    plt.imshow(logX)
    plt.savefig(os.path.join('plots', 'logX.pdf'))
    plt.clf()

    plt.imshow(fX * logX)
    plt.savefig(os.path.join('plots', 'fX.logX.pdf'))
    plt.clf()

    plt.imshow(embeddings @ embeddings.T)
    plt.savefig(os.path.join('plots', 'logX_.pdf'))
    plt.clf()
