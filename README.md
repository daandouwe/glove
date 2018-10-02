# GloVe embeddings in PyTorch
A PyTorch implementation of [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).

## Data
We use [text8](http://mattmahoney.net/dc/textdata.html). To get the data, run:
```bash
cd data
./get-data.sh
```
To obtain the cooccurrence-counts and construct the sparse matrices, run:
```bash
cd data
mkdir vocab pairs cooccur
./get-cooccurrences.sh
```

## Usage
To train 100 dimensional vectors on the cooccurence matrices constructed above, run:
```bash
mkdir vec
./main.py train --name text8 --emb-dim 100 --out-dir vec
```
The vectors are saved in `vec/text8.100d.txt`.

To plot (a number of) these vectors, use:
```bash
./main.py plot --vec-dir vec/text8.100d.txt
```
The plots are saved as html in `plots`. An example can be seen [here](https://github.com/daandouwe/glove/blob/master/plots). (Github does not render html files. To render, download and open, or use [this link](http://htmlpreview.github.com/?https://raw.githubusercontent.com/daandouwe/glove/master/plots/text8.10k.50d.tsne.html).)

# Requirements
```bash
torch==0.4.1
numpy
tqdm
bokeh     # for t-sne plot
sklearn   # for t-sne plot
```

## TODO
- [ ] Add vector evaluation tests.
- [ ] Why so slow on GPU?
- [X] Hogwild training, for fun.
