# GloVe
An implementation of Global Vectors (GloVe) in PyTorch.

## Data
We can use `text8` and `text9`. To get the data, run:
```bash
mkdir data
./get-data.sh
```
To obtain the cooccurence counts and construct matrices, run:
```bash
mkdir vocab pairs cooccur
./cooccur.sh
```

## Usage
To train 100 dimensional vectors on the cooccurence matrices constructed above, run:
```bash
mkdir vec
./main.py train --name text8 --emb-dim 100 --out-dir vec
```

To plot (a number of) these vectors, use:
./main.py plot --vec-dir vec/text8.100d.txt

# Requirements
```bash
torch==0.4.1
numpy
bokeh    # For t-SNE plots
sklearn  # For t-SNE plots
```

## TODO
- [ ] Add vector evaluation tests.
- [ ] Why so slow on GPU?
- [ ] Hogwild training, for fun.
