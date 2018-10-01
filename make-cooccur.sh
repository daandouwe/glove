#!/usr/bin/env bash

CORPUS=data/fil8
NAME=text8
VOCAB_SIZE=10000
WINDOW_SIZE=15
CONTEXT=harmonic
STANFORD_COUNTING=true  # faster counting

set -x

# Construct vocabulary.
python scripts/construct-vocab.py $CORPUS \
    --max-size $VOCAB_SIZE --out vocab/$NAME.vocab

# Count cooccurrences.
if [[ $STANFORD_COUNTING ]]; then
    # Use the counting from Stanford's glove implementation. (Much faster!)
    scripts/stanford/cooccur \
        -vocab-file vocab/$NAME.vocab < $CORPUS > cooccur/cooccurrences.bin

    python scripts/cbytes2text.py vocab/$NAME.vocab cooccur/cooccurrences.bin \
        --out pairs/$NAME.pairs

    rm -r cooccur/cooccurences.bin
else
    # Use own python script. (Much slower!)
    python scripts/get-cooccur-pairs.py $CORPUS vocab/$NAME.vocab \
       --window-size $WINDOW_SIZE --context $CONTEXT --out pairs/$NAME.pairs
fi

# Construct sparse cooccurence matrices.
python scripts/construct-cooccur-matrix.py vocab/$NAME.vocab pairs/$NAME.pairs cooccur/$NAME
