#!/bin/bash

echo "Downloading GloVe embeddings..."

mkdir -p glove
cd glove

if [ ! -f "glove.6B.300d.txt" ]; then
    echo "Downloading GloVe 6B..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.zip
    echo "GloVe downloaded successfully"
else
    echo "GloVe already exists"
fi

cd ..
