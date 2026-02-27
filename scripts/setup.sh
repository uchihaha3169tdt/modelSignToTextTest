#!/bin/bash

echo "Setting up YouTube Sign Language Generation environment..."

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml
conda activate signsam

# Install additional dependencies
echo "Installing additional packages..."
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
pip install chumpy blobfile ffmpeg sentence_transformers
pip install numpy==1.23.5

# Download GloVe embeddings
echo "Downloading GloVe embeddings..."
bash prepare/download_glove.sh

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your data in dataset/YOUTUBE_SIGN/"
echo "2. Run: bash scripts/prepare_data.sh"
echo "3. Run: bash scripts/train_youtube_sign.sh"
