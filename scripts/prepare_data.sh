#!/bin/bash

echo "Preparing YouTube Sign dataset..."

DATASET_DIR="./dataset/YOUTUBE_SIGN"

# Check if data exists
if [ ! -d "$DATASET_DIR/new_joints" ]; then
    echo "Error: $DATASET_DIR/new_joints not found!"
    echo "Please place your extracted data in $DATASET_DIR/"
    exit 1
fi

if [ ! -d "$DATASET_DIR/texts" ]; then
    echo "Error: $DATASET_DIR/texts not found!"
    exit 1
fi

# Check split files
for split in train val test all; do
    if [ ! -f "$DATASET_DIR/${split}.txt" ]; then
        echo "Error: $DATASET_DIR/${split}.txt not found!"
        exit 1
    fi
done

echo "Data files found"

# Calculate statistics
echo "Calculating Mean and Std..."
python prepare/calculate_stats.py --dataset YOUTUBE_SIGN

echo "Data preparation complete!"
