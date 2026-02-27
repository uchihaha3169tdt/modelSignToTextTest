#!/bin/bash

echo "Training YouTube Sign Language Generation Model..."

python -m train.train_mdm \
    --arch sam_stunet \
    --lr 1e-4 \
    --overwrite \
    --save_interval 1000 \
    --num_steps 400000 \
    --dataset youtube_sign \
    --save_dir ./save/youtube_sign_model \
    --batch_size 64 \
    --diffusion_steps 1000 \
    --device 0

echo "Training complete!"
