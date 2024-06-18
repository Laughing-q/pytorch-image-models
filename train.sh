#/bin/bash

torchrun --standalone --nnodes=1 --nproc-per-node=4  train.py \
--pretrained \
--model nextvit_large \
--data-dir /sdata/datasets/accident3 \
--batch-size 96 \
--validation-batch-size 96 \
--epochs 15 \
--img-size 224 \
--num-classes 2 \
--amp \
--model-ema \
--scale 0.6 1.0 \
--checkpoint-hist 20 \
# --mixup 1.0 \
