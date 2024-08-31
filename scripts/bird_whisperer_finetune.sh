#!/bin/bash

DATASET_ROOT=input_path_here
MODEL_NAME='whisper'
SAVE_MODEL_ROOT=input_path_here
TRAINING_MODE='fine-tuning'



python main.py --model_name $MODEL_NAME \
    --save_model_path  $SAVE_MODEL_ROOT\
    --dataset_root $DATASET_ROOT \
    --training_mode $TRAINING_MODE \
    --augmented_run \
    --spec_aug \
    --n_epochs 20 \
    --start_epoch 0 \
    --batch_size 16 \
    --num_workers 4 \
    --lr 0.01 \
    --seed 42 \
    --do_logging