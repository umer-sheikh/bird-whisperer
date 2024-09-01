#!/bin/bash

DATASET_ROOT=<PATH_OF_DATASET_FOLDER>
MODEL_NAME='whisper'
SAVE_MODEL_ROOT=<PATH_TO_SAVE_MODEL_WEIGHTS>
TRAINING_MODE='random-init'



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