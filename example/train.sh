#!/bin/bash

# 配置可见GPU
all_gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -z "$1" ]; then
    GPUS=$all_gpu_ids
    echo "[LOG]: No visible GPUs specified, using all available GPUs: $GPUS"
else
    GPUS=$1
    echo "[LOG]: Visible GPUs: $GPUS"
fi

IFS=',' read -r -a array <<< "$GPUS"

nproc_per_node=${#array[@]}

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$nproc_per_node --master_port=12345 train.py