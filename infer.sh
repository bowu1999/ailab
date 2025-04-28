#!/bin/bash

CONFIG_FILE="/mnt/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/006-Character_recognition/版本2-使用类型描述编码为数据标签进行训练/ailab-train-configs/example.py"

CHECKPOINT_FILE="/mnt/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/006-Character_recognition/版本2-使用类型描述编码为数据标签进行训练/ailab-workspace/epoch_172.pth"

INPUT_FILE=""

OUTPUT_FILE=""


# 配置可见GPU
all_gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -z "$1" ]; then
    GPUS=$all_gpu_ids
    echo "[LOG]: No visible GPUs specified, using all available GPUs: $GPUS"
else
    GPUS=$1
    echo "[LOG]: Visible GPUs: $GPUS"
fi

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=12345 tools/infer.py \
    --config $CONFIG_FILE \
    --ckpt $CHECKPOINT_FILE \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE
