#!/bin/bash



# 切换代码执行路径
CONFIG_FILE="/mnt/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/006-Character_recognition/版本2-使用类型描述编码为数据标签进行训练/ailab-train-configs/example.py"

# 配置可见GPU
all_gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -z "$1" ]; then
    GPUS=$all_gpu_ids
    echo "[LOG]: No visible GPUs specified, using all available GPUs: $GPUS"
else
    GPUS=$1
    echo "[LOG]: Visible GPUs: $GPUS"
fi

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=12345 tools/train.py --config $CONFIG_FILE