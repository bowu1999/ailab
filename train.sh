#!/bin/bash



# 切换代码执行路径
CONFIG_FILE="/mnt/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/006-Character_recognition/版本2-使用类型描述编码为数据标签进行训练/ailab-train-configs/example.yaml"

torchrun --nproc_per_node=4 --master_port=12345 tools/train.py --config $CONFIG_FILE