#!/bin/bash

# your_ckpt=results/Checkpoints/internvla_full_h10w_joint_real_vr_left_1500_pre_2.4/checkpoints/steps_60000_pytorch_model.pt
your_ckpt=$1  # $1 是第一个参数
base_port=10093
# export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

export DEBUG=0

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16