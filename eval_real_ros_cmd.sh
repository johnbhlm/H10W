#!/bin/bash


your_ckpt=$1

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
export PYTHONPATH=$PYTHONPATH:/home/diana/intern-vla_test/starVLA
PYTHONWARNINGS=ignore::UserWarning \
python ./examples/H10W/deploy_policy_real_ros_cmd_tongbu_img_detect_0403.py  \
    #--args.pretrained_path ${your_ckpt}
