#!/bin/bash

policy_name=InternVLA
task_name=${1}
task_config=${2}
# ckpt_setting=${3:-"/home/maintenance/vla_ws/InternVLA-M1/results/Checkpoints/act_freezeqwen_h10w_joint_real_vr611_pretrained/checkpoints/steps_60000_pytorch_model.pt"}
seed=${4:-0}
gpu_id=${5:-0} # default is 0

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python deploy_policy_real_state.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} 