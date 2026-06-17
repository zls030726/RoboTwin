#!/bin/bash
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# 使用下面的指令查看 nvidia_icd.json 的地址
# find /usr/share /etc -name "*nvidia*icd*.json" 2>/dev/null 
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

policy_name=openvla-oft
task_name=${1}
task_config=${2}
checkpoint_path=${3}
checkpint_num=${4}
seed=${5}
gpu_id=${6}
unnorm_key=${7}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES=${6} python script/eval_policy.py --config policy/Your_Policy/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --checkpoint_path ${checkpoint_path} \
    --ckpt_setting ${checkpint_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --unnorm_key ${unnorm_key} \
    --stage_2 True \
    > /data7/Users/zls/VLA/3DCogVLA++/experiments/logs/robotwin_bbh_${checkpint_num}_$(date +%Y%m%d_%H%M%S).log 2>&1

# example usage 
# bash policy/Your_Policy/eval.sh beat_block_hammer demo_randomized /data5/Users/zls/model/openvla/cogvla-robotwin2/hard/aloha_beat_block_hammer/openvla-libero-all-base+aloha_beat_block_hammer+b8+lr-0.0002+lora-r32+dropout-0.0--image_aug--stage1--final_chkpt+aloha_beat_block_hammer+b8+lr-0.0002+lora-r32+dropout-0.0--image_aug--stage2--25000_chkpt 25000 2 7 aloha_beat_block_hammer