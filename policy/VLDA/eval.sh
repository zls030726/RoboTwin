#!/bin/bash
set -e

policy_name=VLDA
task_name=${1:-beat_block_hammer}
gpu_id=${2:-0}
checkpoint_id=${3:-80000}
task_config=${4:-demo_randomized}
seed=${5:-42}
train_config_name=${6:-vlda_robotwin}
exp_name=${7:-robotwin_50}
action_chunk_steps=${8:-50}
eval_video_log=${9:-False}
ckpt_setting=${checkpoint_id}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export ROBOTWIN_FFMPEG=${ROBOTWIN_FFMPEG:-/usr/bin/ffmpeg}
PYTHON_BIN=${LIBERO_PY:-/workspace/VLA/VLDA/RoboTwin/.venv/bin/python}
echo -e "[33mgpu id (to use): ${gpu_id}[0m"
echo -e "[33mffmpeg: ${ROBOTWIN_FFMPEG}[0m"
echo -e "[33mpython: ${PYTHON_BIN}[0m"

LOG_DIR="/workspace/VLA/VLDA/logs/eval/robotwin/${task_name}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${ckpt_setting}_$(date +%Y%m%d_%H%M%S).log"

PYTHONWARNINGS=ignore::UserWarning "${PYTHON_BIN}" script/eval_policy.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --train_config_name ${train_config_name} \
    --exp_name ${exp_name} \
    --checkpoint_id ${checkpoint_id} \
    --action_chunk_steps ${action_chunk_steps} \
    --eval_video_log ${eval_video_log} \
    > "${LOG_FILE}" 2>&1