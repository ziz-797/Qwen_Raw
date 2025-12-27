#!/bin/bash
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

dataset="mimic_cxr"
annotation="/mnt/sda/yingshu/qwen3/dataset/annotation_with_clinic_info.json"
base_dir="/mnt/sdb/datasets/mimic_original/2.0.0/files"

version="v2_deep"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 2 \
    --val_batch_size 2 \
    --freeze_vm False \
    --freeze_llm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 200 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --learning_rate 1e-5 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 1 \
    --limit_val_batches 1 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt
