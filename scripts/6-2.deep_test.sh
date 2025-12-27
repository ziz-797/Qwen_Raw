#!/bin/bash

dataset="mimic_cxr"
annotation="/mnt/sda/yingshu/qwen3/dataset/annotation_with_clinic_info.json"
base_dir="/mnt/sdb/datasets/mimic_original/2.0.0/files"
delta_file="/mnt/sda/shaoyang/model/R2GenGPT/R2GenGPT/save/mimic_cxr/v2_deep/checkpoints/checkpoint_epoch0_step67696_bleu0.147731.pth"

version="v3_deep"
savepath="./save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --max_length 120 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --freeze_llm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 12 \
    --devices 2 \
    2>&1 |tee -a ${savepath}/log.txt
