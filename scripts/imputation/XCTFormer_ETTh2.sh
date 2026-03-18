#!/bin/bash
for mask_rate in 0.125 0.25 0.375 0.5; do
  python -u run.py \
    `# Data` \
    --task_name imputation \
    --is_training 1 \
    --model_id ETTh2_mask_${mask_rate} \
    --data ETTh2 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 1024 \
    --pred_len 0 \
    --mask_rate $mask_rate \
    `# Training` \
    --learning_rate 0.005 \
    `# Model` \
    --model XCTFormer \
    --e_layers 3 \
    --d_model 160 \
    --d_ff 320 \
    --n_heads 1 \
    --patch_len 64 \
    --stride 32 \
    `# XCTFormer` \
    --attn_dropout 0.3
done
