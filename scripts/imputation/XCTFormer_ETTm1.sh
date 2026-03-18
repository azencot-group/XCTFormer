#!/bin/bash
for mask_rate in 0.125 0.25 0.375 0.5; do
  python -u run.py \
    `# Data` \
    --task_name imputation \
    --is_training 1 \
    --model_id ETTm1_mask_${mask_rate} \
    --data ETTm1 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
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
    --d_model 96 \
    --d_ff 192 \
    --n_heads 4 \
    `# XCTFormer` \
    --attn_dropout 0.1
done
