#!/bin/bash
for mask_rate in 0.125 0.25 0.375 0.5; do
  python -u run.py \
    `# Data` \
    --task_name imputation \
    --is_training 1 \
    --model_id ETTm2_mask_${mask_rate} \
    --data ETTm2 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --features M \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 1024 \
    --pred_len 0 \
    --mask_rate $mask_rate \
    `# Training` \
    --learning_rate 0.001 \
    `# Model` \
    --model XCTFormer \
    --e_layers 2 \
    --d_model 128 \
    --d_ff 256 \
    --n_heads 1 \
    `# XCTFormer` \
    --attn_dropout 0.5
done
