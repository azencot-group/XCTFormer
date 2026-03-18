#!/bin/bash
for mask_rate in 0.125 0.25 0.375 0.5; do
  python -u run.py \
    `# Data` \
    --task_name imputation \
    --is_training 1 \
    --model_id Electricity_mask_${mask_rate} \
    --data custom \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --features M \
    --enc_in 321 \
    `# Experimental Setting` \
    --seq_len 1024 \
    --pred_len 0 \
    --mask_rate $mask_rate \
    `# Training` \
    --learning_rate 0.005 \
    `# Model` \
    --model XCTFormer \
    --e_layers 2 \
    --d_model 192 \
    --d_ff 384 \
    --n_heads 2 \
    --patch_len 64 \
    --stride 32 \
    `# XCTFormer` \
    --attn_dropout 0.7 \
    --include_decop \
    --k 128
done
