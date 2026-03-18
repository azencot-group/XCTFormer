#!/bin/bash
for mask_rate in 0.125 0.25 0.375 0.5; do
  python -u run.py \
    `# Data` \
    --task_name imputation \
    --is_training 1 \
    --model_id weather_mask_${mask_rate} \
    --data custom \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --features M \
    --enc_in 21 \
    `# Experimental Setting` \
    --seq_len 1024 \
    --pred_len 0 \
    --mask_rate $mask_rate \
    `# Training` \
    --learning_rate 0.001 \
    `# Model` \
    --model XCTFormer \
    --e_layers 3 \
    --d_model 192 \
    --d_ff 384 \
    --n_heads 1 \
    `# XCTFormer` \
    --attn_dropout 0.8
done
