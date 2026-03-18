#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id traffic_96_${pred_len} \
    --data custom \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --features M \
    --enc_in 862 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    `# Training` \
    --batch_size 8 \
    --learning_rate 0.001 \
    `# Model` \
    --model XCTFormer \
    --e_layers 3 \
    --d_model 248 \
    --d_ff 496 \
    --n_heads 4 \
    `# XCTFormer` \
    --include_decop \
    --k 192 \
    --attn_dropout 0.6
done
