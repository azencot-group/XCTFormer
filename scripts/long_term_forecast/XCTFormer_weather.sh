#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id weather_96_${pred_len} \
    --data custom \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --features M \
    --enc_in 21 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    `# Training` \
    --learning_rate 0.0005 \
    `# Model` \
    --model XCTFormer \
    --e_layers 3 \
    --d_model 248 \
    --d_ff 496 \
    --n_heads 2 \
    `# XCTFormer` \
    --attn_dropout 0.8
done
