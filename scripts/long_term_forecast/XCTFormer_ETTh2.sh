#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh2_96_${pred_len} \
    --data ETTh2 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    `# Training` \
    --learning_rate 0.01 \
    `# Model` \
    --model XCTFormer \
    --e_layers 3 \
    --d_model 30 \
    --d_ff 60 \
    --n_heads 1 \
    --fc_dropout 0.2 \
    `# XCTFormer` \
    --attn_dropout 0.8
done
