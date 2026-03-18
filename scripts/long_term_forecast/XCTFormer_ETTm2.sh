#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTm2_96_${pred_len} \
    --data ETTm2 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --features M \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    `# Training` \
    --learning_rate 0.005 \
    `# Model` \
    --model XCTFormer \
    --e_layers 1 \
    --d_model 224 \
    --d_ff 448 \
    --n_heads 1 \
    `# XCTFormer` \
    --attn_dropout 0.8
done
