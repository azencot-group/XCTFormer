#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_96_${pred_len} \
    --data ETTh1 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    `# Training` \
    --learning_rate 0.001 \
    `# Model` \
    --model XCTFormer \
    --e_layers 1 \
    --d_model 8 \
    --d_ff 16 \
    --n_heads 1 \
    --dropout 0.2 \
    --fc_dropout 0.3 \
    `# XCTFormer` \
    --attn_dropout 0.6
done
