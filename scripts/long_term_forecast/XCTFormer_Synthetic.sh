#!/bin/bash
for pred_len in 96 192 336 720; do
  python -u run.py \
    `# Data` \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id Synthetic_96_${pred_len} \
    --data Synthetic \
    --root_path ./datasets/Synthetic/ \
    --data_path synthetic.csv \
    --features MS \
    --target target \
    --freq h \
    --enc_in 7 \
    `# Experimental Setting` \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    `# Training` \
    --learning_rate 0.005 \
    `# Model` \
    --model XCTFormer \
    --e_layers 2 \
    --d_model 32 \
    --d_ff 64 \
    --n_heads 4 \
    `# XCTFormer` \
    --attn_dropout 0.8
done
