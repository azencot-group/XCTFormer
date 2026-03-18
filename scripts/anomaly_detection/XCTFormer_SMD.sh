#!/bin/bash
python -u run.py \
  `# Data` \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id SMD \
  --data SMD \
  --root_path ./datasets/SMD/ \
  --enc_in 38 \
  `# Experimental Setting` \
  --seq_len 100 \
  `# Training` \
  --batch_size 128 \
  --learning_rate 0.001 \
  `# Model` \
  --model XCTFormer \
  --e_layers 2 \
  --d_model 168 \
  --d_ff 336 \
  --n_heads 1 \
  `# XCTFormer` \
  --attn_dropout 0.3
