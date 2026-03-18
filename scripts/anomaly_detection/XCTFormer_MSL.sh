#!/bin/bash
python -u run.py \
  `# Data` \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id MSL \
  --data MSL \
  --root_path ./datasets/MSL/ \
  --enc_in 55 \
  `# Experimental Setting` \
  --seq_len 100 \
  `# Training` \
  --batch_size 128 \
  --learning_rate 0.01 \
  `# Model` \
  --model XCTFormer \
  --e_layers 2 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  `# XCTFormer` \
  --attn_dropout 0.7
