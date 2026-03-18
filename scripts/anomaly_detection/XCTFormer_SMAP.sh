#!/bin/bash
python -u run.py \
  `# Data` \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id SMAP \
  --data SMAP \
  --root_path ./datasets/SMAP/ \
  --enc_in 25 \
  `# Experimental Setting` \
  --seq_len 100 \
  `# Training` \
  --batch_size 128 \
  --learning_rate 0.005 \
  `# Model` \
  --model XCTFormer \
  --e_layers 3 \
  --d_model 256 \
  --d_ff 128 \
  --n_heads 1 \
  `# XCTFormer` \
  --attn_dropout 0.3
