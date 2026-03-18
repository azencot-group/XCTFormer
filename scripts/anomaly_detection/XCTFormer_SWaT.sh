#!/bin/bash
python -u run.py \
  `# Data` \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id SWaT \
  --data SWaT \
  --root_path ./datasets/SWaT/ \
  --enc_in 51 \
  `# Experimental Setting` \
  --seq_len 100 \
  `# Training` \
  --batch_size 128 \
  --learning_rate 0.0005 \
  `# Model` \
  --model XCTFormer \
  --e_layers 1 \
  --d_model 216 \
  --d_ff 432 \
  --n_heads 2 \
  `# XCTFormer` \
  --attn_dropout 0.4
