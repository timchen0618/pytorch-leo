#!/bin/bash
python3 main.py -train -num_step 200000 -K 1 -verbose \
		-outer_lr 0.0001 \
		-l2_penalty_weight 1e-4 \
		-orthogonality_penalty_weight 303 \
		-kl_weight 0.01 \
		-encoder_penalty_weight 1e-9 \
		-model_dir train_model_val200_leo
