#!/bin/bash
python3 main.py -test \
		-N 5 \
		-K 1 \
		-verbose \
		-load ./miniImageNet_1shot/ort0188_kl001_drop03_l285e_6_ence6_test_val_swap_deepmind/100k_1.2674250.5928_model.pth
