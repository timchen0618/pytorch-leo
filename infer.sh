#!/bin/bash
python3 main.py -test \
		-N 5 \
		-K 1 \
		-verbose \
		-embedding_dir ../embeddings/ \
		-load ./miniImageNet_1shot/toy-0511/model.pth
