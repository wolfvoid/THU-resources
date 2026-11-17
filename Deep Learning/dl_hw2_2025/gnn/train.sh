#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python segmentation.py train \
--dataset fusiongallery \
--dataset_path ./brep_data \
--max_epochs 50 \
--batch_size 64 \
--accelerator gpu \
--devices 1 \
--desc "uv_net" \
--graph_encoder_type "GNN" \
--seed 0 \
--conv_type 'Custom' \
--experiment_name Custom

