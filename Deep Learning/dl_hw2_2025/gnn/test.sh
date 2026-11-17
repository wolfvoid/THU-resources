#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
python segmentation.py test \
--dataset fusiongallery \
--dataset_path ./brep_data \
--desc " " \
--accelerator gpu \
--devices 1 \
--conv_type 'Custom' \
--checkpoint $YOUR_EXPERIMENT_PATH/home/meibingyin/Projects/DeepLearning-2025/dl_hw2_2025/gnn/results/Custom/1204/133818_uv_net/best.ckpt
