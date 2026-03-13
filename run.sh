#!/bin/bash

# run
# CUDA_VISIBLE_DEVICES=0 python demo_viser.py \
#     --input data/long_scannet_s3/scene0000_00/color_1000 \
#     --config ckpts/LoGeR_star/original_config.yaml \
#     --model_name ckpts/LoGeR_star/latest.pt \
#     --window_size 64 \
#     --se3 \
#     --port 8080 \
#     --share

# save ply and traj
# python pt2ply.py --input results_pi3/scene0000_00_color_1000_0_-1_1.pt --image_dir data/long_scannet_s3/scene0000_00/color_1000 --stride 3 --conf_threshold 0.2 --subsample 8

# eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_scannet.sh \
  LoGeR_star \
  --num-processes 8 --port 29122 --window-size 64 --overlap-size 3 --se3 \
  --datasets 'scannet_s3_1000,scannet_s3_50,scannet_s3_90,scannet_s3_100,scannet_s3_150,scannet_s3_200,scannet_s3_300,scannet_s3_400,scannet_s3_500,scannet_s3_600,scannet_s3_700,scannet_s3_800,scannet_s3_900'