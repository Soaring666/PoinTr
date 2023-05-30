#!/usr/bin/env bash

set -x

# CUDA_VISIBLE_DEVICES=1 python main.py --config './cfgs/PCN_models/Seedformer.yaml' 
# python main.py --config './cfgs/PCN_models/Seedformer_posenc.yaml' 
# python main_sh.py --config './cfgs/PCN_models/Seedformer_density.yaml' 
# CUDA_VISIBLE_DEVICES=1 python main_sh.py --config './cfgs/PCN_models/Seedformer_newup.yaml' 
# CUDA_VISIBLE_DEVICES=0 python main.py --config './cfgs/PCN_models/Seedformer_denoise.yaml' 
# CUDA_VISIBLE_DEVICES=0 python main_sh.py --config './cfgs/PCN_models/Seedformer_fusion.yaml' 

# CUDA_VISIBLE_DEVICES=0 python main.py --config './cfgs/PCN_models/SnowFlakeNet.yaml' 
# CUDA_VISIBLE_DEVICES=0 python main_sh.py --config './cfgs/PCN_models/Snow_fusion.yaml' 
# CUDA_VISIBLE_DEVICES=1 python main_sh.py --config './cfgs/PCN_models/Snow_fusion.yaml' 

# CUDA_VISIBLE_DEVICES=0 python main_sh.py --config './cfgs/PCN_models/Snow_fusion.yaml' 

# CUDA_VISIBLE_DEVICES=1 python main.py --config './cfgs/PCN_models/Snow_new.yaml' 
CUDA_VISIBLE_DEVICES=0 python main.py --config './cfgs/PCN_models/Snow_trick.yaml' 