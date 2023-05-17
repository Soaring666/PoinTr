#!/usr/bin/env bash

set -x

# python main.py --config './cfgs/PCN_models/Seedformer.yaml' 
# python main.py --config './cfgs/PCN_models/Seedformer_posenc.yaml' 
# python main_sh.py --config './cfgs/PCN_models/Seedformer_density.yaml' 
# python main.py --config './cfgs/PCN_models/Seedformer_newup.yaml' 
python main.py --config './cfgs/PCN_models/Seedformer_denoise.yaml' 