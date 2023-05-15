#!/usr/bin/env bash

set -x

python main.py
python main.py --config './cfgs/PCN_models/Seedformer_posenc.yaml' 
python main_sh.py --config './cfgs/PCN_models/Seedformer_density.yaml' 
