#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py --test --ckpts experiments/Seedformer_AE/PCN_models/default/ckpt-last.pth 
