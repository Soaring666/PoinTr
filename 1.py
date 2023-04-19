import torch
import os
import numpy as np
import wandb
import open3d as o3d
import torch.nn as nn
import PIL
from torch.nn import functional as F
from easydict import EasyDict
from tqdm import tqdm
from utils import parser, dist_utils, misc
from utils.config import *
from my_datasets.my_PCN import PCN
from utils.misc import *
from tools import builder
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 8
mask_drop = 0.1
con_mask = torch.bernoulli(torch.zeros(batch_size)+mask_drop)
con_mask = torch.ones(batch_size) - con_mask
print(con_mask)




















'''
#track实验数据
run = wandb.init(project='Folding_AE')

# im = Image.open('experiments/FoldingNet_AE/PCN_models/save_img/epoch140/recon/8c3e8ef43fdacf68230ea05136ea3925.jpg')
# im = im.convert('RGB')
# im.save('example.jpg')
# im = Image.fromarray(im)
for ep in range(100):
    if ep % 20 == 0:
        gt_list = []
        recon_list = []
        for i in range(3):
            a_gt = torch.randn(200, 3)
            a_gt[:, 1] = 0
            a_recon = torch.randn(500, 3)
            a_recon[:, 0] = 0
            a_gt = a_gt.numpy()
            a_recon = a_recon.numpy()
            gt_list.append(a_gt)
            recon_list.append(a_recon)
        wandb.log({'gt': [wandb.Object3D(i) for i in gt_list],'recon': [wandb.Object3D(i) for i in recon_list], 'ep': ep})
'''


'''
#导入arg参数
args = parser.get_args()
args.distributed = False
config = get_config(args)
config.dataset.train.others.bs = config.total_bs
model = builder.model_builder(config.model).to(device)
# (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
#                                                         builder.dataset_builder(args, config.dataset.val)
'''

'''
#保存图片
a = torch.randn(100, 3)
a = a.numpy()
x, y, z = a.transpose(1, 0)
input_gt = misc.get_ptcloud_img(a)
im = Image.fromarray(input_gt)
im.show()
print(type(im))

image = wandb.Image(input_gt)
run = wandb.init(project='image')
wandb.log({'image': image})
'''

'''
#自建数据集
train_dataset = PCN(config.dataset.train._base_, config.dataset.train.others)
test_dataset = PCN(config.dataset.val._base_, config.dataset.val.others)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.dataset.train.others.bs,
                                                shuffle = True, 
                                                drop_last = True,
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                shuffle = False, 
                                                drop_last = False,
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
'''




