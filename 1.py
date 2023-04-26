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
from torch import distributed as dist
import torch.multiprocessing as mp


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

mp.set_start_method('spawn')

rank = int(os.environ['RANK'])
print('environ: ', rank)
num_gpus = torch.cuda.device_count()
torch.cuda.set_device(rank % num_gpus)
dist.init_process_group(backend='nccl')

# print('environ:1 ', rank)
# print('environ:2 ', rank)

# rank = dist.get_rank()
# world_size = dist.get_world_size()
# print(rank, world_size)











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




