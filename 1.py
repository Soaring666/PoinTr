import torch
# import os
# import numpy as np
# import wandb
# import open3d as o3d
# import torch.nn as nn
# import PIL
# from torch.nn import functional as F
# from easydict import EasyDict
# from tqdm import tqdm
# from utils import parser, dist_utils, misc
# from utils.config import *
# from my_datasets.my_PCN import PCN
# from utils.misc import *
# from tools import builder
# from PIL import Image
import os
import torch
import torch.multiprocessing as mp
import argparse
import wandb
from torch import distributed as dist


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

shape = [3, 4, 5]
a = torch.randn(shape)
B ,N, C = a.shape

print(B, N, C)



























'''
##########distributed initialization##########
#distributed initialization
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

if args.local_rank == 0:
    run = wandb.init(project='test_dist') 

# rank = int(os.environ['RANK'])
# num_gpus = torch.cuda.device_count()
# torch.cuda.set_device(rank % num_gpus)
# dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=2)
# # a = torch.randn(3,4).cuda()
# print(a.device)
# print("rank: ", dist.get_rank())
device = torch.device('cuda', args.local_rank)
print("local_rank:", args.local_rank)
# b = torch.randn(3,4).cuda()
# print('b:', b.device)
# a = torch.randn(3,4).to(device)
# print(a.device)
##################测试多卡跑代码在wandb上track的情况#########
a = [1, 5, 10, 3]
if args.local_rank == 0:
    for i in range(len(a)):
        run.log({'a': a[i]})

                # if args.distributed:
                #     if args.local_rank == 0:
                # else:
'''


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




