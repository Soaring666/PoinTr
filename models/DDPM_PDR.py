import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn

''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''
from diffusion_utils import UNet, UNet_new, PointNet2CloudCondition
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL2_split, ChamferDistanceL1, ChamferDistanceL1_PM
import numpy as np
import wandb
from .build import MODELS


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 #average beta (n_T,)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t) #log函数使得后面的alpha乘法变成加法
    bar_alpha = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_bar_alpha = torch.sqrt(bar_alpha)
    one_over_sqrta = 1 / torch.sqrt(alpha_t)

    sqrt_one_minus_bar_alpha = torch.sqrt(1 - bar_alpha)
    oma_over_somba = (1 - alpha_t) / sqrt_one_minus_bar_alpha 

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "one_over_sqrta": one_over_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "bar_alpha": bar_alpha,  # \bar{\alpha_t}
        "sqrt_bar_alpha": sqrt_bar_alpha,  # \sqrt{\bar{\alpha_t}}
        "sqrt_omba": sqrt_one_minus_bar_alpha,  # \sqrt{1-\bar{\alpha_t}}
        "oma_over_somba": oma_over_somba,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

@MODELS.register_module()
class DDPM_PDR(nn.Module):
    def __init__(self, config, **kwargs):
        """
        unet params:
            T: time step
            ch: hidden dim
            condition_num: input condition num
            N: input
            ch_mult: channel multiplier
            attn: attention
            num_res_blocks: number of residual blocks
            dropout: conv dropout
        ddpm params:
            betas: betas
            n_T: n_T
            drop_prob: mask dropout
        """
        super(DDPM_PDR, self).__init__()
        self.nn_model = PointNet2CloudCondition()
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(config.betas[0], config.betas[1], config.n_T).items():
            self.register_buffer(k, v)

        self.CD = ChamferDistanceL1()
        self.n_T = config.n_T
        self.loss_mse = nn.MSELoss()

    def forward(self, x, condition_xyz, condition_feat):
        """
        this method is used in training, so samples t and noise randomly
        input:
            x: (B, 576, 64) ***** (B, 512, 3)
            condition: (B, 288)  残缺点云经过编码后的向量(以后可以加入点云标签) ***** seed_feat (B, 128, 256)
        """
        B, N, C = x.shape

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)  (B,)
        noise = torch.randn_like(x).to(x.device)  # eps ~ N(0, 1)

        x_t = (
            self.sqrt_bar_alpha[_ts, None, None] * x
            + self.sqrt_omba[_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        # con_mask = torch.bernoulli(torch.zeros(B)+self.drop_prob).to(x.device)

        # loss_mse = self.loss_mse(noise, self.nn_model(x_t, t, condition, con_mask))
        loss_mse = self.loss_mse(noise, self.nn_model(x_t, _ts, condition_xyz, condition_feat))
        # loss_cd = self.CD(noise, self.nn_model(x_t, t, condition, con_mask))
        loss_all = loss_mse
        # MSE between added noise, and our predicted noise
        return loss_all

    def sample(self, shape, condition_xyz, conditio_feat):
        '''
        shape: B, 512, 3   (B, N, C)表示生成张量的形状
        condition: B, 128, 256

        output:
            x_i: (B, 512, 3)
        '''
        B, N, C = shape
        use_shape = [B, N, C]   #转换为输入到nn_module的形状
        x_i = torch.randn(use_shape).to(condition_xyz.device)  # x_T ~ N(0, 1), sample initial noise

        for i in range(self.n_T - 1, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i], dtype=int).to(condition_xyz.device)
            t_is = t_is.repeat(B,)
            z = torch.randn(use_shape).to(condition_xyz.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is, condition_xyz, conditio_feat)
            x_i = (
                self.one_over_sqrta[i] * (x_i - eps * self.oma_over_somba[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i



    def sample_w(self, condition, shape, guide_w = 0.0):
        '''
        shape: B, 512, 3   (B, N, C)表示生成张量的形状
        condition: B, 128, 256
        '''
        B, N, C = shape
        use_shape = [B, C, N]   #转换为输入到nn_module的形状

        x_i = torch.randn(use_shape).to(condition.device)  # x_T ~ N(0, 1), sample initial noise
        c_i = condition.to(condition.device) # context for us just cycles throught the mnist labels

        # don't drop context at test time
        con_mask = torch.zeros(B)     #取0是掩码
        con_mask = torch.ones(B) - con_mask
        con_mask = con_mask.to(condition.device)

        # double the batch
        c_i = c_i.repeat(2, 1 ,1)
        con_mask = con_mask.repeat(2)
        con_mask[B:] = 0. # makes second half of batch condition free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T - 1, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i], dtype=int).to(condition.device)
            t_is = t_is.repeat(B,)

            # double batch
            x_i = x_i.repeat(2,1,1)
            t_is = t_is.repeat(2)

            z = torch.randn(use_shape).to(condition.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is, c_i, con_mask)
            eps1 = eps[:B]
            eps2 = eps[B:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:B]
            x_i = (
                self.one_over_sqrta[i] * (x_i - eps * self.oma_over_somba[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store



# def train_test():
#     device = 'cpu'

#     # hardcoding these here
#     batch_size = 1
#     n_T = 1000 # 500
#     n_feat = 128 # 128 ok, 256 better (but slower)
#     lrate = 1e-4
#     ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    

#     with torch.no_grad():
#         ddpm = DDPM(T=1000, ch=128, condition_num=256, N=512,  
#                     ch_mult=[1, 2, 2, 2], attn=[2], 
#                     num_res_blocks=2, dropout=0.1, 
#                     betas=(1e-4, 0.02), n_T=n_T, drop_prob=0.1)
#         ddpm = ddpm.to(device)
#         x = torch.randn(batch_size, 512, 3).to(device)        #B, N, 3
#         shape = x.shape
#         condition = torch.randn(batch_size, 128, 256).to(device)
#         loss = ddpm(x, condition)
#         print(loss)
#         x_i, x_i_store_list = ddpm.sample(condition, shape, guide_w = 0.0)
#         print('x_i.shape:', x_i.shape)
#         print('x_i_store_list:', len(x_i_store_list))
    

# if __name__ == "__main__":
#     train_test()





