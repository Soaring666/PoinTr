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
from diffusion_utils import UNet, UNet_new
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
class DDPM_new(nn.Module):
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
        super(DDPM_new, self).__init__()
        self.nn_model = UNet_new(config.T, config.ch, config.condition_num, config.N,
                                 config.posdim, config.alpha, config.beta,
                                 config.ch_mult, config.attn, config.crossattn, 
                                 config.num_res_blocks, config.dropout)
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(config.betas[0], config.betas[1], config.n_T).items():
            self.register_buffer(k, v)

        self.CD = ChamferDistanceL1()
        self.n_T = config.n_T
        self.drop_prob = config.drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, condition):
        """
        this method is used in training, so samples t and noise randomly
        input:
            x: (B, 576, 64) ***** (B, 512, 3)
            condition: (B, 288)  残缺点云经过编码后的向量(以后可以加入点云标签) ***** seed_feat (B, 128, 256)
        """
        x = x.transpose(-2, -1)     #(B, 3, 512)
        B, C, N = x.shape

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)  (B,)
        noise = torch.randn_like(x).to(x.device)  # eps ~ N(0, 1)

        x_t = (
            self.sqrt_bar_alpha[_ts, None, None] * x
            + self.sqrt_omba[_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        con_mask = torch.bernoulli(torch.zeros(B)+self.drop_prob).to(x.device)

        loss_mse = self.loss_mse(noise, self.nn_model(x_t, _ts, condition, con_mask))
        # loss_cd = self.CD(noise, self.nn_model(x_t, t, condition, con_mask))
        loss_all = loss_mse
        # MSE between added noise, and our predicted noise
        return loss_all

    def sample(self, condition, shape, guide_w = 0.0):
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


# def train_mnist():
#     device = 'cuda'

#     # hardcoding these here
#     n_epoch = 20
#     batch_size = 4
#     n_T = 1000 # 500
#     n_classes = 10
#     n_feat = 128 # 128 ok, 256 better (but slower)
#     lrate = 1e-4
#     save_model = False
#     save_dir = './data/diffusion_outputs10/'
#     ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    

#     unet = UNet(
#         T=1000, ch=576, ch_mult=[1, 2, 2, 2], attn=[2],
#         num_res_blocks=2, dropout=0.1).to(device)
#     ddpm = DDPM(nn_model=unet, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
#     ddpm = ddpm.to(device)
#     x = torch.randn(batch_size, 576, 64).to(device)        #B, C, N  N表示encoder之后的中心点数量，C表示中心点的特征维度
#     condition = torch.randn(batch_size, 2304).to(device)
#     loss = ddpm(x, condition)
#     print(loss)

#     # optionally load a model
#     # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))


#     optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    # for ep in range(n_epoch):
    #     print(f'epoch {ep}')
    #     ddpm.train()

    #     # linear lrate decay
    #     optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

    #     pbar = tqdm(dataloader)
    #     loss_ema = None
    #     wandb.init(project="conditional mnist diffusion")
    #     for x, c in pbar:
    #         optim.zero_grad()
    #         x = x.to(device) #(256, 1, 28, 28)
    #         c = c.to(device)
    #         loss = ddpm(x, c)
    #         loss.backward()
    #         if loss_ema is None:
    #             loss_ema = loss.item()
    #         else:
    #             loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
    #         wandb.log({'loss': loss_ema, 'epoch': ep})
    #         pbar.set_description(f"loss: {loss_ema:.4f}")
    #         optim.step()
        
    #     # for eval, save an image of currently generated samples (top rows)
    #     # followed by real images (bottom rows)
    #     ddpm.eval()
    #     with torch.no_grad():
    #         n_sample = 4*n_classes
    #         for w_i, w in enumerate(ws_test):
    #             x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

    #             # append some real images at bottom, order by class also
    #             x_real = torch.Tensor(x_gen.shape).to(device)
    #             for k in range(n_classes):
    #                 for j in range(int(n_sample/n_classes)):
    #                     try: 
    #                         idx = torch.squeeze((c == k).nonzero())[j]
    #                     except:
    #                         idx = 0
    #                     x_real[k+(j*n_classes)] = x[idx]

    #             x_all = torch.cat([x_gen, x_real])
    #             grid = make_grid(x_all*-1 + 1, nrow=10)
    #             save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
    #             print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

    #             if ep%5==0 or ep == int(n_epoch-1):
    #                 # create gif of images evolving over time, based on x_gen_store
    #                 fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
    #                 def animate_diff(i, x_gen_store):
    #                     print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
    #                     plots = []
    #                     for row in range(int(n_sample/n_classes)):
    #                         for col in range(n_classes):
    #                             axs[row, col].clear()
    #                             axs[row, col].set_xticks([])
    #                             axs[row, col].set_yticks([])
    #                             # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
    #                             plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
    #                     return plots
    #                 ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
    #                 ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
    #                 print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
    #     # optionally save model
    #     if save_model and ep == int(n_epoch-1):
    #         torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
    #         print('saved model at ' + save_dir + f"model_{ep}.pth")

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





