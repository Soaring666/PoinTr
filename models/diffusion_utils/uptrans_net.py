import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from numpy.random import randn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class_emb = nn.Embedding(10, 128)

class ConditionEmbedding(nn.Module):
    def __init__(self, condition_num, input_num):
        super().__init__()
        # self.conditionembedding = nn.Conv1d(dim, dim, 1, 1, 0)
        self.layer = nn.Sequential(
            nn.Linear(condition_num, input_num),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=False)
        )
    #     self.initialize()

    # def initialize(self):
    #     for module in self.modules():
    #         if isinstance(module, (nn.Conv1d, nn.Linear)):
    #             init.xavier_uniform_(module.weight)
    #             init.zeros_(module.bias)  

    def forward(self, x):
        '''
        input:
            x: condition information (B, 128, condition_num) 
        retutn: 
            x: condition embedding (B, 128, input_num)
        '''
        x = self.layer(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)     #使用stack可以交替进行sin和cos编码
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.ReLU(inplace=False),
            nn.Linear(dim, dim),
        )
    #     self.initialize()

    # def initialize(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             init.xavier_uniform_(module.weight)
    #             init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

#use trigonometric functions to embedding coordinates
class PosEncode(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.out_dim = out_dim  #the out featdim, 6的倍数
        self.alpha, self.beta = alpha, beta     #default alpha=100, beta=1000
   
        
    def forward(self, xyz):
        '''
        xyz: coordinates of point cloud (B, 3, N)
        x: feature of point cloud (B, C, N)
        output: feature (B, out_dim, N)
        '''
        B, _, N = xyz.shape
        feat_dim = self.out_dim // 6    

        feat_range = torch.arange(feat_dim).float().to(xyz.device)     
        dim_embed = torch.pow(self.beta, feat_range / feat_dim)
        div_embed = torch.div(self.alpha * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 3, 2).contiguous()
        position_embed = position_embed.view(B, self.out_dim, N)

        return position_embed

class Head_pos(nn.Module):
    '''
    use cross attention to embedding the input 512x3 point cloud with seed-feat
    input:
        posdim: the dim of pos-encoding
        alpha: default=100
        beta: default=1000
        ch: input channle
    output: embedded point cloud (B, 128, 512)
    '''
    def __init__(self, posdim, alpha, beta, ch):
        super().__init__()
        self.pos_en = PosEncode(posdim, alpha, beta)
        self.layer = nn.Sequential(
            nn.Conv1d(posdim, ch, 3, 1, 1),
            nn.GroupNorm(32, ch),
            nn.ReLU(inplace=False),
        )       
        self.initialize()

    def initialize(self):
        for module in self.layer.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)  

    def forward(self, x):
        x = self.layer(self.pos_en(x))
        
        return x

class Head(nn.Module):
    '''
    only use conv1d to trans input
    '''
    def __init__(self, ch):
        super().__init__()
        self.layer = nn.Conv1d(3, ch, 3, 1, 1)
    #     self.initialize()

    # def initialize(self):
    #     for module in self.modules():
    #         if isinstance(module, (nn.Conv1d, nn.Linear)):
    #             init.xavier_uniform_(module.weight)
    #             init.zeros_(module.bias)  

    def forward(self, x):
        x = self.layer(x)
        return x

class CrossAttn(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.q_map = nn.Conv1d(dim, out_dim, 1, 1, 0)
        self.k_map = nn.Conv1d(dim, out_dim, 1, 1, 0)
        self.v_map = nn.Conv1d(dim, out_dim, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv1d(out_dim, out_dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)
    #     self.initialize()

    # def initialize(self):
    #     for module in [self.q_map, self.k_map, self.v_map, self.proj]:
    #         init.xavier_uniform_(module.weight)
    #         init.zeros_(module.bias)
        
    def forward(self, q, v):
        '''
        q: pos_embedding (B, dim, N)
        v: seed_feat (B, dim, N)
        output: head_feat (B, dim, N)
        '''
        #q,v除了B形状不一样
        B, C, N = q.shape
        C = self.out_dim
        k = v

        q = self.q_map(q).view(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 3, 1)  #(B, 8, N, C/8)
        k = self.k_map(k).view(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 3, 1)
        v = self.v_map(v).view(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 3, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale       #(B, 8, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-2, -1).reshape(B, C, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x        

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c_main = nn.Conv1d(128, 128, 3, stride=2, padding=1)
    #     self.initialize()

    # def initialize(self):
    #     init.xavier_uniform_(self.main.weight)
    #     init.zeros_(self.main.bias)
    #     init.xavier_uniform_(self.c_main.weight)
    #     init.zeros_(self.c_main.bias)

    def forward(self, x, temb, condition, con_mask):
        x = self.main(x)
        condition = self.c_main(condition)
        return x, condition


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.c_main = nn.Conv1d(128, 128, 3, stride=1, padding=1)
    #     self.initialize()

    # def initialize(self):
    #     init.xavier_uniform_(self.main.weight)
    #     init.xavier_uniform_(self.c_main.weight)
    #     init.zeros_(self.main.bias)
    #     init.zeros_(self.c_main.bias)

    def forward(self, x, temb, condition, con_mask):
        _, _, N = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)

        condition = F.interpolate(
            condition, scale_factor=2, mode='nearest')
        condition = self.c_main(condition)
        return x, condition


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv1d(in_ch, in_ch, 1, stride=1, padding=0)
    #     self.initialize()

    # def initialize(self):
    #     for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
    #         init.xavier_uniform_(module.weight)
    #         init.zeros_(module.bias)

    def forward(self, x):
        B, C, N = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, N, N]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, N, C]
        h = h.transpose(-2, -1)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, crossattn=False):
        #方便起见，先设置con_dim=tdim
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=False),
        )
        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, out_ch),
            nn.ReLU(inplace=False),
        )
        self.cond_proj = nn.Sequential(
            nn.Conv1d(tdim//4, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=False),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.crossattn = crossattn
        if self.crossattn:
            self.crossattn = CrossAttn(out_ch, out_ch)
    #     self.initialize()

    # def initialize(self):
    #     for module in self.modules():
    #         if isinstance(module, (nn.Conv1d, nn.Linear)):
    #             init.xavier_uniform_(module.weight)
    #             init.zeros_(module.bias)

    def forward(self, x, temb, condition, con_mask):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None]
        h = h + self.cond_proj(condition*con_mask)
        h = self.block2(h)
        h = h + self.shortcut(x)

        h = self.attn(h)
        if self.crossattn:
            h = self.crossattn(h, self.cond_proj(condition*con_mask))
        
        #return的condition不变，只有在下采样和上采样的时候才会和input的形状一起变化
        return h, condition


class UNet_new(nn.Module):
    def __init__(self, T, ch, condition_num, N,
                 posdim, alpha, beta, 
                 ch_mult, attn, crossattn, num_res_blocks, dropout):
        '''
        T: time steps
        ch: input hidden channel
        condition_num: condition dim(-1), represent as the point number of seed feat
        N: the point number of input (i.e. (B, 3, 512) so N=512)
        posdim: position embedding dim
        alpha: the parameter of the position embedding, default=100
        beta: the parameter of the position embedding, default=1000

        ch_mult: channel multiplier
        attn(list): which layer to  use self-attn
        crossattn(list): which layer to use cross-attn
        num_res_blocks: number of residual blocks
        dropout: dropout rate
        '''
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.condition_embedding = ConditionEmbedding(condition_num, N)

        self.head = Head(ch)
        # self.headcrossattn = CrossAttn(ch, ch)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), crossattn=(i in crossattn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, crossattn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False, crossattn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), crossattn=(i in crossattn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Conv1d(now_ch, 3, 3, stride=1, padding=1)
        
    #     self.initialize()

    # def initialize(self):
    #     init.xavier_uniform_(self.tail.weight)
    #     init.zeros_(self.tail.bias)

    def forward(self, x, t, condition, con_mask):
        # Timestep embedding
        temb = self.time_embedding(t)   #(B, ch*4)
        condition = self.condition_embedding(condition)  #(B, ch, N)
        con_mask = con_mask[:, None, None]    #(B, 1, 1)
        # Downsampling
        h = self.head(x)    #(B, ch, N)
        # h = self.headcrossattn(h, condition)
        hs = [h]
        for layer in self.downblocks:
            h, condition = layer(h, temb, condition, con_mask)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h, condition = layer(h, temb, condition, con_mask)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h, condition = layer(h, temb, condition, con_mask)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 2
    model = UNet_new(
        T=1000, ch=128, condition_num=256, N=512, 
        posdim=126, alpha=100, beta=1000,
        ch_mult=[1, 2, 2, 2], attn=[2], crossattn=[2],
        num_res_blocks=2, dropout=0.1)
    mask_drop = 0.1
    con_mask = torch.bernoulli(torch.zeros(batch_size)+mask_drop)   #取0表示掩码
    con_mask = torch.ones(batch_size) - con_mask
    x = torch.randn(batch_size, 3, 512)        #B, C, N  N表示encoder之后的中心点数量，C表示中心点的特征维度
    condition = torch.randn(batch_size, 128, 256)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t, condition, con_mask)
    print(y.shape)