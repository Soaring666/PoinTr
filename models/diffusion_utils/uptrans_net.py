import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from numpy.random import randn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConditionEmbedding(nn.Module):
    def __init__(self, condition_num, input_num):
        super().__init__()
        # self.conditionembedding = nn.Conv1d(dim, dim, 1, 1, 0)
        self.layer = nn.Sequential(
            nn.Linear(condition_num, input_num),
            Swish()
        )
    
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
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c_main = nn.Conv1d(128, 128, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, condition, con_mask):
        x = self.main(x)
        condition = self.c_main(condition)
        return x, condition


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.c_main = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

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
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

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
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        #方便起见，先设置con_dim=tdim
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            nn.GroupNorm(32, tdim//4),
            Swish(),
            nn.Conv1d(tdim//4, out_ch, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, condition, con_mask):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None]
        h += self.cond_proj(condition*con_mask)
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h, condition


class UNet(nn.Module):
    def __init__(self, T, ch, condition_num, N, 
                 ch_mult, attn, num_res_blocks, dropout):
        '''
        T: time steps
        ch: input hidden channel
        condition_num: condition dim(-1), represent as the point number of seed feat
        N: the point number of input (i.e. (B, 3, 512) so N=512)
        ch_mult: channel multiplier
        attn: 
        num_res_blocks: number of residual blocks
        dropout: dropout rate
        '''
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.condition_embedding = ConditionEmbedding(condition_num, N)

        self.head = nn.Conv1d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv1d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, condition, con_mask):
        # Timestep embedding
        temb = self.time_embedding(t)   #(B, ch*4)
        condition = self.condition_embedding(condition)  #(B, ch, N)
        con_mask = con_mask[:, None, None]    #(B, 1, 1)
        # Downsampling
        h = self.head(x)    #(B, ch, N)
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
    model = UNet(
        T=1000, ch=128, condition_num=256, N=512, 
        ch_mult=[1, 2, 2, 2], attn=[2],
        num_res_blocks=2, dropout=0.1)
    
    mask_drop = 0.1
    con_mask = torch.bernoulli(torch.zeros(batch_size)+mask_drop)   #取0表示掩码
    con_mask = torch.ones(batch_size) - con_mask
    x = torch.randn(batch_size, 3, 512)        #B, C, N  N表示encoder之后的中心点数量，C表示中心点的特征维度
    condition = torch.randn(batch_size, 128, 256)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t, condition, con_mask)
    print(y.shape)