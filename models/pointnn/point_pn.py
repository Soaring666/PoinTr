# Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

from .model_utils import *


class Downsample(nn.Module):
    def __init__(self, center_num):
        super().__init__()
        self.center_num = center_num
    
    def forward(self, xyz):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.center_num).long()
        center_xyz = index_points(xyz, fps_idx)

        return center_xyz


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, type):
        super().__init__()
        self.type = type
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)


    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalization
        if self.type == 'mn40':
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == 'scan':
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)    #(B, 512, 40, 72)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K)    #(B, 36, 512, 40)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)    #(B, 36, 512, 40)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x
    

# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
    

# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
   
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

    
# Parametric Encoder
class EncP(nn.Module):  
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]    #72, 144, 144, 288
            group_num = group_num // 2          #512, 256, 128, 64
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, LGA_block[i], dim_expansion[i], type))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        #xyz (B, 1024, 3)
        #x (B, 3, 1024)
        x = self.raw_point_embed(x)     #(B, 36, 1024)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN  xyz:center coordinate(B, 512, 3), lc_x:center feature(B, 512, 36), knn_xyz:(B, 512, 40, 3)
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)   #(B, 72, 512), (B, 144, 256), (B, 288, 128), (B,576, 64)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)   #(B, 288)
        return x


# Parametric Network for ModelNet40
class Point_PN(nn.Module):
    def __init__(self, in_channels=3, input_points=2048, num_stages=4, embed_dim=36, k_neighbors=40,
                 beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,2], type='mn40'):
        super().__init__()
        # Parametric Encoder

        self.downsample = Downsample(input_points)

        self.EncP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type)
        


    def forward(self, x):
        '''
        input:
            x: (B, 3, N)
        hiddens
            xyz: regard as coodinates
            x: regard as features
        output:
            x: global feature
        '''
        xyz = x.permute(0, 2, 1)    #(B, N, 3)
        xyz = self.downsample(xyz)

        # Parametric Encoder
        x = self.EncP(xyz, x)   #(B, 288)

        return x
    


