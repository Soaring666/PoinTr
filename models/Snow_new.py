#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
import torch.nn as nn
from torch import nn, einsum
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_PM
from .SnowFlakeNet_utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer, MLP_Res, grouping_operation, query_knn
from .build import MODELS
from snow_utils.utils import Label_emb, Label_mlp

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

        self.label_mlp1 = Label_mlp(128) 
        self.label_mlp2 = Label_mlp(256)
        self.label_mlp3 = Label_mlp(512)

    def forward(self, point_cloud, label_emb):
        """
        Args:
             point_cloud: b, 3, n
             label_emb: b, 128
        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.label_mlp1(label_emb, l1_points)   #(B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.label_mlp2(label_emb, l2_points)   #(B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 512, 1)
        l3_points = self.label_mlp3(label_emb, l3_points)   #(B, 512, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 3 + dim_feat, layer_dims=[256, 128])

        self.mlp_label = Label_mlp(128)

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, label_emb, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            label_emb: (B, 128)
            K_prev: Tensor, (B, 128, N_prev)
        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2)),
                            label_emb.unsqueeze(2).repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)
        Q = self.mlp_label(label_emb, Q)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial, label_emb, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
            label_emb: (b, 128)
        output:
            arr_pcd: [
                pcd1: coarse point (B, 256, 3),
                pcd2: (B, 512, 3),
                pcd3: (B, 2048, 3),
                pcd4: (B, 16384, 3)
            ]
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, label_emb, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd

@MODELS.register_module()
class Snow_new(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super().__init__()
        dim_feat = config.dim_feat
        num_pc = config.num_pc
        num_p0 = config.num_p0
        radius = config.radius
        up_factors = config.up_factors
        # dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):

        self.feat_extractor = FeatureExtractor(out_dim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors)
        self.build_loss_func()

        self.labelemb = Label_emb()

    def build_loss_func(self):
        self.loss_func_CD = ChamferDistanceL1()
        self.loss_func_PM = ChamferDistanceL1_PM()  #单边cd距离

    def get_loss(self, recon, partial, gt, epoch=1,**kwargs):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3, partial]
        """

        Pc, P1, P2, P3 = recon

        gt_2 = fps(gt, P2.shape[1])
        gt_1 = fps(gt_2, P1.shape[1])
        gt_c = fps(gt_1, Pc.shape[1])

        cdc = self.loss_func_CD(Pc, gt_c)
        cd1 = self.loss_func_CD(P1, gt_1)
        cd2 = self.loss_func_CD(P2, gt_2)
        cd3 = self.loss_func_CD(P3, gt)

        partial_matching = self.loss_func_PM(partial, P3)

        loss_sum = (cdc + cd1 + cd2 + cd3 + partial_matching)
        loss_list = [cdc, cd1, cd2, cd3, partial_matching]

        return loss_sum, loss_list, [gt_c, gt_1, gt_2, gt]

    def forward(self, point_cloud, label, return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
            lable: (B,)
        output:
            if training:
                out:[
                    pcd1: (B, 256, 3),
                    pcd2: (B, 512, 3),
                    pcd3: (B, 2048, 3),
                    pcd4: (B, 16384, 3),
                ]
        """

        ####lable embedding
        label_emb = self.labelemb(label)    #(B, 128)

        pcd_bnc = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        feat = self.feat_extractor(point_cloud, label_emb)     #(B, 512, 1)
        out = self.decoder(feat, pcd_bnc, label_emb, return_P0=return_P0)

        return out