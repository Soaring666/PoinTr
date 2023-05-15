from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
# import pointnet2_utils
# from pointnet2_ops.attention import AttentionModule
from pointnet2_ops.attention import GlobalAttentionModule

import copy

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out

def count_to_mask(count, K):
    # counts is of shape (B, npoint)
    # its value range from 0 to K-1
    # return a mask of shape (B, npoint, K)
    mask = torch.arange(K, device=count.device, dtype=count.dtype)
    B, npoint = count.size()
    mask = mask.repeat(B, npoint).view(B, npoint,-1) # shape (B, npoint, K)
    mask = mask < count.unsqueeze(-1)
    return mask

class AttentionModule(nn.Module):
    def __init__(self, c_q, c_k, c_v):
        super(AttentionModule, self).__init__()
        c_q1 = max(c_q, 32)
        c_k1 = max(c_k, 32)
        self.feat_conv = nn.Conv2d(c_q, c_q1, kernel_size=1)
        self.grouped_feat_conv = nn.Conv2d(c_k, c_k1, kernel_size=1)

        self.weight_conv = nn.Sequential(
                    nn.ReLU(inplace=True),
                    MyGroupNorm(32, c_q1+c_k1),
                    nn.Conv2d(c_q1+c_k1, c_v, kernel_size=1),
                    nn.ReLU(inplace=True),
                    MyGroupNorm(32, c_v),
                    nn.Conv2d(c_v, c_v,kernel_size=1))

        self.feat_out_conv = nn.Sequential(
            nn.Conv2d(c_v, c_v, kernel_size=1),
            MyGroupNorm(32, c_v),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, grouped_feat, grouped_feat_out, count):
        # feat (B, c_q, N), acts like query
        # grouped_feat (B, c_k, N, K), acts like key
        # grouped_feat_out (B, c_v, N, K) # acts like value
        # count is of shape (B,N)
        K = grouped_feat.shape[-1]
        feat1 = self.feat_conv(feat.unsqueeze(-1)) # (B, c_q1, N, 1)
        feat1 = feat1.expand(-1,-1,-1,K) # (B, c_q1, N, K)

        grouped_feat1 = self.grouped_feat_conv(grouped_feat) # (B, c_k1, N, K)

        total_feat = torch.cat([feat1, grouped_feat1], dim=1) # (B, c_q1+c_k1, N, K)
        scores = self.weight_conv(total_feat) # (B, c_v, N, K)

        if not count == 'all':
            count = torch.clamp(count, min=1)
            mask = count_to_mask(count, K) # (B,N,K)
            mask = mask.unsqueeze(1).float() # (B,1, N,K)
            scores = scores * mask + (-1e9)*(1-mask)

        weight = F.softmax(scores, dim=-1) # (B, c_v, N, K)
        # pdb.set_trace()
        grouped_feat_out = self.feat_out_conv(grouped_feat_out)  # (B, c_v, N, K)
        out = grouped_feat_out * weight  # (B, c_v, N, K)
        out = out.sum(dim=-1)  # (B, c_v, N)
        return out

class Cross_Attn(nn.Module):
    def __init__(self, c_q, c_k, c_v, c_out):
        super(Cross_Attn, self).__init__()
        self.feat_conv = nn.Conv2d(c_q, c_out, kernel_size=1)
        self.grouped_feat_conv = nn.Conv2d(c_k, c_out, kernel_size=1)
        self.scale = c_out ** -0.5

        self.feat_out_conv = nn.Sequential(
            nn.Conv2d(c_v, c_out, kernel_size=1),
            MyGroupNorm(32, c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, grouped_feat, count):
        # feat (B, c_q, N), acts like query
        # grouped_feat (B, c_k, N, K), acts like key and value
        # count is of shape (B,N)
        K = grouped_feat.shape[-1]
        feat1 = self.feat_conv(feat.unsqueeze(-1)) # (B, c_out, N, 1)
        feat1 = feat1.expand(-1,-1,-1,K) # (B, c_out, N, K)
        feat1 = feat1.permute(0,2,3,1) # (B, N, K, c_out)

        grouped_feat_k = self.grouped_feat_conv(grouped_feat).transpose(1, 2) # (B, N, c_out, K)

        attn = torch.matmul(feat1, grouped_feat_k) * self.scale

        if not count == 'all':
            count = torch.clamp(count, min=1)
            mask = count_to_mask(count, K) # (B, N, K)
            mask = mask.unsqueeze(1).float().transpose(1, 2) # (B, N, 1, K)
            attn = attn * mask + (-1e9)*(1-mask)

        attn = attn.softmax(dim=-1) #(B, N, K, K)

        grouped_feat_v = self.feat_out_conv(grouped_feat)  # (B, c_out, N, K)
        grouped_feat_v = grouped_feat_v.permute(0,2,3,1)  # (B, N, K, c_out)
        out = (attn @ grouped_feat_v).transpose(-2, -1) #(B, N, c_out, K)
        out = out.sum(dim=-1).transpose(-2, -1)  # (B, c_out, N)
        return out


class Mlp_plus_t_emb(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Mlp_plus_t_emb, self).__init__()

        self.t_fc = nn.Linear(512, hidden_dim)
        self.condition_fc = nn.Linear(1024, hidden_dim)

        self.first_mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU()
            )
        self.second_mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )
        self.last_mlp = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )
        self.res_connect = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, feature, t_emb=None, condition_emb=None):
        h = self.first_mlp(feature)
        if t_emb is not None:
            t_emb = self.t_fc(t_emb).unsqueeze(2).unsqueeze(3) # shape (B, out_dim, 1, 1)
            h = self.second_mlp(h + t_emb)
            condition_emb = self.condition_fc(condition_emb).unsqueeze(2).unsqueeze(3) # shape (B, out_dim, 1, 1) 
            h = self.last_mlp(h + condition_emb)
            h = h + self.res_connect(feature)
        else:
            h = self.second_mlp(h)
            h = self.last_mlp(h)
            h = h + self.res_connect(feature)

        return h

class Conv1d_plus_t_emb(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv1d_plus_t_emb, self).__init__()

        self.t_fc = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, in_dim),
            nn.ReLU()
        ) 

        self.condition_fc = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )

        self.first_mlp = nn.Sequential(
            nn.Conv1d(2*in_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )
        self.second_mlp = nn.Sequential(
            nn.Conv1d(2*out_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )
        self.res_connect = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )

    def forward(self, feature, t_emb, condition_emb):
        h = self.feature_mlp(feature)     #(B, out_dim, N)
        t_emb = self.t_fc(t_emb).unsqueeze(2).repeat(1, 1, feature.shape[2]) # shape (B, in_dim, N)
        h = self.first_mlp(torch.cat([feature, t_emb], 1))  #(B, out_dim, N)
        condition_emb = self.condition_fc(condition_emb).unsqueeze(2).repeat(1, 1, feature.shape[2]) # shape (B, out_dim, N) 
        h = self.second_mlp(torch.cat([h, condition_emb], 1))
        h = h + self.res_connect(feature)   #(B, out_dim, N)

        return h
class Global_cond(nn.Module):
    def __init__(self):
        super(Global_cond, self).__init__()
        self.mlp1 = Mlp_plus_t_emb(128, 128, 256)
        self.mlp2 = Mlp_plus_t_emb(512, 512, 1024)
    def forward(self, x):
        feature = self.mlp1(x.unsqueeze(-1))      #(B, 256, num_points, 1)
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])   #(B, 256, 1, 1)
        global_feature = global_feature.expand(-1,-1,feature.size(2),-1)  #(B, 256, num_points, 1)
        feature = torch.cat([feature, global_feature], dim=1)   #(B, 512, num_points, 1)

        feature = self.mlp2(feature)       #(B, 1024, num_points, 1)
        global_feature = F.max_pool2d(feature, kernel_size=[feature.size(2), 1])   #(B, 1024, 1, 1)
        global_feature = global_feature.squeeze(-1).squeeze(-1)     #(B, 1024)
        return global_feature

class Ball_query(nn.Module):
    '''
    暂时先用论文中给出的ballquery聚合特征的方法
    '''
    def __init__(self):
        super(Ball_query, self).__init__()

    def forward(self):
        pass


class _PointnetSAModuleBase(nn.Module):
    # set abstraction module, down sampling
    def __init__(self, npoint, radius, nsample, c_hidden, c_fps, c_group, c_out):
        super(_PointnetSAModuleBase, self).__init__()
        self.c_group = c_out + 9
        self.npoint = npoint
        self.conv_emb = Conv1d_plus_t_emb(c_fps, c_out) 
        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample)
        # self.mlp = Mlp_plus_t_emb(c_group, c_hidden, c_out)
        self.attention_module = Cross_Attn(c_out, self.c_group, self.c_group, c_out)

    def forward(self, xyz, features, t_emb=None, condition_emb=None):
        '''
        input:
            xyz : (B, N, 3)
            features : (B, C, N)
            t_emb : time step embedding (B, t_dim)
            condition_emb : global condition feature (B, 1024)
        output:
            new_xyz : torch.Tensor
                (B, npoint, 3) tensor of the new features' xyz
            new_features : torch.Tensor
                (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        '''

        # reduce the number of points per shape from N to npoint
        # each new point in the npoint points have nsample neighbors in the original N points

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous() # shape (B,3,N)
        
        assert self.npoint is not None
        furthest_point_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, furthest_point_idx).transpose(1, 2).contiguous() #(B, npoint, 3)

        #add t_emb and condition_emb on features
        features = self.conv_emb(features, t_emb=t_emb, condition_emb=condition_emb)   #(B, c_out, N)
        new_xyz_feat = pointnet2_utils.gather_operation(features.contiguous(), furthest_point_idx)   #(B, c_out, npoint)

        #ball query
        grouped_features, count = self.grouper(xyz, new_xyz, features)  #ball query (B, c_out+9, npoint, nsample)
        
        # if t_emb is not None:
        #     out_features = self.mlp(grouped_features, t_emb=t_emb, condition_emb=condition_emb)     #out_feature (B, c_out, npoint, nsample)
        # else:
        #     out_features = self.mlp(grouped_features)
        
        new_features = self.attention_module(new_xyz_feat, grouped_features, count)  #(B, c_out, npoint)

        new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class FeatureMapModule(nn.Module):
    def __init__(self, radius, nsample, c_noisy, c_group, c_out):
        super(FeatureMapModule, self).__init__()

        self.mlp = Mlp_plus_t_emb(c_group, c_out, c_out)
        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample)
        self.use_attention_module = True
        self.attention_module = AttentionModule(c_noisy, c_group, c_out)

    def forward(self, xyz, features, noisy_xyz, noisy_feature):
        # xyz (B,N,3), features (B,C,N), new_xyz (B, npoint, 3)
        grouped_features, count = self.grouper(xyz, noisy_xyz, features, subset=False) # (B, C+9, npoint, nsample)
        out_features = self.mlp(grouped_features) # (B, c_out, npoint, nsample)

        max_features = self.attention_module(noisy_feature, grouped_features, out_features, count)  #(B, c_out, npoint)

        return max_features


class PointnetKnnFPModule(nn.Module):
    r"""Propigates the features of one set to another
    """
    def __init__(self, nsample, c_now, c_up, c_out):
        super(PointnetKnnFPModule, self).__init__()
        self.nsample = nsample
        self.c_group = c_now + 11
        self.conv_now_emb = Conv1d_plus_t_emb(c_now, c_now)
        self.conv_up_emb = Conv1d_plus_t_emb(c_up, c_out)

        self.attention_module = Cross_Attn(c_out, self.c_group, self.c_group, c_out)

        self.mlp = nn.Sequential(
            nn.Conv1d(2*c_out, c_out, 1, 1, 0),
            nn.GroupNorm(32, c_out),
            nn.ReLU(),           
            nn.Conv1d(c_out, c_out, 1, 1, 0),
            nn.GroupNorm(32, c_out),
            nn.ReLU()           
        )

    def forward(self, up_xyz, now_xyz, up_feats, now_feats, t_emb=None, condition_emb=None):
        r"""
        Parameters
        ----------
        up_xyz : torch.Tensor
            (B, N, 3) uplayer xyz positions
        now_xyz : torch.Tensor
            (B, M, 3) nowlayer xyz positions
        up_feats : torch.Tensor
            (B, c_up, N) uplayer features
        now_feats : torch.Tensor
            (B, c_now, M) nowlayer features
        
        Return
        ----------
        new_features : torch.Tensor
            (B, c_out, N) uplayer features
        """
        #add emb to feats
        now_feats = self.conv_now_emb(now_feats, t_emb, condition_emb)  #(B, c_now, M)
        up_feats = self.conv_up_emb(up_feats, t_emb, condition_emb)  #(B, c_out, N)

        #use knn to group features
        grouped_feats = pointnet2_utils.group_knn(up_xyz, now_xyz, now_feats, self.nsample)  #(B, c_now+11, N, K)

        #use attn to gather features
        interpolated_feats = self.attention_module(up_feats, grouped_feats, count='all')  #(B, c_out, N)

        #use mlp to get new features       
        new_features = self.mlp(torch.cat([interpolated_feats, up_feats], dim=1))  #(B, c_out, N)

        return new_features



