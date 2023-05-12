'''
modified:
Release Version for FBNet: Feedback network for point cloud completion
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import sys

# proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(proj_dir, "..","utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_ops.pointnet2_utils import furthest_point_sample, grouping_operation, ball_query, three_interpolate
from pointnet2_ops.pointnet2_utils import gather_operation

def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    input:
        ts: (B,)
        t_emb_dim: int
    output: 
        t_emb: (B, t_emb_dim)
    """
    assert t_emb_dim % 2 == 0

    # input is of shape (B) of integer time steps
    # output is of shape (B, t_emb_dim)
    ts = ts.unsqueeze(1)    #(B, 1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device)   #(half_dim,)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)
    
    return t_emb

class Time_fc(nn.Module):
    def __init__(self, t_dim):
        super(Time_fc, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.ReLU(),
            nn.Linear(t_dim, t_dim),
            nn.ReLU(),
        )

    def forward(self, t):
        return self.layer(t)

class Mlp_plus_t_emb(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Mlp_plus_t_emb, self).__init__()

        self.first_mlp = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
            )

        self.second_mlp = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1, 1, 0),
            nn.GroupNorm(32, out_dim),
            nn.ReLU()
        )

        self.res_connect = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, feature, t_emb):
        h = self.first_mlp(feature)
        t_emb = t_emb.unsqueeze(2)   # (B, in_dim, 1)
        h = self.second_mlp(h + t_emb)
        h = h + self.res_connect(feature)

        return h


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    _, idx = sqrdists.topk(nsample, largest=False)
    return idx.int()

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class MLP_CONV(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_CONV, self).__init__()
        self.mlp = nn.Sequential(
           nn.Conv1d(in_dim, out_dim, 1, 1, 0),
           nn.BatchNorm1d(out_dim),
           nn.ReLU(),
           nn.Conv1d(out_dim, out_dim, 1, 1, 0)
        )

    def forward(self, inputs):
        return self.mlp(inputs)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def group_local(xyz, k=20, return_idx=False):
    """
    Input:
        x: point cloud, [B, 3, N]
    Return:
        group_xyz: [B, 3, N, K]
    """
    xyz = xyz.transpose(2, 1).contiguous()
    idx = query_knn_point(k, xyz, xyz)
    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz.permute(0, 3, 1, 2)
    if return_idx:
        return group_xyz, idx

    return group_xyz

class EdgeConv(torch.nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C2, N]
    """

    def __init__(self, input_channel, output_channel, k):
        super(EdgeConv, self).__init__()
        self.num_neigh = k

        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.ReLU(),
            nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.ReLU(),
            nn.Conv2d(output_channel // 2, output_channel, kernel_size=1)
        )

    def forward(self, inputs):
        batch_size, dims, num_points = inputs.shape
        if self.num_neigh is not None:
            neigh_feature = group_local(inputs, k=self.num_neigh).contiguous()
            central_feat = inputs.unsqueeze(dim=3).repeat(1, 1, 1, self.num_neigh)
        else:
            central_feat = torch.zeros(batch_size, dims, num_points, 1).to(inputs.device)
            neigh_feature = inputs.unsqueeze(-1)
        edge_feature = central_feat - neigh_feature
        feature = torch.cat((edge_feature, central_feat), dim=1)
        feature = self.conv(feature)
        central_feature = feature.max(dim=-1, keepdim=False)[0]
        return central_feature

class CrossTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(CrossTransformer, self).__init__()
        self.n_knn = n_knn

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, in_channel, 1)
        )

    def forward(self, pcd, feat, pcd_feadb, feat_feadb):
        """
        Args:
            pcd: (B, N, 3)
            feat: (B, in_channel, N)
            pcd_feadb: (B, N2, 3)
            feat_feadb: (B, in_channel, N2)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        b, num_point, _ = pcd.shape
        pcd = pcd.transpose(-2, -1).contiguous()
        pcd_feadb = pcd_feadb.transpose(-2, -1).contiguous()

        fusion_pcd = torch.cat((pcd, pcd_feadb), dim=2)
        fusion_feat = torch.cat((feat, feat_feadb), dim=2)

        key_point = pcd
        key_feat = feat

        # Preception processing between pcd and fusion_pcd
        key_point_idx = query_knn(self.n_knn, fusion_pcd.transpose(2,1).contiguous(), key_point.transpose(2,1).contiguous(), include_self=True)

        group_point = grouping_operation(fusion_pcd, key_point_idx)
        group_feat = grouping_operation(fusion_feat, key_point_idx)

        
        qk_rel = key_feat.reshape((b, -1, num_point, 1)) - group_feat
        pos_rel = key_point.reshape((b, -1, num_point, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding) # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1) # b, in_channel + 3, n, n_knn

        group_feat = group_feat + pos_embedding  #
        refined_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)
        
        return refined_feat


class Circul(nn.Module):
    def __init__(self):
        """
        """
        super(Circul, self).__init__()

        self.ext = EdgeConv(3, 128, 16)
        self.mlp = MLP_CONV(in_dim=128 * 2, out_dim=128)
        self.t_mlp = Mlp_plus_t_emb(128, 128)

        self.mlp_delta = nn.Sequential(
            EdgeConv(128, 128, 8),
            MLP_CONV(in_dim=128, out_dim=128),
            MLP_CONV(in_dim=128, out_dim=3)
        )

        self.crosstr = CrossTransformer(in_channel=128, dim=64)



    def forward(self, pcd, pcd_pre, feat_pre, t_emb):
        """
        Args:
            pcd: Tensor, (B, N, 3)
            pcd_next: Tensor, (B, N, 3) 
            feat_next: Tensor, (B, 128, N)
            t_emb: (B, 128)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N)
        """
        # Step 1: Feature Extraction
        feat = self.ext(pcd.transpose(-2, -1).contiguous())
        feat = self.mlp(torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1))

        # Step 2: cross attention
        if pcd_pre == None and feat_pre == None:
            feat = self.crosstr(pcd, feat, pcd, feat)
        else:
            feat = self.crosstr(pcd, feat, pcd_pre, feat_pre)
        feat = self.t_mlp(feat, t_emb)

        # Step 3: Coordinate Generation
        delta = self.mlp_delta(feat)
        pcd_child = pcd + delta.transpose(-2, -1).contiguous()

        return pcd_child, feat

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.num_p0 = 512

        self.circuls = nn.Sequential()
        for _ in range(3):
            self.circuls.append(Circul())

        print('#circul steps:{}'.format(len(self.circuls)))

    def forward(self, pcd, t_emb, condition):
        """
        Args:
            pcd: noisy point (B, 512, 3)
            t_emb: (B, 128)
            condition: seed point (B, 256, 3)
        output:
            x: predicted epsilon (B, 512, 3)
        """
        # Initialize input

        pcd_list = []
        feat_list = []
        # Unfolding across time steps
        for i, circul in enumerate(self.circuls):
            if i == 0:
                pcd = pcd.contiguous()  # (B, 512, 3)
                pcd = fps_subsample(torch.cat([pcd, condition], 1), self.num_p0)    # (B, 512, 3)
                pcd, feat = circul(pcd, None, None, t_emb)    #self crosstr  (B, 512, 3)
            else:
                pcd = pcd_list[i-1].contiguous()  # (B, 512, 3)
                pcd = fps_subsample(torch.cat([pcd_list[i-1], condition], 1), self.num_p0)    # (B, 512, 3)
                pcd, feat = circul(pcd, pcd_list[i-1], feat_list[i-1], t_emb)

            pcd_list.append(pcd)
            feat_list.append(feat)

        return pcd


class FBNet(nn.Module):
    def __init__(self):
        super(FBNet, self).__init__()
        self.t_dim = 128
        self.time_fc = Time_fc(self.t_dim)
        self.refine = RefineNet()

    def forward(self, x, ts, condition):
        '''
        input:
            x: noisy point cloud (B, 512 ,3)
            ts: time embedding (B,)
            condition: seed point cloud (B, 256, 3)
        output:
            x: predicted epsilon (B, 512, 3)
        '''
        # feedback refinement stage
        t_emb = calc_t_emb(ts, self.t_dim)
        t_emb = self.time_fc(t_emb)  #(B, 128)
        pcd = self.refine(x, t_emb, condition)

        return pcd

        
if __name__ == '__main__':
    model = FBNet().cuda()
    x = torch.rand(1, 512, 3).cuda()
    ts = torch.randint(100, (1,)).cuda()
    condition = torch.rand(1, 256, 3).cuda()

    out = model(x, ts, condition)
    print(out.shape)