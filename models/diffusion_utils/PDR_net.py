import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
from pointnet2.models.pnet import Pnet2Stage

from .PDR_net_utils import Global_cond, _PointnetSAModuleBase, FeatureMapModule, PointnetKnnFPModule


class Time_fc(nn.Module):
    def __init__(self, t_dim):
        super(Time_fc, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(t_dim, 4*t_dim),
            nn.ReLU(),
            nn.Linear(4*t_dim, 4*t_dim),
            nn.ReLU(),
        )

    def forward(self, t):
        return self.layer(t)

class PointNet2CloudCondition(nn.Module):
    def __init__(self):
        super(PointNet2CloudCondition, self).__init__()
        self.global_pnet = Global_cond()

        self.t_dim = 128
        self.time_fc = Time_fc(self.t_dim)
        
        # build SA module for condition point cloud
        npoint_condition = [128, 64, 16]
        radius_condition = [0.1, 0.4, 0.8]
        nsample_condition = [32, 32, 32]
        c_fps = [128, 256, 256]
        c_group = [137, 265, 265]
        c_out = [256, 256, 512]
        self.SA_modules_condition = nn.Sequential()
        for i in range(len(npoint_condition)):
            self.SA_modules_condition.append(_PointnetSAModuleBase(
                npoint=npoint_condition[i],
                radius=radius_condition[i],nsample=nsample_condition[i],
                c_hidden=c_out[i], c_fps=c_fps[i], 
                c_group=c_group[i], c_out=c_out[i]
            ))
        

        # build feature transfer modules from condition point cloud to the noisy point cloud x_t at encoder
        radius_fp = [0.1, 0.2, 0.4, 0.8, 0.4, 0.2, 0.1]
        nsample_fp = [32, 32, 32, 32, 32, 32, 32]
        c_noisy = [32, 64, 256, 512, 256, 256, 128]
        c_group = [137, 265, 265, 521, 265, 265, 137]
        c_out = [32, 64, 256,  512, 256, 256, 128]
        self.encoder_feature_map = nn.Sequential()
        for i in range(len(radius_fp)):
            self.encoder_feature_map.append(FeatureMapModule(
                radius=radius_fp[i], nsample=nsample_fp[i],
                c_noisy=c_noisy[i], c_group=c_group[i], c_out=c_out[i]
            ))
    
        # build SA module for the noisy point cloud x_t
        npoint_noisy = [256, 128, 64, 16]
        radius_noisy = [0.1, 0.2, 0.4, 0.8]
        nsample_noisy = [32, 32, 32, 32]
        c_fps = [3, 64, 128, 512]
        c_group = [12, 73, 137, 521]
        c_out = [32, 64, 256, 512]
        self.SA_modules = nn.Sequential()
        for i in range(len(npoint_noisy)):
            self.SA_modules.append(_PointnetSAModuleBase(
                npoint=npoint_noisy[i],
                radius=radius_noisy[i],nsample=nsample_noisy[i],
                c_hidden=c_out[i], c_fps=c_fps[i], 
                c_group=c_group[i], c_out=c_out[i]
            ))
            

        # build FP module for condition cloud
        nsample_fp_cond = [32, 32, 32]
        c_up = [256, 256, 128]
        c_group = [523, 267, 267]
        c_out = [256, 256, 128]
        self.FP_modules_condition = nn.Sequential()
        for i in range(len(nsample_fp_cond)):
            self.FP_modules_condition.append(PointnetKnnFPModule(
                nsample=nsample_fp_cond[i], 
                c_up=c_up[i], c_group=c_group[i], c_out=c_out[i]
            ))   

        # build FP module for noisy point cloud x_t

        nsample_fp = [32, 32, 32, 32]
        c_up = [256, 64, 32, 3]
        c_group = [1035, 523, 523, 267]
        c_out = [256, 256, 128, 128]
        self.FP_modules = nn.Sequential()
        for i in range(len(nsample_fp)):
            self.FP_modules.append(PointnetKnnFPModule(
                nsample=nsample_fp[i],
                c_up=c_up[i], c_group=c_group[i], c_out=c_out[i]
            ))   


        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, kernel_size=1),
            )



    def forward(self, pc, ts, cond_xyz, cond_feat):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pc: input point cloud (B, 512, 3)
            cond_xyz: seed point cloud coordinate (B, 256, 3)
            cond_feat: seed point cloud feature (B, 128, 256)
            ts: diffusion steps (B,)
        """
        features = copy.deepcopy(pc).transpose(-2, -1)
        
        #calc time step embedding
        t_emb = calc_t_emb(ts, self.t_dim)
        t_emb = self.time_fc(t_emb)  #(B, 512)

        #get global feature from condition point
        global_feature = self.global_pnet(cond_feat)  #(B, 1024)
        condition_emb = global_feature

        l_uvw, l_cond_features = [cond_xyz], [cond_feat]
        l_xyz, l_features = [pc], [features]

        li_xyz, li_features = self.SA_modules[0](l_xyz[0], l_features[0], t_emb=t_emb, condition_emb=condition_emb)
        l_xyz.append(li_xyz)
        l_features.append(li_features)

        for i in range(len(self.SA_modules_condition)):

            #condition point cloud pointnet
            li_uvw, li_cond_features = self.SA_modules_condition[i](l_uvw[i], l_cond_features[i])
            l_uvw.append(li_uvw)
            l_cond_features.append(li_cond_features)

            #feature transfer from condition point to noisy point
            mapped_feature = self.encoder_feature_map[i](l_uvw[i], l_cond_features[i], l_xyz[i+1], l_features[i+1])
            input_feature = torch.cat([mapped_feature, l_features[i+1]], dim=1)

            #noisy point cloud pointnet
            li_xyz, li_features = self.SA_modules[i+1](l_xyz[i+1], input_feature, t_emb=t_emb, condition_emb=condition_emb)
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(len(self.FP_modules_condition)):
            #condition feature fp
            l_cond_features[-i-2] = self.FP_modules_condition[i](
                                        l_uvw[-i-2], l_uvw[-i-1], l_cond_features[-i-2], l_cond_features[-i-1],
                                        t_emb = None, condition_emb=None)
            
            #feature transfer from condition point to noisy point
            mapped_feature = self.encoder_feature_map[i+3](l_uvw[-i-1], l_cond_features[-i-1], l_xyz[-i-1], l_features[-i-1]) 
            input_feature = torch.cat([mapped_feature, l_features[-i-1]], dim=1)

            #noisy feature fp
            l_features[-i-2] = self.FP_modules[i](
                l_xyz[-i-2], l_xyz[-i-1], l_features[-i-2], input_feature,
                t_emb = t_emb, condition_emb=condition_emb)
        
        #fp l_cond_features[0] 
        mapped_feature = self.encoder_feature_map[-1](l_uvw[0], l_cond_features[0], l_xyz[1], l_features[1])
        input_feature = torch.cat([ mapped_feature,l_features[1] ], dim=1)    #(B, 256, 256)

        out_feature = self.FP_modules[-1](
            l_xyz[0], l_xyz[1], l_features[0], input_feature,
            t_emb=t_emb, condition_emb=condition_emb)   #(B, 128, 512)

        
        out = self.fc_lyaer(out_feature)
        out = torch.transpose(out, 1,2)

        return out  

    def report_feature_map_neighbor_stats(self, FM_module, module_name='FM_module'):
        with torch.no_grad():
            # num_groupers_per_layer = len(SA_module[0].groupers)
            neigh_stats = []
            neigh_quantile = []
            for i in range(len(FM_module)):
                neigh_stats.append(FM_module[i].mapper.neighbor_stats)
                neigh_quantile.append(FM_module[i].mapper.neighbor_num_quantile)
            
            neigh_stats = torch.stack(neigh_stats, dim=0)
            neigh_quantile = torch.stack(neigh_quantile, dim=0)

        print('%s: neighbor number (min, mean, max)' % (module_name))
        print(neigh_stats)
        print('%s: neighbor quantile (0-0.1-1)' % (module_name))
        print(neigh_quantile)

    def report_neighbor_stats(self):
        if not self.record_neighbor_stats:
            print('neighbor stats is not recorded')
            return
        self.report_SA_module_neighbor_stats(self.SA_modules, module_name='Input cloud SA_module')
        if self.include_local_feature:
            self.report_SA_module_neighbor_stats(self.SA_modules_condition, module_name='Condition cloud SA_module')

        self.report_FP_module_neighbor_stats(self.FP_modules, module_name='Input cloud FP_module')
        if self.include_local_feature:
            self.report_FP_module_neighbor_stats(self.FP_modules_condition, module_name='Condition cloud FP_module')

        if self.include_local_feature:
            self.report_feature_map_neighbor_stats(self.encoder_feature_map, module_name='Encoder feature mapper')
            self.report_feature_map_neighbor_stats(self.decoder_feature_map, module_name='Decoder feature mapper')
        # self.report_feature_map_neighbor_stats([self.last_map], module_name='Last mapper')


if __name__ == '__main__':
    net = PointNet2CloudCondition()
    net = net.cuda()
    pc = torch.randn(3, 512, 3).cuda()
    cond_xyz = torch.randn(3, 256, 3).cuda()
    cond_feat = torch.randn(3, 128, 256).cuda()
    ts = torch.tensor([1, 2, 3]).cuda()
    out = net(pc, ts, cond_xyz, cond_feat)
    print(out.shape)
    
