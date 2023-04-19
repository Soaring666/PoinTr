import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL2_split
from pointnn import Point_PN

@MODELS.register_module()
class FoldingNet_AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_pred = config.num_pred
        self.encoder_channel = config.encoder_channel
        self.grid_size = int(pow(self.num_pred,0.5) + 0.5)

        self.pointNN_encoder = Point_PN()

        '''
        #原始encoder方案
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )
        '''

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N
        self.build_loss_func()

    def build_loss_func(self):
        
        self.loss_func = ChamferDistanceL2()
        # self.loss_func = ChamferDistanceL2_split()

    def get_loss(self, ret, gt, epoch):

        '''
        #lambda_loss increase with epoch increase
        lambda_loss = 1 * (epoch / 50) + 1
        loss_coarse_dist1, loss_coarse_dist2 = self.loss_func(ret[0], gt)
        loss_fine_dist1, loss_fine_dist2 = self.loss_func(ret[1], gt)
        loss_coarse = lambda_loss * loss_coarse_dist1 + loss_coarse_dist2
        loss_fine = lambda_loss * loss_fine_dist1 + loss_fine_dist2
        '''
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine
        
    
    def encoder(self, x):
        bs , n , _ = x.shape

        feature_global = self.pointNN_encoder(x.transpose(-2, -1))  #(B, 288)
        
        return feature_global

        '''
        #原始encoder
        feature = self.first_conv(x.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        '''

    def forward(self, xyz):
        # folding decoder
        feature_global = self.encoder(xyz)
        fd1, fd2 = self.decoder(feature_global) # B N 3
        return (fd1, fd2) # FoldingNet producing final result directly
        
    def decoder(self,x):
        num_sample = self.grid_size * self.grid_size
        bs = x.size(0)
        features = x.view(bs, self.encoder_channel, 1).expand(bs, self.encoder_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd1.transpose(2,1).contiguous() , fd2.transpose(2,1).contiguous()
        # return fd2.transpose(2,1).contiguous()