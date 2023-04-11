import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc


class GraphLayer(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        x = x.transpose(-2, -1)     # B, N, C
        knn_idx = knn_point(self.k, x, x)   # B, N, k  
        # knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x, knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, C, N)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    """
    Graph based encoder.
    只设置坐标位置单个特征进行变换
    input:
        x : the coordinate of points, (b, c, n)
    output:
        x: the global feature (b, 512)
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.k = 16
        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        b, n, c = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn_point(self.k, x, x)   # B, N, k=16  
        # knn_idx = knn(x, k=16)
        knn_x = index_points(x, knn_idx)  # (b, n, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)   # (B, N, 1, 3)
        knn_x = knn_x - mean      # (b, n, 16, 3)
        #folding net 中的协方差, (b, 9, n)
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x.transpose(-2, -1), covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, N)


        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))  # (B, 512, N)

        x = torch.max(x, dim=-1)[0]  #(B, 512)
        return x


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    input:
        grids: initial grid, (b, 2, 45*45)
        codewords: global featurem, (b, 512, 45*45)
    output:
        recon: recon points, (b, 3, 45*45)
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    input:
        x: global feature, (b, 512)
    output:
        recon: recon points, (b, 3, 45*45)
    """

    def __init__(self, in_channel=512):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (b, 512)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2


class FoldingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_func = ChamferDistanceL1()
    
    def get_loss(self, x, gt):
        loss = self.loss_func(x, gt)
        return loss

    def forward(self, x):
        global_feature = self.encoder(x)   #(b, 512)
        x = self.decoder(global_feature)   #(b, 3, 45*45)
        return x
